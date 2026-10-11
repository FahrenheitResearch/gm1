"""The sounding a CM1 namelist runs on: read and checked as CM1 r21.1 reads it.

``isnd = 5`` is base.F's analytic Weisman-Klemp profile and needs no file.
``isnd = 7`` and ``17`` read ``input_sounding`` (base.F:495-607): one surface
record -- surface pressure in hPa, potential temperature in K, water-vapour
mixing ratio in g/kg -- then one record per level: height in m, potential
temperature, mixing ratio and, for ``isnd = 7``, the two wind components.
SounderPy's ``to_file('cm1')`` writes this layout.

:func:`check_sounding` reads a file the way CM1's list-directed reads do and
names every way CM1 would stop on it or read it wrongly:

* the surface record is not three numbers, or a level record is short
  (base.F:546-557 stops on a read error);
* fewer than two levels above the surface (base.F:569-576 stops; ``isnd = 7``
  also extrapolates the surface wind from the first two levels, base.F:617-618);
* heights that do not rise from above the ground (base.F:686-690 brackets each
  model level by the first sounding height above it, so a height that does
  not rise interpolates between the wrong records);
* a top below the model's highest scalar level.  CM1 r21.1 itself stops there
  (base.F:678-683, "zmax of sounding < zmax of grid"), and GM1 refuses the
  same file before the GPU is opened (``cartesian_door.require_sounding_extent``).

Values CM1 would read without stopping but that look like a unit slip (a
surface pressure in Pa, a mixing ratio in kg/kg) are warnings.

:func:`environment` is the t=0 environment block (``getcape.F``'s parcels, the
bulk shear, Bunkers motion and helicity) from the Rust initializer library
(``sounding_report.initial_sounding_report``), when it is installed.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

#: The sounding branches that read ``input_sounding``.
FILE_SOUNDINGS = frozenset({7, 17})


@dataclass(frozen=True)
class SoundingProblem:
    """One way CM1 r21.1 would stop on this file or read it wrongly."""

    code: str
    text: str
    fix: str
    line: int | None = None

    def as_dict(self) -> dict[str, Any]:
        row = {"code": self.code, "text": self.text, "fix": self.fix}
        if self.line is not None:
            row["line"] = self.line
        return row


@dataclass(frozen=True)
class SoundingCheck:
    isnd: int
    path: str | None
    sha256: str | None
    surface: Mapping[str, float] | None
    heights_m: tuple[float, ...]
    highest_mass_level_m: float | None
    problems: tuple[SoundingProblem, ...]
    warnings: tuple[str, ...] = ()
    environment: Mapping[str, Any] | None = None
    environment_status: str = "not-computed"
    environment_reason: str | None = None

    @property
    def reads_file(self) -> bool:
        return self.isnd in FILE_SOUNDINGS

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "isnd": self.isnd,
            "reads_file": self.reads_file,
            "path": self.path,
            "sha256": self.sha256,
            "levels": len(self.heights_m),
            "surface": dict(self.surface) if self.surface else None,
            "top_m": self.heights_m[-1] if self.heights_m else None,
            "first_level_m": self.heights_m[0] if self.heights_m else None,
            "highest_model_scalar_level_m": self.highest_mass_level_m,
            "problems": [problem.as_dict() for problem in self.problems],
            "warnings": list(self.warnings),
            "environment_status": self.environment_status,
        }
        if self.environment_reason:
            payload["environment_reason"] = self.environment_reason
        if self.environment is not None:
            payload["environment"] = dict(self.environment)
        return payload


def _number(token: str) -> float:
    # Fortran list-directed input takes a D exponent as well as an E.
    return float(token.replace("d", "e").replace("D", "e"))


def _records(text: str) -> list[tuple[int, list[str]]]:
    rows = []
    for number, line in enumerate(text.splitlines(), start=1):
        tokens = line.replace(",", " ").split()
        if tokens:
            rows.append((number, tokens))
    return rows


def highest_mass_level_m(effective: Mapping[str, Any]) -> float | None:
    """zh(nk) of the namelist's own vertical coordinate, or None if it does not build."""

    from ..cm1_vertical_grid import VerticalGridError, build_vertical_levels

    try:
        stretch_z = int(effective.get("stretch_z", 0))
        nz = int(effective["nz"])
        dz = float(effective["dz"])
        if stretch_z == 0:
            levels = build_vertical_levels(stretch_z=0, nk=nz, dz_m=dz, ztop_m=nz * dz)
        elif stretch_z == 1:
            levels = build_vertical_levels(
                stretch_z=1, nk=nz, dz_m=dz, ztop_m=float(effective["ztop"]),
                str_bot_m=float(effective["str_bot"]), str_top_m=float(effective["str_top"]),
                dz_bot_m=float(effective["dz_bot"]), dz_top_m=float(effective["dz_top"]),
            )
        else:
            return None
        return float(levels.highest_mass_level_m)
    except (KeyError, TypeError, ValueError, VerticalGridError):
        return None


def check_sounding(path: str | Path | None, effective: Mapping[str, Any]) -> SoundingCheck:
    """Read ``path`` as CM1 r21.1's base.F would for this namelist's ``isnd``."""

    isnd = int(effective.get("isnd", 0))
    top = highest_mass_level_m(effective)
    if isnd not in FILE_SOUNDINGS:
        return SoundingCheck(isnd=isnd, path=None, sha256=None, surface=None, heights_m=(),
                             highest_mass_level_m=top, problems=())
    if path is None:
        return SoundingCheck(
            isnd=isnd, path=None, sha256=None, surface=None, heights_m=(), highest_mass_level_m=top,
            problems=(SoundingProblem(
                code="CM1-IMPORT-SOUNDING-MISSING",
                text=(
                    f"isnd = {isnd} reads the input_sounding file (base.F:495-607), and none was "
                    "given or found beside the namelist"
                ),
                fix="pass --sounding with the input_sounding file this namelist runs on",
            ),),
        )
    source = Path(path)
    try:
        payload = source.read_bytes()
    except OSError as error:
        return SoundingCheck(
            isnd=isnd, path=str(source), sha256=None, surface=None, heights_m=(), highest_mass_level_m=top,
            problems=(SoundingProblem(
                code="CM1-IMPORT-SOUNDING-UNREADABLE",
                text=f"the input_sounding cannot be read: {error}",
                fix="pass --sounding with a readable file",
            ),),
        )
    digest = hashlib.sha256(payload).hexdigest()
    problems: list[SoundingProblem] = []
    warnings: list[str] = []
    try:
        text = payload.decode("utf-8")
    except UnicodeError as error:
        return SoundingCheck(
            isnd=isnd, path=str(source), sha256=digest, surface=None, heights_m=(), highest_mass_level_m=top,
            problems=(SoundingProblem(
                code="CM1-IMPORT-SOUNDING-FORMAT",
                text=f"the input_sounding is not text: {error}",
                fix="supply the plain-text input_sounding CM1 reads",
            ),),
        )
    records = _records(text)
    per_level = 5 if isnd == 7 else 3
    names = "height, theta, qv, u, v" if isnd == 7 else "height, theta, qv"
    surface: dict[str, float] | None = None
    heights: list[float] = []
    thetas: list[float] = []
    qvs: list[float] = []
    if not records:
        problems.append(SoundingProblem(
            code="CM1-IMPORT-SOUNDING-FORMAT",
            text="the input_sounding is empty",
            fix="supply the input_sounding this namelist runs on",
        ))
    else:
        number, tokens = records[0]
        try:
            values = [_number(token) for token in tokens[:3]]
            if len(values) < 3:
                raise ValueError("fewer than three values")
            surface = {"pressure_hpa": values[0], "theta_k": values[1], "qv_g_per_kg": values[2]}
        except ValueError:
            problems.append(SoundingProblem(
                code="CM1-IMPORT-SOUNDING-FORMAT",
                text=(
                    f"line {number}, the surface record, is not three numbers (surface pressure in "
                    "hPa, potential temperature in K, mixing ratio in g/kg); CM1 stops reading it "
                    "(base.F:546)"
                ),
                fix="start the file with: surface pressure (hPa), theta (K), qv (g/kg)",
                line=number,
            ))
        for number, tokens in records[1:]:
            try:
                values = [_number(token) for token in tokens[:per_level]]
                if len(values) < per_level:
                    raise ValueError("short record")
            except ValueError:
                problems.append(SoundingProblem(
                    code="CM1-IMPORT-SOUNDING-FORMAT",
                    text=(
                        f"line {number} is not {per_level} numbers ({names}); CM1 reads {per_level} "
                        f"values per level for isnd = {isnd} and stops on this line (base.F:553-557)"
                    ),
                    fix=f"write each level as {names}",
                    line=number,
                ))
                break
            heights.append(values[0])
            thetas.append(values[1])
            qvs.append(values[2])
    if not problems and len(heights) < 2:
        problems.append(SoundingProblem(
            code="CM1-IMPORT-SOUNDING-LEVELS",
            text=(
                f"the file has {len(heights)} level(s) above the surface record; CM1 stops with "
                "fewer than two (base.F:569-576)"
            ),
            fix="supply at least two levels above the surface record",
        ))
    if heights and heights[0] <= 0.0:
        problems.append(SoundingProblem(
            code="CM1-IMPORT-SOUNDING-HEIGHTS",
            text=(
                f"the first level is at {heights[0]:g} m; CM1 puts the surface record at 0 m "
                "(base.F:605) and brackets each model level between sounding heights, so every "
                "level must lie above the ground"
            ),
            fix="give heights in metres above the ground, starting above 0 m",
        ))
    for index in range(1, len(heights)):
        if not heights[index] > heights[index - 1]:
            problems.append(SoundingProblem(
                code="CM1-IMPORT-SOUNDING-HEIGHTS",
                text=(
                    f"level {index + 1} ({heights[index]:g} m) does not rise above level {index} "
                    f"({heights[index - 1]:g} m); CM1 brackets each model level by the first "
                    "sounding height above it (base.F:686-690), so the profile would be "
                    "interpolated between the wrong records"
                ),
                fix="list the levels in increasing height, each height once",
            ))
            break
    if heights and top is not None and heights[-1] < top:
        problems.append(SoundingProblem(
            code="CM1-IMPORT-SOUNDING-TOP",
            text=(
                f"the sounding stops at {heights[-1]:g} m, below the model's highest scalar level "
                f"({top:g} m). CM1 r21.1 stops here too (base.F:678-683: zmax of sounding < zmax "
                "of grid)"
            ),
            fix=f"extend the sounding to at least {math.ceil(top):d} m, or lower the model top (nz x dz)",
        ))
    if surface is not None:
        pressure = surface["pressure_hpa"]
        if not 300.0 <= pressure <= 1100.0:
            hint = " (it looks like Pa)" if 30000.0 <= pressure <= 110000.0 else ""
            warnings.append(
                f"surface pressure {pressure:g} is outside 300-1100; CM1 reads it in hPa (mb){hint}"
            )
        if not 150.0 <= surface["theta_k"] <= 500.0:
            warnings.append(f"surface theta {surface['theta_k']:g} K is outside 150-500 K")
    if qvs and max(qvs + ([surface["qv_g_per_kg"]] if surface else [])) < 0.1 and any(q > 0 for q in qvs):
        warnings.append(
            "every mixing ratio is below 0.1 g/kg; CM1 reads it in g/kg (base.F:561, 596), so a "
            "profile written in kg/kg is almost dry"
        )
    if thetas and not all(150.0 <= theta <= 2000.0 for theta in thetas):
        warnings.append("some potential temperatures are outside 150-2000 K")
    return SoundingCheck(
        isnd=isnd,
        path=str(source),
        sha256=digest,
        surface=surface,
        heights_m=tuple(heights),
        highest_mass_level_m=top,
        problems=tuple(problems),
        warnings=tuple(warnings),
    )


def environment(
    check: SoundingCheck,
    effective: Mapping[str, Any],
    *,
    library: str | Path | None = None,
) -> SoundingCheck:
    """``check`` with the t=0 environment block, or the reason it is not computed.

    Computed by the Rust initializer library (the same one that builds the
    base state), for ``isnd = 5`` or a file with no problems.  Nothing here
    opens a GPU.
    """

    from dataclasses import replace

    if check.problems:
        return replace(check, environment_status="not-computed",
                       environment_reason="the sounding has the problems listed above")
    try:
        from ..cartesian_door import resolve_initializer_library
        from ..errors import CM1Refusal, fail_fast
        from ..sounding_report import SoundingReportSelection, initial_sounding_report

        with fail_fast():
            path = resolve_initializer_library(None if library is None else Path(library))
            payload = Path(check.path).read_bytes() if check.reads_file and check.path else b""
            nz = int(effective["nz"])
            dz = float(effective["dz"])
            levels = None
            if int(effective.get("stretch_z", 0)) == 1:
                from ..cm1_vertical_grid import build_vertical_levels

                levels = build_vertical_levels(
                    stretch_z=1, nk=nz, dz_m=dz, ztop_m=float(effective["ztop"]),
                    str_bot_m=float(effective["str_bot"]), str_top_m=float(effective["str_top"]),
                    dz_bot_m=float(effective["dz_bot"]), dz_top_m=float(effective["dz_top"]),
                )
            imove = int(effective.get("imove", 0)) == 1
            report = initial_sounding_report(
                path,
                payload,
                isnd=check.isnd,
                iwnd=int(effective.get("iwnd", 0)),
                nz=nz,
                dz_m=dz,
                levels=levels,
                selection=SoundingReportSelection(),
                u_move_m_s=float(effective.get("umove", 0.0)) if imove else 0.0,
                v_move_m_s=float(effective.get("vmove", 0.0)) if imove else 0.0,
            )
    except CM1Refusal as refusal:
        return replace(check, environment_status="not-computed", environment_reason=refusal.detail.reason)
    except (OSError, ValueError, KeyError, TypeError) as error:
        return replace(check, environment_status="not-computed", environment_reason=str(error))
    return replace(check, environment=report, environment_status="computed", environment_reason=None)


def locate(namelist_path: str | Path, explicit: str | Path | None) -> Path | None:
    """``explicit`` if given, else ``input_sounding`` beside the namelist (CM1's run directory)."""

    if explicit is not None:
        return Path(explicit)
    beside = Path(namelist_path).resolve().parent / "input_sounding"
    return beside if beside.is_file() else None


__all__ = [
    "FILE_SOUNDINGS",
    "SoundingCheck",
    "SoundingProblem",
    "check_sounding",
    "environment",
    "highest_mass_level_m",
    "locate",
]
