"""What every quick-look footer says, read from the run's own records.

A footer carries SIMULATED, the GM1 version, the microphysics and its checked
state, dx, the real height of the level shown, the wind frame (ground- or storm-relative) and T+.

Where each comes from:

* the GM1 version: ``run.json`` (``gm1.version``, written when the run was imported), else the newest segment's
  receipt (``version``), else the version of the package drawing the picture, said as such;
* the microphysics: the frame's own publication record (``config.microphysics``, ``config.dense_ice``), so a live
  frame is labelled before its segment has a receipt;
* its checked state: the importer's coverage report (``import-report.json``), which classifies the run's
  ``ptype`` as Same as CM1, Runs differently from CM1 or Not yet checked; without a report, the receipt's
  unvalidated-capability register;
* dx and the level heights: the frame's grid;
* the name: ``run.json`` ``name``, else the run folder's name, unless the caller gives a label;
* ``iorigin`` (where x = y = 0 is, for positions in km): the run's namelist, else its deck.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping

from . import style as S

PLAIN_SCHEME = {"morrison-mp10": "Morrison", "wsm6-mp6": "WSM6", "thompson-mp8": "Thompson", "kessler": "Kessler"}
#: The importer's report sections, as the footer says them.
CHECKED_WORDS = {
    "same": "same as CM1 in the importer's check",
    "differs": "runs differently from CM1",
    "not_checked": "not yet checked against CM1",
}


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def scheme_words(config: Mapping[str, Any]) -> str:
    """'Morrison with hail', 'WSM6', ... from a frame's publication config."""

    component = str(config.get("microphysics") or "")
    name = PLAIN_SCHEME.get(component, component or "unknown microphysics")
    dense = config.get("dense_ice") or {}
    category = dense.get("category") if isinstance(dense, dict) else None
    # CM1 r21.1 passes ihail to both schemes (param.F:3407 graupel_init, param.F:3697-3706 mp_wsm6_init hail_opt)
    if component in ("morrison-mp10", "wsm6-mp6") and category in ("hail", "graupel"):
        name = f"{name} with {category}"
    return name


def _ptype_row(report: Mapping[str, Any]) -> tuple[str | None, Mapping[str, Any] | None]:
    for section in report.get("sections") or []:
        for row in section.get("rows") or []:
            keys = row.get("keys") or ([row["key"]] if row.get("key") else [])
            if "ptype" in keys:
                return str(section.get("id")), row
    return None, None


def checked_state(root: Path, receipt: Mapping[str, Any] | None, component: str) -> tuple[str, str]:
    """(short words for the footer, where they came from)."""

    report = _read_json(root / "import-report.json")
    if isinstance(report, dict):
        section, row = _ptype_row(report)
        if section == "same":
            comparison = report.get("comparison") or {}
            rows = comparison.get("rows") or []
            if comparison.get("scheme") == component and rows and all(r.get("inside_target") for r in rows):
                return "within 5 % of CM1 on CM1's stock supercell (measured)", "import-report.json"
            return CHECKED_WORDS["same"], "import-report.json"
        if section in CHECKED_WORDS:
            return CHECKED_WORDS[section], "import-report.json"
    if isinstance(receipt, dict):
        register = receipt.get("unvalidated") or {}
        for row in register.get("capabilities") or []:
            if component and component in (row.get("values") or []):
                return "not checked against CM1 on this setup", "receipt unvalidated register"
    # Only the importer compares a run's settings with CM1's; a run made another way claims nothing.
    return "no CM1 comparison recorded for this run", "no import report"


@dataclass
class RunFacts:
    name: str
    version: str
    version_source: str
    scheme: str
    component: str
    checked: str
    checked_source: str
    dx_m: float
    dy_m: float
    iorigin: int = 1
    namelist_words: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def scheme_line(self) -> str:
        return f"{self.scheme}, {self.checked}"

    def origin_m(self, nx: int, ny: int) -> tuple[float, float]:
        """The x, y (m) of the domain's south-west corner in the run's own coordinates (iorigin)."""

        if self.iorigin == 2:
            return -0.5 * nx * self.dx_m, -0.5 * ny * self.dy_m
        return 0.0, 0.0

    def footer(self, *, t: float, level: str = "", wind: str = "", notes: str = "") -> list[str]:
        """The two footer lines: the run's facts, then the picture's method notes."""

        first = S.data_line(S.SIMULATED, f"GM1 {self.version}", self.scheme_line, f"dx {S.fmt_grid(self.dx_m)}",
                            level, f"wind {wind}" if wind else "", S.fmt_clock(t))
        return [first, notes] if notes else [first]

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "gm1_version": self.version, "gm1_version_source": self.version_source,
                "microphysics": self.scheme, "component": self.component, "checked": self.checked,
                "checked_source": self.checked_source, "dx_m": self.dx_m, "dy_m": self.dy_m,
                "iorigin": self.iorigin}


def run_facts(layout: Any, config: Mapping[str, Any], dx_m: float, dy_m: float, *, label: str | None = None,
              settings: Any = None) -> RunFacts:
    """The facts of the run ``layout`` (a :class:`cm1core.cm1out.frames.RunLayout`), with ``config`` a frame's
    publication config."""

    root = Path(layout.root if layout.kind != "frames" else layout.root.parent)
    run = _read_json(root / "run.json") if (root / "run.json").is_file() else None
    receipt = None
    for segment in reversed(layout.segments):
        base = segment.frames_root.parent
        receipt = _read_json(base / "receipt.json")
        if isinstance(receipt, dict):
            break
    version, source = None, ""
    if isinstance(run, dict) and (run.get("gm1") or {}).get("version"):
        version, source = str(run["gm1"]["version"]), "run.json"
    elif isinstance(receipt, dict) and receipt.get("version"):
        version, source = str(receipt["version"]), "receipt"
    else:
        from .._version import __version__

        version, source = f"{__version__} (the run's version is not recorded)", "this package"
    component = str(config.get("microphysics") or "")
    checked, checked_source = checked_state(root, receipt if isinstance(receipt, dict) else None, component)
    name = label or (str(run.get("name")) if isinstance(run, dict) and run.get("name") else root.name)
    iorigin = 1
    if settings is not None:
        try:
            iorigin = int(settings.int("iorigin", 1))
        except (TypeError, ValueError):
            iorigin = 1
    return RunFacts(name=name, version=version, version_source=source, scheme=scheme_words(config),
                    component=component, checked=checked, checked_source=checked_source, dx_m=float(dx_m),
                    dy_m=float(dy_m), iorigin=iorigin)


__all__ = ["RunFacts", "checked_state", "run_facts", "scheme_words"]
