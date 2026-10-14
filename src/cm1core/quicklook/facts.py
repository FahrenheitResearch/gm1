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
* ``iorigin`` (where x = y = 0 is, for positions in km): the run's namelist, else its deck;
* the kind of storm: ``"hurricane"`` for a run that starts from CM1's hurricane swirl (``iinit = 7``), which gets
  the hurricane looks and ranges (:mod:`.products`); every other run is a thunderstorm run and says nothing.

Pictures of CM1 netCDF files take their facts from the files (:meth:`.ncread.NcLayout.run_facts`).  CM1's own
output says so where a GM1 run says its version: the footer names the CM1 version the file records in place of
"GM1 x.y.z", nothing is said about a comparison with CM1, and the mark beside "GM1" says "picture" (GM1 made the
picture; CM1 simulated the storm).  A GM1 run drawn from its cm1out files keeps its own footer, and its name says
where the picture was read from.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
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


#: CM1's ``iinit`` for the hurricane swirl (README.namelist: "7 = hurricane vortex").
HURRICANE_IINIT = 7
#: What a hurricane picture's footer says in place of the scheme's thunderstorm comparison, which does not speak
#: for a hurricane.  GM1 and CM1 r21.1 have both run one hurricane, the 8 km preset's changed settings (GM1 on a
#: fixed 7.5 s step, CM1 on its own adaptive one); records/GM1-HURRICANE-RUNS.json has both series and the card
#: says how they compare.  The footer says only that it was compared, not how well.  A run with those settings
#: says so, whatever its storm time or picture interval (neither changes the storm); any other grid, levels, sea
#: temperature, time step, drag or cloud droplet number, or any other setting that changes the storm, says it is
#: not compared.  The same test puts CM1's curves on the time series (:mod:`.series`) and CM1's panel beside
#: GM1's on the looks by radius and height (:mod:`.rings`).
HURRICANE_CHECKED = "compared with CM1 on this 8 km case (measured)"
#: The word beside the GM1 mark on a picture of another model's output (GM1's own storms say "simulated").
CM1_MARK_WORD = "picture"
HURRICANE_NOT_COMPARED = "not compared with CM1 at these settings"
#: The settings of that one compared hurricane (the 8 km preset's namelist).
HURRICANE_COMPARED = (("nx", 128), ("ny", 128), ("nz", 59), ("dx", 8000.0), ("dy", 8000.0), ("tsk0", 301.15),
                      ("adapt_dt", 0), ("dtl", 7.5), ("cnstcd", 0.0024), ("cnstce", 0.0012), ("fcor", 5e-5),
                      ("ptype", 5),
                      # the levels, the kind of drag and the cloud droplet number of that run
                      ("stretch_z", 1), ("ztop", 25000.0), ("str_bot", 150.0), ("str_top", 4000.0),
                      ("dz_bot", 50.0), ("dz_top", 500.0), ("cecd", 1), ("ndcnst", 250.0))


#: The preset whose settings CM1 r21.1 has also been run with (records/GM1-HURRICANE-RUNS.json ``runs.cm1-8km``):
#: its shipped namelist, not a site's own preset folder.
HURRICANE_COMPARED_PRESET = "hurricane-8km"


@lru_cache(maxsize=1)
def compared_storm_settings() -> dict[str, Any] | None:
    """Every setting that changes the storm (:func:`.series.storm_settings`) of the compared hurricane preset, from
    the namelist this package ships; None when it cannot be read."""

    from ..cm1out import settings as _settings
    from .series import storm_settings

    path = Path(__file__).resolve().parents[1] / "presets" / HURRICANE_COMPARED_PRESET / "namelist.input"
    try:
        return storm_settings(_settings.from_namelist(path).values)
    except (_settings.SettingsError, OSError, ValueError):
        return None


def hurricane_differences(settings: Any) -> list[str] | None:
    """The settings that change the storm and differ from the compared hurricane preset's (empty when none do), or
    None when either side's settings cannot be read as a whole (a frame's words only, or no namelist parser)."""

    from .series import _same, storm_settings

    theirs = compared_storm_settings()
    values = getattr(settings, "values", None)
    if theirs is None or not isinstance(values, Mapping):
        return None
    mine = storm_settings(values)
    return sorted(key for key in set(mine) | set(theirs) if not _same(mine.get(key), theirs.get(key)))


def hurricane_checked(settings: Any) -> str:
    """The footer words for a hurricane run with these CM1 settings: compared when the settings of
    :data:`HURRICANE_COMPARED` are the 8 km preset's and, when the run's settings can be read as a whole, every
    other setting that changes the storm is too (the storm time and the picture interval never change it)."""

    try:
        same = all(abs(settings.float(key, float("nan")) - value) <= 1e-9 + 1e-6 * abs(value)
                   for key, value in HURRICANE_COMPARED)
    except (TypeError, ValueError, AttributeError):
        same = False
    if same and isinstance(getattr(settings, "values", None), Mapping):
        # A run's whole settings: every other setting that changes the storm must be the preset's too (when the
        # preset's own cannot be read, nothing is claimed).
        same = hurricane_differences(settings) == []
    return HURRICANE_CHECKED if same else HURRICANE_NOT_COMPARED


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
    #: ``"hurricane"`` for a run started from CM1's hurricane swirl (iinit = 7); ``""`` otherwise.
    storm_kind: str = ""
    #: The model that simulated the storm, where it is not GM1 (CM1's own netCDF output: "CM1 r21.1").
    made_by: str = ""
    #: The model's short name for the time series' label ("GM1" unless :attr:`made_by` says otherwise).
    model: str = "GM1"
    #: The word beside the GM1 mark: "simulated" on GM1's storms, :data:`CM1_MARK_WORD` on CM1's own output.
    mark_word: str = S.SIMULATED
    #: Where the picture was read from when it was not the run's frames ("cm1out netCDF", ...).
    source: str = ""

    @property
    def scheme_line(self) -> str:
        return f"{self.scheme}, {self.checked}" if self.checked else self.scheme

    def origin_m(self, nx: int, ny: int) -> tuple[float, float]:
        """The x, y (m) of the domain's south-west corner in the run's own coordinates (iorigin)."""

        if self.iorigin == 2:
            return -0.5 * nx * self.dx_m, -0.5 * ny * self.dy_m
        return 0.0, 0.0

    def clock(self, t: float) -> str:
        """A picture's storm time: a hurricane's in hours and days, as its pages say it; any other run's H:MM:SS."""

        return S.fmt_long_clock(t) if self.storm_kind == "hurricane" else S.fmt_clock(t)

    def footer(self, *, t: float, level: str = "", wind: str = "", notes: str = "") -> list[str]:
        """The two footer lines: the run's facts, then the picture's method notes."""

        # A hurricane picture's footer says the time as its clock and the pages do (hours and days).
        clock = self.clock(t)
        first = S.data_line(S.SIMULATED, self.made_by or f"GM1 {self.version}", self.scheme_line,
                            f"dx {S.fmt_grid(self.dx_m)}", level, f"wind {wind}" if wind else "", clock)
        lines = [first, notes] if notes else [first]
        # The mark's word travels with the footer lines into the picture (style.Overlay.footer).
        return S.Footer(lines, self.mark_word) if self.mark_word != S.SIMULATED else lines

    def as_dict(self) -> dict[str, Any]:
        record = {"name": self.name, "gm1_version": self.version, "gm1_version_source": self.version_source,
                  "microphysics": self.scheme, "component": self.component, "checked": self.checked,
                  "checked_source": self.checked_source, "dx_m": self.dx_m, "dy_m": self.dy_m,
                  "iorigin": self.iorigin}
        if self.storm_kind:
            # Only a run with a kind carries the key, so every other run's record is what it was.
            record["storm_kind"] = self.storm_kind
        # Likewise the keys of pictures drawn from netCDF files (GM1's own run drawn from its frames has none).
        for key, value, plain in (("made_by", self.made_by, ""), ("model", self.model, "GM1"),
                                  ("mark_word", self.mark_word, S.SIMULATED), ("source", self.source, "")):
            if value != plain:
                record[key] = value
        return record


def read_settings(layout: Any) -> tuple[Any, str | None]:
    """The run's CM1 settings from the namelist or deck beside its frames, and ``None`` with one plain line when
    they cannot be read: without them a hurricane run is drawn as a thunderstorm run (its kind is CM1's
    ``iinit``) and positions count from the domain's corner (CM1's ``iorigin``), and the line says so."""

    from ..cm1out import settings as _settings

    try:
        return _settings.resolve(layout), None
    except (_settings.SettingsError, OSError, ValueError) as error:
        files = [*layout.candidate_namelists(), *layout.candidate_decks()]
        why = (str(error) if any(path.is_file() for path in files)
               else "there is no CM1 namelist or GM1 deck beside these frames")
        return None, settings_line(why)


def settings_line(why: str) -> str:
    """The one line a picture pass prints when it draws without the run's settings."""

    return (f"Drawn without the run's settings ({why.rstrip('.')}): a hurricane run is drawn as a thunderstorm run "
            "and x, y count from the domain's corner. Give them with --namelist FILE or --deck FILE.")


def run_facts(layout: Any, config: Mapping[str, Any], dx_m: float, dy_m: float, *, label: str | None = None,
              settings: Any = None) -> RunFacts:
    """The facts of the run ``layout`` (a :class:`cm1core.cm1out.frames.RunLayout`), with ``config`` a frame's
    publication config.  CM1 netCDF files (:class:`.ncread.NcLayout`) say their own."""

    own = getattr(layout, "run_facts", None)
    if own is not None:
        return own(config, dx_m, dy_m, label=label, settings=settings)
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
    storm_kind = ""
    if settings is not None:
        try:
            iorigin = int(settings.int("iorigin", 1))
        except (TypeError, ValueError):
            iorigin = 1
        try:
            # CM1's hurricane start (init3d.F: a swirl over a warm ocean) is the one initializer that is one.
            storm_kind = "hurricane" if int(settings.int("iinit", 1)) == HURRICANE_IINIT else ""
        except (TypeError, ValueError):
            storm_kind = ""
    if storm_kind == "hurricane":
        checked = hurricane_checked(settings)
        checked_source = ("GM1's and CM1's runs of the 8 km hurricane preset" if checked == HURRICANE_CHECKED
                          else "no CM1 run with these hurricane settings")
    return RunFacts(name=name, version=version, version_source=source, scheme=scheme_words(config),
                    component=component, checked=checked, checked_source=checked_source, dx_m=float(dx_m),
                    dy_m=float(dy_m), iorigin=iorigin, storm_kind=storm_kind)


__all__ = ["CM1_MARK_WORD", "HURRICANE_CHECKED", "HURRICANE_COMPARED", "HURRICANE_COMPARED_PRESET",
           "HURRICANE_NOT_COMPARED", "RunFacts", "checked_state", "compared_storm_settings", "hurricane_checked",
           "hurricane_differences", "run_facts", "scheme_words"]
