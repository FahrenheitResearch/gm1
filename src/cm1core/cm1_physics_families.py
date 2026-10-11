"""CM1 r21.1's radiation, boundary-layer and cumulus families as one selection.

This module is the CM1 authority for three namelist families GM1 previously
could not name at all:

* ``param11``'s ``radopt`` with its ``dtrad`` cadence and the solar geometry
  (``ctrlat``/``ctrlon`` and the start date) that gives a zenith angle,
* ``param2``'s ``ipbl`` together with ``cm1setup``, which is what decides
  *which* closure mixes a column, and
* ``param2``'s ``cuparam``.

It is deliberately import-light: nothing here imports CuPy, numpy or the
ArWen engine, so a configuration can be admitted or refused on the host with
no CUDA context and no installed physics estate.  The mapping from these CM1
selectors onto ArWen's component surface lives in
:mod:`cm1core.arwen_physics_component`; the scientific meaning lives here.

Authority
---------
``cm1setup``        README.namelist:122-147, src/param.F:752-753, :1176-1191
``idoles/idopbl``   src/param.F:752-753 (the only place either is set)
``ipbl``            README.namelist:310-333, src/param.F:1446-1478, :2744-2749,
                    src/turb.F:1885-2620, tendency at src/solve1.F:1086-1113
``radopt``          README.namelist:835-847, src/radiation_driver.F:119,:314,
                    cadence at src/cm1.F:3196-3199,:3691-3713,
                    tendency at src/solve1.F:1056-1085
``dtrad``/geometry  README.namelist:871-901, src/param.F:6344-6347
``cuparam``         README.namelist:513-517, src/cu_ntiedtke.F,
                    tendency at src/solve1.F:1508-1541
``rterm``           README.namelist:555-558, src/solve1.F:1003-1054,
                    src/param.F:2283-2294
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from .errors import refuse


PHYSICS_FAMILIES_GATE = "config-grammar"

#: src/param.F:752-753.  These two logicals are the whole of CM1's scale
#: selection: nothing else in the model decides whether an LES closure or a
#: PBL parameterization owns the vertical mixing.
CM1SETUP_LABELS: Mapping[int, str] = MappingProxyType(
    {
        0: "Euler equations; no subgrid turbulence model and no explicit diffusion",
        1: "large-eddy simulation; sgsmodel owns the vertical mixing",
        2: "mesoscale with a PBL parameterization; ipbl owns the vertical mixing",
        3: "direct numerical simulation",
        4: "LES within a mesoscale model; both closures, on different meshes",
    }
)

#: README.namelist:310-333 plus one GM1 extension.  11 is not a CM1 value:
#: it is WRF's own ``bl_pbl_physics=11`` for the scale-aware Shin-Hong
#: scheme, which CM1 r21.1 does not ship and which ArWen does.  It is
#: outside CM1's 0..6 so it can never be mistaken for one of CM1's, and a
#: run that selects it is recorded as not expressible as a CM1 namelist.
IPBL_LABELS: Mapping[int, str] = MappingProxyType(
    {
        0: "no PBL parameterization",
        1: "YSU (Hong et al. 2006)",
        2: "CM1 simple Louis-type scheme (Bryan and Rotunno 2009)",
        3: "GFS-EDMF as configured in HWRF v4.0a",
        4: "MYNN level 2.5 (Nakanishi and Niino 2006)",
        5: "MYNN level 3 (Nakanishi and Niino 2006)",
        6: "MYJ (Mellor-Yamada-Janjic)",
        11: "Shin-Hong scale-aware (WRF bl_pbl_physics=11; not a CM1 r21.1 value)",
    }
)
IPBL_GM1_EXTENSIONS: frozenset[int] = frozenset({11})

#: README.namelist:835-847.
RADOPT_LABELS: Mapping[int, str] = MappingProxyType(
    {
        0: "no atmospheric radiation",
        1: "NASA-Goddard longwave and shortwave (src/irrad3d.F, src/sorad3d.F)",
        2: "RRTMG longwave and shortwave (src/module_ra_rrtmg_lw.F, _sw.F)",
    }
)

#: README.namelist:513-517.
CUPARAM_LABELS: Mapping[int, str] = MappingProxyType(
    {
        0: "no convection parameterization",
        1: "new Tiedtke (src/cu_ntiedtke.F)",
    }
)

#: The grid spacing below which a deep-convection parameterization is a
#: double count rather than a closure.  CM1 does not refuse this itself --
#: every deck it ships simply sets ``cuparam=0`` -- so the number is GM1's,
#: and it is stated rather than implied.  New Tiedtke is scale-dependent:
#: ``src/cu_ntiedtke.F:221-230`` multiplies the deep adjustment timescale by
#: ``(1.06133+log(15000/dx))**3``, which is 53.6 at dx=1 km, so the closure
#: is slowed as the grid refines -- but it is never switched off, and a
#: slowed mass-flux scheme on a column that is already overturning
#: explicitly still removes the same CAPE twice.
CUMULUS_RESOLVED_CONVECTION_DX_M = 4000.0
#: Between this and :data:`CUMULUS_RESOLVED_CONVECTION_DX_M` the answer is
#: genuinely ambiguous (the convective grey zone).  A run there is admitted
#: and marked, not refused.
CUMULUS_GREY_ZONE_DX_M = 10000.0

#: src/param.F:6344-6347.  ``radtim`` starts at zero, which is why radiation
#: fires on the very first large step of a run.
RADIATION_CLOCK_START_S = 0.0


def cm1setup_flags(cm1setup: int) -> tuple[bool, bool]:
    """``(idoles, idopbl)`` exactly as ``src/param.F:752-753`` sets them."""

    return (cm1setup in (1, 4), cm1setup in (2, 4))


@dataclass(frozen=True, slots=True)
class Cm1TurbulenceScale:
    """Which closure mixes a column: CM1's ``cm1setup``/``sgsmodel``/``ipbl``.

    GM1's Cartesian storm lane is ``cm1setup=1`` with ``sgsmodel=1``, which
    is what it has always integrated.  Declaring nothing keeps exactly that.
    """

    cm1setup: int
    sgsmodel: int
    ipbl: int

    @property
    def idoles(self) -> bool:
        return cm1setup_flags(self.cm1setup)[0]

    @property
    def idopbl(self) -> bool:
        return cm1setup_flags(self.cm1setup)[1]

    def record(self) -> dict[str, object]:
        return {
            "cm1setup": self.cm1setup,
            "cm1setup_label": CM1SETUP_LABELS[self.cm1setup],
            "sgsmodel": self.sgsmodel,
            "ipbl": self.ipbl,
            "ipbl_label": IPBL_LABELS[self.ipbl],
            "ipbl_is_cm1_value": self.ipbl not in IPBL_GM1_EXTENSIONS,
            "idoles": self.idoles,
            "idopbl": self.idopbl,
            "authority": (
                "CM1 r21.1 src/param.F:752-753 (idoles/idopbl), :1446-1478 "
                "(ipbl>=1 requires cm1setup=2,4), README.namelist:122-147,310-333"
            ),
        }


def validate_turbulence_scale(
    selection: Cm1TurbulenceScale, *, option_prefix: str = "closures"
) -> Cm1TurbulenceScale:
    """Refuse every scale selection that would mix one column twice.

    Each rule below is one ``src/param.F`` stop, or one silent CM1 rewrite
    that GM1 refuses instead so a sealed receipt never names a closure the
    run did not integrate.
    """

    if selection.cm1setup not in CM1SETUP_LABELS:
        refuse(
            code="CM1-CFG-CM1SETUP-RANGE",
            gate=PHYSICS_FAMILIES_GATE,
            reason=(
                "src/param.F:1176-1191 stops the model for any cm1setup outside "
                "0..4; an out-of-range value would leave idoles and idopbl both "
                "false (src/param.F:752-753) and the column would be mixed by "
                "nothing at all while the receipt named a closure"
            ),
            remedy="set closures.cm1setup to 0, 1, 2, 3 or 4",
            option=f"{option_prefix}.cm1setup",
            value=selection.cm1setup,
        )
    if selection.ipbl not in IPBL_LABELS:
        refuse(
            code="CM1-CFG-IPBL-RANGE",
            gate=PHYSICS_FAMILIES_GATE,
            reason=(
                "src/turb.F:1885-2620 dispatches on ipbl 1..6 and GM1 adds "
                "11 for ArWen's scale-aware Shin-Hong scheme; any other value "
                "reaches no branch, so the boundary layer would be left "
                "unmixed while the run reported a PBL scheme"
            ),
            remedy=(
                "set closures.ipbl to one of "
                + ", ".join(str(value) for value in sorted(IPBL_LABELS))
            ),
            option=f"{option_prefix}.ipbl",
            value=selection.ipbl,
        )
    idoles, idopbl = cm1setup_flags(selection.cm1setup)
    if selection.ipbl >= 1 and not idopbl:
        refuse(
            code="CM1-CFG-PBL-WITHOUT-MESOSCALE-SETUP",
            gate=PHYSICS_FAMILIES_GATE,
            reason=(
                f"src/param.F:1465-1478 stops the model for ipbl={selection.ipbl} "
                f"with cm1setup={selection.cm1setup}, because src/param.F:752-753 "
                "leaves idopbl false and src/solve1.F:1086-1113 then never adds "
                "the PBL tendency: the scheme would be configured, reported, and "
                "silently inert, and the run would be an unmixed one wearing a "
                "boundary-layer label"
            ),
            remedy=(
                "set closures.cm1setup to 2 (mesoscale) or 4 (LES inside "
                "mesoscale) to use a PBL scheme, or set closures.ipbl to 0"
            ),
            option=f"{option_prefix}.ipbl",
            value=selection.ipbl,
        )
    if idopbl and selection.ipbl <= 0:
        refuse(
            code="CM1-CFG-MESOSCALE-SETUP-WITHOUT-PBL",
            gate=PHYSICS_FAMILIES_GATE,
            reason=(
                f"src/param.F:1446-1462 stops the model for cm1setup="
                f"{selection.cm1setup} with ipbl={selection.ipbl}; the mesoscale "
                "setup switches the LES closure off and names no replacement, so "
                "the lowest kilometre would receive no vertical mixing at all"
            ),
            remedy="set closures.ipbl to a scheme, or set closures.cm1setup to 1",
            option=f"{option_prefix}.ipbl",
            value=selection.ipbl,
        )
    # src/param.F:752-753 leaves idoles false for cm1setup=2, and
    # README.namelist:334-336 says sgsmodel is "only used for cm1setup = 1".
    # CM1 silently ignores the sgsmodel it was given; GM1 runs an always-on
    # prognostic-TKE closure, so ignoring it here would mean running both.
    if selection.cm1setup == 2 and selection.sgsmodel != 0:
        refuse(
            code="CM1-CFG-DOUBLE-COUNTED-VERTICAL-MIXING",
            gate=PHYSICS_FAMILIES_GATE,
            reason=(
                f"cm1setup=2 with closures.sgs_model={selection.sgsmodel} asks for "
                "a PBL parameterization and an LES subgrid closure on the same "
                "column; CM1 resolves this by ignoring sgsmodel "
                "(README.namelist:334-336, src/param.F:752-753 leaves idoles "
                "false), but GM1's Cartesian runtime always integrates its "
                "prognostic-TKE closure, so the column would be mixed twice and "
                "the boundary layer would come out roughly twice as deep and far "
                "too well mixed"
            ),
            remedy=(
                "set closures.sgs_model to 0 alongside cm1setup=2, or keep "
                "cm1setup=1 and leave the LES closure in charge"
            ),
            option=f"{option_prefix}.sgs_model",
            value=selection.sgsmodel,
        )
    if selection.cm1setup == 4:
        refuse(
            code="CM1-CFG-NESTED-LES-IN-MESOSCALE",
            gate=PHYSICS_FAMILIES_GATE,
            reason=(
                "cm1setup=4 runs an LES closure on an inner mesh and a PBL "
                "parameterization beyond it (README.namelist:145-148, CM1's "
                "param17 section); GM1 has no param17 inner-mesh boundary, so "
                "both closures would act on every column of the single mesh and "
                "the vertical flux would be counted twice everywhere"
            ),
            remedy=(
                "choose cm1setup=1 for an LES run or cm1setup=2 for a "
                "mesoscale run; the nested-LES boundary is not ported"
            ),
            option=f"{option_prefix}.cm1setup",
            value=selection.cm1setup,
        )
    if selection.cm1setup == 3:
        refuse(
            code="CM1-CFG-DNS-SETUP",
            gate=PHYSICS_FAMILIES_GATE,
            reason=(
                "cm1setup=3 integrates the Navier-Stokes equations with explicit "
                "molecular diffusivities from CM1's param7 section "
                "(README.namelist:138-142); GM1 declares no param7 table, so the "
                "run would proceed with the LES closure switched off and no "
                "diffusion at all in its place"
            ),
            remedy="use cm1setup=1 (LES) or cm1setup=2 (mesoscale with a PBL scheme)",
            option=f"{option_prefix}.cm1setup",
            value=selection.cm1setup,
        )
    if selection.cm1setup == 0 and selection.sgsmodel != 0:
        refuse(
            code="CM1-CFG-INVISCID-SETUP-WITH-SGS",
            gate=PHYSICS_FAMILIES_GATE,
            reason=(
                f"cm1setup=0 integrates the Euler equations and ignores sgsmodel "
                f"outright (README.namelist:126-130), but closures.sgs_model="
                f"{selection.sgsmodel} asks GM1's always-on prognostic-TKE "
                "closure to run; the run would be a subgrid-mixed one described "
                "as adiabatic and inviscid"
            ),
            remedy="set closures.sgs_model to 0 for cm1setup=0",
            option=f"{option_prefix}.sgs_model",
            value=selection.sgsmodel,
        )
    return selection


@dataclass(frozen=True, slots=True)
class Cm1RadiationSelection:
    """CM1's ``param11``: a radiation scheme, its cadence and its sun.

    ``dtrad_s`` is CM1's ``dtrad``.  The eight geometry words are CM1's
    ``ctrlat``, ``ctrlon`` and the start date; they exist because a radiation
    scheme cannot compute a solar zenith angle without a place and a time.
    GM1's idealized lane is deliberately location-free, so they are refused
    as a group unless radiation is on, and required as a group when it is:
    an idealized run given a silent default latitude would compute the sun's
    angle for a place nobody chose.
    """

    radopt: int
    dtrad_s: float
    ctrlat: float
    ctrlon: float
    year: int
    month: int
    day: int
    hour: int
    minute: int
    second: int

    @property
    def active(self) -> bool:
        return self.radopt >= 1

    def record(self) -> dict[str, object]:
        return {
            "radopt": self.radopt,
            "radopt_label": RADOPT_LABELS[self.radopt],
            "dtrad_s": self.dtrad_s,
            "ctrlat": self.ctrlat,
            "ctrlon": self.ctrlon,
            "start_utc": (
                f"{self.year:04d}-{self.month:02d}-{self.day:02d}"
                f"T{self.hour:02d}:{self.minute:02d}:{self.second:02d}Z"
            ),
            "location_free": not self.active,
            "authority": (
                "CM1 r21.1 README.namelist:835-901, src/radiation_driver.F:119,314, "
                "src/cm1.F:3196-3199,3691-3713 (dtrad cadence), "
                "src/solve1.F:1056-1085 (thraten applied to thten1)"
            ),
        }


#: A radiation-free run, and the exact selection an absent [radiation] table
#: binds.  src/param.F:6347 sets radtim to 1.0d60 for radopt<=0, so the
#: clock below never fires either.
NO_RADIATION = Cm1RadiationSelection(
    radopt=0,
    dtrad_s=0.0,
    ctrlat=0.0,
    ctrlon=0.0,
    year=0,
    month=0,
    day=0,
    hour=0,
    minute=0,
    second=0,
)


def validate_radiation_selection(
    selection: Cm1RadiationSelection,
    *,
    option_prefix: str = "radiation",
    rterm: int = 0,
) -> Cm1RadiationSelection:
    """Refuse every radiation selection CM1 itself stops on, plus GM1's own."""

    if selection.radopt not in RADOPT_LABELS:
        refuse(
            code="CM1-CFG-RADOPT-RANGE",
            gate=PHYSICS_FAMILIES_GATE,
            reason=(
                "src/radiation_driver.F dispatches on radopt=1 (:119) and "
                "radopt=2 (:314) only; any other nonzero value would leave "
                "lwten/swten/thraten at zero while src/solve1.F:1056-1085 "
                "added that zero to thten1, so the run would report a "
                "radiation scheme and integrate none"
            ),
            remedy="set radiation.radopt to 0, 1 or 2",
            option=f"{option_prefix}.radopt",
            value=selection.radopt,
        )
    if selection.radopt >= 1 and rterm == 1:
        refuse(
            code="CM1-CFG-RADOPT-WITH-RTERM",
            gate=PHYSICS_FAMILIES_GATE,
            reason=(
                "src/param.F:2283-2294 stops the model when radopt and rterm are "
                "both on: src/solve1.F:1006-1053 and :1056-1085 would each add a "
                "radiative theta tendency to thten1, so the column would be "
                "cooled by a real radiation scheme and by the Rotunno-Emanuel "
                "relaxation that stands in for one"
            ),
            remedy="set closures.rterm to 0, or radiation.radopt to 0",
            option=f"{option_prefix}.radopt",
            value=selection.radopt,
        )
    if not selection.active:
        # README.namelist:832-834: nothing in param11 is read for radopt=0.
        # The refusal here is GM1's: a document that types a latitude and a
        # date with the sun switched off is describing a run it is not asking
        # for, and the sealed receipt would carry a location the run never had.
        declared = [
            name
            for name, value in (
                ("dtrad_s", selection.dtrad_s),
                ("ctrlat", selection.ctrlat),
                ("ctrlon", selection.ctrlon),
                ("year", selection.year),
                ("month", selection.month),
                ("day", selection.day),
                ("hour", selection.hour),
                ("minute", selection.minute),
                ("second", selection.second),
            )
            if value
        ]
        if declared:
            refuse(
                code="CM1-CFG-RADIATION-GEOMETRY-INERT",
                gate=PHYSICS_FAMILIES_GATE,
                reason=(
                    "radopt=0 reads nothing else in param11 "
                    "(README.namelist:832-834), so "
                    + ", ".join(sorted(declared))
                    + " would be sealed into the receipt as this run's place, "
                    "date and radiation cadence while no radiation was computed "
                    "and the model had no sun at all"
                ),
                remedy=(
                    "leave every other [radiation] key at zero for radopt=0, or "
                    "switch radiation on"
                ),
                option=f"{option_prefix}.radopt",
                value=selection.radopt,
            )
        return selection
    if not (math.isfinite(selection.dtrad_s) and selection.dtrad_s > 0.0):
        refuse(
            code="CM1-CFG-DTRAD-RANGE",
            gate=PHYSICS_FAMILIES_GATE,
            reason=(
                "src/cm1.F:3708-3713 advances radtim by dtrad until it passes the "
                "model time; a non-positive or non-finite dtrad never passes it, "
                "so that loop would not terminate"
            ),
            remedy="set radiation.dtrad_s to a positive number of seconds",
            option=f"{option_prefix}.dtrad_s",
            value=selection.dtrad_s,
        )
    if not (-90.0 <= selection.ctrlat <= 90.0):
        refuse(
            code="CM1-CFG-CTRLAT-RANGE",
            gate=PHYSICS_FAMILIES_GATE,
            reason=(
                "the solar zenith angle is formed from the sine and cosine of the "
                "latitude; outside -90..90 the declared place is not on the Earth "
                "and the run would be lit by a sun at an angle no site ever sees"
            ),
            remedy="set radiation.ctrlat between -90 and 90 degrees",
            option=f"{option_prefix}.ctrlat",
            value=selection.ctrlat,
        )
    if not (-180.0 <= selection.ctrlon <= 180.0):
        refuse(
            code="CM1-CFG-CTRLON-RANGE",
            gate=PHYSICS_FAMILIES_GATE,
            reason=(
                "the hour angle is the declared longitude offset from the model "
                "clock; outside -180..180 the run's local noon lands on the wrong "
                "side of the planet and the diurnal cycle is shifted by hours"
            ),
            remedy="set radiation.ctrlon between -180 and 180 degrees",
            option=f"{option_prefix}.ctrlon",
            value=selection.ctrlon,
        )
    calendar = (
        ("year", selection.year, 1, 9999),
        ("month", selection.month, 1, 12),
        ("day", selection.day, 1, 31),
        ("hour", selection.hour, 0, 23),
        ("minute", selection.minute, 0, 59),
        ("second", selection.second, 0, 59),
    )
    for name, value, low, high in calendar:
        if not (low <= value <= high):
            refuse(
                code="CM1-CFG-RADIATION-CLOCK-RANGE",
                gate=PHYSICS_FAMILIES_GATE,
                reason=(
                    f"{name}={value} is not a calendar {name}; CM1 turns the start "
                    "date into a Julian day (src/radiation_driver.F) and an "
                    "out-of-range field moves the declination, so the run would be "
                    "lit by the wrong season"
                ),
                remedy=f"set radiation.{name} between {low} and {high}",
                option=f"{option_prefix}.{name}",
                value=value,
            )
    return selection


@dataclass(slots=True)
class RadiationClock:
    """CM1's ``radtim`` scheduler, in CM1's own arithmetic.

    ``src/cm1.F:3196-3199`` decides on the time at the *end* of the step
    about to be taken; ``src/cm1.F:3708-3713`` then walks ``radtim`` forward
    by ``dtrad`` until it passes the model time.  ``src/param.F:6344`` starts
    ``radtim`` at zero, which is why radiation fires on the first large step.

    Both are binary64 here because CM1 declares ``mtime`` and ``radtim``
    ``double precision`` (``src/cm1.F:61``).
    """

    dtrad_s: float
    dt_s: float
    radtim_s: float = RADIATION_CLOCK_START_S

    def due(self, completed_steps: int) -> bool:
        """Whether the step numbered ``completed_steps + 1`` computes radiation."""

        mtime = float(completed_steps) * float(self.dt_s)
        return (mtime + self.dt_s) >= (self.radtim_s - 0.1 * self.dt_s)

    def advance(self, completed_steps: int) -> float:
        """Walk ``radtim`` past the model time, as ``src/cm1.F:3708-3713`` does."""

        mtime = float(completed_steps) * float(self.dt_s)
        while True:
            self.radtim_s = self.radtim_s + self.dtrad_s
            if self.radtim_s > mtime:
                return self.radtim_s

    def steps_between_calls(self) -> int:
        """How many large steps one radiation interval spans, for a receipt."""

        return max(1, int(math.floor(self.dtrad_s / self.dt_s + 1.0e-9)))


@dataclass(frozen=True, slots=True)
class Cm1CumulusSelection:
    """CM1's ``cuparam`` plus the grid spacing it is being asked to act on."""

    cuparam: int
    dx_m: float
    dy_m: float

    @property
    def active(self) -> bool:
        return self.cuparam >= 1

    @property
    def grey_zone(self) -> bool:
        spacing = min(self.dx_m, self.dy_m)
        return self.active and spacing < CUMULUS_GREY_ZONE_DX_M

    def record(self) -> dict[str, object]:
        return {
            "cuparam": self.cuparam,
            "cuparam_label": CUPARAM_LABELS[self.cuparam],
            "convective_grey_zone": self.grey_zone,
            "resolved_convection_dx_m": CUMULUS_RESOLVED_CONVECTION_DX_M,
            "grey_zone_dx_m": CUMULUS_GREY_ZONE_DX_M,
            "authority": (
                "CM1 r21.1 README.namelist:513-517, src/cu_ntiedtke.F, "
                "src/solve1.F:1508-1541 (thcuten/ucuten/vcuten/qvcuten applied)"
            ),
        }


def validate_cumulus_selection(
    selection: Cm1CumulusSelection, *, option_prefix: str = "physics"
) -> Cm1CumulusSelection:
    """Refuse a convection scheme on a grid that lifts its own updrafts."""

    if selection.cuparam not in CUPARAM_LABELS:
        refuse(
            code="CM1-CFG-CUPARAM-RANGE",
            gate=PHYSICS_FAMILIES_GATE,
            reason=(
                "CM1 r21.1 ships one convection parameterization "
                "(README.namelist:513-517); any other value reaches no "
                "initializer in src/init_physics.F:141-156 and "
                "src/solve1.F:1508-1541 would add uninitialized cumulus "
                "tendencies to the prognostic fields"
            ),
            remedy="set physics.cuparam to 0 or 1",
            option=f"{option_prefix}.cuparam",
            value=selection.cuparam,
        )
    if not selection.active:
        return selection
    spacing = min(selection.dx_m, selection.dy_m)
    if spacing < CUMULUS_RESOLVED_CONVECTION_DX_M:
        refuse(
            code="CM1-CFG-CUMULUS-AT-RESOLVED-SCALE",
            gate=PHYSICS_FAMILIES_GATE,
            reason=(
                f"a deep-convection parameterization was selected at {spacing:g} m "
                f"spacing, below {CUMULUS_RESOLVED_CONVECTION_DX_M:g} m, where this "
                "model lifts its own convective updrafts; the scheme's own "
                "scale-dependency factor (src/cu_ntiedtke.F:221-230) slows its deep "
                "closure as the grid refines but never switches it off, so "
                "src/solve1.F:1508-1541 would still add a mass-flux scheme's heating "
                "and drying to columns that are already overturning explicitly -- the "
                "same CAPE removed twice, leaving the resolved updraft starved of the "
                "buoyancy that drives it, in a column left too warm and too dry"
            ),
            remedy=(
                "set physics.cuparam to 0 at storm-resolving spacing, or run the "
                f"scheme on a grid of at least "
                f"{CUMULUS_RESOLVED_CONVECTION_DX_M:g} m"
            ),
            option=f"{option_prefix}.cuparam",
            value=selection.cuparam,
        )
    return selection


__all__ = [
    "CM1SETUP_LABELS",
    "CUMULUS_GREY_ZONE_DX_M",
    "CUMULUS_RESOLVED_CONVECTION_DX_M",
    "CUPARAM_LABELS",
    "Cm1CumulusSelection",
    "Cm1RadiationSelection",
    "Cm1TurbulenceScale",
    "IPBL_GM1_EXTENSIONS",
    "IPBL_LABELS",
    "NO_RADIATION",
    "PHYSICS_FAMILIES_GATE",
    "RADIATION_CLOCK_START_S",
    "RADOPT_LABELS",
    "RadiationClock",
    "cm1setup_flags",
    "validate_cumulus_selection",
    "validate_radiation_selection",
    "validate_turbulence_scale",
]
