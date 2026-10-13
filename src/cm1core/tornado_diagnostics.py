"""CM1 r21.1 rotation diagnostics, surface swaths and a tornado report.

GM1 has produced a splitting supercell and cannot say so from its own output.
This module is the part that says so: CM1's updraft helicity, its three
vorticity components, its Ertel potential vorticity, its whole-run surface
swaths, and one report that names what the run actually produced without
letting a mesocyclone be read as a tornado.

Everything here is transcribed from CM1 r21.1, Cartesian grid, and every
formula carries its ``file:line``.  Nothing is invented; where CM1 has no
answer -- whether a vortex is a tornado -- the answer comes from the
literature, is named by citation, is fixed rather than configurable, and is
reported beside the grid spacing that decides whether it can mean anything.

Array-module agnostic.  Every routine takes ``xp``: pass the runtime's CuPy
module and the whole computation is device-resident, with only scalars and
one plane per published frame crossing to the host; pass NumPy and the same
source runs on the host, which is how a machine with no free GPU verifies it.
The two agree bit for bit here because every operation is an IEEE-754
binary32 add, subtract, multiply, maximum or minimum on identically shaped
operands, and the one accumulation (the updraft-helicity column integral)
runs as CM1's own serial ``k`` loop rather than a reduction, so no
summation order is left to the backend.

Why not ArWen's ``uh_diag.cu``
------------------------------
ArWen ships ``gpuwm/core/kernels/uh_diag.cu``, and it was read before a line
of this was written.  It is a transcription of **WRF** v4.6.1's
``cal_helicity`` (``dyn_em/module_diffusion_em.F:7132-7579``), which is a
different diagnostic from CM1's:

* it reads WRF mass-coordinate carriers -- ``ph``/``phb`` geopotential,
  ``msfu``/``msfv`` map factors, ``fnm``/``fnp``/``dn``/``dnw`` coordinate
  columns, ``cf1..cf3`` -- none of which exists anywhere in GM1, whose
  Cartesian lane is a height coordinate with unit map factors;
* it integrates only over layers wholly inside 2-5 km and zeroes the whole
  column as soon as one layer's ``wavg`` is non-positive, where CM1 keeps the
  part of each layer that lies inside 2-5 km and clips ``w``
  and ``zeta`` to zero per layer (``misclibs.F:1520-1527``);
* it applies a nine-point smoother and a ``use_column`` gate that CM1 has
  nowhere.

Feeding it a synthesized ``ph``/``phb`` would not be "connecting" it; it
would publish WRF's number under CM1's name, and CM1 r21.1 is this project's
mathematical authority.  So the CM1 definition is transcribed here, and
ArWen's kernel is left where it is, in a repository this port does not edit.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any


TORNADO_DIAGNOSTIC_SCHEMA = "gm1.tornado-diagnostics/v1"
TORNADO_SWATH_SCHEMA = "gm1.tornado-surface-swaths/v1"
TORNADO_REPORT_SCHEMA = "gm1.tornado-report/v1"

SOURCE_ANCHORS = MappingProxyType({
    # calcuh: vertically integrated updraft helicity, 2-5 km AGL.
    "updraft_helicity": "src/misclibs.F:1417-1532",
    "updraft_helicity_layer": "src/misclibs.F:1441-1442",
    "updraft_helicity_integrand": "src/misclibs.F:1516-1530",
    "updraft_helicity_corner_vorticity": "src/misclibs.F:1454-1461",
    "updraft_helicity_reference": "src/misclibs.F:1423 (Kain et al. 2008, WAF 23, 931-952)",
    "updraft_helicity_output_switch": "src/writeout.F:724-770",
    # calcvort: the three vorticity components and Ertel PV at scalar points.
    "vorticity_x": "src/misclibs.F:1626-1685",
    "vorticity_y": "src/misclibs.F:1862-1921",
    "vorticity_z": "src/misclibs.F:2037-2049",
    "vorticity_semi_slip_surface": "src/misclibs.F:2165-2172",
    "potential_vorticity_theta": "src/misclibs.F:1585-1612",
    "potential_vorticity_x_term": "src/misclibs.F:1686-1707",
    "potential_vorticity_y_term": "src/misclibs.F:2007-2030",
    "potential_vorticity_z_term": "src/misclibs.F:2151-2172",
    "vorticity_output_switch": "src/writeout.F:4330-4400",
    "potential_vorticity_output_switch": "src/writeout.F:4403-4426",
    "ground_relative_winds_for_calcvort": "src/writeout.F:291-308",
    # Whole-run surface swaths.
    "swath_initialization": "src/init3d.F:181-203",
    "swath_max_surface_wind": "src/solve3.F:855-878",
    "swath_max_vertical_vorticity": "src/solve3.F:883-965",
    "swath_min_surface_pressure": "src/solve3.F:968-993",
    "swath_max_rainwater": "src/solve3.F:994-1019",
    "swath_max_graupel": "src/solve3.F:1022-1047",
    "swath_max_updraft_5km": "src/solve3.F:1051-1108",
    "swath_max_updraft_helicity": "src/solve3.F:1110-1136",
    "swath_translation": "src/adv_routines.F:8884-9178 (movesfc)",
    "swath_slot_count": "src/param.F:4697-4698 (nrain)",
    "swath_output_switches": "src/writeout.F:408-545 and :568-720",
    # The swaths cross a restart: restart_write.F writes sws..shs and, under
    # nrain=2, the translated sws2..shs2; restart_read.F reads them back.
    "swath_restart_write": "src/restart_write.F:696-758",
    "swath_restart_read": "src/restart_read.F:1064-1126",
    "domain_location_integral": "src/cm1.F:3512-3513",
    # Surface layer, for the semi-slip lowest-level horizontal vorticity.
    "surface_layer_lowest_wind": "src/turb.F:532-545",
    "surface_layer_getcecd_constant_cd": "src/sfcphys.F:120-136",
    "karman": "src/constants.F:23",
    # Fornberg quadratic ends used by the PV vertical theta gradient.
    "fornberg_ends": "src/param.F:8988-9052",
})

TORNADO_DIAGNOSTIC_CLAIM_LIMIT = (
    "CM1 r21.1 Cartesian-grid updraft helicity, vorticity components, Ertel "
    "potential vorticity and whole-run surface swaths on one flat "
    "uniform-horizontal domain, with the outermost scalar row and column of "
    "every corner-averaged field closed from the nearest source-valid "
    "interior cell because this runtime publishes no metric halo; not a CM1 "
    "output-file reproduction, not a tornado detection unless the report "
    "block says the grid resolves one, and never a forecast-skill claim"
)

# ---------------------------------------------------------------------------
# CM1 constants
# ---------------------------------------------------------------------------

#: ``misclibs.F:1441-1442``.  The integration layer, metres above ground.
UPDRAFT_HELICITY_BOTTOM_M = 2000.0
UPDRAFT_HELICITY_TOP_M = 5000.0
#: ``solve3.F:1090``.  The height ``sus`` samples ``w`` at.
UPDRAFT_SWATH_HEIGHT_M = 5000.0
#: ``init3d.F:191-201``.  CM1's own initial swath values.
SWATH_INITIAL_VALUES = MappingProxyType({
    "sws": 0.0, "srs": 0.0, "sgs": 0.0, "shs": 0.0,
    "sps": 200000.0, "svs": -1000.0, "sus": -1000.0,
})
#: Which extremum each swath keeps (``solve3.F:867,897,978,1005,1033,1094,1122``).
SWATH_EXTREMUM = MappingProxyType({
    "sws": "max", "svs": "max", "sps": "min", "srs": "max",
    "sgs": "max", "sus": "max", "shs": "max",
})
SWATH_UNITS = MappingProxyType({
    "sws": "m s-1", "svs": "s-1", "sps": "Pa", "srs": "kg kg-1",
    "sgs": "kg kg-1", "sus": "m s-1", "shs": "m2 s-2",
})
#: ``writeout.F:410-545``.  CM1's own descriptions, kept verbatim.
SWATH_MEANINGS = MappingProxyType({
    "sws": "max horiz wind speed at lowest model level",
    "svs": "max vert vorticity at lowest model level",
    "sps": "min pressure perturbation at lowest model level",
    "srs": "max qr at lowest model level",
    "sgs": "max qg at lowest model level",
    "sus": "max w at 5 km AGL",
    "shs": "max integrated updraft helicity",
})
SWATH_NAMES = tuple(SWATH_MEANINGS)
#: ``writeout.F:571-720``.  CM1 writes the translated slot under name + "2".
SWATH_GROUND_SUFFIX = "2"

#: ``constants.F:23``.
CM1_KARMAN = 0.40
#: CM1's ``nqr`` and ``nqg`` (one-based 3 and 6) as zero-based rows: the same
#: for ptype 3, 5 and 7 (param.F:3322-3398, 3414-3473, 3641-3678), and for every
#: component GM1 binds, which all carry CM1's six mass rows first
#: (moisture_species.py ``require_cm1_mass_prefix``).
RAINWATER_INDEX = 2
GRAUPEL_INDEX = 5
#: The names this module used while only WSM6 was admitted; the rows are CM1's.
WSM6_RAINWATER_INDEX = RAINWATER_INDEX
WSM6_GRAUPEL_INDEX = GRAUPEL_INDEX


def _has_rain_graupel_rows(component: str) -> bool:
    """Whether the component's scalar axis has qr at CM1's nqr and qg at CM1's nqg (solve3.F:999-1000, 1025-1026)."""

    from .moisture_species import MOISTURE_COMPONENT_IDS, moisture_species_profile

    if component not in MOISTURE_COMPONENT_IDS:
        return False
    names = moisture_species_profile(component).mass_names
    return names[RAINWATER_INDEX:RAINWATER_INDEX + 1] == ("qr",) and names[GRAUPEL_INDEX:GRAUPEL_INDEX + 1] == ("qg",)

# ---------------------------------------------------------------------------
# Detection thresholds.  Fixed, cited, and deliberately not configurable:
# a threshold a user can lower until the report says "tornado" is not a
# detector, it is a wish.
# ---------------------------------------------------------------------------

#: Kain, Weiss, Bright et al. 2008, *Wea. Forecasting* **23**, 931-952 -- the
#: paper ``misclibs.F:1423`` itself cites.  25 m2/s2 is the value that paper
#: uses to flag a rotating updraft; 100 m2/s2 is its strong-rotation value and
#: the one SPC and the operational convection-allowing models carry as the
#: supercell/tornado proxy.
UH_ROTATING_UPDRAFT_M2_S2 = 25.0
UH_STRONG_ROTATION_M2_S2 = 100.0
#: Wicker and Wilhelmson 1995, *J. Atmos. Sci.* **52**, 2675-2703, and the
#: tornado-like-vortex literature that follows it: 0.1 s-1 of vertical
#: vorticity at the lowest model level is the usual TLV criterion.
NEAR_SURFACE_VORTEX_S_INV = 0.1
#: The same literature's sustaining requirement: a two-minute life is the
#: shortest that is conventionally called a vortex rather than a transient.
VORTEX_MINIMUM_DURATION_S = 120.0
#: WSEC 2006, *A Recommendation for an Enhanced Fujita Scale*, Texas Tech
#: University.  Three-second gusts at 10 m above ground, m/s, converted from
#: the published mph bounds.  A model's lowest-level grid-volume mean wind is
#: NOT a three-second gust at 10 m, which is why ``tornado_report`` reports
#: the band a wind would fall in and refuses to call it an EF rating.
EF_SCALE_WIND_M_S = ((0, 29.0), (1, 38.0), (2, 50.0), (3, 61.0), (4, 74.0), (5, 89.0))
#: Bryan, Wyngaard and Fritsch 2003, *Mon. Wea. Rev.* **131**, 2394-2416: a
#: simulated feature is resolved only at roughly six grid lengths.
EFFECTIVE_RESOLUTION_CELLS = 6.0
#: A significant tornado's core is a few hundred metres across; 200 m is the
#: conventional working diameter for "can this grid hold a tornado at all".
TORNADO_CORE_DIAMETER_M = 200.0
#: A tornado is a surface-interaction phenomenon: the corner-flow layer that
#: makes it is tens of metres deep, so the lowest model layer has to be too.
TORNADO_NEAR_GROUND_LAYER_M = 50.0

THRESHOLD_AUTHORITY = MappingProxyType({
    "updraft_helicity": (
        "Kain et al. 2008, Wea. Forecasting 23, 931-952 -- the reference CM1 "
        "itself cites at src/misclibs.F:1423"
    ),
    "near_surface_vorticity": (
        "Wicker and Wilhelmson 1995, J. Atmos. Sci. 52, 2675-2703 and the "
        "tornado-like-vortex literature following it"
    ),
    "wind_bands": (
        "WSEC 2006, A Recommendation for an Enhanced Fujita Scale, Texas Tech "
        "University; three-second gust at 10 m, which a model grid-volume "
        "mean wind is not"
    ),
    "resolution": (
        "Bryan, Wyngaard and Fritsch 2003, Mon. Wea. Rev. 131, 2394-2416"
    ),
})


# ---------------------------------------------------------------------------
# Selection: CM1's own output_* switches
# ---------------------------------------------------------------------------

#: Every CM1 ``output_*`` switch this module implements, in ``input.F`` order.
OUTPUT_SWITCH_NAMES = (
    "output_uh", "output_vort", "output_pv",
    "output_sws", "output_svs", "output_sps",
    "output_srs", "output_sgs", "output_sus", "output_shs",
)
#: The switches whose field is a whole-run surface swath.
SWATH_SWITCH_BY_NAME = MappingProxyType({
    "sws": "output_sws", "svs": "output_svs", "sps": "output_sps",
    "srs": "output_srs", "sgs": "output_sgs", "sus": "output_sus",
    "shs": "output_shs",
})


@dataclass(frozen=True, slots=True)
class TornadoDiagnosticSelection:
    """CM1's ``output_*`` switches plus one GM1 report switch.

    Every name but ``tornado_report`` is CM1 r21.1's own namelist name and
    carries CM1's own 0/1 value, so a reader of a GM1 receipt can find the
    branch it selected in ``src/writeout.F`` and ``src/solve3.F`` by name.
    ``tornado_report`` has no CM1 counterpart because CM1 makes no tornado
    claim; it is GM1's, and it is a boolean.

    All-off is the default and is the state every GM1 run integrated before
    this module existed: nothing is computed, nothing is allocated, nothing is
    published, and the run's bytes are exactly what they were.
    """

    output_uh: int = 0
    output_vort: int = 0
    output_pv: int = 0
    output_sws: int = 0
    output_svs: int = 0
    output_sps: int = 0
    output_srs: int = 0
    output_sgs: int = 0
    output_sus: int = 0
    output_shs: int = 0
    tornado_report: bool = False

    def __post_init__(self) -> None:
        for name in OUTPUT_SWITCH_NAMES:
            value = getattr(self, name)
            if isinstance(value, bool) or type(value) is not int:
                raise TypeError(
                    f"{name} must be CM1's integer 0/1 switch, not {type(value).__name__}"
                )
            if value not in (0, 1):
                raise ValueError(
                    f"{name}={value} is not one of CM1's two switch values; "
                    "param.F:4677 clamps it to max(0,min(1,...)), so CM1 would "
                    "run a different branch than the number the receipt records"
                )
        if type(self.tornado_report) is not bool:
            raise TypeError("tornado_report must be a boolean")

    @property
    def active(self) -> bool:
        """True when this selection asks for any work at all."""

        return bool(self.tornado_report or any(
            getattr(self, name) for name in OUTPUT_SWITCH_NAMES
        ))

    @property
    def swath_names(self) -> tuple[str, ...]:
        """The swaths this selection keeps, in CM1's ``solve3.F`` order."""

        return tuple(
            name for name in SWATH_NAMES if getattr(self, SWATH_SWITCH_BY_NAME[name])
        )

    @property
    def needs_updraft_helicity(self) -> bool:
        """``shs`` is a running maximum of ``calcuh``'s own output field."""

        return bool(self.output_uh or self.output_shs)

    @property
    def needs_vorticity(self) -> bool:
        return bool(self.output_vort or self.output_pv)

    def record(self) -> dict[str, Any]:
        row: dict[str, Any] = {name: getattr(self, name) for name in OUTPUT_SWITCH_NAMES}
        row["tornado_report"] = self.tornado_report
        row["authority"] = (
            "CM1 r21.1 src/input.F:120 output_vort/output_pv/output_uh, "
            "src/writeout.F:408-545 the surface swaths, src/solve3.F:855-1136 "
            "their running extrema; tornado_report is GM1's own and CM1 has no "
            "counterpart"
        )
        return row


DEFAULT_TORNADO_DIAGNOSTICS = TornadoDiagnosticSelection()


def validate_tornado_diagnostic_selection(
    selection: TornadoDiagnosticSelection,
    *,
    grid: Any,
    microphysics: str,
    bbc: int,
    option_prefix: str = "diagnostics",
) -> None:
    """Refuse every selection this runtime would publish as something else.

    Each refusal names the concrete wrong number that would otherwise reach an
    artifact.  A selection that asks for nothing is admitted without reading
    the grid at all, so the default costs nothing and can refuse nothing.
    """

    if not isinstance(selection, TornadoDiagnosticSelection):
        raise TypeError("selection must be TornadoDiagnosticSelection")
    if not selection.active:
        return
    nx, ny, nz = int(grid.nx), int(grid.ny), int(grid.nz)
    ztop = float(grid.ztop_m)
    faces = tuple(float(value) for value in grid.face_heights_m)
    heights = tuple(float(value) for value in grid.mass_heights_m)

    if selection.needs_vorticity or selection.needs_updraft_helicity or selection.output_svs:
        if nx < 4 or ny < 4:
            raise ValueError(
                f"{option_prefix}: a corner-averaged vorticity needs four "
                f"cells in each horizontal direction and this grid is "
                f"{nx}x{ny}; with fewer there is no interior cell whose four "
                "corners are all differenceable, so every published word "
                "would be boundary closure and none of it model vorticity"
            )
    if selection.needs_updraft_helicity:
        if ztop < UPDRAFT_HELICITY_TOP_M:
            raise ValueError(
                f"{option_prefix}.output_uh/output_shs: this grid's top is "
                f"{ztop} m and CM1's updraft-helicity layer "
                f"(misclibs.F:1441-1442) runs to {UPDRAFT_HELICITY_TOP_M} m "
                "AGL; the column integral would stop at the model top and be "
                "published under the 2-5 km name while covering less of the "
                "layer than it claims"
            )
        if not any(
            UPDRAFT_HELICITY_BOTTOM_M <= value <= UPDRAFT_HELICITY_TOP_M
            for value in heights
        ):
            raise ValueError(
                f"{option_prefix}.output_uh/output_shs: no scalar level of "
                f"this grid lies between {UPDRAFT_HELICITY_BOTTOM_M} and "
                f"{UPDRAFT_HELICITY_TOP_M} m, so misclibs.F:1520's layer test "
                "selects no layer and every published updraft helicity would "
                "be exactly zero regardless of the storm"
            )
    if selection.output_sus and ztop < UPDRAFT_SWATH_HEIGHT_M:
        raise ValueError(
            f"{option_prefix}.output_sus: this grid's top is {ztop} m and "
            f"solve3.F:1088-1092 scans upward for the first level at or above "
            f"{UPDRAFT_SWATH_HEIGHT_M} m AGL; the scan would stop at the model "
            "top and the swath would record the maximum w at the model top "
            "under the 5 km name"
        )
    if (selection.output_srs or selection.output_sgs) and not _has_rain_graupel_rows(microphysics):
        raise ValueError(
            f"{option_prefix}.output_srs/output_sgs: solve3.F:999-1000 and "
            f"1025-1026 guard both swaths on the microphysics having a "
            f"rainwater and a graupel species, and {microphysics!r} carries "
            "no qr and qg rows at CM1's nqr = 3 and nqg = 6; the swath would "
            "read whichever tracer happened to occupy index 2 and 5 and "
            "publish it as qr and qg"
        )
    if int(bbc) not in (1, 2, 3):
        raise ValueError(
            f"{option_prefix}: calcvort has a lower-boundary branch for "
            f"bbc 1, 2 and 3 only (misclibs.F:1634-1659, 2165-2172) and this "
            f"runtime declares bbc={bbc}; the lowest level of xvort and yvort "
            "would be left holding the free-slip copy of the level above "
            "while the receipt named another boundary"
        )
    if selection.tornado_report and not (
        selection.output_svs and selection.output_sws and selection.output_sps
    ):
        raise ValueError(
            f"{option_prefix}.tornado_report: the report states peak "
            "near-surface vertical vorticity, peak near-surface wind and "
            "minimum surface pressure perturbation, and those three numbers "
            "are exactly the svs, sws and sps swaths; with any of them off "
            "the report would print its headline numbers as absent for a run "
            "that produced them -- set output_svs, output_sws and output_sps"
        )
    if len(faces) != nz + 1 or len(heights) != nz:
        raise ValueError(
            f"{option_prefix}: this grid reports {len(heights)} scalar and "
            f"{len(faces)} face heights for nz={nz}; a vertical coordinate "
            "that does not describe its own levels would place every layer "
            "thickness in the updraft-helicity integral at the wrong height"
        )


# ---------------------------------------------------------------------------
# Small typed helpers
# ---------------------------------------------------------------------------

def _f32(xp: Any, value: float) -> Any:
    return xp.float32(value)


def _require_shape(name: str, array: Any, shape: tuple[int, ...]) -> None:
    if tuple(array.shape) != tuple(shape):
        raise ValueError(
            f"{name} has extent {tuple(array.shape)} but this grid needs "
            f"{tuple(shape)}; reading it anyway would place one column's "
            "words at another column's position"
        )


@dataclass(frozen=True, slots=True)
class VerticalColumn:
    """CM1's own vertical coordinate for one Cartesian column, zero-based.

    ``zh``/``zf`` are heights above ground in metres; ``mh``/``mf`` are CM1's
    stretching metrics (``param.F:8168-8176``), both exactly 1.0 on the
    uniform coordinate ``param.F:7789-7811`` builds.  ``rdz`` is CM1's
    ``1/dz`` reference, which is the number every ``*rdz*mf(k)`` product is
    divided by, so the pair reproduces a stretched column as well as a
    uniform one without a second code path.
    """

    zh_m: tuple[float, ...]
    zf_m: tuple[float, ...]
    mh: tuple[float, ...]
    mf: tuple[float, ...]
    rdz: float
    dgs: tuple[float, float, float]
    dgt: tuple[float, float, float]

    @property
    def nz(self) -> int:
        return len(self.zh_m)


def vertical_column(grid: Any) -> VerticalColumn:
    """Read one Cartesian geometry's CM1 vertical coordinate.

    ``mf`` is CM1-indexed ``1..nk+1``: the metric on the w faces.  A uniform
    grid carries none of these arrays, so the uniform values ``param.F:8170``
    would produce are used directly rather than materialized.
    """

    import numpy as np

    from .cm1_vertical_grid import fornberg_first_derivative_weights

    nz = int(grid.nz)
    zh = tuple(float(value) for value in grid.mass_heights_m)
    zf = tuple(float(value) for value in grid.face_heights_m)
    rdz = 1.0 / float(grid.dz_m)
    if grid.vertical is None:
        one = tuple(1.0 for _ in range(nz))
        # param.F:9042-9052 on equally spaced ends.  The quadratic through
        # three levels of one spacing has first-derivative weights
        # (-2, 3, -1) at the ground face and (2, -3, 1) at the top face, in
        # units of the reference dz; the sign differs because the top face
        # lies above its three nodes and the ground face below its own.
        return VerticalColumn(
            zh, zf, one, one + (1.0,), rdz, (-2.0, 3.0, -1.0), (2.0, -3.0, 1.0)
        )
    levels = grid.vertical
    mh_words = np.asarray(levels.mh)
    mf_words = np.asarray(levels.mf)
    if mh_words.shape[0] < nz + 1 or mf_words.shape[0] < nz + 2:
        raise ValueError(
            f"this vertical coordinate carries {mh_words.shape[0]} mh and "
            f"{mf_words.shape[0]} mf words for nz={nz}; CM1 indexes mh 1..nk "
            "and mf 1..nk+1, so a shorter array would supply the wrong layer "
            "metric at the model top"
        )
    mh = tuple(float(value) for value in mh_words[1:nz + 1])
    mf = tuple(float(value) for value in mf_words[1:nz + 2])
    dgs = fornberg_first_derivative_weights(levels.sigma, levels.sigmaf, nk=nz, top=False)
    dgt = fornberg_first_derivative_weights(levels.sigma, levels.sigmaf, nk=nz, top=True)
    return VerticalColumn(zh, zf, mh, mf, rdz, dgs, dgt)


def _close_y_edges(xp: Any, plane: Any) -> Any:
    """Copy each outer ``y`` row from the nearest interior row."""

    ny = plane.shape[0]
    plane[0, :] = plane[1, :]
    plane[ny - 1, :] = plane[ny - 2, :]
    return plane


def _close_x_edges(xp: Any, plane: Any) -> Any:
    """Copy each outer ``x`` column from the nearest interior column."""

    nx = plane.shape[1]
    plane[:, 0] = plane[:, 1]
    plane[:, nx - 1] = plane[:, nx - 2]
    return plane


def _close_boundary_ring(xp: Any, plane: Any, interior: Any) -> Any:
    """Write an interior result into a full plane, closing the outer ring.

    ``interior`` covers cells ``1..ny-2`` by ``1..nx-2``: the only cells whose
    four corners are all differenceable in a runtime that publishes no metric
    halo.  The outer ring is copied from the nearest interior cell, which is
    exactly the source-index clamp ``cuda_storm_diagnostics.py`` already
    applies to the same corner average, and is display closure rather than
    model vorticity.  Nothing is extrapolated and nothing is invented.
    """

    ny, nx = plane.shape
    plane[1:ny - 1, 1:nx - 1] = interior
    return _close_x_edges(xp, _close_y_edges(xp, plane))


#: How many outer rows and columns each field could not difference.
#:
#: ``"none"``  every word is model output: no horizontal difference is taken
#:             (``sws``, ``sps``, ``srs``, ``sgs``, ``sus``).
#: ``"y"``     the two ``y`` edge rows (``xvort``: no ``x`` difference at all).
#: ``"x"``     the two ``x`` edge columns (``yvort``).
#: ``"first"`` the first row and first column (``svs``: CM1 stores the
#:             vorticity point itself, so only the corner below and left of
#:             the domain is unavailable).
#: ``"both"``  the whole ring (every corner-averaged field: ``zvort``, ``uh``,
#:             ``shs``, ``pv``).
CLOSED_EDGES = ("none", "x", "y", "first", "both")


def boundary_closure_words(nx: int, ny: int, *, edges: str = "both") -> dict[str, int]:
    """How many words of one plane are closure rather than model output."""

    if edges not in CLOSED_EDGES:
        raise ValueError(f"edges must be one of {CLOSED_EDGES}")
    if edges == "first":
        interior = max(nx - 1, 0) * max(ny - 1, 0)
    else:
        inner_x = max(nx - 2, 0) if edges in ("x", "both") else nx
        inner_y = max(ny - 2, 0) if edges in ("y", "both") else ny
        interior = inner_x * inner_y
    return {
        "plane_words": nx * ny,
        "exact_interior_words": interior,
        "display_closed_boundary_words": nx * ny - interior,
        "closed_edges": edges,
    }


#: Which edges each swath's own field could not difference.
SWATH_CLOSED_EDGES = MappingProxyType({
    "sws": "none", "svs": "first", "sps": "none", "srs": "none",
    "sgs": "none", "sus": "none", "shs": "both",
})


# ---------------------------------------------------------------------------
# Corner vertical vorticity -- misclibs.F:1454-1461 and :2041-2044
# ---------------------------------------------------------------------------

def corner_vertical_vorticity(
    xp: Any, *, u: Any, v: Any, level: int, rdx: Any, rdy: Any,
    out: Any | None = None, scratch: Any | None = None,
) -> Any:
    """``zeta`` on the interior corners of one model level.

    ``misclibs.F:1457-1458`` (calcuh) and ``misclibs.F:2041-2042`` (calcvort)
    form the same expression: the ``x`` difference of ``v`` across the corner
    times ``rdx*uf``, minus the ``y`` difference of ``u`` across it times
    ``rdy*vf``, in that order.

    ``uf``/``vf`` are CM1's horizontal stretching factors and are exactly 1.0
    on this lane's uniform grid (``param.F``'s uniform-horizontal branch, the
    same words ``cuda_cartesian_dynamics.py:184-186`` fills).  The returned
    plane is indexed ``[cj-1, ci-1]`` for corner ``(ci, cj)`` with
    ``ci = 1..nx-1`` and ``cj = 1..ny-1``: the corners a runtime with no halo
    can difference.
    """

    nz, ny, nx = v.shape[0], v.shape[1] - 1, v.shape[2]
    if u.shape != (nz, ny, nx + 1):
        raise ValueError(
            f"u has extent {tuple(u.shape)} and v {tuple(v.shape)}; the two "
            "do not describe one staggered Cartesian grid, so the difference "
            "would pair a u face with a v face one cell away"
        )
    k = int(level)
    if not 0 <= k < nz:
        raise ValueError(f"level {k} is outside this grid's {nz} scalar levels")
    plane = xp.empty((ny - 1, nx - 1), dtype=xp.float32) if out is None else out
    _require_shape("corner vorticity plane", plane, (ny - 1, nx - 1))
    term = xp.empty_like(plane) if scratch is None else scratch
    _require_shape("corner vorticity scratch", term, (ny - 1, nx - 1))
    xp.subtract(v[k, 1:ny, 1:nx], v[k, 1:ny, 0:nx - 1], out=plane)
    xp.multiply(plane, rdx, out=plane)
    xp.subtract(u[k, 1:ny, 1:nx], u[k, 0:ny - 1, 1:nx], out=term)
    xp.multiply(term, rdy, out=term)
    xp.subtract(plane, term, out=plane)
    return plane


def _cell_from_corners(xp: Any, corners: Any, out: Any, scratch: Any | None = None) -> Any:
    """``0.25*(zeta(i,j)+zeta(i+1,j)+zeta(i,j+1)+zeta(i+1,j+1))``.

    ``misclibs.F:1524-1525`` and ``misclibs.F:2047``.  Fortran evaluates the
    parenthesised sum left to right, so the grouping is ``((A+B)+C)+D`` and
    that grouping is kept: on binary32 it is not the same number as any other.
    """

    ny, nx = out.shape
    if ny < 4 or nx < 4:
        raise ValueError(
            f"a corner average needs four cells in each direction and this "
            f"plane is {ny}x{nx}; with fewer there is no cell whose four "
            "corners are all differenceable"
        )
    interior = (
        xp.empty((ny - 2, nx - 2), dtype=xp.float32) if scratch is None else scratch
    )
    _require_shape("cell average scratch", interior, (ny - 2, nx - 2))
    xp.add(corners[0:ny - 2, 0:nx - 2], corners[0:ny - 2, 1:nx - 1], out=interior)
    xp.add(interior, corners[1:ny - 1, 0:nx - 2], out=interior)
    xp.add(interior, corners[1:ny - 1, 1:nx - 1], out=interior)
    xp.multiply(interior, _f32(xp, 0.25), out=interior)
    return _close_boundary_ring(xp, out, interior)


def vertical_vorticity_level(
    xp: Any, *, u: Any, v: Any, level: int, rdx: Any, rdy: Any, out: Any | None = None
) -> Any:
    """CM1 ``zvort`` at scalar points on one model level.

    ``misclibs.F:2037-2049``.  Level zero is the lowest model level, which is
    where a tornado shows up and where ``solve3.F:889-899`` takes the ``svs``
    swath.
    """

    ny, nx = v.shape[1] - 1, v.shape[2]
    plane = xp.empty((ny, nx), dtype=xp.float32) if out is None else out
    _require_shape("vertical vorticity plane", plane, (ny, nx))
    corners = corner_vertical_vorticity(xp, u=u, v=v, level=level, rdx=rdx, rdy=rdy)
    return _cell_from_corners(xp, corners, plane)


# ---------------------------------------------------------------------------
# calcuh -- misclibs.F:1417-1532
# ---------------------------------------------------------------------------

def updraft_helicity(
    xp: Any,
    *,
    u: Any,
    v: Any,
    w: Any,
    column: VerticalColumn,
    rdx: Any,
    rdy: Any,
    out: Any | None = None,
) -> Any:
    """CM1's vertically integrated updraft helicity, m2/s2.

    ``misclibs.F:1417-1532``, Cartesian branch, operation for operation: for
    every level whose scalar height lies in 2-5 km, ``wbar`` is half the sum of
    ``w`` on the level's two faces and ``zbar`` a quarter of the sum of
    ``zeta`` on the cell's four corners, each clipped below at zero, and the
    column total gains the part of the layer inside 2-5 km times ``wbar``
    times ``zbar``, multiplied in that order.

    Both clips are CM1's and they are the whole physical content of the
    diagnostic: only a cyclonically rotating updraft contributes, so a
    downdraft over anticyclonic rotation -- which has the same sign of
    product -- adds nothing (``misclibs.F:1521-1522``).

    The accumulation is CM1's serial ``k`` loop, not a reduction, so the
    result does not depend on which array backend evaluates it.
    """

    nz, ny, nx = v.shape[0], v.shape[1] - 1, v.shape[2]
    _require_shape("w", w, (nz + 1, ny, nx))
    if column.nz != nz:
        raise ValueError(
            f"the vertical column describes {column.nz} levels and the wind "
            f"field {nz}; the layer thicknesses would be taken from another "
            "grid's heights"
        )
    plane = xp.zeros((ny, nx), dtype=xp.float32) if out is None else out
    _require_shape("updraft helicity plane", plane, (ny, nx))
    plane[...] = _f32(xp, 0.0)
    zero = _f32(xp, 0.0)
    half = _f32(xp, 0.5)
    zbar = xp.empty((ny, nx), dtype=xp.float32)
    wbar = xp.empty((ny, nx), dtype=xp.float32)
    corners = xp.empty((ny - 1, nx - 1), dtype=xp.float32)
    corner_scratch = xp.empty((ny - 1, nx - 1), dtype=xp.float32)
    cell_scratch = xp.empty((ny - 2, nx - 2), dtype=xp.float32)
    for k in range(nz):
        height = column.zh_m[k]
        if not (UPDRAFT_HELICITY_BOTTOM_M <= height <= UPDRAFT_HELICITY_TOP_M):
            continue
        thickness = _f32(xp, (
            min(column.zf_m[k + 1], UPDRAFT_HELICITY_TOP_M)
            - max(column.zf_m[k], UPDRAFT_HELICITY_BOTTOM_M)
        ))
        xp.add(w[k], w[k + 1], out=wbar)
        xp.multiply(wbar, half, out=wbar)
        xp.maximum(wbar, zero, out=wbar)
        corner_vertical_vorticity(
            xp, u=u, v=v, level=k, rdx=rdx, rdy=rdy, out=corners, scratch=corner_scratch
        )
        _cell_from_corners(xp, corners, zbar, cell_scratch)
        xp.maximum(zbar, zero, out=zbar)
        # ``(thickness*wbar)*zbar``: Fortran's own left-to-right grouping.
        xp.multiply(wbar, thickness, out=wbar)
        xp.multiply(wbar, zbar, out=wbar)
        xp.add(plane, wbar, out=plane)
    return plane


# ---------------------------------------------------------------------------
# calcvort -- misclibs.F:1540-2174
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class LowerBoundaryState:
    """What ``calcvort``'s ``bbc`` branches need beyond the wind field.

    ``bbc=2`` needs only the frame velocity, because ``misclibs.F:1655`` and
    ``misclibs.F:1891`` reconstruct the no-slip image point from the
    ground-relative wind at the lowest level.  ``bbc=3`` needs the surface
    layer's ``ust``, ``znt`` and lowest-level wind, which
    ``misclibs.F:2165-2172`` divides by to build the log-layer horizontal
    vorticity; ``cnstcd`` is enough to rebuild them for ``cecd=1``, CM1's
    constant drag coefficient branch (``sfcphys.F:120-136``).  For the
    wind-dependent branches (``cecd=2`` and ``3``, ``sfcphys.F:138-195``) the
    roughness length is the run's own carried state and is handed to
    :func:`surface_layer_state` beside this record.
    """

    bbc: int = 1
    tbc: int = 1
    u_move_m_s: float = 0.0
    v_move_m_s: float = 0.0
    cecd: int = 1
    cnstcd: float = 0.001


def surface_layer_state(
    xp: Any, *, u: Any, v: Any, lower: LowerBoundaryState, za_m: float, znt: Any = None
) -> dict[str, Any]:
    """CM1's ``getsfc``/``getcecd`` pair for the constant-Cd branch, and the
    friction velocity of the wind-dependent branches from the run's own
    roughness length.

    ``turb.F:539-543`` averages the *ground-relative* lowest-level wind to the
    scalar point; ``sfcphys.F:126-135`` turns the declared constant drag
    coefficient into ``znt`` and ``ust``.  ``za`` is the lowest scalar level
    above ground, ``turb.F:544``.

    This is recomputed here rather than read from the closure because the
    closure's surface carriers live inside its held phase and are released
    with it; a diagnostic at the post-solve3 boundary can see none of them.
    The arithmetic is the same source, so the numbers are the same numbers.

    ``cecd=2`` (Deacon) and ``cecd=3`` (Fairall/Donelan) make the roughness
    length over water a function of the 10 m wind and find it by an
    iteration that starts from the last known ``z0`` (``sfcphys.F:138-178``),
    so it cannot be rebuilt from the wind alone.  It does not have to be:
    ``getcecd`` writes it back, and the closure keeps that plane between
    steps (its ``surface_state["znt"]``, which is what the parcel step reads
    too).  Given that plane as ``znt``, ``sfcphys.F:190`` turns it into the
    friction velocity with the same statement the constant branch's
    ``sfcphys.F:135`` uses, over water and land alike, and ``cd`` follows
    from inverting ``sfcphys.F:167``.  The roughness is the one the run's
    last ``getcecd`` call left, and the wind is this boundary's, as it is on
    the constant branch.  Without ``znt`` a wind-dependent ``cecd`` is
    refused as before: a roughness invented here would report a friction
    velocity the run never used.
    """

    wind_dependent = int(lower.cecd) != 1
    if wind_dependent and int(lower.cecd) not in (2, 3):
        raise ValueError(
            f"cecd={lower.cecd} is not one of sfcphys.F's three drag branches (1, 2, 3)"
        )
    if wind_dependent and znt is None:
        raise ValueError(
            f"cecd={lower.cecd} selects one of sfcphys.F's wind-dependent "
            "drag branches, which iterate on the 10 m wind; only cecd=1, the "
            "constant-Cd branch of sfcphys.F:120-136, is transcribed, and "
            "running it under another cecd would report a friction velocity "
            "the run never used"
        )
    ny, nx = v.shape[1] - 1, v.shape[2]
    half = _f32(xp, 0.5)
    umove = _f32(xp, lower.u_move_m_s)
    vmove = _f32(xp, lower.v_move_m_s)
    u1 = xp.empty((ny, nx), dtype=xp.float32)
    v1 = xp.empty((ny, nx), dtype=xp.float32)
    s1 = xp.empty((ny, nx), dtype=xp.float32)
    # turb.F:296-307 forms ugr/vgr; turb.F:539-541 averages them.
    xp.add(xp.add(u[0, :, 0:nx], umove), xp.add(u[0, :, 1:nx + 1], umove), out=u1)
    xp.multiply(u1, half, out=u1)
    xp.add(xp.add(v[0, 0:ny, :], vmove), xp.add(v[0, 1:ny + 1, :], vmove), out=v1)
    xp.multiply(v1, half, out=v1)
    xp.multiply(u1, u1, out=s1)
    xp.add(s1, xp.multiply(v1, v1), out=s1)
    xp.sqrt(s1, out=s1)
    if wind_dependent:
        roughness = xp.asarray(znt, dtype=xp.float32)
        if tuple(roughness.shape) != (ny, nx):
            raise ValueError(
                f"znt has shape {tuple(roughness.shape)}, but the surface layer of this grid is "
                f"({ny}, {nx}): the roughness length must be the run's own whole-domain plane"
            )
        if not bool(xp.all(roughness > _f32(xp, 0.0))):
            raise ValueError(
                "znt must be positive everywhere: sfcphys.F:189 divides by it, and a zero or "
                "negative roughness length has no logarithmic layer above it"
            )
        # sfcphys.F:189-190: rznt = 1.0/znt; ust = max(s1*karman/alog((za+znt)*rznt), 1.0e-6)
        rznt = xp.divide(_f32(xp, 1.0), roughness)
        level_plane = xp.log(xp.multiply(xp.add(roughness, _f32(xp, float(za_m))), rznt))
        ust = xp.maximum(
            xp.divide(xp.multiply(s1, _f32(xp, CM1_KARMAN)), level_plane), _f32(xp, 1.0e-6)
        )
        # sfcphys.F:167 inverted: znt = 10/(exp(karman/sqrt(cd)) - 1), so cd = (karman/alog((10+znt)*rznt))**2.
        cd_plane = xp.square(xp.divide(
            _f32(xp, CM1_KARMAN), xp.log(xp.multiply(xp.add(roughness, _f32(xp, 10.0)), rznt))
        ))
        return {"u1": u1, "v1": v1, "s1": s1, "znt": roughness, "ust": ust, "cd": cd_plane,
                "za_m": float(za_m)}
    # sfcphys.F:126-128
    cd = max(1.0e-10, float(lower.cnstcd))
    znt = 10.0 / (math.exp(CM1_KARMAN / math.sqrt(cd)) - 1.0)
    if not math.isfinite(znt) or znt <= 0.0:
        raise ValueError(
            f"cnstcd={lower.cnstcd} puts sfcphys.F:128's exp(karman/sqrt(Cd)) "
            "outside the range where the roughness length is finite and "
            "positive; every ust built from it would be infinite or zero"
        )
    level = math.log((float(za_m) + znt) / znt)
    # sfcphys.F:135
    ust = xp.maximum(
        xp.multiply(s1, _f32(xp, CM1_KARMAN / level)), _f32(xp, 1.0e-6)
    )
    return {"u1": u1, "v1": v1, "s1": s1, "znt": znt, "ust": ust, "cd": cd, "za_m": float(za_m)}


def _allocator(xp: Any, allocate: Any) -> Any:
    """One named-allocation callable for every scratch this module takes.

    ``allocate(name, shape, dtype)`` lets a caller route every temporary
    through its own device registry, so a diagnostic's transient peak is
    priced and released like every other phase in this runtime.  With no
    allocator the arrays come straight from the array module, which is what
    a host test wants.
    """

    if allocate is not None:
        return lambda name, shape, zero=False: (
            _zeroed(xp, allocate(name, shape, xp.float32), zero)
        )
    return lambda name, shape, zero=False: (
        xp.zeros(shape, dtype=xp.float32) if zero
        else xp.empty(shape, dtype=xp.float32)
    )


def _zeroed(xp: Any, array: Any, zero: bool) -> Any:
    if zero:
        array[...] = xp.float32(0.0)
    return array


def _apply_face_metric(xp: Any, array: Any, column: VerticalColumn, kf: int) -> Any:
    """Multiply by ``rdz`` and then by ``mf(k)``, in CM1's own two steps.

    ``misclibs.F:1629`` multiplies by ``rdz`` and then by the mean of the two
    ``mf`` words either side of the corner.  On this
    lane's horizontally uniform grid the two ``mf`` words are the same word,
    so the average is that word exactly; the two multiplications are kept
    separate because ``x*rdz*mf`` and ``x*float32(rdz*mf)`` are not the same
    binary32 number on a stretched column.
    """

    xp.multiply(array, _f32(xp, column.rdz), out=array)
    xp.multiply(array, _f32(xp, column.mf[kf]), out=array)
    return array


def vorticity_components(
    xp: Any,
    *,
    u: Any,
    v: Any,
    w: Any,
    column: VerticalColumn,
    rdx: Any,
    rdy: Any,
    lower: LowerBoundaryState,
    allocate: Any = None,
    release: Any = None,
    out: Mapping[str, Any] | None = None,
    surface_znt: Any = None,
) -> dict[str, Any]:
    """CM1 ``xvort``/``yvort``/``zvort`` at scalar points, 1/s.

    ``surface_znt`` is the run's own roughness-length plane, which a
    ``bbc=3`` boundary with a wind-dependent drag (``cecd=2`` or ``3``)
    needs for its log-layer vorticity (:func:`surface_layer_state`); the
    constant-drag branch does not read it.

    ``misclibs.F:1540-2174``, Cartesian grid without terrain.  The winds are
    the *ground-relative* ones (``misclibs.F:1551``, ``writeout.F:291-296``),
    which matters only at a non-free-slip lower boundary: the frame velocity
    cancels out of every interior difference and does not cancel out of the
    image point ``misclibs.F:1655`` builds, so it is carried in ``lower``
    rather than added to the whole field.

    ``xvort`` is CM1's ``dw/dy - dv/dz``, ``yvort`` its ``du/dz - dw/dx``, and
    ``zvort`` its ``dv/dx - du/dy``; each is formed on its own vorticity point
    and then averaged to the scalar point with CM1's four-term average in
    CM1's term order.

    Which outer rows and columns are closure rather than model output differs
    per component and is not glossed over: ``xvort`` takes no ``x``
    difference, so only its two ``y`` edge rows are closed; ``yvort`` only its
    two ``x`` edge columns; ``zvort``, a corner average, the whole ring.
    """

    nz, ny, nx = v.shape[0], v.shape[1] - 1, v.shape[2]
    _require_shape("w", w, (nz + 1, ny, nx))
    _require_shape("u", u, (nz, ny, nx + 1))
    if column.nz != nz:
        raise ValueError(
            f"the vertical column describes {column.nz} levels and the wind "
            f"field {nz}; every vertical derivative would use another grid's "
            "layer depths"
        )
    if nx < 4 or ny < 4:
        raise ValueError(
            f"calcvort differences neighbouring vorticity points and this "
            f"grid is {nx}x{ny}; with fewer than four cells in a direction "
            "every published word would be boundary closure"
        )
    bbc, tbc = int(lower.bbc), int(lower.tbc)
    if bbc not in (1, 2, 3):
        raise ValueError(
            f"calcvort has a lower-boundary branch for bbc 1, 2 and 3 only "
            f"and this state declares bbc={bbc}"
        )
    if tbc not in (1, 2):
        raise ValueError(
            f"calcvort has an upper-boundary branch for tbc 1 and 2 only and "
            f"this state declares tbc={tbc}"
        )
    quarter = _f32(xp, 0.25)
    two = _f32(xp, 2.0)
    minus_two = _f32(xp, -2.0)
    new = _allocator(xp, allocate)
    drop = release if release is not None else (lambda *names: None)
    # The three components are the caller's persistent carriers when it owns
    # them; only the temporaries below go through the registry, and every one
    # is released before the next is taken.
    if out is not None:
        for name in ("xvort", "yvort", "zvort"):
            _require_shape(name, out[name], (nz, ny, nx))

    # -- xvort: tem lives at (scalar x, v face y, w face z) ----------------
    tem = new("vort.tem_x", (nz + 1, ny + 1, nx), zero=True)
    scratch = new("vort.scratch_x", (ny - 1, nx))
    for kf in range(1, nz):
        # misclibs.F:1628-1629
        first = xp.subtract(w[kf, 1:ny, :], w[kf, 0:ny - 1, :])
        xp.multiply(first, rdy, out=first)
        xp.subtract(v[kf, 1:ny, :], v[kf - 1, 1:ny, :], out=scratch)
        _apply_face_metric(xp, scratch, column, kf)
        xp.subtract(first, scratch, out=tem[kf, 1:ny, :])
    if bbc == 1:
        # misclibs.F:1634-1642
        tem[0] = tem[1]
    else:
        # misclibs.F:1644-1659: the no-slip image point.  bbc=3's lowest
        # level is overwritten by the log-layer form further down.
        xp.add(v[0, 1:ny, :], _f32(xp, lower.v_move_m_s), out=scratch)
        xp.multiply(scratch, minus_two, out=scratch)
        _apply_face_metric(xp, scratch, column, 0)
        tem[0, 1:ny, :] = scratch
    if tbc == 1:
        # misclibs.F:1661-1669
        tem[nz] = tem[nz - 1]
    else:
        # misclibs.F:1671-1679
        xp.multiply(v[nz - 1, 1:ny, :], two, out=scratch)
        _apply_face_metric(xp, scratch, column, nz)
        tem[nz, 1:ny, :] = scratch
    xvort = new("vort.xvort", (nz, ny, nx)) if out is None else out["xvort"]
    inner = new("vort.inner_x", (ny - 2, nx))
    for k in range(nz):
        # misclibs.F:1684
        xp.add(tem[k, 1:ny - 1, :], tem[k, 2:ny, :], out=inner)
        xp.add(inner, tem[k + 1, 1:ny - 1, :], out=inner)
        xp.add(inner, tem[k + 1, 2:ny, :], out=inner)
        xp.multiply(inner, quarter, out=inner)
        xvort[k, 1:ny - 1, :] = inner
        _close_y_edges(xp, xvort[k])
    del tem, scratch, inner
    drop("vort.tem_x", "vort.scratch_x", "vort.inner_x")

    # -- yvort: tem lives at (u face x, scalar y, w face z) ----------------
    tem = new("vort.tem_y", (nz + 1, ny, nx + 1), zero=True)
    scratch = new("vort.scratch_y", (ny, nx - 1))
    for kf in range(1, nz):
        # misclibs.F:1864-1865
        xp.subtract(u[kf, :, 1:nx], u[kf - 1, :, 1:nx], out=scratch)
        _apply_face_metric(xp, scratch, column, kf)
        second = xp.subtract(w[kf, :, 1:nx], w[kf, :, 0:nx - 1])
        xp.multiply(second, rdx, out=second)
        xp.subtract(scratch, second, out=tem[kf, :, 1:nx])
    if bbc == 1:
        tem[0] = tem[1]
    else:
        # misclibs.F:1879-1894
        xp.add(u[0, :, 1:nx], _f32(xp, lower.u_move_m_s), out=scratch)
        xp.multiply(scratch, two, out=scratch)
        _apply_face_metric(xp, scratch, column, 0)
        tem[0, :, 1:nx] = scratch
    if tbc == 1:
        tem[nz] = tem[nz - 1]
    else:
        xp.multiply(u[nz - 1, :, 1:nx], minus_two, out=scratch)
        _apply_face_metric(xp, scratch, column, nz)
        tem[nz, :, 1:nx] = scratch
    yvort = new("vort.yvort", (nz, ny, nx)) if out is None else out["yvort"]
    inner = new("vort.inner_y", (ny, nx - 2))
    for k in range(nz):
        # misclibs.F:1920
        xp.add(tem[k, :, 1:nx - 1], tem[k, :, 2:nx], out=inner)
        xp.add(inner, tem[k + 1, :, 1:nx - 1], out=inner)
        xp.add(inner, tem[k + 1, :, 2:nx], out=inner)
        xp.multiply(inner, quarter, out=inner)
        yvort[k, :, 1:nx - 1] = inner
        _close_x_edges(xp, yvort[k])
    del tem, scratch, inner
    drop("vort.tem_y", "vort.scratch_y", "vort.inner_y")

    # -- zvort: misclibs.F:2041-2047 ---------------------------------------
    zvort = new("vort.zvort", (nz, ny, nx)) if out is None else out["zvort"]
    corners = new("vort.corners", (ny - 1, nx - 1))
    corner_scratch = new("vort.corner_scratch", (ny - 1, nx - 1))
    cell_scratch = new("vort.cell_scratch", (ny - 2, nx - 2))
    for k in range(nz):
        corner_vertical_vorticity(
            xp, u=u, v=v, level=k, rdx=rdx, rdy=rdy, out=corners, scratch=corner_scratch
        )
        _cell_from_corners(xp, corners, zvort[k], cell_scratch)

    if bbc == 3:
        # misclibs.F:2165-2172.  The semi-slip lower boundary replaces the
        # lowest level of the horizontal vorticity with the log-layer form,
        # because the free-slip copy-down above would report no surface
        # vorticity on a boundary that is generating it -- and that surface
        # vorticity is exactly what a low-level mesocyclone tilts upward.
        surface = surface_layer_state(
            xp, u=u, v=v, lower=lower, za_m=column.zh_m[0],
            znt=surface_znt if int(lower.cecd) != 1 else None,
        )
        if isinstance(surface["znt"], float):
            depth = _f32(xp, CM1_KARMAN * (column.zh_m[0] + surface["znt"]))
        else:
            # A wind-dependent drag: the roughness is a plane, so karman*(zh + znt) is one too.
            depth = xp.multiply(
                xp.add(surface["znt"], _f32(xp, float(column.zh_m[0]))), _f32(xp, CM1_KARMAN)
            )
        scale = xp.divide(surface["ust"], depth)
        speed = xp.maximum(surface["s1"], _f32(xp, 0.01))
        xvort[0] = xp.multiply(
            xp.multiply(scale, _f32(xp, -1.0)), xp.divide(surface["v1"], speed)
        )
        yvort[0] = xp.multiply(scale, xp.divide(surface["u1"], speed))

    del corners, corner_scratch, cell_scratch
    drop("vort.corners", "vort.corner_scratch", "vort.cell_scratch")
    return {"xvort": xvort, "yvort": yvort, "zvort": zvort}


def potential_vorticity(
    xp: Any,
    *,
    theta: Any,
    rr: Any,
    vorticity: Mapping[str, Any],
    column: VerticalColumn,
    rdx: Any,
    rdy: Any,
    allocate: Any = None,
    release: Any = None,
    out: Any | None = None,
) -> Any:
    """CM1's Ertel potential vorticity at scalar points, K m2 kg-1 s-1.

    ``misclibs.F:1686-1707`` adds ``xvort * d(theta)/dx``,
    ``misclibs.F:2013-2030`` adds ``yvort * d(theta)/dy``, and
    ``misclibs.F:2151-2172`` adds ``zvort * d(theta)/dz`` and multiplies the
    whole sum by ``rr = 1/rho``.  ``theta`` is the *full* potential
    temperature ``th0 + tha`` (``misclibs.F:1596``), not the perturbation: a
    perturbation-only PV would miss the base-state stratification, which is
    most of it.

    The horizontal gradients are formed on the faces and averaged back to the
    scalar point exactly as CM1 does; the vertical gradient uses CM1's own
    Fornberg quadratic at the two ends (``param.F:9042-9052``).  The outer
    ring is nearest-interior closure, because every term in the sum needs a
    difference this runtime's halo-free carriers cannot take there.
    """

    nz, ny, nx = theta.shape
    _require_shape("rr", rr, (nz, ny, nx))
    for name in ("xvort", "yvort", "zvort"):
        _require_shape(name, vorticity[name], (nz, ny, nx))
    if column.nz != nz:
        raise ValueError(
            f"the vertical column describes {column.nz} levels and theta "
            f"{nz}; the vertical theta gradient would use another grid's "
            "layer depths"
        )
    half = _f32(xp, 0.5)
    new = _allocator(xp, allocate)
    drop = release if release is not None else (lambda *names: None)
    pv = new("pv.field", (nz, ny, nx), zero=True) if out is None else out
    if out is not None:
        _require_shape("pv", pv, (nz, ny, nx))
        pv[...] = _f32(xp, 0.0)

    # misclibs.F:1690 then :1702 -- d(theta)/dx on u faces, averaged back.
    dtdx = new("pv.dtdx", (nz, ny, nx + 1), zero=True)
    xp.multiply(
        xp.subtract(theta[:, :, 1:nx], theta[:, :, 0:nx - 1]), rdx, out=dtdx[:, :, 1:nx]
    )
    term = xp.add(dtdx[:, :, 1:nx - 1], dtdx[:, :, 2:nx])
    xp.multiply(term, half, out=term)
    xp.add(pv[:, :, 1:nx - 1], xp.multiply(vorticity["xvort"][:, :, 1:nx - 1], term),
           out=pv[:, :, 1:nx - 1])
    del dtdx
    drop("pv.dtdx")

    # misclibs.F:2010 then :2026 -- d(theta)/dy on v faces, averaged back.
    dtdy = new("pv.dtdy", (nz, ny + 1, nx), zero=True)
    xp.multiply(
        xp.subtract(theta[:, 1:ny, :], theta[:, 0:ny - 1, :]), rdy, out=dtdy[:, 1:ny, :]
    )
    term = xp.add(dtdy[:, 1:ny - 1, :], dtdy[:, 2:ny, :])
    xp.multiply(term, half, out=term)
    xp.add(pv[:, 1:ny - 1, :], xp.multiply(vorticity["yvort"][:, 1:ny - 1, :], term),
           out=pv[:, 1:ny - 1, :])
    del dtdy
    drop("pv.dtdy")

    # misclibs.F:2154-2165 -- d(theta)/dz on w faces, Fornberg-closed ends.
    dtdz = new("pv.dtdz", (nz + 1, ny, nx), zero=True)
    for kf in range(1, nz):
        xp.subtract(theta[kf], theta[kf - 1], out=dtdz[kf])
        _apply_face_metric(xp, dtdz[kf], column, kf)
    bottom = xp.multiply(theta[2], _f32(xp, column.dgs[2]))
    xp.add(bottom, xp.multiply(theta[1], _f32(xp, column.dgs[1])), out=bottom)
    xp.add(bottom, xp.multiply(theta[0], _f32(xp, column.dgs[0])), out=bottom)
    xp.multiply(bottom, _f32(xp, column.rdz), out=bottom)
    xp.multiply(bottom, _f32(xp, column.mh[0]), out=dtdz[0])
    top = xp.multiply(theta[nz - 3], _f32(xp, column.dgt[2]))
    xp.add(top, xp.multiply(theta[nz - 2], _f32(xp, column.dgt[1])), out=top)
    xp.add(top, xp.multiply(theta[nz - 1], _f32(xp, column.dgt[0])), out=top)
    xp.multiply(top, _f32(xp, column.rdz), out=top)
    xp.multiply(top, _f32(xp, column.mh[nz - 1]), out=dtdz[nz])
    term = xp.add(dtdz[0:nz], dtdz[1:nz + 1])
    xp.multiply(term, half, out=term)
    xp.add(pv, xp.multiply(vorticity["zvort"], term), out=pv)
    xp.multiply(pv, rr, out=pv)
    for k in range(nz):
        _close_x_edges(xp, _close_y_edges(xp, pv[k]))
    del dtdz
    drop("pv.dtdz")
    return pv


# ---------------------------------------------------------------------------
# Whole-run surface swaths -- solve3.F:855-1136, init3d.F:181-203
# ---------------------------------------------------------------------------

def _round_half_away(value: float) -> int:
    """Round to the nearest cell, halves away from zero.

    ``round()`` in Python rounds halves to even, which would make the
    ground-anchored swath jump by a cell at a different model time depending
    on the parity of the offset.  A swath is a damage track; its cell
    boundaries must not depend on parity.
    """

    return int(math.floor(value + 0.5)) if value >= 0.0 else -int(math.floor(-value + 0.5))


@dataclass(frozen=True, slots=True)
class GroundAnchor:
    """The exact map from this run's moving grid to fixed ground.

    CM1 keeps a second, ground-fixed copy of every swath and moves it with
    ``movesfc`` (``adv_routines.F:8884-9178``), a three-stage WENO advection
    of the swath field at ``-umove``.  That is CM1's only option: CM1's
    ``solve3.F`` swath block knows the frame velocity but not where the
    domain *is*.

    GM1 does know.  ``RuntimeClock`` integrates ``cm1.F:3512-3513``'s
    ``domainlocx``/``domainlocy`` and publishes the pair with every frame, so
    the ground position of grid column ``i`` at any model time is
    ``domainlocx + i*dx`` exactly.  This anchor therefore *places* each
    update on a ground-fixed canvas instead of advecting the accumulated
    maximum, and the difference is not cosmetic: three WENO stages per step
    over a fourteen-thousand-step run diffuse a swath maximum away, and the
    peak of a tornado swath is the one number the run exists to report.

    What remains inexact is only the sub-cell part: the canvas has this grid's
    own cell size, so an update lands in the cell whose centre is nearest and
    the remainder is recorded as ``residual``, never hidden.  It is bounded by
    half a cell by construction, and it is zero whenever the domain has moved
    a whole number of cells.
    """

    dx_m: float
    dy_m: float
    u_move_m_s: float
    v_move_m_s: float
    dt_s: float
    total_steps: int

    @property
    def imove(self) -> int:
        """CM1's ``imove`` for this run (``param.F:4697-4698``)."""

        return int(self.u_move_m_s != 0.0 or self.v_move_m_s != 0.0)

    def final_offset_cells(self) -> tuple[int, int]:
        """Where the domain's own origin ends up, in whole cells."""

        seconds = self.dt_s * self.total_steps
        return (
            _round_half_away(self.u_move_m_s * seconds / self.dx_m),
            _round_half_away(self.v_move_m_s * seconds / self.dy_m),
        )

    def canvas_shape(self, ny: int, nx: int) -> tuple[int, int]:
        """The ground-fixed extent this run's whole track needs, in cells.

        One cell of margin in each direction, because the run supplies the
        clock's accumulated ``domainlocx`` rather than a recomputed product:
        a float64 sum of a constant can land a half-cell either side of the
        closed form at the last step, and a swath that fell off the canvas
        there would either raise at the end of a fourteen-thousand-step run
        or silently record the final minute on the wrong ground.
        """

        ox, oy = self.final_offset_cells()
        return ny + abs(oy) + 1, nx + abs(ox) + 1

    def origin_cells(self) -> tuple[int, int]:
        ox, oy = self.final_offset_cells()
        return min(0, ox), min(0, oy)

    def place(self, domainlocx_m: float, domainlocy_m: float) -> tuple[int, int, float, float]:
        """Canvas offset of the domain origin, and the sub-cell remainder."""

        ox0, oy0 = self.origin_cells()
        ox = _round_half_away(domainlocx_m / self.dx_m)
        oy = _round_half_away(domainlocy_m / self.dy_m)
        return (
            ox - ox0,
            oy - oy0,
            domainlocx_m - ox * self.dx_m,
            domainlocy_m - oy * self.dy_m,
        )

    def record(self) -> dict[str, Any]:
        ox, oy = self.final_offset_cells()
        return {
            "imove": self.imove,
            "u_move_m_s": self.u_move_m_s,
            "v_move_m_s": self.v_move_m_s,
            "dt_s": self.dt_s,
            "total_steps": self.total_steps,
            "final_offset_cells_xy": [ox, oy],
            "authority": SOURCE_ANCHORS["domain_location_integral"],
            "cm1_method": SOURCE_ANCHORS["swath_translation"],
            "gm1_method": (
                "exact placement on a ground-fixed canvas from the "
                "domainlocx/domainlocy integral, quantized to this grid's own "
                "cell; CM1 advects the accumulated maximum with three WENO "
                "stages per step instead, which diffuses the peak a swath "
                "exists to report"
            ),
        }


#: Which per-step scalar series the report reads.  Each is one float32 word
#: per large step, held on the device and converted once at report time, so a
#: fourteen-thousand-step run costs 56 KB and not one host synchronization.
SERIES_NAMES = ("svs_max", "sws_max", "sps_min", "shs_max")
#: The swath each per-step series is taken from, and its units.
SERIES_SOURCE = MappingProxyType({
    "svs_max": "svs", "sws_max": "sws", "sps_min": "sps", "shs_max": "shs",
})
SERIES_UNITS = MappingProxyType({
    "svs_max": "s-1", "sws_max": "m s-1", "sps_min": "Pa", "shs_max": "m2 s-2",
})


# ---------------------------------------------------------------------------
# The swaths across a restart -- restart_write.F:696-758, restart_read.F:1064-1126
# ---------------------------------------------------------------------------
#
# CM1 writes sws..shs into every restart file and, when the domain moves
# (nrain = 2), the translated sws2..shs2 as well; restart_read.F reads them
# straight back, so a resumed run's swath is the uninterrupted run's.  GM1's
# parent restart carries the same planes in its own ``swath`` group
# (cm1core.cartesian_restart), plus what GM1 keeps that CM1 does not: the
# ground canvas's origin and extent (CM1's sws2 is advected on the domain's
# own grid; GM1's is placed on a ground-fixed canvas sized from the run's
# length, so a resume that runs longer or shorter needs to know where the old
# canvas sat), and the per-step series the report reads.

SWATH_RESTART_SCHEMA = "gm1.swath-restart/v1"
SWATH_RESTART_GROUP = "swath"
SWATH_RESTART_PLANE = "swath.{name}"
SWATH_RESTART_CANVAS = "swath.{name}" + SWATH_GROUND_SUFFIX
SWATH_RESTART_SERIES = "swath.series.{name}"
SWATH_RESTART_INDEX = "swath.series.svs_index"
SWATH_RESTART_LOCATION = "swath.series.domain_location"
#: Restart scalars (the envelope's scalar table).
SWATH_RESTART_UPDATES = "swath.updates"
SWATH_RESTART_OPENED = "swath.opened_step"
SWATH_RESTART_FIRST = "swath.first_step.{name}"
SWATH_RESTART_RESIDUAL_X = "swath.residual_x_m"
SWATH_RESTART_RESIDUAL_Y = "swath.residual_y_m"
SWATH_RESTART_ORIGIN_X = "swath.ground.origin_x_cells"
SWATH_RESTART_ORIGIN_Y = "swath.ground.origin_y_cells"

# A restart carries binary32 words only, and every one must be finite.  The
# per-step domain location is binary64 (cm1.F:266); its 64 bits travel as
# three chunks of at most 22 bits, each an exact small integer in binary32.
LOCATION_CHUNK_BITS = 22
LOCATION_CHUNKS = 3
_CHUNK_MASK = (1 << LOCATION_CHUNK_BITS) - 1
_TOP_CHUNK_MASK = (1 << (64 - 2 * LOCATION_CHUNK_BITS)) - 1
#: The per-step svs maximum's flat index travels as binary32, exact below 2**24.
_INDEX_EXACT_LIMIT = 1 << 24


def binary64_as_binary32_chunks(values: Any) -> Any:
    """Exact binary32 chunks of binary64 values: shape ``values.shape + (3,)``."""

    import numpy

    bits = numpy.ascontiguousarray(values, dtype="<f8").view("<u8")
    out = numpy.empty(bits.shape + (LOCATION_CHUNKS,), dtype=numpy.float32)
    for index in range(LOCATION_CHUNKS):
        shift = numpy.uint64(index * LOCATION_CHUNK_BITS)
        out[..., index] = ((bits >> shift) & numpy.uint64(_CHUNK_MASK)).astype(numpy.float32)
    return out


def binary32_chunks_as_binary64(chunks: Any) -> Any:
    """The binary64 values :func:`binary64_as_binary32_chunks` encoded, bit for bit."""

    import numpy

    words = numpy.asarray(chunks, dtype=numpy.float32)
    if words.ndim < 1 or words.shape[-1] != LOCATION_CHUNKS:
        raise ValueError(f"binary64 chunks need a trailing extent of {LOCATION_CHUNKS}")
    limits = numpy.array([_CHUNK_MASK, _CHUNK_MASK, _TOP_CHUNK_MASK], dtype=numpy.float64)
    wide = words.astype(numpy.float64)
    if not bool(numpy.all((wide >= 0.0) & (wide <= limits) & (wide == numpy.floor(wide)))):
        raise ValueError(
            "a binary64 chunk is not a whole number in its 22-bit range; the "
            "restart's domain-location words are not the ones this encoding wrote"
        )
    ints = wide.astype(numpy.uint64)
    bits = ints[..., 0]
    for index in range(1, LOCATION_CHUNKS):
        bits = bits | (ints[..., index] << numpy.uint64(index * LOCATION_CHUNK_BITS))
    return numpy.ascontiguousarray(bits).view("<f8")


@dataclass(frozen=True, slots=True)
class SwathRestartLayout:
    """What a restart carries of one run's swaths, and where it sits on the ground.

    The shapes follow the run that writes the restart: the ground canvas and
    the per-step series are sized from that run's length, so a restart of a
    two-hour run and a restart of the same run planned for three hours differ
    here and nowhere in the model state.  That is why the layout is recorded
    beside the state rather than folded into it.
    """

    names: tuple[str, ...]
    surface_shape: tuple[int, int]
    total_steps: int
    canvas_shape: tuple[int, int] | None
    canvas_origin_cells_xy: tuple[int, int] | None

    def __post_init__(self) -> None:
        unknown = [name for name in self.names if name not in SWATH_MEANINGS]
        if unknown or len(set(self.names)) != len(self.names) or not self.names:
            raise ValueError(f"a swath restart layout names CM1's swaths once each, not {self.names!r}")
        if (self.canvas_shape is None) != (self.canvas_origin_cells_xy is None):
            raise ValueError("a ground canvas is carried with its origin, or neither is")
        if type(self.total_steps) is not int or self.total_steps < 1:
            raise ValueError("total_steps must be a positive integer")
        ny, nx = self.surface_shape
        if ny * nx >= _INDEX_EXACT_LIMIT:
            raise ValueError(
                f"a {nx}x{ny} surface has flat indices past 2**24, which binary32 "
                "cannot hold exactly; the per-step vortex position would be carried "
                "to the wrong cell"
            )

    @classmethod
    def for_anchor(
        cls, names: Sequence[str], *, ny: int, nx: int, anchor: GroundAnchor, ground: bool,
    ) -> "SwathRestartLayout":
        canvas = anchor.canvas_shape(int(ny), int(nx)) if (ground and anchor.imove) else None
        return cls(
            tuple(names), (int(ny), int(nx)), int(max(anchor.total_steps, 1)),
            None if canvas is None else (int(canvas[0]), int(canvas[1])),
            None if canvas is None else tuple(int(v) for v in anchor.origin_cells()),
        )

    def carriers(self) -> tuple[tuple[str, str, str, str, tuple[int, ...], str], ...]:
        """(name, meaning, units, stagger, shape, key) of every carried array."""

        rows: list[tuple[str, str, str, str, tuple[int, ...], str]] = []
        for name in self.names:
            rows.append((
                SWATH_RESTART_PLANE.format(name=name),
                f"run-long {SWATH_MEANINGS[name]} (restart_write.F:696-723)",
                SWATH_UNITS[name], "surface", tuple(self.surface_shape), name,
            ))
        if self.canvas_shape is not None:
            for name in self.names:
                rows.append((
                    SWATH_RESTART_CANVAS.format(name=name),
                    f"ground-anchored run-long {SWATH_MEANINGS[name]} "
                    "(restart_write.F:724-758 sws2; GM1 ground canvas)",
                    SWATH_UNITS[name], "ground-canvas", tuple(self.canvas_shape),
                    f"{name}{SWATH_GROUND_SUFFIX}",
                ))
        steps = (int(self.total_steps),)
        for series in SERIES_NAMES:
            rows.append((
                SWATH_RESTART_SERIES.format(name=series),
                f"per-large-step {series} of the {SERIES_SOURCE[series]} fold, "
                "slot n-1 for step n; 0 in slots no fold wrote",
                SERIES_UNITS[series], "step-series", steps, f"series.{series}",
            ))
        rows.append((
            SWATH_RESTART_INDEX,
            "flat j*nx+i index of each step's svs maximum, slot n-1 for step n; "
            "-1 in slots no fold wrote",
            "1", "step-series", steps, "series.svs_index",
        ))
        rows.append((
            SWATH_RESTART_LOCATION,
            "cm1.F:3512-3513 domainlocx/domainlocy at each folded step, binary64 "
            "bits as three exact 22-bit binary32 chunks",
            "m (binary64 bits)", "step-series", (*steps, 2, LOCATION_CHUNKS),
            "series.domain_location",
        ))
        return tuple(rows)

    def record(self) -> dict[str, Any]:
        return {
            "schema": SWATH_RESTART_SCHEMA,
            "names": list(self.names),
            "surface_shape": list(self.surface_shape),
            "total_steps": int(self.total_steps),
            "ground_canvas_shape": None if self.canvas_shape is None else list(self.canvas_shape),
            "ground_canvas_origin_cells_xy": (
                None if self.canvas_origin_cells_xy is None
                else list(self.canvas_origin_cells_xy)
            ),
            "carriers": [row[0] for row in self.carriers()],
            "authority": {
                "write": SOURCE_ANCHORS["swath_restart_write"],
                "read": SOURCE_ANCHORS["swath_restart_read"],
            },
            "gm1_additions": (
                "the ground canvas's origin and extent (GM1 places the "
                "ground-fixed swath on a canvas sized from the run's length "
                "instead of advecting it with movesfc) and the per-step series "
                "and domain locations the tornado report reads"
            ),
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "SwathRestartLayout":
        if record.get("schema") != SWATH_RESTART_SCHEMA:
            raise ValueError(
                f"a swath restart layout is {SWATH_RESTART_SCHEMA}, not {record.get('schema')!r}"
            )
        canvas = record.get("ground_canvas_shape")
        origin = record.get("ground_canvas_origin_cells_xy")
        return cls(
            tuple(str(name) for name in record["names"]),
            tuple(int(v) for v in record["surface_shape"]),  # type: ignore[arg-type]
            int(record["total_steps"]),
            None if canvas is None else (int(canvas[0]), int(canvas[1])),
            None if origin is None else (int(origin[0]), int(origin[1])),
        )


@dataclass(frozen=True, slots=True)
class SwathRestartState:
    """One restart's swath words and counters, on the host, before they are applied.

    ``first_step`` is the first large step each carried swath folded: 1 for a
    whole-run swath, later for one that started at an earlier resume.
    ``opened_step`` is where the carried lineage's per-step series begin.
    """

    layout: SwathRestartLayout
    step: int
    planes: Mapping[str, Any]
    canvases: Mapping[str, Any]
    series: Mapping[str, Any]
    vortex_index: Any
    domain_location: Any
    updates: int
    opened_step: int
    first_step: Mapping[str, int]
    residual_m: tuple[float, float]

    def record(self) -> dict[str, Any]:
        return {
            "names": list(self.layout.names),
            "step": int(self.step),
            "updates": int(self.updates),
            "opened_step": int(self.opened_step),
            "first_step": {name: int(value) for name, value in self.first_step.items()},
            "ground_canvas_shape": (
                None if self.layout.canvas_shape is None else list(self.layout.canvas_shape)
            ),
            "ground_canvas_origin_cells_xy": (
                None if self.layout.canvas_origin_cells_xy is None
                else list(self.layout.canvas_origin_cells_xy)
            ),
            "ground_placement_residual_m_xy": list(self.residual_m),
        }


class SurfaceSwaths:
    """CM1's whole-run surface extrema for one live Cartesian domain.

    ``init3d.F:181-203`` sets the initial values, ``solve3.F:855-1136``
    updates them once per completed large step, and ``writeout.F:408-720``
    writes them.  Every one of those three is reproduced here; what is not
    reproduced is ``movesfc``, and :class:`GroundAnchor` says why.

    The object is array-module agnostic: with CuPy every plane and every
    series word stays on the device and only :meth:`record` crosses to the
    host.
    """

    def __init__(
        self,
        xp: Any,
        *,
        grid: Any,
        selection: TornadoDiagnosticSelection,
        anchor: GroundAnchor,
        column: VerticalColumn,
        allocate: Any = None,
        ground: bool = True,
        opened_step: int = 0,
    ) -> None:
        # ``ground=False``: no ground-fixed twin, whatever the frame velocity.
        # A live nest child is the one caller (cm1core.nest_live): its
        # lattice moves inside the parent as well as with the frame, so the
        # canvas GroundAnchor sizes from the frame velocity alone would put
        # every post-move update on the wrong ground.
        if type(ground) is not bool:
            raise TypeError("ground must be a boolean")
        # ``opened_step``: the completed large steps when these swaths open.
        # Zero for a run from its initial condition; a resumed run opens them
        # at its restart step, and either continues a restart's swaths from
        # there (continue_from) or starts them there.
        if type(opened_step) is not int or not 0 <= opened_step <= int(anchor.total_steps):
            raise ValueError(
                f"opened_step must be an integer from 0 to the {anchor.total_steps} "
                "steps these swaths are sized for"
            )
        self.xp = xp
        self.grid = grid
        self.selection = selection
        self.anchor = anchor
        self.column = column
        self.nx, self.ny, self.nz = int(grid.nx), int(grid.ny), int(grid.nz)
        self.names = selection.swath_names
        # Folds this object made; ``restored_updates`` are the folds the
        # restart it continued had already made.
        self.updates = 0
        self.restored_updates = 0
        self.opened_step = opened_step
        self.last_step = opened_step
        # The first large step each swath folded.  None: carried by a restart
        # that does not say (a nest child's, cm1core.nest_restart).
        self.first_step: dict[str, int | None] = {
            name: opened_step + 1 for name in self.names
        }
        self.max_residual_m = (0.0, 0.0)
        self._allocate = allocate if allocate is not None else (
            lambda name, shape, dtype: xp.empty(shape, dtype=dtype)
        )
        # solve3.F:1086-1092: one scan of the face heights, done once.
        self.updraft_level = self._updraft_swath_level()
        self.frame: dict[str, Any] = {}
        self.ground: dict[str, Any] = {}
        for name in self.names:
            plane = self._allocate(f"swath.{name}", (self.ny, self.nx), xp.float32)
            plane[...] = _f32(xp, SWATH_INITIAL_VALUES[name])
            self.frame[name] = plane
        if anchor.imove and ground:
            shape = anchor.canvas_shape(self.ny, self.nx)
            for name in self.names:
                canvas = self._allocate(
                    f"swath.{name}{SWATH_GROUND_SUFFIX}", shape, xp.float32
                )
                canvas[...] = _f32(xp, SWATH_INITIAL_VALUES[name])
                self.ground[name] = canvas
        steps = max(int(anchor.total_steps), 1)
        self.series: dict[str, Any] = {
            name: self._allocate(f"swath.series.{name}", (steps,), xp.float32)
            for name in SERIES_NAMES
        }
        for name in SERIES_NAMES:
            self.series[name][...] = _f32(xp, float("nan"))
        self.vortex_index = self._allocate("swath.series.svs_index", (steps,), xp.int32)
        self.vortex_index[...] = -1
        # Entry n-1 is the domain location at step n; None before these
        # swaths (or the lineage they continue) opened.
        self.domain_location: list[tuple[float, float] | None] = [None] * opened_step

    # -- helpers -----------------------------------------------------------

    def _updraft_swath_level(self) -> int:
        """``solve3.F:1086-1092``'s upward scan, resolved once for the column."""

        if not self.selection.output_sus:
            return 0
        kf = 1
        while self.column.zf_m[kf] < UPDRAFT_SWATH_HEIGHT_M and kf < self.nz - 1:
            kf += 1
        return kf

    def _fold(self, name: str, plane: Any, *, canvas_x: int, canvas_y: int) -> None:
        xp = self.xp
        keep = xp.maximum if SWATH_EXTREMUM[name] == "max" else xp.minimum
        keep(self.frame[name], plane, out=self.frame[name])
        canvas = self.ground.get(name)
        if canvas is None:
            return
        height, width = canvas.shape
        if not (0 <= canvas_y and canvas_y + self.ny <= height
                and 0 <= canvas_x and canvas_x + self.nx <= width):
            raise ValueError(
                f"the domain is at ground cell ({canvas_x}, {canvas_y}) and the "
                f"ground-anchored swath canvas is {width}x{height} for a "
                f"{self.nx}x{self.ny} domain; folding here would either wrap "
                "the swath or record this step's extremum on ground the "
                "domain never covered"
            )
        view = canvas[canvas_y:canvas_y + self.ny, canvas_x:canvas_x + self.nx]
        keep(view, plane, out=view)

    # -- CM1's seven fields ------------------------------------------------

    def surface_wind_speed(self, *, u: Any, v: Any, out: Any) -> Any:
        """``solve3.F:863-865``: ground-relative wind speed at level one."""

        xp, nx, ny = self.xp, self.nx, self.ny
        half = _f32(xp, 0.5)
        term = xp.empty_like(out)
        xp.add(u[0, :, 0:nx], u[0, :, 1:nx + 1], out=out)
        xp.multiply(out, half, out=out)
        xp.add(out, _f32(xp, self.anchor.u_move_m_s), out=out)
        xp.multiply(out, out, out=out)
        xp.add(v[0, 0:ny, :], v[0, 1:ny + 1, :], out=term)
        xp.multiply(term, half, out=term)
        xp.add(term, _f32(xp, self.anchor.v_move_m_s), out=term)
        xp.multiply(term, term, out=term)
        xp.add(out, term, out=out)
        xp.sqrt(out, out=out)
        return out

    def surface_vertical_vorticity(self, *, u: Any, v: Any, out: Any) -> Any:
        """``solve3.F:889-899``: corner vorticity at the lowest model level.

        CM1 stores this on the vorticity points and writes the ``1..ni,1..nj``
        rectangle of that array, so the value published at cell ``(i,j)`` is
        the vorticity at that cell's south-west corner -- sharper than a cell
        average, which is why it is the swath CM1 keeps.  Only the first row
        and first column are unavailable to a runtime with no halo, and they
        are closed from their neighbour.
        """

        xp, nx, ny = self.xp, self.nx, self.ny
        corners = corner_vertical_vorticity(
            xp, u=u, v=v, level=0, rdx=self.rdx, rdy=self.rdy
        )
        out[1:ny, 1:nx] = corners
        out[0, 1:nx] = out[1, 1:nx]
        out[:, 0] = out[:, 1]
        return out

    # -- the per-step update ----------------------------------------------

    def update(
        self,
        *,
        step: int,
        u: Any,
        v: Any,
        w: Any,
        prs: Any,
        prs0_column: Any,
        q3d: Any,
        updraft_helicity_plane: Any | None,
        domainlocx_m: float,
        domainlocy_m: float,
    ) -> None:
        """Fold one completed large step into every selected swath.

        ``solve3.F`` runs this block at the end of every large step, and so
        does this: a swath that sampled only at history cadence would miss the
        minute a tornado exists in.
        """

        xp, nx, ny = self.xp, self.nx, self.ny
        if type(step) is not int or step < 1:
            raise ValueError("step must be a positive integer")
        if step > self.anchor.total_steps:
            raise ValueError(
                f"step {step} is past the {self.anchor.total_steps} steps this "
                "swath was sized for; the ground canvas has no cell for the "
                "domain's position at that time, so the swath would record "
                "this step's extremum on ground the domain had already left"
            )
        if step != self.last_step + 1:
            raise ValueError(
                f"the swaths stand at step {self.last_step} and were asked to fold "
                f"step {step}; solve3.F:855-1136 folds every large step, so a "
                "skipped or repeated step is a run-long extremum that silently "
                "missed or double-counted part of the run"
            )
        canvas_x, canvas_y, rx, ry = self.anchor.place(domainlocx_m, domainlocy_m)
        self.max_residual_m = (
            max(self.max_residual_m[0], abs(rx)),
            max(self.max_residual_m[1], abs(ry)),
        )
        self.domain_location.append((float(domainlocx_m), float(domainlocy_m)))
        slot = step - 1
        plane = xp.empty((ny, nx), dtype=xp.float32)
        sel = self.selection
        if sel.output_sws:
            self.surface_wind_speed(u=u, v=v, out=plane)
            self.series["sws_max"][slot] = xp.amax(plane)
            self._fold("sws", plane, canvas_x=canvas_x, canvas_y=canvas_y)
        if sel.output_svs:
            self.surface_vertical_vorticity(u=u, v=v, out=plane)
            self.series["svs_max"][slot] = xp.amax(plane)
            self.vortex_index[slot] = xp.argmax(plane)
            self._fold("svs", plane, canvas_x=canvas_x, canvas_y=canvas_y)
        if sel.output_sps:
            # solve3.F:977: prs - prs0 at the lowest model level.
            xp.subtract(prs[0], prs0_column, out=plane)
            self.series["sps_min"][slot] = xp.amin(plane)
            self._fold("sps", plane, canvas_x=canvas_x, canvas_y=canvas_y)
        if sel.output_srs:
            # solve3.F:1003
            xp.copyto(plane, q3d[RAINWATER_INDEX][0])
            self._fold("srs", plane, canvas_x=canvas_x, canvas_y=canvas_y)
        if sel.output_sgs:
            # solve3.F:1031
            xp.copyto(plane, q3d[GRAUPEL_INDEX][0])
            self._fold("sgs", plane, canvas_x=canvas_x, canvas_y=canvas_y)
        if sel.output_sus:
            # solve3.F:1093
            xp.copyto(plane, w[self.updraft_level])
            self._fold("sus", plane, canvas_x=canvas_x, canvas_y=canvas_y)
        if sel.output_shs:
            if updraft_helicity_plane is None:
                raise ValueError(
                    "output_shs keeps the running maximum of calcuh's own "
                    "field (solve3.F:1119-1124) and no updraft-helicity plane "
                    "was supplied; folding anything else would publish a "
                    "different diagnostic under CM1's shs name"
                )
            xp.copyto(plane, updraft_helicity_plane)
            self.series["shs_max"][slot] = xp.amax(plane)
            self._fold("shs", plane, canvas_x=canvas_x, canvas_y=canvas_y)
        self.updates += 1
        self.last_step = step

    # -- reporting ---------------------------------------------------------

    @property
    def rdx(self) -> Any:
        return _f32(self.xp, 1.0 / float(self.grid.dx_m))

    @property
    def rdy(self) -> Any:
        return _f32(self.xp, 1.0 / float(self.grid.dy_m))

    def host_series(self) -> dict[str, list[float]]:
        """One host copy of every per-step series, taken once.

        Entry ``n-1`` is step ``n``, from the run's first step to the last one
        folded.  Steps no fold of this swath's lineage saw (before a swath
        that started at a resume) are NaN, with index -1.
        """

        xp = self.xp
        to_host = getattr(xp, "asnumpy", None)
        out: dict[str, list[float]] = {}
        for name in SERIES_NAMES:
            values = self.series[name]
            words = to_host(values) if to_host is not None else values
            out[name] = [float(value) for value in words[:self.last_step]]
        index = self.vortex_index
        words = to_host(index) if to_host is not None else index
        out["svs_index"] = [int(value) for value in words[:self.last_step]]
        return out

    # -- across a restart --------------------------------------------------

    def restart_layout(self) -> SwathRestartLayout:
        """What a restart written now carries of these swaths."""

        return SwathRestartLayout.for_anchor(
            self.names, ny=self.ny, nx=self.nx, anchor=self.anchor, ground=bool(self.ground),
        )

    def restart_words(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """The arrays and counters a restart at this step carries, by carrier name.

        The frame planes and ground canvases are the live allocations: the
        restart snapshot copies them at the quiescent boundary it is taken
        at.  The series are copies whose unwritten slots (NaN) read 0, and the
        vortex index and the domain locations are re-encoded as binary32,
        because the envelope carries finite binary32 words and nothing else.
        """

        xp = self.xp
        layout = self.restart_layout()
        arrays: dict[str, Any] = {}
        for name in self.names:
            arrays[SWATH_RESTART_PLANE.format(name=name)] = self.frame[name]
        if layout.canvas_shape is not None:
            for name in self.names:
                arrays[SWATH_RESTART_CANVAS.format(name=name)] = self.ground[name]
        zero = _f32(xp, 0.0)
        for series in SERIES_NAMES:
            words = self.series[series]
            # NaN is the only value that differs from itself.
            arrays[SWATH_RESTART_SERIES.format(name=series)] = xp.ascontiguousarray(
                xp.where(words == words, words, zero)
            )
        arrays[SWATH_RESTART_INDEX] = xp.ascontiguousarray(
            self.vortex_index.astype(xp.float32)
        )
        import numpy

        location = numpy.zeros((layout.total_steps, 2), dtype=numpy.float64)
        for index, value in enumerate(self.domain_location):
            if value is not None:
                location[index] = value
        arrays[SWATH_RESTART_LOCATION] = xp.ascontiguousarray(
            xp.asarray(binary64_as_binary32_chunks(location))
        )
        counters = {
            "step": int(self.last_step),
            "updates": int(self.restored_updates + self.updates),
            "opened_step": int(self.opened_step),
            "first_step": {
                name: (0 if value is None else int(value))
                for name, value in self.first_step.items()
            },
            "residual_m": (float(self.max_residual_m[0]), float(self.max_residual_m[1])),
            "canvas_origin_cells_xy": layout.canvas_origin_cells_xy,
        }
        return arrays, counters

    def continue_from(self, state: SwathRestartState) -> dict[str, Any]:
        """Continue a restart's swaths: ``restart_read.F:1064-1126``.

        Every swath both select continues word for word.  One this run
        selects and the restart does not carry starts at the restart step,
        and the returned record says so (``restart_here``); one the restart
        carries and this run does not select is not continued.  A ground
        canvas is re-placed by its recorded origin, so a resume planned
        longer or shorter than the run that wrote the restart puts every
        carried word on the same ground cell.
        """

        xp = self.xp
        if self.updates:
            raise RuntimeError(
                "the swaths have already folded a step of this run; continuing a "
                "restart's swaths over them would discard it"
            )
        if int(state.step) != self.last_step:
            raise ValueError(
                f"the restart's swaths stand at step {state.step} and these open at "
                f"step {self.last_step}; continuing them would fold one model time's "
                "extremum into another's"
            )
        if tuple(state.layout.surface_shape) != (self.ny, self.nx):
            raise ValueError(
                f"the restart's swaths are {tuple(state.layout.surface_shape)} and this "
                f"grid's are {(self.ny, self.nx)}; copying them would put one column's "
                "extremum at another column's ground"
            )
        carried = set(state.planes)
        continued = tuple(name for name in self.names if name in carried)
        restart_here = tuple(name for name in self.names if name not in carried)
        not_continued = tuple(name for name in state.layout.names if name not in self.names)
        if bool(self.ground) != bool(state.canvases) and continued:
            raise ValueError(
                "the restart "
                + ("carries" if state.canvases else "carries no")
                + " ground-anchored swath canvas and this run "
                + ("keeps one" if self.ground else "keeps none")
                + "; the frame velocity must have changed, which the restart "
                "profile refuses before this point"
            )
        step = int(state.step)
        for name in continued:
            value = xp.asarray(state.planes[name])
            _require_shape(f"restored {name} swath", value, (self.ny, self.nx))
            xp.copyto(self.frame[name], value)
            self.first_step[name] = int(state.first_step[name]) or None
            if self.ground:
                self._place_restored_canvas(
                    name, state.canvases[name], state.layout.canvas_origin_cells_xy,
                )
        for series, source in SERIES_SOURCE.items():
            if source not in continued:
                continue
            first = int(state.first_step[source])
            if 1 <= first <= step:
                window = xp.asarray(state.series[series][first - 1:step])
                self.series[series][first - 1:step] = window
                if series == "svs_max":
                    index = xp.asarray(state.vortex_index[first - 1:step]).astype(xp.int32)
                    self.vortex_index[first - 1:step] = index
        if continued:
            for number in range(int(state.opened_step) + 1, step + 1):
                x, y = state.domain_location[number - 1]
                self.domain_location[number - 1] = (float(x), float(y))
            self.opened_step = int(state.opened_step)
            self.restored_updates = int(state.updates)
            self.max_residual_m = tuple(float(v) for v in state.residual_m)  # type: ignore[assignment]
        return {
            "step": step,
            "continued": list(continued),
            "restart_here": {name: step + 1 for name in restart_here},
            "not_continued": list(not_continued),
            "ground_canvas": (
                None if not (self.ground and continued) else {
                    "restart_shape": list(state.layout.canvas_shape or ()),
                    "restart_origin_cells_xy": list(state.layout.canvas_origin_cells_xy or ()),
                    "run_shape": list(self.anchor.canvas_shape(self.ny, self.nx)),
                    "run_origin_cells_xy": list(self.anchor.origin_cells()),
                }
            ),
        }

    def _place_restored_canvas(
        self, name: str, canvas: Any, origin: tuple[int, int] | None,
    ) -> None:
        """Put a restart's ground canvas on this run's canvas, cell for cell.

        Canvas cell ``(cy, cx)`` is ground cell ``(cy + oy0, cx + ox0)``
        (:meth:`GroundAnchor.place`).  Every cell the restarted run folded is
        inside this run's canvas, because both hold the domain from step 0
        to at least the restart step; the rest of the old canvas can only
        hold CM1's initial value, and a word there that does not is refused.
        """

        import numpy

        xp = self.xp
        target = self.ground[name]
        if origin is None:
            raise ValueError(f"the restart carries the {name} ground canvas without its origin")
        old = numpy.asarray(canvas, dtype=numpy.float32)
        new_ox, new_oy = self.anchor.origin_cells()
        shift_x, shift_y = int(origin[0]) - int(new_ox), int(origin[1]) - int(new_oy)
        height, width = old.shape
        new_height, new_width = (int(extent) for extent in target.shape)
        x0, x1 = max(0, -shift_x), min(width, new_width - shift_x)
        y0, y1 = max(0, -shift_y), min(height, new_height - shift_y)
        sentinel = numpy.float32(SWATH_INITIAL_VALUES[name])
        outside = numpy.ones(old.shape, dtype=bool)
        if x0 < x1 and y0 < y1:
            outside[y0:y1, x0:x1] = False
        if bool(numpy.any(old[outside] != sentinel)):
            raise ValueError(
                f"the restart's {name} ground canvas holds folded words on ground "
                "this run's canvas does not cover; they would be dropped and the "
                "swath would lose part of its track"
            )
        if x0 < x1 and y0 < y1:
            window = xp.asarray(numpy.ascontiguousarray(old[y0:y1, x0:x1]))
            view = target[y0 + shift_y:y1 + shift_y, x0 + shift_x:x1 + shift_x]
            xp.copyto(view, window)

    def extrema(self) -> dict[str, float | None]:
        """The whole-run extremum of every selected swath, as one host row."""

        xp = self.xp
        to_host = getattr(xp, "asnumpy", None)
        out: dict[str, float | None] = {}
        for name in self.names:
            plane = self.frame[name]
            value = xp.amin(plane) if SWATH_EXTREMUM[name] == "min" else xp.amax(plane)
            out[name] = float(to_host(value) if to_host is not None else value)
        return out

    def record(self) -> dict[str, Any]:
        closure = {
            name: boundary_closure_words(
                self.nx, self.ny, edges=SWATH_CLOSED_EDGES[name]
            )
            for name in self.names
        }
        return {
            "schema": TORNADO_SWATH_SCHEMA,
            "fields": list(self.names),
            "meanings": {name: SWATH_MEANINGS[name] for name in self.names},
            "units": {name: SWATH_UNITS[name] for name in self.names},
            "extremum_kept": {name: SWATH_EXTREMUM[name] for name in self.names},
            "initial_values": {name: SWATH_INITIAL_VALUES[name] for name in self.names},
            "updates": self.restored_updates + self.updates,
            "updates_this_segment": self.updates,
            "opened_step": self.opened_step,
            "last_step": self.last_step,
            "first_step": dict(self.first_step),
            "whole_run": all(value == 1 for value in self.first_step.values()),
            "extrema": self.extrema(),
            "ground_slot": bool(self.ground),
            "ground_canvas_shape": (
                list(self.anchor.canvas_shape(self.ny, self.nx)) if self.ground else None
            ),
            "ground_canvas_origin_cells_xy": (
                list(self.anchor.origin_cells()) if self.ground else None
            ),
            "ground_placement_residual_m_xy": list(self.max_residual_m),
            "ground_placement_residual_bound_m_xy": [
                0.5 * float(self.grid.dx_m), 0.5 * float(self.grid.dy_m)
            ],
            "moving_frame": self.anchor.record(),
            "updraft_swath_level_index": self.updraft_level,
            "updraft_swath_level_height_m": (
                self.column.zf_m[self.updraft_level] if self.selection.output_sus else None
            ),
            "boundary_closure": closure,
            "authority": {
                "initialization": SOURCE_ANCHORS["swath_initialization"],
                "update": SOURCE_ANCHORS["swath_max_surface_wind"],
                "slots": SOURCE_ANCHORS["swath_slot_count"],
                "output": SOURCE_ANCHORS["swath_output_switches"],
                "restart": (
                    f"{SOURCE_ANCHORS['swath_restart_write']} and "
                    f"{SOURCE_ANCHORS['swath_restart_read']}"
                ),
            },
        }


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------

def wind_band(speed_m_s: float) -> dict[str, Any]:
    """Name the EF wind band a speed falls in, and refuse to rate it.

    ``EF_SCALE_WIND_M_S`` is a table of three-second gusts at ten metres
    above ground (WSEC 2006).  A model reports a grid-volume mean at the
    lowest scalar level, which is neither a gust nor at ten metres, so the
    band is reported as "the EF band this wind speed falls in", never as the
    tornado's rating.  On a kilometre grid the two differ by more than the
    width of the scale.
    """

    band: int | None = None
    for index, threshold in EF_SCALE_WIND_M_S:
        if speed_m_s >= threshold:
            band = index
    return {
        "speed_m_s": speed_m_s,
        "band": band,
        "band_label": "below EF0" if band is None else f"EF{band} wind band",
        "is_an_ef_rating": False,
        "why_not": (
            "the EF scale rates observed damage against three-second gusts at "
            "10 m; this number is a model grid-volume mean at the lowest "
            "scalar level, so it names a band and not a rating"
        ),
        "authority": THRESHOLD_AUTHORITY["wind_bands"],
    }


def resolution_verdict(grid: Any, column: VerticalColumn) -> dict[str, Any]:
    """Can this grid hold a tornado at all?

    Bryan, Wyngaard and Fritsch 2003 put a simulated feature's effective
    resolution at roughly six grid lengths, so a grid resolves a
    ``TORNADO_CORE_DIAMETER_M`` core only when ``6*dx`` is no larger than it.
    A tornado is also a surface-interaction phenomenon: its corner flow lives
    in the lowest few tens of metres, so the lowest model layer has to be
    that thin as well.  Both tests are reported, both are reported by value,
    and a run that fails either cannot have its vortex called a tornado no
    matter what the vorticity says.
    """

    spacing = max(float(grid.dx_m), float(grid.dy_m))
    resolved_m = EFFECTIVE_RESOLUTION_CELLS * spacing
    lowest_layer_m = column.zf_m[1] - column.zf_m[0]
    horizontal = resolved_m <= TORNADO_CORE_DIAMETER_M
    vertical = lowest_layer_m <= TORNADO_NEAR_GROUND_LAYER_M
    return {
        "horizontal_spacing_m": spacing,
        "effective_resolution_m": resolved_m,
        "effective_resolution_cells": EFFECTIVE_RESOLUTION_CELLS,
        "tornado_core_diameter_m": TORNADO_CORE_DIAMETER_M,
        "lowest_layer_depth_m": lowest_layer_m,
        "required_lowest_layer_depth_m": TORNADO_NEAR_GROUND_LAYER_M,
        "resolves_a_tornado_core": horizontal,
        "resolves_the_near_ground_layer": vertical,
        "can_hold_a_tornado": bool(horizontal and vertical),
        "authority": THRESHOLD_AUTHORITY["resolution"],
    }


def _exceedance_episodes(
    values: Sequence[float], threshold: float, dt_s: float
) -> list[dict[str, Any]]:
    """Contiguous runs of steps at or above a threshold, with durations."""

    episodes: list[dict[str, Any]] = []
    start: int | None = None
    peak = -math.inf
    peak_step = 0
    for index, value in enumerate(values):
        above = math.isfinite(value) and value >= threshold
        if above:
            if start is None:
                start, peak, peak_step = index, value, index + 1
            elif value > peak:
                peak, peak_step = value, index + 1
        elif start is not None:
            episodes.append({
                "first_step": start + 1,
                "last_step": index,
                "steps": index - start,
                "duration_s": (index - start) * dt_s,
                "peak_value": peak,
                "peak_step": peak_step,
            })
            start, peak = None, -math.inf
    if start is not None:
        episodes.append({
            "first_step": start + 1,
            "last_step": len(values),
            "steps": len(values) - start,
            "duration_s": (len(values) - start) * dt_s,
            "peak_value": peak,
            "peak_step": peak_step,
        })
    return episodes


def _track(
    series: Mapping[str, Sequence[float]],
    *,
    grid: Any,
    anchor: GroundAnchor,
    domain_location: Sequence[tuple[float, float] | None],
    threshold: float,
) -> list[dict[str, Any]]:
    """Where the near-surface vorticity maximum was, while it mattered.

    The index is the flat position of the maximum in the ``(ny, nx)`` ``svs``
    plane.  That plane holds CM1's own vorticity-point values
    (``solve3.F:889-899``), so entry ``(j, i)`` is the vorticity at the south-
    west corner of cell ``(j, i)``: frame position ``(i*dx, j*dy)``.  The
    ground position adds the run's own ``domainlocx``/``domainlocy`` at that
    step, whose origin is where the frame's ``(0, 0)`` stood at model time
    zero (``cm1.F:348-349`` start the integral there).  Nothing is resampled,
    nothing is rounded, and no reader is left to form the product.
    """

    nx, dx, dy = int(grid.nx), float(grid.dx_m), float(grid.dy_m)
    rows: list[dict[str, Any]] = []
    values = series["svs_max"]
    indices = series["svs_index"]
    for step_index, value in enumerate(values):
        if not math.isfinite(value) or value < threshold:
            continue
        flat = indices[step_index]
        if flat < 0:
            continue
        location = (
            domain_location[step_index] if step_index < len(domain_location) else None
        )
        if location is None:
            continue
        cell_y, cell_x = divmod(int(flat), nx)
        locx, locy = location
        rows.append({
            "step": step_index + 1,
            "model_seconds": (step_index + 1) * anchor.dt_s,
            "vertical_vorticity_s_inv": value,
            "frame_x_m": cell_x * dx,
            "frame_y_m": cell_y * dy,
            "ground_x_m": locx + cell_x * dx,
            "ground_y_m": locy + cell_y * dy,
            "cell_xy": [cell_x, cell_y],
        })
    return rows


def tornado_report(
    *,
    swaths: SurfaceSwaths,
    grid: Any,
    column: VerticalColumn,
    lower: LowerBoundaryState,
    track_budget: int = 4096,
) -> dict[str, Any]:
    """Say plainly what this run produced, and refuse to overstate it.

    The report is four things: the headline numbers a tornado run is judged
    on, the duration of every episode that passed the near-surface vortex
    threshold, the ground track of the vorticity maximum through those
    episodes, and a verdict that cannot say "tornado" unless the grid could
    have held one.

    A strong mesocyclone is not a tornado.  The verdict ladder keeps them
    apart by construction: updraft helicity is reported as rotation aloft and
    can reach its highest label without the word tornado appearing, while the
    vortex labels require near-surface vorticity, a sustained life, and a grid
    whose effective resolution reaches a tornado core.
    """

    if type(track_budget) is not int or track_budget < 1:
        raise ValueError("track_budget must be a positive integer")
    series = swaths.host_series()
    extrema = swaths.extrema()
    anchor = swaths.anchor
    dt_s = anchor.dt_s
    resolution = resolution_verdict(grid, column)

    peak_uh = extrema.get("shs")
    peak_vorticity = extrema.get("svs")
    peak_wind = extrema.get("sws")
    min_pressure = extrema.get("sps")

    episodes = _exceedance_episodes(
        series["svs_max"], NEAR_SURFACE_VORTEX_S_INV, dt_s
    ) if "svs" in swaths.names else []
    longest = max((row["duration_s"] for row in episodes), default=0.0)
    sustained = [row for row in episodes if row["duration_s"] >= VORTEX_MINIMUM_DURATION_S]

    track = _track(
        series, grid=grid, anchor=anchor,
        domain_location=swaths.domain_location,
        threshold=NEAR_SURFACE_VORTEX_S_INV,
    ) if "svs" in swaths.names else []
    track_dropped = max(len(track) - track_budget, 0)
    if track_dropped:
        keep = track_budget // 2
        track = track[:keep] + track[len(track) - (track_budget - keep):]

    rotation_aloft = "none"
    if peak_uh is not None:
        if peak_uh >= UH_STRONG_ROTATION_M2_S2:
            rotation_aloft = "strong mid-level rotation"
        elif peak_uh >= UH_ROTATING_UPDRAFT_M2_S2:
            rotation_aloft = "rotating updraft"

    near_surface = "none"
    if peak_vorticity is not None and peak_vorticity >= NEAR_SURFACE_VORTEX_S_INV:
        near_surface = "transient near-surface vortex"
        if sustained:
            near_surface = "sustained near-surface vortex"

    if near_surface == "sustained near-surface vortex" and resolution["can_hold_a_tornado"]:
        verdict = "tornado-like vortex"
    elif near_surface != "none":
        verdict = "near-surface vortex, not resolvable as a tornado on this grid"
    elif rotation_aloft != "none":
        verdict = f"{rotation_aloft} aloft, no near-surface vortex"
    else:
        verdict = "no rotating updraft and no near-surface vortex"

    statement = _statement(
        verdict=verdict,
        peak_uh=peak_uh,
        peak_vorticity=peak_vorticity,
        peak_wind=peak_wind,
        min_pressure=min_pressure,
        longest=longest,
        resolution=resolution,
    )

    return {
        "schema": TORNADO_REPORT_SCHEMA,
        "verdict": verdict,
        "statement": statement,
        "headline": {
            "peak_near_surface_vertical_vorticity_s_inv": peak_vorticity,
            "peak_near_surface_wind_m_s": peak_wind,
            "minimum_surface_pressure_perturbation_pa": min_pressure,
            "peak_updraft_helicity_m2_s2": peak_uh,
            "peak_updraft_at_5km_m_s": extrema.get("sus"),
        },
        "wind_band": None if peak_wind is None else wind_band(peak_wind),
        "rotation_aloft": {
            "label": rotation_aloft,
            "rotating_updraft_threshold_m2_s2": UH_ROTATING_UPDRAFT_M2_S2,
            "strong_rotation_threshold_m2_s2": UH_STRONG_ROTATION_M2_S2,
            "authority": THRESHOLD_AUTHORITY["updraft_helicity"],
            "is_a_tornado": False,
            "why_not": (
                "updraft helicity is a 2-5 km AGL integral (misclibs.F:1441-1442); "
                "it measures a mesocyclone, and every operational use of it is "
                "a proxy for storms that may produce tornadoes, never a "
                "detection of one"
            ),
        },
        "near_surface_vortex": {
            "label": near_surface,
            "threshold_s_inv": NEAR_SURFACE_VORTEX_S_INV,
            "minimum_duration_s": VORTEX_MINIMUM_DURATION_S,
            "authority": THRESHOLD_AUTHORITY["near_surface_vorticity"],
            "episodes": episodes,
            "sustained_episodes": len(sustained),
            "longest_episode_s": longest,
            "level_index": 0,
            "level_height_m": column.zh_m[0],
        },
        "track": {
            "rows": track,
            "dropped": track_dropped,
            "threshold_s_inv": NEAR_SURFACE_VORTEX_S_INV,
            "coordinates": (
                "frame_x/frame_y are the vorticity point (the cell's "
                "south-west corner) of the moving grid, in metres from the "
                "frame origin; ground_x/ground_y add this run's own "
                "cm1.F:3512-3513 domainlocx/domainlocy at that step, whose "
                "origin is where the frame origin stood at model time zero"
            ),
        },
        "resolution": resolution,
        "lower_boundary": {
            "bbc": int(lower.bbc),
            "label": {1: "free slip", 2: "no slip", 3: "semi slip"}[int(lower.bbc)],
            "generates_surface_vorticity": int(lower.bbc) != 1,
            "why_it_matters": (
                "a free-slip bottom exerts no surface stress at all "
                "(turb.F:4536-4545), so any near-surface rotation in such a "
                "run was produced without the surface drag that makes a real "
                "tornado's vorticity"
            ),
        },
        "swaths": swaths.record(),
        "claim_limit": (
            "thresholds applied to this run's own CM1 surface swaths; the "
            "verdict names what the fields show at this grid spacing and is "
            "not a tornado observation, a damage rating, or forecast skill"
        ),
    }


def _statement(
    *,
    verdict: str,
    peak_uh: float | None,
    peak_vorticity: float | None,
    peak_wind: float | None,
    min_pressure: float | None,
    longest: float,
    resolution: Mapping[str, Any],
) -> str:
    """One sentence a person can read, with the caveat attached, not appended."""

    parts = [verdict]
    if peak_vorticity is not None:
        parts.append(
            f"peak near-surface vertical vorticity {peak_vorticity:.4f} /s"
        )
    if peak_wind is not None:
        parts.append(f"peak near-surface wind {peak_wind:.1f} m/s")
    if min_pressure is not None:
        parts.append(f"minimum surface pressure perturbation {min_pressure:.0f} Pa")
    if peak_uh is not None:
        parts.append(f"peak 2-5 km updraft helicity {peak_uh:.0f} m2/s2")
    if longest > 0.0:
        parts.append(f"longest near-surface vortex episode {longest:.0f} s")
    if not resolution["can_hold_a_tornado"]:
        parts.append(
            "this grid cannot resolve a tornado: "
            f"{resolution['horizontal_spacing_m']:.0f} m spacing resolves "
            f"{resolution['effective_resolution_m']:.0f} m features and the "
            f"lowest layer is {resolution['lowest_layer_depth_m']:.0f} m deep"
        )
    return "; ".join(parts)


__all__ = (
    "CM1_KARMAN",
    "GroundAnchor",
    "SERIES_NAMES",
    "SurfaceSwaths",
    "SwathRestartLayout",
    "SwathRestartState",
    "binary32_chunks_as_binary64",
    "binary64_as_binary32_chunks",
    "resolution_verdict",
    "tornado_report",
    "wind_band",
    "DEFAULT_TORNADO_DIAGNOSTICS",
    "EFFECTIVE_RESOLUTION_CELLS",
    "EF_SCALE_WIND_M_S",
    "NEAR_SURFACE_VORTEX_S_INV",
    "OUTPUT_SWITCH_NAMES",
    "SOURCE_ANCHORS",
    "CLOSED_EDGES",
    "SWATH_CLOSED_EDGES",
    "SWATH_EXTREMUM",
    "SWATH_GROUND_SUFFIX",
    "SWATH_INITIAL_VALUES",
    "SWATH_MEANINGS",
    "SWATH_NAMES",
    "SWATH_RESTART_CANVAS",
    "SWATH_RESTART_FIRST",
    "SWATH_RESTART_GROUP",
    "SWATH_RESTART_INDEX",
    "SWATH_RESTART_LOCATION",
    "SWATH_RESTART_OPENED",
    "SWATH_RESTART_ORIGIN_X",
    "SWATH_RESTART_ORIGIN_Y",
    "SWATH_RESTART_PLANE",
    "SWATH_RESTART_RESIDUAL_X",
    "SWATH_RESTART_RESIDUAL_Y",
    "SWATH_RESTART_SCHEMA",
    "SWATH_RESTART_SERIES",
    "SWATH_RESTART_UPDATES",
    "SERIES_SOURCE",
    "SERIES_UNITS",
    "SWATH_SWITCH_BY_NAME",
    "SWATH_UNITS",
    "THRESHOLD_AUTHORITY",
    "TORNADO_CORE_DIAMETER_M",
    "TORNADO_DIAGNOSTIC_CLAIM_LIMIT",
    "TORNADO_DIAGNOSTIC_SCHEMA",
    "TORNADO_NEAR_GROUND_LAYER_M",
    "TORNADO_REPORT_SCHEMA",
    "TORNADO_SWATH_SCHEMA",
    "UH_ROTATING_UPDRAFT_M2_S2",
    "UH_STRONG_ROTATION_M2_S2",
    "UPDRAFT_HELICITY_BOTTOM_M",
    "UPDRAFT_HELICITY_TOP_M",
    "UPDRAFT_SWATH_HEIGHT_M",
    "VORTEX_MINIMUM_DURATION_S",
    "LowerBoundaryState",
    "TornadoDiagnosticSelection",
    "VerticalColumn",
    "boundary_closure_words",
    "corner_vertical_vorticity",
    "potential_vorticity",
    "surface_layer_state",
    "updraft_helicity",
    "validate_tornado_diagnostic_selection",
    "vertical_column",
    "vertical_vorticity_level",
    "vorticity_components",
)
