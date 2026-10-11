"""CM1 r21.1 ``parcel.F`` Lagrangian parcels, integrated inline on the device.

Authority
---------
``src/parcel.F:10`` ``parcel_driver`` is the integrator; ``:656``
``parcel_interp`` samples the fields onto the parcels; ``:1898``
``setup_parcel_vars`` is the attribute inventory; ``src/init3d.F:355-411``
seeds the initial positions; ``src/param.F:6239-6282`` resolves the sixteen
``&param13`` switches against the run's other words and ``:6385-6500`` builds
the attribute offsets from what it resolved; ``src/solve3.F:438-466`` and
``src/solve2.F:1290-1301`` say *which* velocity a ``psolver=3`` run advects a
parcel with -- the acoustic small-step **time average** ``rru``/``rrv``/
``rrw``, not the end-of-step wind.

Why inline
----------
The classic tornado-research workflow computes trajectories after the fact
from history output, which forces a dump every few seconds and still loses
the path between dumps.  GM1 already holds ``rru``/``rrv``/``rrw`` on the
device at the end of every large step (``cuda_psolver3_terrain_open.py``
writes the ``tavg`` average there), so the trajectory can be integrated at
the model timestep for the cost of one small kernel, with no extra history
at all.

What is CM1's and what is GM1's
-------------------------------
The integrator, the interpolation stencil, the boundary treatment, the
attribute list and the seeding *formula* are CM1's, transcribed.  Three
things go beyond CM1 and are marked ``GM1 EXTENSION`` at their definition:

* **seeding modes.**  CM1 hard-codes one 60x60x10 grid in ``init3d.F:381``.
  GM1 exposes that grid as parameters and adds a box and a
  vorticity-threshold mode, so a circuit can be seeded around the vortex
  rather than over the whole domain.
* **backward trajectories.**  ``direction = -1`` integrates the same RK2 with
  ``-dt``, against a stored velocity history.
* **structure of arrays.**  CM1 stores ``pdata(nparcels,npvals)``; this module
  stores ``(npvals,nparcels)`` so a device thread per parcel reads coalesced.
  The published order is CM1's attribute order, so a reader of
  ``cm1out_pdata.dat`` and a reader of this arena see the same variables.
"""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .cuda_psolver3_terrain_open import CM1_OPEN_NGXY


#: v1 recorded the switches as requested; v2 added ``number_concentration_rows``
#: for a double-moment axis.  v3 records the switches as ``param.F:6239-6282``
#: resolves them, beside what was requested, the rule that turned each one off,
#: the switches that resolve on and still lay out no column, and the words the
#: resolution read -- so every record changed shape, and says so.
LAGRANGIAN_PARCEL_SCHEMA = "gpuwm-cm1.lagrangian-parcels/v3"

#: Every CM1 line this module transcribes.  A reader checks the port against
#: the authority from here, not from a summary.
SOURCE_ANCHORS: Mapping[str, str] = {
    "parcel_driver": "src/parcel.F:10-620",
    "bottom_free_slip_extrapolation": "src/parcel.F:60-70",
    "bottom_no_slip": "src/parcel.F:71-93",
    "bottom_semi_slip_log_layer": "src/parcel.F:470-515",
    "get2d": "src/parcel.F:1719-1787",
    "semi_slip_roughness_carrier": "src/solve3.F:769 (zntmp), src/cm1.F:2192-2194, src/turb.F:1161,1233-1237",
    "top_free_slip_extrapolation": "src/parcel.F:98-120",
    "top_w_zero": "src/parcel.F:124-128",
    "cell_search": "src/parcel.F:153-260",
    "u_point_weights": "src/parcel.F:272-336",
    "v_point_weights": "src/parcel.F:338-390",
    "w_point_weights": "src/parcel.F:392-461",
    "tri_interp": "src/parcel.F:1693-1711",
    "rk2_update": "src/parcel.F:521-568",
    "below_ground_clamp": "src/parcel.F:560-567",
    "periodic_exit": "src/parcel.F:577-593",
    "recorded_winds": "src/parcel.F:1134-1343,1530-1589 (parcel_interp on u3d/v3d/w3d, src/cm1.F:4010-4020)",
    "integrator_transcription": "cm1core.nest_parcels (host reference and kernel text)",
    "attribute_inventory": "src/parcel.F:1898-2130",
    "attribute_offsets": "src/param.F:6385-6500",
    "switch_resolution": "src/param.F:6239-6282",
    "idoles_idopbl": "src/param.F:752-753 (cm1.F:392-393 default both false)",
    "bbc_rewrite": "src/param.F:873-889",
    "sgsmodel_rewrite": "src/param.F:1074-1085",
    "iusetke": "src/param.F:4989-4997,5107-5108 (cm1.F:370 default false)",
    "switch_resolution_oracle": "evidence/oracle/parcel-switch-resolution/cm1-parcel-switch-resolution.json",
    "zvort_attribute": "src/parcel.F:1061-1072",
    "buoyancy_attribute": "src/parcel.F:768-775,822-840",
    "vpg_attribute": "src/parcel.F:777-783,844-860",
    "seed_grid": "src/init3d.F:355-411",
    "horizontal_origin": "src/param.F:6646-6664",
    "domain_extent": "src/param.F:7406-7407,7743-7744,8193",
    "psolver3_advecting_velocity": "src/solve3.F:438-466",
    "small_step_time_average": "src/solve2.F:1290-1301, src/sound.F:806-832",
    "parcel_driver_call_site": "src/solve3.F:768-769",
    "cgs_cgt_extrapolation_weights": "src/param.F:9038-9052",
}

#: ``param.F:9038-9052``'s three-point one-sided Fornberg weights, the exact
#: binary32 words the packaged kernels already carry
#: (``cuda_storm_diagnostics.py``).
CM1_CGS1_F32 = 1.875
CM1_CGS2_F32 = -1.25
CM1_CGS3_F32 = 0.375

#: ``constants.F`` -- the two the attribute sampler needs.
CM1_G = 9.81
CM1_CP = 1005.7
CM1_RD = 287.04
CM1_RV = 461.5
CM1_REPS = CM1_RV / CM1_RD
CM1_REPSM1 = CM1_RV / CM1_RD - 1.0
CM1_KARMAN = 0.4

#: ``parcel.F:1898-2130``'s seven unconditional attributes, in CM1's order
#: (``param.F:6391-6399``).
PARCEL_BASE_ATTRIBUTES: tuple[str, ...] = ("x", "y", "z", "u", "v", "w", "mtime")

#: The sixteen ``&param13`` switches (``input.F:330-333``) and the most each
#: one can add, in ``param.F:6401-6494``'s order.  What a switch *does* add is
#: decided by :func:`parcel_attribute_layout` against the run's
#: :class:`ParcelSwitchContext`, the way ``param.F`` decides it: first
#: ``:6257-6282`` overrides the request (:func:`resolve_parcel_switches`), then
#: ``:6385-6500`` gates the columns -- ``prcl_q`` adds ``qname(nqv)``,
#: ``qname(nql1..nql2)`` and, only with ``iice = 1``, ``qname(nqs1..nqs2)``
#: (``:6422-6434``); ``prcl_nc`` adds ``qname(nnc1..nnc2)`` only with ``idm = 1``
#: (``:6435-6439``, :func:`parcel_number_concentrations`); ``prcl_km``,
#: ``prcl_kh`` and ``prcl_tke`` add columns only under
#: ``cm1setup >= 1 .or. ipbl >= 1`` (``:6443``); ``prcl_qsat``'s ``qsi`` needs
#: ``iice = 1`` (``:6483``).  ``prcl_ptra`` adds ``pt1..ptnpt`` in CM1; this port
#: carries no passive-tracer family, so its context holds ``iptra = 0`` and
#: ``:6257-6259`` turns the switch off.
PARCEL_SWITCH_ATTRIBUTES: Mapping[str, tuple[str, ...]] = {
    "prcl_th": ("th",),
    "prcl_t": ("t",),
    "prcl_prs": ("prs",),
    "prcl_ptra": (),
    "prcl_q": ("qv", "qc", "qr", "qi", "qs", "qg"),
    "prcl_nc": (),
    "prcl_km": ("kmh", "kmv"),
    "prcl_kh": ("khh", "khv"),
    "prcl_tke": ("tke",),
    "prcl_dbz": ("dbz",),
    "prcl_b": ("b",),
    "prcl_vpg": ("vpg",),
    "prcl_vort": ("zvort",),
    "prcl_rho": ("rho",),
    "prcl_qsat": ("qsl", "qsi"),
    "prcl_sfc": ("znt", "ust"),
}
PARCEL_SWITCH_ORDER: tuple[str, ...] = tuple(PARCEL_SWITCH_ATTRIBUTES)

#: ``parcel.F:1898-2130``'s units, so the published table says what a column is.
PARCEL_ATTRIBUTE_UNITS: Mapping[str, str] = {
    "x": "m", "y": "m", "z": "m", "u": "m/s", "v": "m/s", "w": "m/s",
    "mtime": "s", "th": "K", "t": "K", "prs": "Pa",
    "qv": "kg/kg", "qc": "kg/kg", "qr": "kg/kg", "qi": "kg/kg",
    "qs": "kg/kg", "qg": "kg/kg",
    "kmh": "m^2/s", "kmv": "m^2/s", "khh": "m^2/s", "khv": "m^2/s",
    "tke": "m^2/s^2", "dbz": "dBZ", "b": "m/s/s", "vpg": "m/s/s",
    "zvort": "1/s", "rho": "kg/m^3", "qsl": "kg/kg", "qsi": "kg/kg",
    "znt": "m", "ust": "m/s",
}

#: ``parcel.F:2013-2024``: every ``prcl_nc`` column is published in ``1/kg``.
CM1_PARCEL_NUMBER_CONCENTRATION_UNIT = "1/kg"


def parcel_number_concentrations(profile: Any) -> tuple[tuple[str, int], ...]:
    """``param.F:6434-6438``: the ``prcl_nc`` attributes and their ``q3d`` rows.

    CM1 adds ``qname(n)`` for ``n = nnc1..nnc2`` when ``prcl_nc = 1`` and
    ``idm = 1``, and ``parcel_interp`` samples ``q3d(n)`` for each at the
    scalar point (``parcel.F:1430-1431``).  A single-moment axis contributes
    none.  Rows are zero-based.
    """

    if int(profile.idm) != 1:
        return ()
    rows = tuple(range(int(profile.nnc1) - 1, int(profile.nnc2)))
    names = profile.names
    if not rows or rows[-1] >= len(names):
        raise ValueError(
            f"{profile.component_id} declares idm=1 but no nnc1..nnc2 rows on its axis"
        )
    return tuple((names[row], row) for row in rows)


def _number_concentration_names() -> frozenset[str]:
    from .moisture_species import MOISTURE_COMPONENT_IDS, moisture_species_profile

    return frozenset(
        name
        for component in MOISTURE_COMPONENT_IDS
        for name, _ in parcel_number_concentrations(moisture_species_profile(component))
    )


def parcel_attribute_units(name: str) -> str:
    """CM1's unit for a CM1 attribute; the GM1 vorticity budget's otherwise."""

    if name in PARCEL_ATTRIBUTE_UNITS:
        return PARCEL_ATTRIBUTE_UNITS[name]
    if name in _number_concentration_names():
        return CM1_PARCEL_NUMBER_CONCENTRATION_UNIT
    if name in ("xvortb", "yvortb", "zvortb"):
        return "1/s"
    if name[:6] in ("xvortb", "yvortb", "zvortb") and name[6:7] == "_":
        return "1/s^2"
    raise KeyError(name)


#: Attributes this port declares but cannot fill from a Cartesian-lane
#: carrier, and what a receipt reader sees instead.  Naming them is the
#: difference between a column of zeros that looks like data and a column a
#: reader knows is unfilled.
PARCEL_UNFILLED_ATTRIBUTES: Mapping[str, str] = {
    "kmh": (
        "cuda_cartesian_closure.py:336-339 allocates kmh/kmv/khh/khv as "
        "held-tendency phase scratch and releases them at the end of that "
        "phase, so parcel.F:1372's sample has no live carrier at solve3 time"
    ),
    "kmv": (
        "see kmh: the Cartesian closure's eddy viscosities are phase scratch, "
        "not a retained carrier"
    ),
    "khh": (
        "see kmh: the Cartesian closure's eddy diffusivities are phase "
        "scratch, not a retained carrier"
    ),
    "khv": (
        "see kmh: the Cartesian closure's eddy diffusivities are phase "
        "scratch, not a retained carrier"
    ),
    "qsl": (
        "parcel.F:1029-1050 evaluates cm1libs.F's rslf polynomial fit, which "
        "this port has not transcribed; publishing a different saturation "
        "formula under CM1's name would misreport every supersaturation"
    ),
    "qsi": (
        "parcel.F:1029-1050 evaluates cm1libs.F's rsif polynomial fit, which "
        "this port has not transcribed"
    ),
    "dbz": (
        "the Cartesian lane's reflectivity is produced by the ArWen WSM6 "
        "component per physics block and is not retained past the block, so "
        "parcel.F:1420's qd_dbz sample has no live carrier here"
    ),
    "znt": (
        "parcel.F:1524-1525's prcl_sfc sample of zntmp is not published: the "
        "semi-slip integrator reads the roughness (parcel.F:470-515), but this "
        "column is not recorded from it (param.F:6280-6282 lays the column out "
        "only for bbc=3)"
    ),
    "ust": (
        "friction velocity is surface-layer tile scratch "
        "(cuda_surface_drag.SURFACE_STATE_NAMES), not a retained carrier, so "
        "parcel.F:1526-1527's sample has nothing to read at output time"
    ),
}


def _finite_positive(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and > 0")
    return result


def _count(name: str, value: Any, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


#: The mass rows ``parcel_interp`` samples (``parcel.F:1424-1427``) that this
#: port's sampler fills (``nest_parcels._derived_at``): CM1's six-row prefix.
PARCEL_MASS_ROW_NAMES: tuple[str, ...] = ("qv", "qc", "qr", "qi", "qs", "qg")


def _word(name: str, value: Any, admitted: Sequence[int] | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a CM1 integer word")
    if admitted is not None and value not in admitted:
        raise ValueError(f"{name}={value} is not one of {sorted(admitted)}")
    return value


@dataclass(frozen=True, slots=True)
class ParcelSwitchContext:
    """The CM1 words ``param.F`` resolves and lays out the sixteen switches with.

    ``param.F:6257-6282`` overrides a requested ``prcl_*`` from ``iptra``,
    ``imoist``, ``ptype``, ``idm``, ``sgsmodel``, ``ipbl``, ``iusetke`` and
    ``bbc``; ``:6385-6500`` then gates the columns on ``idm``, ``iice``,
    ``cm1setup``/``ipbl`` and ``terrain_flag``.  Every word is the value CM1
    holds when it reaches those lines -- ``bbc`` after ``:873-889``'s rewrite,
    ``sgsmodel`` after ``:1074-1085``'s, ``iusetke`` as ``:4989-4997`` and
    ``:5107-5108`` leave it -- which is why :func:`cm1_parcel_switch_context`
    derives them from namelist words rather than taking them.  ``mass_rows``
    and ``number_rows`` are the ``prcl_q`` and ``prcl_nc`` columns this axis
    lays out.
    """

    imoist: int
    ptype: int | None
    iice: int
    idm: int
    iptra: int
    cm1setup: int
    sgsmodel: int
    ipbl: int
    iusetke: bool
    bbc: int
    terrain: bool
    mass_rows: tuple[str, ...]
    number_rows: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        for name in ("imoist", "iice", "idm", "iptra"):
            _word(name, getattr(self, name), (0, 1))
        for name in ("cm1setup", "sgsmodel", "ipbl", "bbc"):
            _word(name, getattr(self, name))
        if self.ptype is not None:
            _word("ptype", self.ptype)
        for name in ("iusetke", "terrain"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be a CM1 logical")
        if self.iptra != 0:
            raise ValueError(
                "iptra=1: param.F:6415-6420 lays out pt1..ptnpt and parcel.F:1449-1451 "
                "samples them from pt3d, but this port carries no passive-tracer family, so "
                "those columns would be published with nothing behind them"
            )
        if self.terrain:
            raise ValueError(
                "terrain_flag: param.F:6470-6475 drops zvort, :6494-6499 adds zs and sigma, "
                "and parcel.F:137-619 integrates sigma rather than height; this port's parcel "
                "integrator and sampler carry neither, so the columns would not be CM1's"
            )
        mass = tuple(str(name) for name in self.mass_rows)
        rows = tuple((str(name), int(row)) for name, row in self.number_rows)
        object.__setattr__(self, "mass_rows", mass)
        object.__setattr__(self, "number_rows", rows)
        if self.imoist == 0 and (mass or rows or self.iice or self.idm):
            raise ValueError(
                "imoist=0 carries no moisture axis: param.F:3099-3110 leaves iice=idm=0 and "
                ":6260-6264 lays out no moisture column"
            )
        if self.imoist == 1 and (
            not mass or mass[0] != "qv" or len(set(mass)) != len(mass)
            or any(name not in PARCEL_MASS_ROW_NAMES for name in mass)
        ):
            raise ValueError(
                f"prcl_q's rows {mass} are not qv followed by CM1 mass rows this port samples "
                f"({', '.join(PARCEL_MASS_ROW_NAMES)}); parcel.F:1424-1427 would be published "
                "under names the sampler does not fill"
            )
        taken = set(PARCEL_ATTRIBUTE_UNITS) | set(PARCEL_BASE_ATTRIBUTES)
        if len({name for name, _ in rows}) != len(rows) or any(
            name in taken or row < 0 for name, row in rows
        ):
            raise ValueError(f"number-concentration attributes must be distinct new rows: {rows}")
        if bool(rows) != (self.idm == 1):
            raise ValueError(
                f"idm={self.idm} with number rows {rows}: param.F:6435-6439 lays out "
                "qname(nnc1..nnc2) exactly when idm=1"
            )

    def record(self) -> dict[str, Any]:
        return {
            "imoist": self.imoist, "ptype": self.ptype, "iice": self.iice, "idm": self.idm,
            "iptra": self.iptra, "cm1setup": self.cm1setup, "sgsmodel": self.sgsmodel,
            "ipbl": self.ipbl, "iusetke": self.iusetke, "bbc": self.bbc,
            "terrain_flag": self.terrain, "prcl_q_rows": list(self.mass_rows),
            "cm1_authority": {
                "bbc": SOURCE_ANCHORS["bbc_rewrite"],
                "sgsmodel": SOURCE_ANCHORS["sgsmodel_rewrite"],
                "iusetke": SOURCE_ANCHORS["iusetke"],
                "idoles_idopbl": SOURCE_ANCHORS["idoles_idopbl"],
            },
        }


def cm1_parcel_switch_context(
    *,
    imoist: int,
    profile: Any,
    cm1setup: int,
    sgsmodel: int,
    ipbl: int,
    sfcmodel: int,
    oceanmodel: int,
    bbc: int,
    iptra: int,
    terrain: bool,
) -> ParcelSwitchContext:
    """Derive :class:`ParcelSwitchContext` from namelist words, as ``param.F`` does.

    ``profile`` is the moisture axis (:class:`cm1core.moisture_species.
    MoistureSpeciesProfile`, or anything with its ``cm1_ptype``, ``iice``,
    ``idm``, ``nqv``/``nql1``/``nql2``/``nqs1``/``nqs2``/``nnc1``/``nnc2`` and
    ``names``); ``None`` exactly when ``imoist = 0``.

    Every rewrite ``param.F`` makes to a word the resolution reads, in source
    order: ``:873-889`` forces ``bbc = 3`` when ``sfcmodel >= 1``,
    ``oceanmodel = 2`` or ``ipbl >= 1`` (``:856-870`` only lifts an
    ``oceanmodel <= 0`` to 1, which cannot make it 2); ``:1074-1085`` zeroes
    ``sgsmodel`` without ``idoles``; ``:4989-4997`` and ``:5107-5108`` set
    ``iusetke``.  No admitted GM1 run reaches the ``bbc`` rewrite -- the door
    refuses ``sfcmodel >= 1`` without ``bbc = 3`` by name (``config.py``,
    ``cuda_surface_drag.py``), refuses ``oceanmodel = 2``, and holds this lane
    to ``cm1setup = 1``, where ``ipbl >= 1`` is refused -- but this resolver
    applies it anyway, and the CM1 oracle's ``bbc-forced-by-surface-layer``
    case holds it to CM1.
    """

    from .cm1_physics_families import cm1setup_flags

    _word("imoist", imoist, (0, 1))
    for name, value in (("cm1setup", cm1setup), ("sgsmodel", sgsmodel), ("ipbl", ipbl),
                        ("sfcmodel", sfcmodel), ("oceanmodel", oceanmodel), ("bbc", bbc),
                        ("iptra", iptra)):
        _word(name, value)
    if sfcmodel >= 1 or oceanmodel == 2 or ipbl >= 1:  # param.F:873-889
        bbc = 3
    idoles, idopbl = cm1setup_flags(cm1setup)  # param.F:752-753
    if not idoles and sgsmodel >= 1:  # param.F:1074-1085
        sgsmodel = 0
    iusetke = bool(
        (idopbl and ipbl in (4, 5)) or sfcmodel == 6  # param.F:4989-4997
        or (idoles and sgsmodel in (1, 3, 4, 5))  # param.F:5107-5108
    )
    words = dict(cm1setup=cm1setup, sgsmodel=sgsmodel, ipbl=ipbl, iusetke=iusetke, bbc=bbc,
                 iptra=iptra, terrain=bool(terrain))
    if imoist == 0:
        if profile is not None:
            raise ValueError("imoist=0 names no moisture axis, so no profile is passed")
        return ParcelSwitchContext(imoist=0, ptype=None, iice=0, idm=0, mass_rows=(),
                                   number_rows=(), **words)
    if profile is None:
        raise ValueError("imoist=1 lays out its axis's rows (param.F:6422-6439); pass its profile")
    names = tuple(profile.names)
    iice = int(profile.iice)
    mass = (names[int(profile.nqv) - 1],) + names[int(profile.nql1) - 1:int(profile.nql2)]
    if iice == 1:
        mass += names[int(profile.nqs1) - 1:int(profile.nqs2)]
    return ParcelSwitchContext(
        imoist=1, ptype=profile.cm1_ptype, iice=iice, idm=int(profile.idm),
        mass_rows=mass, number_rows=parcel_number_concentrations(profile), **words,
    )


def parcel_switch_context(config: Any) -> ParcelSwitchContext:
    """The context of one bound runtime contract (:class:`cm1core.gm1_runtime.RuntimeConfig`).

    ``physics.component = "off"`` is CM1's ``imoist = 0``.  ``iptra = 0``: this
    port carries no passive-tracer family.  ``terrain = False``: ``config.py``
    refuses ``grid.terrain`` outside the real-atmosphere lane, and the
    ``[lagrangian]`` table binds only in the Cartesian lane.
    """

    imoist = 0 if getattr(config, "microphysics", None) == "off" else 1
    return cm1_parcel_switch_context(
        imoist=imoist, profile=None if imoist == 0 else config.profile,
        cm1setup=int(config.cm1setup), sgsmodel=int(config.sgsmodel), ipbl=int(config.ipbl),
        sfcmodel=int(config.sfcmodel), oceanmodel=int(config.oceanmodel), bbc=int(config.bbc),
        iptra=0, terrain=False,
    )


#: ``param.F:6257-6282``, one rule per ``if``, in source order.
_OVERRIDE_RULES: tuple[tuple[str, Any, tuple[str, ...]], ...] = (
    ("src/param.F:6257-6259: iptra=0", lambda c: c.iptra == 0, ("prcl_ptra",)),
    ("src/param.F:6260-6264: imoist=0", lambda c: c.imoist == 0,
     ("prcl_q", "prcl_nc", "prcl_dbz", "prcl_qsat")),
    ("src/param.F:6266-6268: ptype=1 or 4", lambda c: c.imoist != 0 and c.ptype in (1, 4),
     ("prcl_dbz",)),
    ("src/param.F:6269-6271: idm=0", lambda c: c.imoist != 0 and c.idm == 0, ("prcl_nc",)),
    ("src/param.F:6273-6276: sgsmodel<=0 or ipbl<=0",
     lambda c: c.sgsmodel <= 0 or c.ipbl <= 0, ("prcl_km", "prcl_kh")),
    ("src/param.F:6277-6279: iusetke false", lambda c: not c.iusetke, ("prcl_tke",)),
    ("src/param.F:6280-6282: bbc/=3", lambda c: c.bbc != 3, ("prcl_sfc",)),
)


def resolve_parcel_switches(
    requested: Mapping[str, int], context: ParcelSwitchContext,
) -> tuple[dict[str, int], dict[str, str]]:
    """``param.F:6257-6282``: the switches CM1 holds, and why each one was turned off.

    ``requested`` is all sixteen, each 0 or 1 (``:6239-6254`` clamps; this
    port's door refuses any other value by name).  The second mapping names the
    rule for every switch requested on that the resolution turned off.
    """

    resolved = {name: int(requested[name]) for name in PARCEL_SWITCH_ORDER}
    overrides: dict[str, str] = {}
    for rule, applies, names in _OVERRIDE_RULES:
        if applies(context):
            for name in names:
                if resolved[name] == 1:
                    overrides[name] = rule
                resolved[name] = 0
    return resolved, overrides


@dataclass(frozen=True, slots=True)
class ParcelAttributeLayout:
    """Which attributes this run records, in CM1's own order.

    ``param.F:6239-6500`` resolves the sixteen switches and builds ``npvals``
    and the ``pr*`` offsets from them exactly this way; the only difference is
    that the offsets here are zero-based Python indices into a
    ``(npvals, nparcels)`` array rather than one-based Fortran column numbers.
    ``switches`` are the resolved values (what ``param.F:6286-6301`` prints);
    ``requested`` what the document asked for.
    """

    names: tuple[str, ...]
    switches: Mapping[str, int]
    #: ``prcl_nc``'s ``(name, zero-based q3d row)`` pairs, in ``nnc1..nnc2``
    #: order; empty unless the switch resolves on (so the axis is double-moment).
    number_rows: tuple[tuple[str, int], ...]
    requested: Mapping[str, int]
    #: switch -> the ``param.F:6257-6282`` rule that turned it off.
    overrides: Mapping[str, str]
    #: switch -> the ``param.F:6385-6500`` gate that lays out no column for it.
    gated: Mapping[str, str]
    context: ParcelSwitchContext

    @property
    def npvals(self) -> int:
        return len(self.names)

    def index(self, name: str) -> int:
        return self.names.index(name)

    @property
    def unfilled(self) -> tuple[str, ...]:
        return tuple(n for n in self.names if n in PARCEL_UNFILLED_ATTRIBUTES)

    def record(self) -> dict[str, Any]:
        return {
            "schema": LAGRANGIAN_PARCEL_SCHEMA,
            "npvals": self.npvals,
            "names": list(self.names),
            "units": [parcel_attribute_units(n) for n in self.names],
            "switches": {k: int(v) for k, v in sorted(self.switches.items())},
            "requested_switches": {k: int(v) for k, v in sorted(self.requested.items())},
            "switch_overrides": dict(sorted(self.overrides.items())),
            "switches_without_columns": dict(sorted(self.gated.items())),
            "switch_context": self.context.record(),
            "cm1_authority": SOURCE_ANCHORS["attribute_offsets"],
            "switch_resolution_authority": SOURCE_ANCHORS["switch_resolution"],
            "unfilled_attributes": {
                name: PARCEL_UNFILLED_ATTRIBUTES[name] for name in self.unfilled
            },
            # Only a double-moment axis with prcl_nc on has any.
            **(
                {"number_concentration_rows": {name: row for name, row in self.number_rows}}
                if self.number_rows else {}
            ),
        }


def parcel_attribute_layout(
    switches: Mapping[str, Any],
    context: ParcelSwitchContext,
) -> ParcelAttributeLayout:
    """``param.F:6239-6500``: resolve the requested switches, then lay out the columns.

    ``context`` is required: a layout built from the request alone publishes
    columns CM1 turns off (``prcl_nc`` on a single-moment axis, ``kmh``..``khv``
    on an LES deck, ``znt``/``ust`` without the semi-slip surface).  A switch
    missing from ``switches`` is requested off.
    """

    if not isinstance(context, ParcelSwitchContext):
        raise TypeError(
            "parcel_attribute_layout needs the run's ParcelSwitchContext: param.F:6257-6282 "
            "resolves every prcl_* switch against the run's own words before one column is "
            "laid out"
        )
    unknown = sorted(set(switches) - set(PARCEL_SWITCH_ATTRIBUTES))
    if unknown:
        raise ValueError(f"unknown &param13 parcel switch: {unknown[0]}")
    requested: dict[str, int] = {}
    for switch in PARCEL_SWITCH_ORDER:
        value = switches.get(switch, 0)
        if isinstance(value, bool):
            value = int(value)
        if not isinstance(value, int) or value not in (0, 1):
            raise ValueError(f"{switch} must be 0 or 1 (CM1 param.F tests .eq.1)")
        requested[switch] = value
    resolved, overrides = resolve_parcel_switches(requested, context)
    turbulence = context.cm1setup >= 1 or context.ipbl >= 1  # param.F:6443
    names: list[str] = list(PARCEL_BASE_ATTRIBUTES)
    gated: dict[str, str] = {}
    for switch in PARCEL_SWITCH_ORDER:
        if resolved[switch] != 1:
            continue
        if switch == "prcl_q":  # param.F:6422-6434
            names.extend(context.mass_rows)
        elif switch == "prcl_nc":  # param.F:6435-6439; resolved on means idm=1
            names.extend(name for name, _ in context.number_rows)
        elif switch in ("prcl_km", "prcl_kh", "prcl_tke") and not turbulence:
            gated[switch] = "src/param.F:6443: cm1setup<1 and ipbl<1 lay out no turbulence column"
        elif switch == "prcl_qsat":  # param.F:6480-6487
            names.extend(("qsl", "qsi") if context.iice == 1 else ("qsl",))
        else:
            names.extend(PARCEL_SWITCH_ATTRIBUTES[switch])
    return ParcelAttributeLayout(
        names=tuple(names), switches=resolved,
        number_rows=context.number_rows if resolved["prcl_nc"] == 1 else (),
        requested=requested, overrides=overrides, gated=gated, context=context,
    )


_DECK_CACHE: dict[tuple[Any, ...], tuple[Any, Any]] = {}


@dataclass(frozen=True, slots=True)
class ParcelGrid:
    """The uniform Cartesian coordinate a parcel position is measured in.

    ``iorigin`` is CM1's own ``&param2`` word (``param.F:6646-6664``);
    ``run/namelist.input:81`` ships ``2``, the domain-centred origin.  It is
    carried here rather than derived because every published parcel position
    is meaningless without it: the same parcel is at ``x=0`` or at
    ``x=nx*dx/2`` depending on this one integer.
    """

    nx: int
    ny: int
    nz: int
    dx_m: float
    dy_m: float
    dz_m: float
    iorigin: int = 2
    #: The grid's CM1 vertical coordinate (``cm1_vertical_grid.VerticalLevels``,
    #: ``zf``/``zh`` CM1-indexed with ghosts).  ``None`` is ``param.F:7789-7811``'s
    #: uniform deck of ``nz`` layers of ``dz_m``; a stretched grid must pass its
    #: own, because ``parcel.F:251-260`` searches the grid's real faces.
    levels: Any = None

    def __post_init__(self) -> None:
        for name in ("nx", "ny", "nz"):
            _count(name, getattr(self, name), minimum=4)
        for name in ("dx_m", "dy_m", "dz_m"):
            object.__setattr__(self, name, _finite_positive(name, getattr(self, name)))
        if self.iorigin not in (1, 2):
            raise ValueError(
                "iorigin must be 1 or 2; param.F:6666-6672 stops CM1 on anything else"
            )
        if self.levels is not None and int(self.levels.nk) != self.nz:
            raise ValueError(
                f"the vertical coordinate has {int(self.levels.nk)} layers but nz is {self.nz}"
            )

    @property
    def minx_m(self) -> float:
        return 0.0 if self.iorigin == 1 else -0.5 * self.dx_m * self.nx

    @property
    def maxx_m(self) -> float:
        return self.minx_m + self.dx_m * self.nx

    @property
    def miny_m(self) -> float:
        return 0.0 if self.iorigin == 1 else -0.5 * self.dy_m * self.ny

    @property
    def maxy_m(self) -> float:
        return self.miny_m + self.dy_m * self.ny

    @property
    def maxz_m(self) -> float:
        """``param.F:8193`` ``maxz = sigmaf(nk+1)``, ``zf(nk+1)`` without terrain."""

        if self.levels is not None:
            return float(self.levels.zf[self.nz + 1])
        return self.dz_m * self.nz

    @property
    def lowest_mass_level_m(self) -> float:
        """``zh(1)`` -- the height ``init3d.F:387`` seeds the first layer at."""

        return self.mass_level_m(1)

    def mass_level_m(self, k: int) -> float:
        """CM1's ``zh(k)`` on this grid's own deck."""

        if self.levels is not None:
            return float(self.levels.zh[int(k)])
        return self.dz_m * (float(k) - 0.5)

    def deck(self, lateral: tuple[int, int, int, int] = (2, 2, 2, 2)) -> Any:
        """This grid as :class:`cm1core.nest_parcels.ParcelDeck`, with the periodic exit.

        ``xf(1)`` is ``minx`` in this grid's ``iorigin`` frame; ``lateral`` is
        ``(wbc, ebc, sbc, nbc)`` for the ``bc.F`` halo rules.
        """

        key = (self.nx, self.ny, self.nz, self.dx_m, self.dy_m, self.dz_m, self.iorigin,
               id(self.levels), tuple(int(v) for v in lateral))
        cached = _DECK_CACHE.get(key)
        if cached is not None and cached[0] is self.levels:
            return cached[1]
        from .cm1_vertical_grid import uniform_vertical_levels
        from .nest_parcels import deck_from_levels

        levels = self.levels if self.levels is not None else uniform_vertical_levels(self.nz, self.dz_m)
        deck = deck_from_levels(
            name="parent", nx=self.nx, ny=self.ny, dx_m=self.dx_m, dy_m=self.dy_m,
            minx_m=self.minx_m, miny_m=self.miny_m, levels=levels,
            lateral=tuple(int(v) for v in lateral), periodic=True,  # type: ignore[arg-type]
        )
        _DECK_CACHE[key] = (self.levels, deck)
        return deck

    def record(self) -> dict[str, Any]:
        return {
            "nx": self.nx, "ny": self.ny, "nz": self.nz,
            "dx_m": self.dx_m, "dy_m": self.dy_m, "dz_m": self.dz_m,
            "iorigin": self.iorigin,
            "minx_m": self.minx_m, "maxx_m": self.maxx_m,
            "miny_m": self.miny_m, "maxy_m": self.maxy_m,
            "maxz_m": self.maxz_m,
            "vertical": "uniform" if self.levels is None else "grid levels",
            "vertical_sha256": None if self.levels is None else str(self.levels.sha256),
            "cm1_authority": SOURCE_ANCHORS["horizontal_origin"],
        }


# --------------------------------------------------------------------------
# Seeding.  GM1 EXTENSION beyond init3d.F's single hard-coded grid.
# --------------------------------------------------------------------------

SEEDING_MODES: tuple[str, ...] = ("grid", "box", "vorticity-threshold")


@dataclass(frozen=True, slots=True)
class ParcelSeeding:
    """Where the parcels start.

    ``mode="grid"`` is ``init3d.F:381-387`` with its three hard-coded counts
    and three hard-coded spacings promoted to parameters; CM1's own values are
    ``60, 60, 10`` at ``2000, 2000, 1000`` metres from ``(minx, miny, zh(1))``,
    which this class reproduces exactly when they are supplied.

    ``mode="box"`` (GM1 EXTENSION) fills a named rectangular box, which is how
    a tornado study actually seeds: a dense cloud around the low-level
    mesocyclone, not a sparse net over a 120 km domain.

    ``mode="vorticity-threshold"`` (GM1 EXTENSION) seeds, at the first step,
    only the grid points whose vertical vorticity exceeds
    ``vorticity_threshold_s_inv``, taking them in ``(k, j, i)`` order until
    ``nparcels`` is reached.  This is the seeding the field wants and cannot
    have post hoc: it needs the vorticity field at model time, on the device.
    """

    mode: str = "grid"
    nparcels: int = 36000
    seed_nx: int = 60
    seed_ny: int = 60
    seed_nz: int = 10
    seed_dx_m: float = 2000.0
    seed_dy_m: float = 2000.0
    seed_dz_m: float = 1000.0
    seed_x0_m: float = 0.0
    seed_y0_m: float = 0.0
    seed_z0_m: float = 0.0
    vorticity_threshold_s_inv: float = 0.01

    def __post_init__(self) -> None:
        if self.mode not in SEEDING_MODES:
            raise ValueError(f"unknown parcel seeding mode {self.mode!r}")
        _count("nparcels", self.nparcels, minimum=1)
        for name in ("seed_nx", "seed_ny", "seed_nz"):
            _count(name, getattr(self, name), minimum=1)
        for name in ("seed_dx_m", "seed_dy_m", "seed_dz_m"):
            _finite_positive(name, getattr(self, name))
        for name in ("seed_x0_m", "seed_y0_m", "seed_z0_m",
                     "vorticity_threshold_s_inv"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, float(value))

    def record(self) -> dict[str, Any]:
        payload = {
            "mode": self.mode,
            "nparcels": self.nparcels,
            "seed_nx": self.seed_nx, "seed_ny": self.seed_ny, "seed_nz": self.seed_nz,
            "seed_dx_m": self.seed_dx_m, "seed_dy_m": self.seed_dy_m,
            "seed_dz_m": self.seed_dz_m,
            "seed_x0_m": self.seed_x0_m, "seed_y0_m": self.seed_y0_m,
            "seed_z0_m": self.seed_z0_m,
            "vorticity_threshold_s_inv": self.vorticity_threshold_s_inv,
            "cm1_authority": SOURCE_ANCHORS["seed_grid"],
        }
        payload["beyond_cm1"] = self.mode != "grid"
        return payload

    @property
    def is_cm1_default_grid(self) -> bool:
        """True when this is literally ``init3d.F:381-387``'s own seeding."""

        return (
            self.mode == "grid"
            and (self.seed_nx, self.seed_ny, self.seed_nz) == (60, 60, 10)
            and (self.seed_dx_m, self.seed_dy_m, self.seed_dz_m) == (2000.0, 2000.0, 1000.0)
            and (self.seed_x0_m, self.seed_y0_m, self.seed_z0_m) == (0.0, 0.0, 0.0)
        )


def seed_parcel_positions(
    seeding: ParcelSeeding,
    grid: ParcelGrid,
    *,
    vertical_vorticity: Any = None,
) -> Any:
    """Return the ``(3, nparcels)`` float32 seed positions ``(x, y, z)``.

    ``init3d.F:363-365`` initialises every parcel to ``-1.0e30`` so that an
    MPI ``ALLREDUCE(MAX)`` can gather them; that sentinel is kept here for any
    parcel the requested seeding does not place, so an unplaced parcel is
    visibly unplaced rather than silently sitting at the origin.
    """

    np = importlib.import_module("numpy")
    n = seeding.nparcels
    out = np.full((3, n), np.float32(-1.0e30), dtype=np.float32)
    if seeding.mode == "grid":
        # init3d.F:381-387, with the counts and spacings promoted.  CM1 stops
        # the model when the triple loop overruns nparcels; overrunning is a
        # configuration error there and here, so it is refused by value.
        wanted = seeding.seed_nx * seeding.seed_ny * seeding.seed_nz
        if wanted > n:
            raise ValueError(
                f"seeding grid asks for {wanted} parcels but nparcels is {n}; "
                "init3d.F:375-383 stops CM1 in exactly this case"
            )
        index = 0
        base_z = grid.lowest_mass_level_m + seeding.seed_z0_m
        for k in range(seeding.seed_nz):
            for j in range(seeding.seed_ny):
                for i in range(seeding.seed_nx):
                    out[0, index] = np.float32(
                        grid.minx_m + seeding.seed_x0_m + seeding.seed_dx_m * i
                    )
                    out[1, index] = np.float32(
                        grid.miny_m + seeding.seed_y0_m + seeding.seed_dy_m * j
                    )
                    out[2, index] = np.float32(base_z + seeding.seed_dz_m * k)
                    index += 1
    elif seeding.mode == "box":
        wanted = seeding.seed_nx * seeding.seed_ny * seeding.seed_nz
        if wanted > n:
            raise ValueError(
                f"seeding box asks for {wanted} parcels but nparcels is {n}"
            )
        index = 0
        for k in range(seeding.seed_nz):
            for j in range(seeding.seed_ny):
                for i in range(seeding.seed_nx):
                    out[0, index] = np.float32(seeding.seed_x0_m + seeding.seed_dx_m * i)
                    out[1, index] = np.float32(seeding.seed_y0_m + seeding.seed_dy_m * j)
                    out[2, index] = np.float32(seeding.seed_z0_m + seeding.seed_dz_m * k)
                    index += 1
    else:
        if vertical_vorticity is None:
            raise ValueError(
                "vorticity-threshold seeding needs the vertical vorticity field; "
                "it is a model-time decision and cannot be made from configuration alone"
            )
        zeta = np.asarray(vertical_vorticity)
        if zeta.ndim != 3 or zeta.shape != (grid.nz, grid.ny, grid.nx):
            raise ValueError(
                "vertical vorticity must be the (nz, ny, nx) scalar-point field"
            )
        threshold = np.float32(seeding.vorticity_threshold_s_inv)
        hits = np.argwhere(np.abs(zeta) >= threshold)
        index = 0
        for k, j, i in hits:
            if index >= n:
                break
            out[0, index] = np.float32(grid.minx_m + grid.dx_m * (float(i) + 0.5))
            out[1, index] = np.float32(grid.miny_m + grid.dy_m * (float(j) + 0.5))
            out[2, index] = np.float32(grid.mass_level_m(int(k) + 1))
            index += 1
    return out


def parcel_circuit_positions(
    *,
    center_x_m: float,
    center_y_m: float,
    center_z_m: float,
    radius_m: float,
    count: int,
) -> Any:
    """Seed one closed material circuit as a horizontal ring of parcels.

    GM1 EXTENSION: CM1 has no circuit diagnostic.  The ring is the discrete
    material circuit whose circulation :mod:`cm1core.circulation_circuits`
    integrates; seeding it here keeps the ordering (counter-clockwise, first
    parcel on the +x radius) that the circulation sum depends on.
    """

    np = importlib.import_module("numpy")
    _count("count", count, minimum=3)
    radius = _finite_positive("radius_m", radius_m)
    angle = np.arange(count, dtype=np.float64) * (2.0 * math.pi / float(count))
    out = np.empty((3, count), dtype=np.float32)
    out[0] = (np.float64(center_x_m) + radius * np.cos(angle)).astype(np.float32)
    out[1] = (np.float64(center_y_m) + radius * np.sin(angle)).astype(np.float32)
    out[2] = np.float32(center_z_m)
    return out


# --------------------------------------------------------------------------
# The integrator.  One transcription serves every grid: cm1core.nest_parcels
# holds parcel.F's searches, stencils, semi-slip log layer and RK2 as a host
# reference and as the packaged kernel text (proved raw-word against each
# other by the g++ harness), and this lane's parent parcels run on it with the
# parent's own deck.  What is kept here is the carrier type, the ghost fill
# the Eulerian budget shares, and host wrappers the analytic tests call.
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ParcelVelocityField:
    """The three padded C-grid velocity carriers a parcel step reads.

    CM1's ``uten1``/``vten1``/``wten1`` for ``psolver=3``: GM1's ``rru``/``rrv``/
    ``rrw`` acoustic small-step averages, copied at ``solve2.F:1290-1301``.
    They carry ``CM1_OPEN_NGXY`` lateral halo cells; ``u`` and ``v`` hold CM1
    levels ``0..nk+1`` and ``w`` levels ``0..nk+2``, each CM1 level ``k`` at
    storage index ``k``.
    """

    u: Any
    v: Any
    w: Any


def fill_parcel_vertical_boundaries(
    field: ParcelVelocityField,
    *,
    nx: int,
    ny: int,
    nz: int,
    bbc: int,
    tbc: int,
    u_move_m_s: float = 0.0,
    v_move_m_s: float = 0.0,
    xp: Any = None,
) -> None:
    """``parcel.F:57-130`` with ``param.F:9038-9052``'s uniform-deck weights.

    The parcel step itself fills with its deck's own ``cgs``/``cgt``
    (:func:`cm1core.nest_parcels.fill_velocity_ghosts`); this form is what the
    Eulerian vorticity budget uses on the uniform lane.
    """

    if xp is None:
        xp = importlib.import_module("numpy")
    f32 = xp.float32
    for name, array in (("u", field.u), ("v", field.v), ("w", field.w)):
        if array.ndim != 3 or array.shape[0] != nz + 2 + (1 if name == "w" else 0):
            raise ValueError(f"parcel velocity carrier {name} has unexpected extent")
    if bbc == 1:
        # free slip: param.F:9038's one-sided quadratic, parcel.F:63-72.
        for array in (field.u, field.v):
            array[0] = (
                f32(CM1_CGS1_F32) * array[1]
                + f32(CM1_CGS2_F32) * array[2]
                + f32(CM1_CGS3_F32) * array[3]
            )
    elif bbc == 2:
        # no slip, parcel.F:73-93.  The stored wind is frame-relative, so a
        # ground-fixed zero is -umove in this frame (init3d.F:32-33).
        field.u[0] = f32(0.0) - f32(u_move_m_s)
        field.v[0] = f32(0.0) - f32(v_move_m_s)
    elif bbc == 3:
        # parcel.F:94-96: u,v below the lowest level come from the log layer
        # at sample time (parcel.F:470-515), not from a ghost fill.
        pass
    else:
        raise ValueError(f"bbc={bbc} has no parcel.F lower-boundary branch")
    if tbc == 1:
        for array in (field.u, field.v):
            array[nz + 1] = (
                f32(CM1_CGS1_F32) * array[nz]
                + f32(CM1_CGS2_F32) * array[nz - 1]
                + f32(CM1_CGS3_F32) * array[nz - 2]
            )
    elif tbc == 2:
        field.u[nz + 1] = f32(0.0)
        field.v[nz + 1] = f32(0.0)
    else:
        raise ValueError(f"tbc={tbc} has no parcel.F upper-boundary branch")
    # parcel.F:124-128: wa(i,j,nk+1) = 0, the model-top face, for every tbc.
    # CM1 level k is storage index k, so that is index nz+1, not nz+2.
    field.w[nz + 1] = f32(0.0)


def _sample_uvw(
    np: Any,
    field: ParcelVelocityField,
    grid: ParcelGrid,
    x: Any,
    y: Any,
    z: Any,
    *,
    surface: Any = None,
) -> tuple[Any, Any, Any]:
    """``parcel.F:251-515`` at one position on ``grid``'s own deck: ``(u, v, w)``."""

    from .nest_parcels import STATUS_REASONS, reference_sample_uvw

    u, v, w, status = reference_sample_uvw(
        np, field.u, field.v, field.w, grid.deck(),
        np.asarray([x], dtype=np.float32), np.asarray([y], dtype=np.float32),
        np.asarray([z], dtype=np.float32),
        second_stage=False, ng=CM1_OPEN_NGXY, surface=surface,
    )
    if int(status[0]) != 0:
        raise ValueError(f"({x}, {y}, {z}) cannot be sampled: {STATUS_REASONS[int(status[0])]}")
    return u[0], v[0], w[0]


def cpu_parcel_advance(
    positions: Any,
    field: ParcelVelocityField,
    grid: ParcelGrid,
    *,
    dt_s: float,
    direction: int = 1,
    surface: Any = None,
    lateral: tuple[int, int, int, int] = (2, 2, 2, 2),
) -> Any:
    """One ``parcel.F:137-619`` RK2 step for every parcel, on ``grid``'s deck.

    ``direction=-1`` (GM1 EXTENSION) runs the same scheme with ``-dt``.  A
    parcel the step cannot take (off the lattice, past the deck ghosts,
    non-finite, or ``init3d.F:363``'s unplaced sentinel) keeps its words.
    """

    np = importlib.import_module("numpy")
    if direction not in (1, -1):
        raise ValueError("direction must be +1 (forward) or -1 (backward)")
    from .nest_parcels import reference_rk2

    out = np.array(positions, dtype=np.float32, copy=True)
    n = int(out.shape[1])
    moved, _ = reference_rk2(
        np, out, np.ones(n, dtype=np.int32), np.zeros(n, dtype=np.int32), code=1,
        u=field.u, v=field.v, w=field.w, deck=grid.deck(lateral),
        dt_s=_finite_positive("dt_s", dt_s) * direction, ng=CM1_OPEN_NGXY, surface=surface,
    )
    return moved


__all__ = [
    "CM1_CGS1_F32", "CM1_CGS2_F32", "CM1_CGS3_F32",
    "PARCEL_ATTRIBUTE_UNITS", "PARCEL_BASE_ATTRIBUTES",
    "PARCEL_SWITCH_ATTRIBUTES", "PARCEL_SWITCH_ORDER",
    "PARCEL_MASS_ROW_NAMES", "PARCEL_UNFILLED_ATTRIBUTES", "ParcelAttributeLayout",
    "ParcelGrid", "ParcelSwitchContext", "cm1_parcel_switch_context",
    "parcel_attribute_units", "parcel_number_concentrations", "parcel_switch_context",
    "resolve_parcel_switches",
    "CM1_PARCEL_NUMBER_CONCENTRATION_UNIT",
    "ParcelSeeding", "ParcelVelocityField", "SEEDING_MODES",
    "SOURCE_ANCHORS", "cpu_parcel_advance",
    "fill_parcel_vertical_boundaries",
    "parcel_attribute_layout", "parcel_circuit_positions",
    "seed_parcel_positions",
]
