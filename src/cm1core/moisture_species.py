"""CM1 r21.1 moisture-scalar inventories, keyed by the selected column component.

CM1 builds one ordered ``q3d(:,:,:,n)`` table per ``ptype`` in ``src/param.F``
and then drives every scalar stage from ``numq`` and the index ranges
``nqv``/``nql1``/``nql2``/``nqs1``/``nqs2``/``nnc1``/``nnc2``.  This module is
the CM1-owned transcription of exactly the two tables GM1 can execute, so no
consumer has to spell a species count or an index range inline.

Importing this module imports neither CuPy, :mod:`gpuwm`, nor NumPy: a
misconfigured species set must refuse before any device work is priced.

Authority (read-only tree ``work/native-reference/cm1r21.1``):

* ``src/param.F:3400-3582`` -- ``ptype=5`` (Morrison double moment).  The
  ``inum`` branch at ``:3417``/``:3476`` is not a namelist choice:
  ``src/morrison.F:325`` hard-codes ``INUM = 1`` and ``GRAUPEL_INIT`` returns
  it through ``cm1inum`` (``src/morrison.F:314,656``), so an unmodified r21.1
  always takes ``src/param.F:3420-3473`` and ``numq = 10``.  The ``numq = 11``
  table that adds ``ncc`` is unreachable and is therefore not transcribed.
* ``src/param.F:3632-3707`` -- ``ptype=7`` (WSM6), ``numq = 6``.
* ``src/param.F:936-950`` -- ``ptype=4,7,50,51,52,53,55`` are force-downgraded
  to ``eqtset=1``; ``ptype=5`` is absent from that list and keeps ``eqtset=2``.
* ``src/turb.F:5521-5535`` -- ``calcnm`` accumulates ``qt`` over exactly
  ``nqv``, ``nql1..nql2`` and ``nqs1..nqs2``; number concentrations are
  excluded by name, which is why both tables share one mass prefix.
* ``src/mp_driver.F:505-520`` -- the ``ptype=5`` positive-definite repair runs
  on scalars 1..6 only.  ``src/mp_driver.F:520`` shows the number-concentration
  ``pdefq`` call commented out in the authority, so GM1 must not add one.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from .errors import refuse


MOISTURE_SPECIES_SCHEMA = "gm1.moisture-species-profile/v1"

#: ``src/param.F:3400-3582`` and ``:3632-3707``.
MOISTURE_SPECIES_AUTHORITY: Mapping[str, str] = MappingProxyType({
    "ptype5_table": "src/param.F:3400-3473",
    "ptype5_inum_is_fixed": "src/morrison.F:314-325,656",
    "ptype7_table": "src/param.F:3632-3678",
    "eqtset_downgrade": "src/param.F:936-950",
    "calcnm_total_water": "src/turb.F:5521-5535",
    "ptype5_positive_definite_repair": "src/mp_driver.F:505-520",
    "ptype7_positive_definite_repair": "src/mp_driver.F:1469-1479",
    "scalar_transport_loop": "src/solve2.F:1435",
    "held_tendency_loops": "src/solve1.F:458,712,1215",
    "scalar_boundary_and_qa": "src/solve3.F:258,625",
})

MASS_MIXING_RATIO = "mass-mixing-ratio"
NUMBER_CONCENTRATION = "number-concentration"


@dataclass(frozen=True, slots=True)
class MoistureScalar:
    """One entry of CM1's ordered ``q3d`` scalar axis."""

    #: CM1's own ``qname(n)``, stripped of its fixed-width padding.
    name: str
    #: CM1's ``qunit(n)``.
    unit: str
    #: :data:`MASS_MIXING_RATIO` or :data:`NUMBER_CONCENTRATION`.
    kind: str
    #: CM1's ``qmag(n)``: the order-of-magnitude the WENO epsilon scales by.
    qmag: float
    #: CM1's ``cloudvar(n)``.
    cloudvar: bool

    @property
    def is_mass(self) -> bool:
        return self.kind == MASS_MIXING_RATIO

    def record(self) -> dict[str, object]:
        return {
            "name": self.name, "unit": self.unit, "kind": self.kind,
            "qmag": self.qmag, "cloudvar": self.cloudvar,
        }


@dataclass(frozen=True, slots=True)
class MoistureSpeciesProfile:
    """The complete CM1 scalar inventory one GM1 component executes.

    Index fields keep CM1's own one-based numbering so a transcribed source
    line stays findable from the profile.  ``-2`` is CM1's own "unset" value
    for ``nnc1``/``nnc2`` (``src/param.F:3111-3112``), which ``ptype=7`` never
    assigns.
    """

    component_id: str
    #: CM1's ``ptype`` for this scheme, or ``None`` where CM1 r21.1 has no
    #: scheme of its own to number.  ``None`` is not "unknown": it is the
    #: recorded fact that the axis below is ArWen's, and :attr:`axis_authority`
    #: says where it came from instead.
    cm1_ptype: int | None
    scalars: tuple[MoistureScalar, ...]
    nqv: int
    nql1: int
    nql2: int
    nqs1: int
    nqs2: int
    nnc1: int
    nnc2: int
    iice: int
    idm: int
    #: CM1's admitted ``eqtset`` for this ``ptype`` after ``param.F:936-950``.
    cm1_eqtset: int
    #: Where this scalar axis is transcribed from, file:line.  A CM1 ptype
    #: cites ``src/param.F``; a scheme CM1 does not carry cites the ArWen
    #: launcher whose required state fields define the axis instead.
    axis_authority: str = "src/param.F"

    @property
    def numq(self) -> int:
        """CM1's ``numq``: the length of the transported scalar axis."""

        return len(self.scalars)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(scalar.name for scalar in self.scalars)

    @property
    def units(self) -> tuple[str, ...]:
        return tuple(scalar.unit for scalar in self.scalars)

    @property
    def qmag(self) -> tuple[float, ...]:
        return tuple(scalar.qmag for scalar in self.scalars)

    @property
    def mass_names(self) -> tuple[str, ...]:
        """The scalars carried in kg kg-1, in axis order."""

        return tuple(scalar.name for scalar in self.scalars if scalar.is_mass)

    @property
    def number_names(self) -> tuple[str, ...]:
        """The scalars carried in kg-1, in axis order."""

        return tuple(scalar.name for scalar in self.scalars if not scalar.is_mass)

    @property
    def mass_indices(self) -> tuple[int, ...]:
        """Zero-based ``q3d`` rows that are mass mixing ratios."""

        return tuple(
            index for index, scalar in enumerate(self.scalars) if scalar.is_mass
        )

    @property
    def number_indices(self) -> tuple[int, ...]:
        """Zero-based ``q3d`` rows that are number concentrations."""

        return tuple(
            index for index, scalar in enumerate(self.scalars) if not scalar.is_mass
        )

    @property
    def repaired_scalars(self) -> int:
        """How many leading scalars CM1's pre-microphysics ``pdefq`` repairs.

        Both admitted tables repair exactly the mass prefix.  ``ptype=5``
        leaves the four number concentrations alone
        (``src/mp_driver.F:514-520``: the ``asq(7)`` call is commented out in
        the authority), and ``ptype=7`` has none to leave alone.
        """

        return len(self.mass_indices)

    @property
    def vapour_index(self) -> int:
        """Zero-based ``q3d`` row of water vapour."""

        return self.nqv - 1

    @property
    def condensate_indices(self) -> tuple[int, ...]:
        """Zero-based mass rows that are condensate rather than vapour."""

        return tuple(
            index for index in self.mass_indices if index != self.vapour_index
        )

    def index(self, name: str) -> int:
        """Zero-based ``q3d`` row of one scalar, refusing an absent name."""

        try:
            return self.names.index(name)
        except ValueError:
            raise KeyError(
                f"{self.component_id} carries no scalar named {name!r}; "
                f"its axis ({self.axis_authority}) is {list(self.names)}"
            ) from None

    def carrier_meaning(self) -> str:
        """The history envelope's one-line description of the packed axis."""

        return "current " + ",".join(self.names) + " moisture scalars"

    def record(self) -> dict[str, object]:
        return {
            "schema": MOISTURE_SPECIES_SCHEMA,
            "component_id": self.component_id,
            "cm1_ptype": self.cm1_ptype,
            "cm1_scheme_present": self.cm1_ptype is not None,
            "cm1_eqtset": self.cm1_eqtset,
            "axis_authority": self.axis_authority,
            "numq": self.numq,
            "scalars": [scalar.record() for scalar in self.scalars],
            "names": list(self.names),
            "mass_names": list(self.mass_names),
            "number_names": list(self.number_names),
            "repaired_scalars": self.repaired_scalars,
            "cm1_index_ranges": {
                "nqv": self.nqv, "nql1": self.nql1, "nql2": self.nql2,
                "nqs1": self.nqs1, "nqs2": self.nqs2,
                "nnc1": self.nnc1, "nnc2": self.nnc2,
            },
            "iice": self.iice,
            "idm": self.idm,
            "authority": dict(MOISTURE_SPECIES_AUTHORITY),
        }


def _mass(name: str, cloudvar: bool = False) -> MoistureScalar:
    # Both admitted tables give every mass scalar qmag=1.0e-2
    # (src/param.F:3464-3469 and :3673-3678).
    return MoistureScalar(name, "kg/kg", MASS_MIXING_RATIO, 1.0e-2, cloudvar)


def _number(name: str, qmag: float = 1.0e6) -> MoistureScalar:
    # src/param.F:3470-3473, the inum=1 table.  Every #/kg row CM1 itself
    # numbers takes 1.0e6 (:3470-3473 for ptype=5, :3384-3385 for ptype=3);
    # the argument exists for the ArWen-only axes below, where CM1 has no row
    # to copy and the value is this port's, recorded as such.
    return MoistureScalar(name, "#/kg", NUMBER_CONCENTRATION, qmag, False)


#: ``src/param.F:3641-3678``.  CM1 never assigns ``nnc1``/``nnc2`` here, so
#: they keep the block default ``-2`` (``src/param.F:3111-3112``), and ``idm``
#: keeps ``0`` (``src/param.F:3103``).
WSM6_PROFILE = MoistureSpeciesProfile(
    component_id="wsm6-mp6",
    cm1_ptype=7,
    scalars=(
        _mass("qv"), _mass("qc", True), _mass("qr"),
        _mass("qi", True), _mass("qs"), _mass("qg"),
    ),
    nqv=1, nql1=2, nql2=3, nqs1=4, nqs2=6, nnc1=-2, nnc2=-2,
    iice=1, idm=0, axis_authority="src/param.F:3641-3678",
    # src/param.F:936-950 force-downgrades ptype=7 to eqtset=1.  GM1's
    # Cartesian lane nevertheless executes the eqtset=2 source order, which
    # is one of the divergences records/CM1-ARWEN-SEAM.md names.
    cm1_eqtset=1,
)

#: ``src/param.F:3414-3473``, the ``inum=1`` table that ``src/morrison.F:325``
#: makes the only reachable one.  The four number concentrations are ordered
#: ice, snow, rain, graupel -- which is *not* the order ArWen's launcher takes
#: its moments in, so every consumer must map by name.
MORRISON_PROFILE = MoistureSpeciesProfile(
    component_id="morrison-mp10",
    cm1_ptype=5,
    scalars=(
        _mass("qv"), _mass("qc", True), _mass("qr"),
        _mass("qi", True), _mass("qs"), _mass("qg"),
        _number("nci"), _number("ncs"), _number("ncr"), _number("ncg"),
    ),
    nqv=1, nql1=2, nql2=3, nqs1=4, nqs2=6, nnc1=7, nnc2=10,
    iice=1, idm=1,
    cm1_eqtset=2,
    axis_authority="src/param.F:3414-3473",
)

#: ``src/param.F:3322-3398``, CM1's ``ptype=3``.  Eight scalars: the same six
#: mass rows as ``ptype=5``/``ptype=7`` followed by two number concentrations,
#: cloud-ice and rain, which is exactly the pair ArWen's classic Thompson
#: launcher requires (``gpuwm/core/microphysics.py`` ``_apply_thompson``:
#: ``required = ("qc","qi","ni","qs","qg","qr","nr")``).  ``src/param.F:936-950``
#: does not list ``ptype=3``, so CM1 keeps ``eqtset=2`` for it.
THOMPSON_PROFILE = MoistureSpeciesProfile(
    component_id="thompson-mp8",
    cm1_ptype=3,
    scalars=(
        _mass("qv"), _mass("qc", True), _mass("qr"),
        _mass("qi", True), _mass("qs"), _mass("qg"),
        _number("nci"), _number("ncr"),
    ),
    nqv=1, nql1=2, nql2=3, nqs1=4, nqs2=6, nnc1=7, nnc2=8,
    iice=1, idm=1,
    cm1_eqtset=2,
    axis_authority="src/param.F:3322-3398",
)

#: WDM6 (ArWen ``mp_physics=16``).  CM1 r21.1 has no WDM6 ``ptype`` -- the
#: whole ``src/param.F:3141-4509`` dispatch never names the scheme -- so this
#: axis is transcribed from the ArWen launcher's own required state instead
#: (``gpuwm/core/wdm6.py:151``: ``qi, qs, qg, nn, nc, nr`` beside ``qv``,
#: ``qc`` and ``qr``).  WDM6 is WSM6's six mass rows plus double-moment warm
#: rain: cloud and rain number, and a prognostic CCN number ``nn``.
#:
#: The mass prefix is CM1's own six, in CM1's own order, which is what lets
#: this axis travel through the sealed six-species post-physics boundary.  The
#: three ``#/kg`` rows take the magnitude CM1 gives every number row it does
#: number (1.0e6); that choice is GM1's, and the profile records it.
WDM6_PROFILE = MoistureSpeciesProfile(
    component_id="wdm6-mp16",
    cm1_ptype=None,
    scalars=(
        _mass("qv"), _mass("qc", True), _mass("qr"),
        _mass("qi", True), _mass("qs"), _mass("qg"),
        _number("ncn"), _number("ncc"), _number("ncr"),
    ),
    nqv=1, nql1=2, nql2=3, nqs1=4, nqs2=6, nnc1=7, nnc2=9,
    iice=1, idm=1,
    # GM1's Cartesian large step executes the eqtset=2 source order, and CM1
    # has no ptype here to downgrade it.
    cm1_eqtset=2,
    axis_authority="gpuwm/core/wdm6.py:151 (ArWen; CM1 r21.1 has no WDM6 ptype)",
)

#: Aerosol-aware Thompson (ArWen ``mp_physics=28``).  CM1 r21.1 has no
#: aerosol-aware Thompson ``ptype``; its ``ptype=3`` is the classic scheme.
#: The axis is the classic eight plus three: cloud-droplet number, and the
#: water- and ice-friendly aerosol numbers the scheme activates from
#: (``gpuwm/core/microphysics_aerosol.py``: ``state.nc``, ``state.nwfa``,
#: ``state.nifa``).  The two 2-D surface aerosol sources ``nwfa2d``/``nifa2d``
#: are not on the transported axis; they are component-owned carriers.
THOMPSON_AEROSOL_PROFILE = MoistureSpeciesProfile(
    component_id="thompson-aerosol-mp28",
    cm1_ptype=None,
    scalars=(
        _mass("qv"), _mass("qc", True), _mass("qr"),
        _mass("qi", True), _mass("qs"), _mass("qg"),
        _number("nci"), _number("ncr"), _number("ncc"),
        # Aerosol number concentrations run four orders of magnitude above a
        # hydrometeor count, so they take their own order-of-magnitude rather
        # than a hydrometeor row's: the WENO positive-definite epsilon
        # src/solve2.F:1472 forms as qmag*epsilon must scale with the field it
        # guards.  1.0e10 is the order of WRF's own naCCN0=3e10 m-3 reference
        # (module_mp_thompson.F), converted per kilogram.
        _number("nwfa", 1.0e10), _number("nifa", 1.0e10),
    ),
    nqv=1, nql1=2, nql2=3, nqs1=4, nqs2=6, nnc1=7, nnc2=11,
    iice=1, idm=1,
    cm1_eqtset=2,
    axis_authority=(
        "gpuwm/core/microphysics_aerosol.py (ArWen; CM1 r21.1 has no "
        "aerosol-aware Thompson ptype)"
    ),
)

#: CM1's six mass mixing ratios, in the order every admitted table numbers them
#: (``src/param.F:3425-3431`` for ``ptype=5``, ``:3645-3651`` for ``ptype=7``).
CM1_MASS_PREFIX_NAMES: tuple[str, ...] = ("qv", "qc", "qr", "qi", "qs", "qg")

#: Where each consumer of the mass prefix reads it, and why a number
#: concentration never enters that consumer.  Every line is CM1's own loop
#: bound, not GM1's inference.
CM1_SPECIES_CLASSIFICATION_AUTHORITY: Mapping[str, str] = MappingProxyType({
    # solve2.F:317-392 forms dum2 (all liquid) over n=nql1..nql2 and dum3 (all
    # solid) over n=nqs1..nqs2; buoyancy, theta-rho and the eqtset=2 cpm/cvm
    # terms (:394-412, :895-921) read dum1/dum2/dum3 only.
    "eos_buoyancy_loading": "src/solve2.F:317-412,895-921",
    # mp_driver.F:603-617 rebuilds prs/pp3d/th3d from rho, T and q3d(nqv).
    "post_physics_eos": "src/mp_driver.F:603-617",
    # mp_driver.F:505-520 calls pdefqtest on q3d(1..6); asq(7) is commented out.
    "positive_definite_repair": "src/mp_driver.F:505-520",
    # turb.F:5521-5535 accumulates qt over nqv, nql1..nql2, nqs1..nqs2.
    "calcnm_total_water": "src/turb.F:5521-5535",
    # init3d.F getset's moist density: rho = prs/((th0+th3d)*pi*(rd+max(0,qv)*rv)).
    "startup_density": "src/init3d.F:2393-2396",
    # solve2.F:1618-1645 apmasscon sums rho*ruh*rvh*rmh and pi0+pp3d; no q3d row.
    "dry_mass_ledger": "src/solve2.F:1618-1645",
})


def require_cm1_mass_prefix(profile: "MoistureSpeciesProfile") -> int:
    """Return 6 when rows 0..5 are CM1's six mass rows; refuse otherwise.

    The eqtset=2 stage terms, the startup density and the post-physics EOS are
    compiled over exactly CM1's six mass rows at fixed offsets.  A component
    whose leading six rows are anything else -- a number concentration in the
    mass window, or a different ordering -- would add a kg-1 count into the
    liquid or solid water loading, which is a buoyancy error of order 10^6.
    """

    names = profile.names[: len(CM1_MASS_PREFIX_NAMES)]
    prefix = tuple(range(len(CM1_MASS_PREFIX_NAMES)))
    if (
        names != CM1_MASS_PREFIX_NAMES
        or profile.mass_indices != prefix
        or (profile.nqv, profile.nql1, profile.nql2, profile.nqs1, profile.nqs2)
        != (1, 2, 3, 4, 6)
    ):
        refuse(
            code="GM1-MOISTURE-MASS-PREFIX",
            gate="moisture-species",
            reason=(
                f"component {profile.component_id!r} does not carry CM1's six "
                "mass mixing ratios qv,qc,qr,qi,qs,qg as its leading rows and "
                "only rows; the eqtset=2 buoyancy/theta-rho terms "
                "(src/solve2.F:317-412) and the post-physics EOS "
                "(src/mp_driver.F:603-617) read those offsets, so a number "
                "concentration or a reordered row would be summed into the "
                "water loading"
            ),
            remedy="select one of " + ", ".join(map(repr, MOISTURE_COMPONENT_IDS)),
            option="physics.component",
            value=profile.component_id,
        )
    return len(CM1_MASS_PREFIX_NAMES)


#: The axes a lane can select.  One transcribed axis above is deliberately
#: absent, and :mod:`cm1core.column_component_blockers` says why: ArWen's WDM6
#: refuses without WRF's XLAND land/sea mask, which this idealized lane has no
#: field for.  That blocker was measured by dispatching the column, not
#: guessed.  Transcribing an axis without binding it is the point: the next
#: reader inherits the work rather than the question.
_PROFILES: Mapping[str, MoistureSpeciesProfile] = MappingProxyType({
    WSM6_PROFILE.component_id: WSM6_PROFILE,
    MORRISON_PROFILE.component_id: MORRISON_PROFILE,
    THOMPSON_PROFILE.component_id: THOMPSON_PROFILE,
    THOMPSON_AEROSOL_PROFILE.component_id: THOMPSON_AEROSOL_PROFILE,
})

#: Axes this port has transcribed for schemes it cannot yet select.
TRANSCRIBED_UNBOUND_PROFILES: Mapping[str, MoistureSpeciesProfile] = MappingProxyType({
    WDM6_PROFILE.component_id: WDM6_PROFILE,
})

#: Every component id the Cartesian storm runtime binds, in a stable order.
MOISTURE_COMPONENT_IDS: tuple[str, ...] = tuple(_PROFILES)

#: The component the Cartesian lane selects when a configuration says nothing.
DEFAULT_COMPONENT_ID = WSM6_PROFILE.component_id


def moisture_species_profile(component_id: object) -> MoistureSpeciesProfile:
    """Return the CM1 scalar inventory for one component, refusing by value.

    The refusal names the concrete wrong result because a component whose
    scalar axis GM1 has not transcribed would be integrated with the wrong
    ``numq``: the dycore would transport the wrong number of scalars and the
    microphysics would read whichever rows happened to line up.
    """

    if not isinstance(component_id, str):
        refuse(
            code="GM1-MOISTURE-COMPONENT-TYPE",
            gate="moisture-species",
            reason=(
                "a column component is selected by its exact semantic id; a "
                "non-string selector cannot name a transcribed CM1 scalar table"
            ),
            remedy="select one of " + ", ".join(map(repr, MOISTURE_COMPONENT_IDS)),
            option="physics.component",
            value=component_id,
        )
    profile = _PROFILES.get(component_id)
    if profile is None:
        from .column_component_blockers import COLUMN_COMPONENT_BLOCKERS, refuse_blocked_component

        if component_id in COLUMN_COMPONENT_BLOCKERS:
            # A scheme ArWen compiles and GM1 has measured, held back by one
            # named piece of GM1 work.  Saying "unsupported" here would be a
            # lie about where the gap is.
            refuse_blocked_component(component_id, option="physics.component")
        refuse(
            code="GM1-MOISTURE-COMPONENT-UNSUPPORTED",
            gate="moisture-species",
            reason=(
                f"component {component_id!r} has no transcribed CM1 q3d scalar "
                "table, so GM1 cannot know how many scalars to transport, which "
                "rows are kg kg-1 and which are kg-1, or which rows the "
                "pre-microphysics positive-definite repair may touch; binding it "
                "would advect an arbitrary number of rows and hand the scheme "
                "whichever ones happened to align"
            ),
            remedy="select one of " + ", ".join(map(repr, MOISTURE_COMPONENT_IDS)),
            option="physics.component",
            value=component_id,
        )
    return profile


def moisture_scalar_count(component_id: object) -> int:
    """CM1's ``numq`` for one component."""

    return moisture_species_profile(component_id).numq


def require_equation_set(component_id: object, equation_set: object) -> int:
    """Admit only the equation set GM1's large step executes for a component.

    ``src/param.F:936-950`` is CM1's own cross-check and is recorded on each
    profile.  GM1's Cartesian large step is built on the ``eqtset=2`` source
    order for both components, so the gate here is GM1's, not CM1's: selecting
    anything else would run the final-RK removal/recapture schedule that the
    bound step composer does not implement.
    """

    profile = moisture_species_profile(component_id)
    if equation_set != 2 or isinstance(equation_set, bool):
        refuse(
            code="GM1-MOISTURE-EQUATION-SET",
            gate="moisture-species",
            reason=(
                "the Cartesian large step executes CM1's eqtset=2 source order "
                "only; another equation set would skip the final-RK held "
                "removal and recapture the bound step composer performs, so "
                "pressure and theta would be rebuilt by a schedule nothing here "
                "implements"
            ),
            remedy="set physics.equation_set = 2",
            option="physics.equation_set",
            value=equation_set,
        )
    return profile.cm1_ptype


__all__ = [
    "CM1_MASS_PREFIX_NAMES",
    "CM1_SPECIES_CLASSIFICATION_AUTHORITY",
    "DEFAULT_COMPONENT_ID",
    "MASS_MIXING_RATIO",
    "MOISTURE_COMPONENT_IDS",
    "MOISTURE_SPECIES_AUTHORITY",
    "MOISTURE_SPECIES_SCHEMA",
    "MORRISON_PROFILE",
    "THOMPSON_AEROSOL_PROFILE",
    "THOMPSON_PROFILE",
    "TRANSCRIBED_UNBOUND_PROFILES",
    "WDM6_PROFILE",
    "MoistureScalar",
    "MoistureSpeciesProfile",
    "NUMBER_CONCENTRATION",
    "WSM6_PROFILE",
    "moisture_scalar_count",
    "moisture_species_profile",
    "require_cm1_mass_prefix",
    "require_equation_set",
]
