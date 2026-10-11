"""CM1 r21.1's ``ihail`` switch: what the dense-ice category actually is.

``ihail`` is one integer in CM1's ``&param2`` namelist and it is not a
preference.  It replaces the constants that set how fast the dense
precipitating ice category falls, and the fall speed of that category is what
decides how far precipitation is carried from an updraft before it reaches
the ground -- which places the cold pool, which sets the low-level
mesocyclone.  This module is the CM1-owned transcription of both branches for
both column components GM1 can bind, so a run records the constants it
executed rather than a flag.

Importing this module imports neither CuPy, :mod:`gpuwm` nor NumPy.

Authority (read-only tree ``work/native-reference/cm1r21.1``):

* ``src/input.F:95,272`` -- ``ihail`` is a ``&param2`` namelist variable.
* ``src/param.F:1567-1580`` -- CM1 stops unless ``0 <= ihail <= 1``.  CM1
  supplies no default: the namelist must say which category this is.
* ``src/param.F:3407`` -- ``ptype=5`` passes it straight through as the
  first argument of ``graupel_init``.
* ``src/param.F:3697-3706`` -- ``ptype=7`` passes it straight through as
  ``call mp_wsm6_init(..., hail_opt=ihail, ...)``.  ``README.namelist:505-506``
  names only ``ptype=2,5``; the r21.1 source is the authority and it reaches
  WSM6 too.  GM1 follows the source.
* ``src/mp_wsm6.F:86-100`` -- the WSM6 branch: ``n0g``, ``deng``, ``avtg``,
  ``bvtg``, ``lamdagmax``.
* ``src/mp_wsm6.F:152,155,939,1551`` -- the WSM6 fall speed those constants
  feed: ``pidn0g`` is the product of pi, ``deng`` and ``n0g``; the slope
  ``lamdag`` is the fourth root of ``pidn0g`` over ``rho*q``; ``pvtg`` is
  ``avtg`` times gamma(4 + ``bvtg``) over 6; and the fall speed ``vt2g`` is
  ``pvtg`` times ``rslope`` to the power ``bvtg`` times ``denfac``.
* ``src/morrison.F:169-172,384,397-412,441-476`` -- the Morrison branch:
  ``AG``/``BG`` in ``V = A*D**B`` and the bulk density ``RHOG`` that also sets
  ``CG = RHOG*PI/6``.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from .errors import refuse


DENSE_ICE_SCHEMA = "gm1.dense-ice-selection/v1"

#: ``src/param.F:1567-1580``.
IHAIL_VALUES: tuple[int, ...] = (0, 1)
IHAIL_NAMES: Mapping[int, str] = MappingProxyType({0: "graupel", 1: "hail"})

#: CM1's own answer.  ``src/param.F:1567-1580`` supplies no compiled default --
#: it stops the model on anything outside 0..1 -- so the authority is what CM1
#: r21.1 actually ships: ``run/namelist.input:56`` and
#: ``run/config_files/supercell/namelist.input:55`` both say ``ihail = 1``, and
#: the supercell deck is the case this lane exists to reproduce.
#:
#: GM1 executed ``0`` from the day it bound WSM6 until this value moved, so
#: every receipt recorded before the move names a graupel-like dense ice
#: category that CM1's own decks do not select.  The move is deliberate and
#: changes results: it is the WSM6 entry of :data:`DEFAULT_IHAIL_BY_COMPONENT`
#: and the signature default of the WSM6-only helpers; the door resolves by
#: component, and a document that wants the old words declares
#: ``physics.ihail = 0`` explicitly.
DEFAULT_IHAIL = 1

#: The dense-ice branch an undeclared ``physics.ihail`` binds, per component.
#:
#: Every component that reads a dense-ice selector binds CM1's own answer,
#: ``ihail = 1``.  CM1 r21.1 ships that value in the distributed namelist
#: (``run/namelist.input:55-56``) and in the supercell deck this lane exists to
#: reproduce (``run/config_files/supercell/namelist.input:54-55``), and
#: ``src/param.F:3407`` and ``:3697-3706`` pass it unchanged into Morrison and
#: WSM6 respectively.  WRF's own ``morr_rimed_ice`` default is 1 for Morrison.
#:
#: ``wsm6-mp6`` previously bound 0 -- the branch this lane had executed since it
#: first bound WSM6, and ArWen's own ``wsm6_hail_opt`` library default.  Keeping
#: it would have preserved already-measured words at the price of running every
#: storm's dense ice as graupel where CM1's own deck runs hail: 100x the
#: number-concentration intercept, 500 rather than 700 kg m-3, and a
#: correspondingly slower fall speed, so precipitation reaches the ground
#: further downshear than CM1 puts it and the cold pool that drives the
#: low-level mesocyclone forms in the wrong place.  Better science beats a
#: preserved number; the move is recorded here and in every receipt.
#:
#: ``wdm6-mp16`` has no CM1 ptype at all, so there is no CM1 answer to move to;
#: it binds WRF's own ``hail_opt`` default of 0 and says so.
DEFAULT_IHAIL_BY_COMPONENT: Mapping[str, int] = MappingProxyType({
    "wsm6-mp6": DEFAULT_IHAIL,
    "morrison-mp10": 1,
    # No CM1 ptype, so no CM1 answer to move to: WRF's Registry.EM_COMMON
    # default for the shared hail_opt key is 0, and ArWen's RunConfig default
    # for wdm6_hail_opt is 0.  A document that wants hail declares it.
    "wdm6-mp16": 0,
})

DENSE_ICE_AUTHORITY: Mapping[str, str] = MappingProxyType({
    "namelist": "src/input.F:95,272",
    "range_check": "src/param.F:1567-1580",
    "ptype5_init": "src/param.F:3407",
    "ptype7_init": "src/param.F:3697-3706",
    "wsm6_constants": "src/mp_wsm6.F:86-100",
    "wsm6_fall_speed": "src/mp_wsm6.F:152,155,939,1551",
    "morrison_constants": "src/morrison.F:397-412,441-476",
    "morrison_fall_speed_form": "src/morrison.F:397",
    "readme_understates_scope": "README.namelist:505-506",
})


@dataclass(frozen=True, slots=True)
class Wsm6DenseIce:
    """``src/mp_wsm6.F:88-99``: the five constants ``hail_opt`` replaces."""

    n0g: float
    deng: float
    avtg: float
    bvtg: float
    lamdagmax: float

    @property
    def pidn0g(self) -> float:
        """``src/mp_wsm6.F:155``: ``pidn0g = pi*deng*n0g``."""

        # pi is spelled out rather than imported so this module stays free of
        # every runtime import; mp_wsm6.F:101 forms it as 4*atan(1).
        return 3.141592653589793 * self.deng * self.n0g

    def mass_weighted_fall_speed_m_s(self, rho_q_kg_m3: float) -> float:
        """``src/mp_wsm6.F:939`` at unit ``denfac``, for one ``rho*qg``.

        ``vt2g = pvtg*rslope**bvtg``, ``rslope = 1/lamdag`` and
        ``lamdag = (pidn0g/(rho*q))**0.25`` (``:1551``), so the whole
        dependence on the selected constants is
        ``avtg * (rho*q/pidn0g)**(bvtg/4)`` times the ``gamma(4+bvtg)/6``
        factor of ``:152``, which is carried separately by
        :func:`wsm6_gamma_4pb_over_6` because both branches share ``bvtg``.
        """

        if not rho_q_kg_m3 > 0.0:
            raise ValueError("a fall speed needs a positive rho*qg")
        return self.avtg * (rho_q_kg_m3 / self.pidn0g) ** (self.bvtg / 4.0)

    def record(self) -> dict[str, object]:
        return {
            "n0g_m4": self.n0g, "deng_kg_m3": self.deng,
            "avtg": self.avtg, "bvtg": self.bvtg,
            "lamdagmax_m_inv": self.lamdagmax, "pidn0g": self.pidn0g,
        }


@dataclass(frozen=True, slots=True)
class MorrisonDenseIce:
    """``src/morrison.F:407-412,441-444``: ``V=A*D**B`` plus the bulk density."""

    ag: float
    bg: float
    rhog: float

    @property
    def cg(self) -> float:
        """``src/morrison.F:476``: ``CG = RHOG*PI/6``."""

        return self.rhog * 3.141592653589793 / 6.0

    def fall_speed_m_s(self, diameter_m: float) -> float:
        """``src/morrison.F:397``: ``V = A*D**B`` for one particle diameter."""

        if not diameter_m > 0.0:
            raise ValueError("a fall speed needs a positive particle diameter")
        return self.ag * diameter_m ** self.bg

    def record(self) -> dict[str, object]:
        return {"ag": self.ag, "bg": self.bg, "rhog_kg_m3": self.rhog, "cg": self.cg}


#: ``src/mp_wsm6.F:94-99`` (the ``else`` branch) and ``:88-93`` (``hail_opt=1``).
WSM6_DENSE_ICE: Mapping[int, Wsm6DenseIce] = MappingProxyType({
    0: Wsm6DenseIce(n0g=4.0e6, deng=500.0, avtg=330.0, bvtg=0.8, lamdagmax=6.0e4),
    1: Wsm6DenseIce(n0g=4.0e4, deng=700.0, avtg=285.0, bvtg=0.8, lamdagmax=2.0e4),
})

#: ``src/morrison.F:407-412`` and ``:441-444``.
MORRISON_DENSE_ICE: Mapping[int, MorrisonDenseIce] = MappingProxyType({
    0: MorrisonDenseIce(ag=19.3, bg=0.37, rhog=400.0),
    1: MorrisonDenseIce(ag=114.5, bg=0.5, rhog=900.0),
})

#: The ArWen ``RunConfig`` field each component takes the same integer in.
#: ArWen spells WSM6's after WRF's ``hail_opt``, Morrison's after WRF Registry
#: ``morr_rimed_ice``, and WDM6's after WRF's shared ``hail_opt`` namelist key;
#: all three carry CM1's ``ihail`` unchanged.
ARWEN_IHAIL_OPTION: Mapping[str, str] = MappingProxyType({
    "wsm6-mp6": "wsm6_hail_opt",
    "morrison-mp10": "morr_rimed_ice",
    "wdm6-mp16": "wdm6_hail_opt",
})

#: Components that have a binding here but no dense-ice branch to select, and
#: why.  This is not "unsupported": the scheme runs, it simply has no
#: ``ihail``-shaped decision, so a declared ``physics.ihail`` would be written
#: into the run's configuration identity and its receipts while changing
#: nothing the scheme computed.  Naming that is the whole point of the table.
DENSE_ICE_SELECTOR_ABSENT: Mapping[str, str] = MappingProxyType({
    # CM1's imoist=0.  No column component is bound at all (the dry iconly
    # document), so there is no dense precipitating ice category to select:
    # src/param.F:3407 and :3697-3706 pass ihail only into a ptype's init.
    "off": (
        "physics.component = \"off\" is CM1's imoist=0, which binds no "
        "microphysics scheme, so no scheme reads ihail and no dense "
        "precipitating ice exists"
    ),
    "thompson-mp8": (
        "Thompson diagnoses its graupel size distribution from the rime mass "
        "and the local graupel content rather than from a pair of compiled "
        "hail/graupel constants, so WRF exposes no hail_opt for mp_physics=8 "
        "and ArWen's RunConfig carries none"
    ),
    "thompson-aerosol-mp28": (
        "aerosol-aware Thompson shares mp_physics=8's ice and graupel "
        "treatment, so it has the same diagnosed graupel distribution and the "
        "same absent hail_opt"
    ),
})

#: Diameters the Morrison record reports ``V=A*D**B`` at.  5 mm is small hail
#: and 1 cm is severe-criterion hail; both are inside the range the dense-ice
#: category actually occupies in a supercell.
_MORRISON_REPORT_DIAMETERS_M = (0.005, 0.01)
#: ``rho*qg`` the WSM6 record reports its mass-weighted fall speed at.  With
#: ``rho`` near 0.7 kg m-3 at the mid-level graupel maximum, 2e-3 kg m-3 is a
#: 3 g/kg graupel core.
_WSM6_REPORT_RHO_Q_KG_M3 = 2.0e-3


@dataclass(frozen=True, slots=True)
class DenseIceAbsent:
    """A bound component that has no dense-ice branch to select.

    It answers the same questions :class:`DenseIceSelection` does so that a
    receipt, a metadata document and a doctor row never have to branch on the
    component to describe what dense ice this run is made of.
    """

    component_id: str
    reason: str

    ihail: None = None
    arwen_option: None = None
    constants: None = None
    alternative: None = None

    @property
    def category(self) -> str:
        return "no-selectable-dense-ice-branch"

    def physical_effect(self) -> str:
        return (
            f"{self.component_id} has no ihail-shaped decision: {self.reason}; "
            "the dense precipitating ice category this run integrates is "
            "whatever the scheme itself diagnoses, and no configuration key "
            "changes it"
        )

    def record(self) -> dict[str, object]:
        return {
            "schema": DENSE_ICE_SCHEMA,
            "component_id": self.component_id,
            "ihail": None,
            "category": self.category,
            "arwen_run_config_option": None,
            "selector_absent_reason": self.reason,
            "physical_effect": self.physical_effect(),
            "authority": dict(DENSE_ICE_AUTHORITY),
        }


@dataclass(frozen=True, slots=True)
class DenseIceSelection:
    """One component's executed dense-ice branch, with its consequence."""

    component_id: str
    ihail: int
    arwen_option: str
    constants: Wsm6DenseIce | MorrisonDenseIce
    #: The other branch, so a receipt states what was not selected.
    alternative: Wsm6DenseIce | MorrisonDenseIce

    @property
    def category(self) -> str:
        return IHAIL_NAMES[self.ihail]

    def fall_speed_ratio_to_alternative(self) -> float:
        """How much faster the selected category falls than the other one.

        For WSM6 both branches share ``bvtg`` (``src/mp_wsm6.F:92,97``), so the
        ``gamma(4+bvtg)/6`` factor of ``:152`` cancels and the ratio is exact
        for every ``rho*qg``.  For Morrison the exponents differ
        (``src/morrison.F:409,412``), so the ratio is evaluated at the larger
        reported diameter and is a point on ``V=A*D**B``, not a constant.
        """

        if isinstance(self.constants, Wsm6DenseIce):
            assert isinstance(self.alternative, Wsm6DenseIce)
            rho_q = _WSM6_REPORT_RHO_Q_KG_M3
            return (
                self.constants.mass_weighted_fall_speed_m_s(rho_q)
                / self.alternative.mass_weighted_fall_speed_m_s(rho_q)
            )
        assert isinstance(self.alternative, MorrisonDenseIce)
        diameter = _MORRISON_REPORT_DIAMETERS_M[-1]
        return (
            self.constants.fall_speed_m_s(diameter)
            / self.alternative.fall_speed_m_s(diameter)
        )

    def physical_effect(self) -> str:
        """One sentence a reader can check against the recorded numbers."""

        ratio = self.fall_speed_ratio_to_alternative()
        other = IHAIL_NAMES[1 - self.ihail]
        direction = "faster" if ratio >= 1.0 else "slower"
        factor = ratio if ratio >= 1.0 else 1.0 / ratio
        return (
            f"{self.component_id} runs its dense precipitating ice as "
            f"{self.category} (ihail={self.ihail}); the selected constants make "
            f"it fall about {factor:.2f}x {direction} than the {other} branch, "
            "so precipitation is carried a correspondingly different distance "
            "downshear of the updraft before it reaches the ground, which is "
            "where the cold pool forms and what the low-level mesocyclone then "
            "develops over"
        )

    def record(self) -> dict[str, object]:
        result: dict[str, object] = {
            "schema": DENSE_ICE_SCHEMA,
            "component_id": self.component_id,
            "ihail": self.ihail,
            "category": self.category,
            "arwen_run_config_option": self.arwen_option,
            "selected_constants": self.constants.record(),
            "alternative_constants": self.alternative.record(),
            "fall_speed_ratio_to_alternative": self.fall_speed_ratio_to_alternative(),
            "physical_effect": self.physical_effect(),
            "authority": dict(DENSE_ICE_AUTHORITY),
        }
        if isinstance(self.constants, Wsm6DenseIce):
            assert isinstance(self.alternative, Wsm6DenseIce)
            rho_q = _WSM6_REPORT_RHO_Q_KG_M3
            result["mass_weighted_fall_speed_m_s"] = {
                "rho_qg_kg_m3": rho_q,
                "selected": self.constants.mass_weighted_fall_speed_m_s(rho_q),
                "alternative": self.alternative.mass_weighted_fall_speed_m_s(rho_q),
                "excludes": "gamma(4+bvtg)/6 and denfac, shared by both branches",
                "authority": DENSE_ICE_AUTHORITY["wsm6_fall_speed"],
            }
        else:
            assert isinstance(self.alternative, MorrisonDenseIce)
            result["fall_speed_m_s"] = {
                "diameters_m": list(_MORRISON_REPORT_DIAMETERS_M),
                "selected": [
                    self.constants.fall_speed_m_s(d)
                    for d in _MORRISON_REPORT_DIAMETERS_M
                ],
                "alternative": [
                    self.alternative.fall_speed_m_s(d)
                    for d in _MORRISON_REPORT_DIAMETERS_M
                ],
                "authority": DENSE_ICE_AUTHORITY["morrison_fall_speed_form"],
            }
        return result


def require_ihail(value: object, *, option: str = "physics.ihail") -> int:
    """Admit CM1's ``ihail`` by value, refusing anything else by name."""

    if isinstance(value, bool) or not isinstance(value, int) or value not in IHAIL_VALUES:
        refuse(
            code="GM1-DENSE-ICE-IHAIL",
            gate="dense-ice",
            reason=(
                "CM1 param.F:1567-1580 stops the model unless ihail is 0 or 1, "
                "because the value selects which set of dense-ice constants the "
                "scheme is initialized with; anything else has no branch, so the "
                "column would be initialized with whichever constants the "
                "else-branch happens to hold and every graupel/hail fall speed, "
                "and therefore where precipitation lands relative to the "
                "updraft, would be the unselected category's"
            ),
            remedy=(
                "set ihail = 0 for a graupel-like dense ice category (many small "
                "slow particles) or ihail = 1 for a hail-like one (fewer large "
                "fast particles)"
            ),
            option=option,
            value=value,
        )
    return int(value)


def default_ihail(component_id: object) -> int | None:
    """The ``ihail`` an undeclared ``physics.ihail`` binds for one component.

    Refused by value for an unknown component because the fallback would
    otherwise be whichever branch this module happens to list first, and the
    run would integrate the unselected dense-ice category's fall speeds while
    its receipt named the component the document asked for.
    """

    if isinstance(component_id, str) and component_id in DENSE_ICE_SELECTOR_ABSENT:
        # A bound component with nothing to select: the undeclared case is the
        # only case, and it binds no integer at all.
        return None
    value = (
        DEFAULT_IHAIL_BY_COMPONENT.get(component_id)
        if isinstance(component_id, str)
        else None
    )
    if value is None:
        refuse(
            code="GM1-DENSE-ICE-DEFAULT-COMPONENT",
            gate="dense-ice",
            reason=(
                f"component {component_id!r} has no recorded dense-ice default, "
                "so an undeclared physics.ihail has no value to bind; the column "
                "would be initialized from another component's continuity "
                "decision and every graupel/hail fall speed, and therefore where "
                "precipitation lands relative to the updraft, would be that "
                "other scheme's"
            ),
            remedy=(
                "declare physics.ihail explicitly, or select one of "
                + ", ".join(map(repr, sorted(DEFAULT_IHAIL_BY_COMPONENT)))
            ),
            option="physics.component",
            value=component_id,
        )
    return value


def dense_ice_selection(
    component_id: object, ihail: object
) -> DenseIceSelection | DenseIceAbsent:
    """Bind one component's transcribed dense-ice branch, refusing by value.

    ``ihail=None`` is the undeclared case, not a bad value: it binds the
    component's own recorded default (:func:`default_ihail`), which for a
    component with no dense-ice branch is ``None`` again.  Every other value
    is admitted or refused by :func:`require_ihail`.
    """

    if isinstance(component_id, str) and component_id in DENSE_ICE_SELECTOR_ABSENT:
        reason = DENSE_ICE_SELECTOR_ABSENT[component_id]
        if ihail is not None:
            refuse(
                code="GM1-DENSE-ICE-SELECTOR-ABSENT",
                gate="dense-ice",
                reason=(
                    f"component {component_id!r} has no dense-ice selector: "
                    f"{reason}.  A declared ihail would be sealed into this "
                    "run's configuration identity and every receipt it "
                    "publishes while the scheme continued to diagnose its own "
                    "dense ice, so the recorded configuration would describe a "
                    "choice the run never made"
                ),
                remedy=(
                    "remove physics.ihail, or select a component that reads it: "
                    + ", ".join(map(repr, sorted(ARWEN_IHAIL_OPTION)))
                ),
                option="physics.ihail",
                value=ihail,
            )
        return DenseIceAbsent(component_id=component_id, reason=reason)
    if ihail is None:
        ihail = default_ihail(component_id)
    option = ARWEN_IHAIL_OPTION.get(component_id) if isinstance(component_id, str) else None
    if option is None:
        refuse(
            code="GM1-DENSE-ICE-COMPONENT",
            gate="dense-ice",
            reason=(
                f"component {component_id!r} has no transcribed ihail branch, so "
                "GM1 cannot know which ArWen option carries CM1's integer; "
                "binding it would leave the scheme on its own library default "
                "while the run recorded the declared value"
            ),
            remedy="select one of " + ", ".join(map(repr, sorted(ARWEN_IHAIL_OPTION))),
            option="physics.component",
            value=component_id,
        )
    selected = require_ihail(ihail)
    # WSM6 and WDM6 share WRF's hail_opt constants: gpuwm/core/wdm6.py takes
    # its rimed-ice pair from the same five-constant block WSM6 does, and
    # gpuwm/core/kernels/wdm6.cu carries the same n0g/deng/avtg/bvtg/lamdagmax.
    table = (
        MORRISON_DENSE_ICE if component_id == "morrison-mp10" else WSM6_DENSE_ICE
    )
    return DenseIceSelection(
        component_id=component_id,
        ihail=selected,
        arwen_option=option,
        constants=table[selected],
        alternative=table[1 - selected],
    )


def arwen_ihail_kwargs(component_id: object, ihail: object) -> dict[str, int]:
    """The ``RunConfig`` keyword this component takes ``ihail`` in, if any.

    A component with no dense-ice selector contributes no keyword at all, so
    the scheme runs on its own diagnosed dense ice and nothing here quietly
    hands it another scheme's flag.
    """

    selection = dense_ice_selection(component_id, ihail)
    if selection.arwen_option is None:
        return {}
    return {selection.arwen_option: selection.ihail}


__all__ = [
    "ARWEN_IHAIL_OPTION",
    "DENSE_ICE_SELECTOR_ABSENT",
    "DenseIceAbsent",
    "DEFAULT_IHAIL",
    "DEFAULT_IHAIL_BY_COMPONENT",
    "DENSE_ICE_AUTHORITY",
    "DENSE_ICE_SCHEMA",
    "DenseIceSelection",
    "IHAIL_NAMES",
    "IHAIL_VALUES",
    "MORRISON_DENSE_ICE",
    "MorrisonDenseIce",
    "WSM6_DENSE_ICE",
    "Wsm6DenseIce",
    "arwen_ihail_kwargs",
    "default_ihail",
    "dense_ice_selection",
    "require_ihail",
]
