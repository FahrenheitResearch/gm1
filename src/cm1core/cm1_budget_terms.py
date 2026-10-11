"""CM1 r21.1's per-process budget term lists, resolved for one Cartesian run.

``README.namelist`` ``&param9`` offers ``output_ubudget``, ``output_vbudget``,
``output_wbudget``, ``output_thbudget`` and ``output_qvbudget``.  Which named
terms each one publishes is decided in ``src/param.F:5302-5875`` from the rest
of the namelist (``ntdiag``/``nqdiag``/``nudiag``/``nvdiag``/``nwdiag`` and
their ``td_*``/``qd_*``/``ud_*``/``vd_*``/``wd_*`` slots), and the names
written to the history file are ``src/writeout.F:5144-8457``'s
``name_output`` strings (``ptb_hadv``, ``qvb_hadv``, ``ub_hadv`` ...).  This
module is that decision, transcribed, for the facts a GM1 Cartesian run can
have, and the refusals for the facts it can have but whose terms this lane
does not capture.

CM1 raises ``dotbud``/``doqbud``/``doubud``/``dovbud``/``dowbud`` only on a
step that writes output (``src/cm1.F:3268-3296``), and every term is the
tendency of that one large step's final RK stage (``nrk == nrkmax``) or of its
``solve1``/``mp_driver`` call.  Nothing is accumulated across steps and
nothing is written to a restart file unless ``restart_file_diags`` is true
(``src/restart_write.F:2083-2092``, ``run/namelist.input:296`` ships
``.false.``), so a restarted run recomputes every term on its next output step.

The vorticity-budget inventory at the bottom is a GM1 extension and is named
so everywhere it appears.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

BUDGET_SWITCHES: tuple[str, ...] = (
    "output_ubudget", "output_vbudget", "output_wbudget",
    "output_thbudget", "output_qvbudget",
)

#: ``writeout.F`` name prefix and the history stagger of each budget.
BUDGET_PREFIX: Mapping[str, str] = {
    "output_ubudget": "ub", "output_vbudget": "vb", "output_wbudget": "wb",
    "output_thbudget": "ptb", "output_qvbudget": "qvb",
}
BUDGET_STAGGER: Mapping[str, str] = {
    "output_ubudget": "u", "output_vbudget": "v", "output_wbudget": "w",
    "output_thbudget": "mass", "output_qvbudget": "mass",
}
BUDGET_UNITS: Mapping[str, str] = {
    "output_ubudget": "m s-2", "output_vbudget": "m s-2", "output_wbudget": "m s-2",
    "output_thbudget": "K s-1", "output_qvbudget": "kg kg-1 s-1",
}
BUDGET_VARIABLE: Mapping[str, str] = {
    "output_ubudget": "u", "output_vbudget": "v", "output_wbudget": "w",
    "output_thbudget": "theta", "output_qvbudget": "qv",
}

#: Where each term is formed in CM1, and what it means.
TERM_AUTHORITY: Mapping[str, tuple[str, str]] = {
    "hadv": ("src/adv.F:1555-1601,2477-2521,3379-3425,665-702; non-diffusive part adv.F:1640-1706,790-850",
             "horizontal advection, sixth-order non-diffusive flux form"),
    "vadv": ("src/adv.F:1561-1601,3385-3425,665-702; non-diffusive part adv.F:1747-1812,852-900",
             "vertical advection, sixth-order non-diffusive flux form"),
    "hidiff": ("src/adv.F:1679-1727,2609-2658,3491-3539,790-850",
               "implicit horizontal diffusion of the odd-order scheme (total minus sixth-order)"),
    "vidiff": ("src/adv.F:1769-1812,2700-2743,3572-3597,852-900",
               "implicit vertical diffusion of the odd-order scheme (total minus sixth-order)"),
    "hturb": ("src/turbtend.F:1326-1347,2092-2114,2852-2871,395-420",
              "subgrid turbulence, horizontal flux divergence"),
    "vturb": ("src/turbtend.F:1326-1347,2092-2114,2852-2871,395-420",
              "subgrid turbulence, vertical flux divergence (implicit column solve)"),
    "pgrad": ("src/solve2.F:925-956,1196-1245",
              "acoustic-step pressure-gradient acceleration: (x3d-xa)/dt minus the explicit tendency"),
    "rdamp": ("src/solve1.F:489-529", "Rayleigh damping"),
    "cor": ("src/solve2.F:694-714", "Coriolis acceleration"),
    "buoy": ("src/solve2.F:896-920", "buoyancy (c1*b(k-1)+c2*b(k))"),
    "mp": ("src/mp_driver.F:151-170,1670-1690",
           "microphysics, including its positive-definite repair: (after-before)/dt"),
    "rad": ("src/solve1.F:1007-1048", "radiative relaxation (rterm)"),
    "div": ("src/solve2.F:1331-1340", "eqtset=2 moist divergence term -t11*thterm"),
    "diss": ("src/solve1.F:920-996", "dissipative heating"),
    "lspg": ("src/solve1.F:590-612",
             "GM1 EXTENSION: lspgrad=1 large-scale pressure gradient, -fcor x (V0 + Vmove) on u "
             "and +fcor x (U0 + Umove) on v; the 3-D budget has no slot for it (param.F:5573-5587)"),
}

#: Terms CM1 r21.1's 3-D ``udiag``/``vdiag`` budgets never carry.  Each is
#: published only when its selector is on, after CM1's own names, and labelled
#: as an extension: without ``lspg`` an lspgrad=1 momentum budget would not sum
#: to the tendency the model took, and every horizontal vorticity curl would
#: carry the base state's removed Coriolis force under no name.  The name is
#: CM1's own domain-average diagnostic's (domaindiag.F:7466-7480, :7924-7938),
#: which writes the ``ulspg``/``vlspg`` profiles -- and those solve1.F:592-593
#: leaves at zero for lspgrad=1, so CM1's own ``ub_lspg`` is not this term.
GM1_EXTENSION_TERMS: frozenset[str] = frozenset({"lspg"})

#: ``writeout.F`` ``name_output`` order within each budget.
_WRITEOUT_ORDER: Mapping[str, tuple[str, ...]] = {
    "output_ubudget": ("hadv", "vadv", "hidiff", "vidiff", "hediff", "vediff", "hturb",
                       "vturb", "pgrad", "rdamp", "cor", "cent", "pbl", "lsw", "lspg"),
    "output_vbudget": ("hadv", "vadv", "hidiff", "vidiff", "hediff", "vediff", "hturb",
                       "vturb", "pgrad", "rdamp", "cor", "cent", "pbl", "lsw", "lspg"),
    "output_wbudget": ("hadv", "vadv", "hidiff", "vidiff", "hediff", "vediff", "hturb",
                       "vturb", "pgrad", "rdamp", "buoy"),
    "output_thbudget": ("hadv", "vadv", "hidiff", "vidiff", "hediff", "vediff", "hturb",
                        "vturb", "mp", "rdamp", "nudge", "rad", "div", "diss", "pbl", "lsw"),
    "output_qvbudget": ("hadv", "vadv", "hidiff", "vidiff", "hediff", "vediff", "hturb",
                        "vturb", "mp", "nudge", "pbl", "lsw"),
}

_MORRISON_COMPONENTS = ("morrison",)


@dataclass(frozen=True)
class BudgetFacts:
    """The namelist facts ``param.F:5302-5875`` branches on, for one run."""

    cm1setup: int = 1
    ipbl: int = 0
    hadvordrs: int = 5
    vadvordrs: int = 5
    hadvordrv: int = 5
    vadvordrv: int = 5
    advwenos: int = 2
    advwenov: int = 0
    idiff: int = 0
    irdamp: int = 1
    hrdamp: int = 0
    icor: int = 0
    lspgrad: int = 0
    rterm: int = 0
    radopt: int = 0
    testcase: int = 0
    imoist: int = 1
    component: str = "wsm6-mp6"
    eqtset: int = 2
    idiss: int = 1
    efall: int = 0
    nudgeobc: int = 0
    stretch_z: int = 0
    nest: bool = False
    nest_feedback: int = 0

    @property
    def dohturb(self) -> bool:
        # param.F: cm1setup=1 (LES) turns on both halves of the 3-D closure.
        return self.cm1setup >= 1

    @property
    def dovturb(self) -> bool:
        return self.cm1setup == 1


def _odd(order: int, weno: int) -> bool:
    return order in (3, 5, 7, 9) or weno >= 1


def resolve_terms(switch: str, facts: BudgetFacts) -> tuple[str, ...]:
    """``param.F``'s slot list for one budget, in ``writeout.F`` order."""

    f = facts
    present: set[str] = {"hadv", "vadv"}
    if switch in ("output_ubudget", "output_vbudget", "output_wbudget"):
        # param.F:5590-5875
        if f.cm1setup >= 1 or f.ipbl >= 1:
            if f.dohturb:
                present.add("hturb")
            if f.dovturb:
                present.add("vturb")
        if _odd(f.hadvordrv, f.advwenov):
            present.add("hidiff")
        if _odd(f.vadvordrv, f.advwenov):
            present.add("vidiff")
        if f.idiff >= 1:
            present.add("hediff")
        present.add("pgrad")
        if f.irdamp >= 1 or f.hrdamp >= 1:
            present.add("rdamp")
        if switch == "output_wbudget":
            present.add("buoy")
        else:
            if f.icor == 1:
                present.add("cor")
            # solve1.F:590-612 adds to uten1/vten1 only.
            if f.lspgrad == 1:
                present.add("lspg")
            if f.ipbl >= 1:
                present.add("pbl")
    elif switch == "output_thbudget":
        # param.F:5302-5419
        if f.cm1setup >= 1:
            if f.dohturb:
                present.add("hturb")
            if f.dovturb:
                present.add("vturb")
        if _odd(f.hadvordrs, f.advwenos):
            present.add("hidiff")
        if _odd(f.vadvordrs, f.advwenos):
            present.add("vidiff")
        if f.idiff == 1:
            present.add("hediff")
        if f.imoist >= 1 and f.component != "off":
            present.add("mp")
        if f.irdamp >= 1:
            present.add("rdamp")
        if f.testcase == 10:
            present.add("nudge")
        if f.rterm == 1 or f.radopt >= 1:
            present.add("rad")
        if f.imoist == 1 and f.eqtset == 2:
            present.add("div")
        if f.idiss == 1 or f.ipbl in (3, 4, 5):
            present.add("diss")
        if f.ipbl >= 1:
            present.add("pbl")
    elif switch == "output_qvbudget":
        # param.F:5488-5570
        if f.cm1setup >= 1 or f.ipbl >= 1:
            if f.dohturb:
                present.add("hturb")
            if f.dovturb:
                present.add("vturb")
        if _odd(f.hadvordrs, f.advwenos):
            present.add("hidiff")
        if _odd(f.vadvordrs, f.advwenos):
            present.add("vidiff")
        if f.idiff == 1:
            present.add("hediff")
        if f.component != "off":
            present.add("mp")
        if f.testcase == 10:
            present.add("nudge")
        if f.ipbl >= 1:
            present.add("pbl")
    else:
        raise KeyError(switch)
    return tuple(name for name in _WRITEOUT_ORDER[switch] if name in present)


def budget_defects(switch: str, facts: BudgetFacts) -> list[tuple[str, str]]:
    """``(option, reason)`` for every fact this lane cannot honour for ``switch``.

    Each reason names the concrete wrong result: a published term that would
    not be the process it is named for, or a set of terms that would not sum
    to the tendency the model took.
    """

    f = facts
    out: list[tuple[str, str]] = []
    prefix = BUDGET_PREFIX[switch]
    if f.nest:
        out.append((
            "nest",
            f"{prefix}_* inside a live nest: the child adds the Davies relaxation "
            "to its held tendencies (nest_live.NestLateralForcing.relax_held) and "
            "re-imposes its specified zone inside every acoustic small step, and "
            "CM1 has no budget term for either, so the relaxation would be "
            f"published as {prefix}_hturb/{prefix}_rdamp and the specified-zone "
            f"overwrite as {prefix}_pgrad"
            + ("" if not f.nest_feedback else
               "; with nest.feedback=1 the parent's own u/v/w/theta/qv are "
               "overwritten by the child after the parent's step, so the "
               "parent's terms would not sum to the tendency of the state its "
               "frame publishes")
        ))
    if f.ipbl >= 1:
        out.append((
            "closures.ipbl",
            f"closures.ipbl={f.ipbl} selects a PBL scheme whose column tendency "
            f"CM1 publishes as {prefix}_pbl (solve1.F:1284-1345); this lane "
            "captures no PBL tendency, so it would be folded into the residual "
            "terms and every other term would still carry CM1's name"
        ))
    if f.cm1setup not in (0, 1):
        out.append((
            "closures.cm1setup",
            f"closures.cm1setup={f.cm1setup} splits the closure into horizontal "
            "and vertical parts CM1 assigns to different schemes; this lane's "
            "split replays only the cm1setup=1 3-D closure"
        ))
    if f.idiff >= 1:
        out.append((
            "dynamics.idiff",
            f"{prefix}_hediff/{prefix}_vediff are CM1's explicit artificial "
            "diffusion, which this lane does not run"
        ))
    if switch in ("output_thbudget",) and f.radopt >= 1:
        out.append((
            "radiation.radopt",
            f"radiation.radopt={f.radopt} makes ptb_rad the radiation scheme's "
            "tendency (solve1.F:1073-1080); this lane captures only the rterm=1 "
            "relaxation's increment, so the scheme's heating would be missing "
            "from ptb_rad and from the sum"
        ))
    if switch in ("output_thbudget", "output_qvbudget") and f.testcase == 10:
        out.append((
            "initialization.testcase",
            "testcase=10 adds the large-scale nudging CM1 publishes as "
            f"{prefix}_nudge (solve1.F:1405-1480); this lane captures no nudging "
            "increment"
        ))
    if switch in ("output_thbudget", "output_qvbudget") and any(
        token in f.component for token in _MORRISON_COMPONENTS
    ):
        out.append((
            "physics.component",
            f"physics.component={f.component!r} is CM1's ptype=5, for which "
            f"param.F:5402-5418,5556-5569 adds the process rates {prefix}_cond, "
            f"{prefix}_evac, {prefix}_evar, {prefix}_dep, {prefix}_subl "
            "(morrison.F:1106-1230); the ArWen Morrison component does not hand "
            "those rates across the seam, so the history would lack terms CM1 "
            "writes under this selection"
        ))
    if f.efall == 1 and switch == "output_thbudget":
        out.append((
            "physics.efall",
            "efall=1 adds ptb_efall (mp_driver.F:214-285), which this lane does not capture"
        ))
    return out


def facts_from_config(config: Any) -> BudgetFacts:
    """Read the facts off a bound ``CM1Config`` without importing ``config``."""

    closures = getattr(config, "closures", None)
    dynamics = getattr(config, "dynamics", None)
    radiation = getattr(config, "radiation", None)
    initialization = getattr(config, "initialization", None)
    nest = getattr(config, "nest", None)
    physics = getattr(config, "physics", None)
    rterm = getattr(closures, "rterm_value", None)
    if rterm is None:
        rterm = getattr(closures, "rterm", 0) or 0
    return BudgetFacts(
        cm1setup=int(getattr(closures, "cm1setup", 1)),
        ipbl=int(getattr(closures, "ipbl", 0)),
        advwenos=int(getattr(dynamics, "scalar_weno", 2)),
        advwenov=int(getattr(dynamics, "velocity_weno", 0)),
        idiff=int(getattr(dynamics, "idiff", 0)),
        irdamp=int(getattr(closures, "upper_damping", 1)),
        hrdamp=int(getattr(closures, "hrdamp", 0)),
        icor=int(getattr(dynamics, "coriolis", 0)),
        lspgrad=int(getattr(dynamics, "lspgrad_value", 0) or 0),
        rterm=int(rterm),
        radopt=int(getattr(radiation, "radopt", 0) or 0),
        testcase=int(getattr(initialization, "testcase", 0) or 0),
        component=str(getattr(physics, "component", "wsm6-mp6")),
        stretch_z=int(getattr(getattr(config, "grid", None), "stretch_z", 0) or 0),
        nest=nest is not None,
        nest_feedback=int(getattr(nest, "feedback", 0) or 0),
    )


def facts_from_runtime(config: Any, *, nest: bool = False) -> BudgetFacts:
    """The same facts from a ``gm1_runtime.RuntimeConfig``."""

    radiation = getattr(config, "radiation", None)
    return BudgetFacts(
        cm1setup=int(getattr(config, "cm1setup", 1)),
        ipbl=int(getattr(config, "ipbl", 0)),
        irdamp=int(getattr(config, "irdamp", 1)),
        hrdamp=int(getattr(config, "hrdamp", 0)),
        icor=int(getattr(config, "icor", 0)),
        lspgrad=int(getattr(config, "lspgrad", 0)),
        rterm=int(getattr(config, "rterm", 0)),
        radopt=int(getattr(radiation, "radopt", 0) or 0),
        component=str(getattr(config, "microphysics", "wsm6-mp6")),
        nudgeobc=int(getattr(config, "nudgeobc", 0)),
        stretch_z=int(getattr(config.grid, "stretch_z", 0)),
        nest=nest,
    )


def selected_budgets(selection: Any) -> tuple[str, ...]:
    if selection is None:
        return ()
    return tuple(key for key in BUDGET_SWITCHES if int(selection[key]) == 1)


def budget_carrier_rows(
    selection: Any, facts: BudgetFacts, *,
    mass: tuple[int, int, int], u: tuple[int, int, int],
    v: tuple[int, int, int], w: tuple[int, int, int], group: str,
) -> tuple[tuple[Any, ...], ...]:
    """History carrier rows, by CM1's ``writeout.F`` names, for a selection."""

    shapes = {"u": u, "v": v, "w": w, "mass": mass}
    rows: list[tuple[Any, ...]] = []
    for switch in selected_budgets(selection):
        prefix = BUDGET_PREFIX[switch]
        for term in resolve_terms(switch, facts):
            meaning = TERM_AUTHORITY.get(term, ("", term))[1]
            name = f"{prefix}_{term}"
            origin = "GM1 extension; not CM1's domaindiag profile of this name" if term in GM1_EXTENSION_TERMS else f"CM1 {name}"
            rows.append((
                name,
                f"{BUDGET_VARIABLE[switch]} budget: {meaning} ({origin})",
                BUDGET_UNITS[switch], BUDGET_STAGGER[switch],
                shapes[BUDGET_STAGGER[switch]], group, name, False,
            ))
    return tuple(rows)


# ---------------------------------------------------------------------------
# GM1 extension: the vorticity budget inventory.
# ---------------------------------------------------------------------------

#: Curl of each CM1 momentum budget term.  ``buoy`` exists only for x and y.
VORTICITY_CURL_TERMS: tuple[str, ...] = (
    "hadv", "vadv", "hidiff", "vidiff", "hturb", "vturb", "rdamp", "cor", "lspg", "pgrad", "buoy",
)
#: The velocity-gradient decomposition of the advective part, and its residual.
VORTICITY_DECOMPOSITION_TERMS: tuple[str, ...] = ("stretch", "tilt", "divg", "advect", "advres")
VORTICITY_AXES: tuple[str, ...] = ("x", "y", "z")

VORTICITY_TERM_MEANING: Mapping[str, str] = {
    "vort": "vorticity component (calcvort, misclibs.F:1618-2056) of the completed state",
    "tend": "sum of every curl term: the step's tendency of that component",
    "hadv": "curl of the horizontal advection budget term",
    "vadv": "curl of the vertical advection budget term",
    "hidiff": "curl of the implicit horizontal diffusion of the advection scheme",
    "vidiff": "curl of the implicit vertical diffusion of the advection scheme",
    "hturb": "curl of the horizontal subgrid-turbulence tendency (friction)",
    "vturb": "curl of the vertical subgrid-turbulence tendency (friction, incl. surface drag)",
    "rdamp": "curl of the Rayleigh damping tendency",
    "cor": "curl of the Coriolis acceleration",
    "lspg": "curl of the lspgrad=1 large-scale pressure gradient (cancels the base state's part of cor)",
    "pgrad": "curl of the acoustic pressure-gradient acceleration",
    "buoy": "curl of the buoyancy acceleration: baroclinic generation (d b/dy for x, -d b/dx for y)",
    "stretch": "stretching, w_i dV_i/dx_i, centred differences at scalar points",
    "tilt": "tilting, sum over j!=i of w_j dV_i/dx_j, centred differences at scalar points",
    "divg": "-w_i (div V), centred differences at scalar points",
    "advect": "-(V.grad) w_i, centred differences at scalar points",
    "advres": "curl of the model's own advection terms minus stretch+tilt+divg+advect, by value",
}


def vorticity_budget_names(facts: BudgetFacts) -> tuple[str, ...]:
    """Carrier names ``xvortb``, ``xvortb_hadv`` ... for this run's terms."""

    momentum = set(resolve_terms("output_ubudget", facts)) | set(
        resolve_terms("output_wbudget", facts)
    )
    names: list[str] = []
    for axis in VORTICITY_AXES:
        base = f"{axis}vortb"
        names.append(base)
        names.append(f"{base}_tend")
        for term in VORTICITY_CURL_TERMS:
            if term not in momentum:
                continue
            if term == "buoy" and axis == "z":
                continue
            names.append(f"{base}_{term}")
        names.extend(f"{base}_{term}" for term in VORTICITY_DECOMPOSITION_TERMS)
    return tuple(names)


def vorticity_carrier_rows(facts: BudgetFacts, *, mass: tuple[int, int, int],
                           group: str) -> tuple[tuple[Any, ...], ...]:
    rows: list[tuple[Any, ...]] = []
    for name in vorticity_budget_names(facts):
        term = name.split("_", 1)[1] if "_" in name else "vort"
        units = "s-1" if term == "vort" else "s-2"
        rows.append((
            name,
            f"GM1 EXTENSION vorticity budget ({name[0]}): {VORTICITY_TERM_MEANING[term]}",
            units, "mass", mass, group, name, False,
        ))
    return tuple(rows)


def refuse_inadmissible_budgets(config: Any, *, refuse: Any) -> None:
    """Door check: every selected budget, every defect, refused by name."""

    selection = getattr(config, "output", None)
    lagrangian = getattr(config, "lagrangian", None)
    wanted = list(selected_budgets(selection))
    vorticity = bool(lagrangian is not None and (
        bool(lagrangian.values.get("vorticity_budget_history", False))
        or bool(lagrangian.values.get("parcel_vorticity_budget", False))
    ))
    if not wanted and not vorticity:
        return
    facts = facts_from_config(config)
    for switch in wanted:
        for option, reason in budget_defects(switch, facts):
            refuse(
                code="CM1-OUTPUT-BUDGET",
                gate="CM1-param9",
                reason=f"output.{switch}=1: {reason}",
                remedy=f"set output.{switch} = 0, or change {option}",
                option=f"output.{switch}",
                value=1,
            )
    if vorticity:
        for option, reason in budget_defects("output_ubudget", facts):
            refuse(
                code="CM1-LAGRANGIAN-VORTICITY-BUDGET",
                gate="CM1-param9",
                reason=(
                    "the vorticity budget is the curl of the momentum budget and "
                    f"inherits its defect: {reason}"
                ),
                remedy=f"turn off lagrangian.vorticity_budget_history/parcel_vorticity_budget, or change {option}",
                option="lagrangian.vorticity_budget_history",
                value=True,
            )


def needs_budget_stage(config: Any) -> bool:
    lagrangian = getattr(config, "lagrangian", None)
    return bool(selected_budgets(getattr(config, "output", None))) or bool(
        lagrangian is not None and (
            bool(lagrangian.values.get("vorticity_budget_history", False))
            or bool(lagrangian.values.get("parcel_vorticity_budget", False))
        )
    )


def iter_all_term_names(facts: BudgetFacts) -> Iterable[str]:
    for switch in BUDGET_SWITCHES:
        for term in resolve_terms(switch, facts):
            yield f"{BUDGET_PREFIX[switch]}_{term}"


__all__ = [
    "BUDGET_PREFIX", "BUDGET_STAGGER", "BUDGET_SWITCHES", "BUDGET_UNITS",
    "BudgetFacts", "TERM_AUTHORITY", "VORTICITY_AXES", "VORTICITY_CURL_TERMS",
    "VORTICITY_DECOMPOSITION_TERMS", "VORTICITY_TERM_MEANING",
    "budget_carrier_rows", "budget_defects", "facts_from_config",
    "facts_from_runtime", "iter_all_term_names", "needs_budget_stage",
    "refuse_inadmissible_budgets", "resolve_terms", "selected_budgets",
    "vorticity_budget_names", "vorticity_carrier_rows",
]
