"""What each CM1 dynamical word was checked against, and what it was not.

Every ``param2``/``param3`` selector this port admits appears here exactly
once, with the check that was actually run behind it.  ``doctor`` prints the
whole register; a Cartesian run receipt carries the rows its own configuration
selects, so a reader who has one artifact and no other context still learns
which of the numbers in front of them rests on a measurement and which does
not.

A row's ``validation`` is never a claim about the future.  It names the run or
the test, and ``not_checked`` names, concretely, what that run could not see.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any


VALIDATED = "validated"
UNVALIDATED = "unvalidated"
INERT = "inert"

DYNAMICS_VALIDATION_SCHEMA = "gm1.dynamics-validation-register/v1"


@dataclass(frozen=True, slots=True)
class DynamicsCapability:
    key: str
    cm1_name: str
    authority: str
    status: str
    validation: str
    not_checked: str
    #: The values every measured run used, for an UNVALIDATED real whose
    #: measurement covers exactly those values.  A run at one of them rests on
    #: those runs; any other value is the unmeasured capability the row names.
    measured_values: tuple[float, ...] = ()

    def measured_at(self, value: Any) -> bool:
        """Whether ``value`` is one the measured runs used.

        Compared to a float32's precision, because the runtime contract
        carries the value the device reads and the document the one it wrote.
        """

        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
        return any(abs(float(value) - item) <= 1e-6 * max(1.0, abs(item))
                   for item in self.measured_values)

    def status_at(self, value: Any) -> str:
        if self.status == UNVALIDATED and self.measured_at(value):
            return VALIDATED
        return self.status

    def as_dict(self, value: Any = None) -> dict[str, Any]:
        row: dict[str, Any] = {
            "key": self.key,
            "cm1_name": self.cm1_name,
            "authority": self.authority,
            "status": self.status_at(value),
            "validation": self.validation,
            "not_checked": self.not_checked,
        }
        if self.measured_values:
            row["value"] = value
            row["measured_values"] = list(self.measured_values)
        return row


DYNAMICS_CAPABILITIES: tuple[DynamicsCapability, ...] = (
    DynamicsCapability(
        key="dynamics.coriolis",
        cm1_name="icor",
        authority="src/solve2.F:631-720",
        status=VALIDATED,
        validation=(
            "differential against CM1 r21.1 at 32x32x40 / 1 km, fcor=1e-4, 280 s: "
            "u3d nrmse 5.35e-5 and v3d 2.01e-4 with the term on, against a "
            "measured icor=0-vs-icor=0 floor of 5.2e-5 and 1.6e-4; switching "
            "GM1's term off multiplies those by 117 and 595.  prs, rho, th3d "
            "and w3d, which solve2.F:639-653 reaches only through the winds, "
            "move by at most 2.7 times their own disagreement.  Repeated in a "
            "translating frame (umove=12.5, vmove=3.0), where "
            "solve2.F:645,647's frame add-back applies.  The operator also "
            "integrates the analytic inertial oscillation to 1e-4 relative over "
            "a full period.  "
            "evidence/oracle/dynamics-cm1-words/coriolis-fplane-32.txt and "
            "coriolis-fplane-moving-32.txt"
        ),
        not_checked=(
            "no run longer than 280 s, and none on a grid with terrain.  "
            "icor=1 with adapt_dt=1 is measured "
            "(evidence/oracle/dynamics-cm1-words/coriolis-adaptive-32.txt); "
            "icor=1 with adapt_dt=1 AND a translating frame is refused, "
            "because the published domain location's exactness proof needs "
            "one dt for the whole run"
        ),
    ),
    DynamicsCapability(
        key="dynamics.beta_plane",
        cm1_name="betaplane",
        authority="src/param.F:8945-8969; src/solve2.F:655-673",
        status=VALIDATED,
        validation=(
            "differential against CM1 r21.1 at 32x512x40 / 1 km -- a 512 km "
            "meridional span, over which param.F:8958's beta0 varies f by 12% "
            "of fcor=1e-4.  Each code's beta-plane run matches the other's at "
            "the floor (v3d nrmse 4.06e-5 at 280 s) and disagrees with the "
            "other's f-plane run by a factor of 71 (2.90e-3).  "
            "evidence/oracle/dynamics-cm1-words/coriolis-betaplane-512.txt"
        ),
        not_checked=(
            "on a domain narrower than about 100 km the beta contribution is "
            "below this pair of codes' own disagreement and no differential can "
            "separate it; the 32 km case was run and could not"
        ),
    ),
    DynamicsCapability(
        key="dynamics.fcor_s_inv",
        cm1_name="fcor",
        authority="src/param.F:8945-8969; src/constants.F:87",
        status=VALIDATED,
        validation=(
            "carried by value into the kernel scalar and measured at 1e-4 in "
            "the icor=1 differentials above; the sign is checked analytically "
            "in both hemispheres"
        ),
        not_checked=(
            "only |fcor| = 1e-4 has been run against CM1; the asin domain bound "
            "for betaplane=1 is a refusal, not a measurement"
        ),
    ),
    DynamicsCapability(
        key="dynamics.lspgrad",
        cm1_name="lspgrad",
        authority="src/solve1.F:590-612; README.namelist:531-538",
        status=UNVALIDATED,
        validation=(
            "the per-level forcing words are refused unless raw-word identical "
            "to the CPU operator evidence/oracles/lspgrad-source-exact-v1 "
            "proves against gfortran -O0/-O2; the kernel text is compiled by "
            "g++ and driven through the real closure on the host "
            "(tests/test_cartesian_lspgrad.py), where it changes only held "
            "u/v and keeps the tiled and untiled closure raw-word identical; "
            "on the RTX 5090 the device kernel equals the host compilation "
            "word for word (tests/test_cuda_gm1_solve1_lspgrad1_closure_gpu.py), "
            "and in the no-storm balance column at cnstcd 0.005 the lspgrad "
            "switch response of the domain-mean wind at 600 s matches CM1's "
            "to 0.0057 m/s at 3-12 km and 0.028 m/s at 1-3 km"
        ),
        not_checked=(
            "the balance column's SRH criterion (0-1 km within 1 m2/s2 of CM1) "
            "fails for a reason lspgrad=0 shares: GM1's periodic column now "
            "keeps every column bit-identical (evidence/oracle/"
            "periodic-momentum-faces), but it carries a horizontally uniform "
            "subgrid TKE where CM1 stays laminar, so the near-surface drift "
            "below about 700 m is not compared (docs/GM1-CARTESIAN-RUNTIME.md)"
        ),
    ),
    DynamicsCapability(
        key="dynamics.alph",
        cm1_name="alph",
        authority="README.namelist:756-759; src/sound.F:78-82",
        status=VALIDATED,
        validation=(
            "differential against CM1 r21.1 at alph=0.60 and alph=0.5 on the "
            "32x32x40 / 1 km acoustic case, 280 s.  Against CM1 at 0.60, GM1 "
            "at 0.60 is closer than GM1 at 0.5 in prs, rho, u3d, v3d and w3d "
            "at every frame (in prs by 1.8 to 12.7 times); against CM1 at 0.5, "
            "GM1 at 0.5 is closer in prs, rho and w3d at every frame.  "
            "tools/oracle/switch_response.py puts GM1's response to "
            "the change at 0.997 to 1.003 of CM1's in prs and rho and 0.998 to "
            "1.012 in w3d, with at most 12%, 11% and 13% of CM1's response not "
            "reproduced.  "
            "evidence/oracle/dynamics-cm1-words/alph-default-move.txt"
        ),
        not_checked=(
            "only 0.5 and 0.60 have been run against CM1, on one flat 32x32x40 "
            "case and at one horizon.  th3d, u3d and v3d leave 0.854, 0.615 "
            "and 0.445 of CM1's response unreproduced and that is not "
            "explained here: the switch_response table the numbers came from "
            "was not kept, only the ranges and maxima above were transcribed, "
            "and the run directories are gone, so the |dCM1| column that "
            "docs/ORACLE-DIFFERENTIAL-HARNESS.md:452 requires -- read the last "
            "column only while |dCM1| exceeds the two codes' baseline "
            "disagreement -- cannot be applied to those three fields by anyone "
            "reading this.  The next capture must ship the table "
            "(switch_response.py --json) beside the file; until it does, treat "
            "prs, rho and w3d as the measured claim and those three as "
            "unexplained.  The earlier finding that GM1 responded at 5-32% of "
            "CM1's strength does not hold on the re-captured runs.  Part of it "
            "came from switch_response.py reading CM1 one output interval "
            "early (fixed there)"
        ),
    ),
    DynamicsCapability(
        key="dynamics.kdiv",
        cm1_name="kdiv",
        authority="README.namelist:751-754",
        status=UNVALIDATED,
        validation=(
            "threaded to the kernel argument and recorded by value; CM1's own "
            "0.10 is the default and is what every measured run used, so a run "
            "at 0.10 rests on those runs and is reported validated"
        ),
        not_checked="no run has used a kdiv other than 0.10 on either code",
        measured_values=(0.10,),
    ),
    DynamicsCapability(
        key="dynamics.adaptive_timestep",
        cm1_name="adapt_dt",
        authority="src/misclibs.F:593-810; src/cm1.F:3013-3048",
        status=VALIDATED,
        validation=(
            "differential against CM1 r21.1 with adapt_dt=1 on both codes, "
            "dtl=1 s, 280 s: CM1 chose dt=1.1 then 2.0 and so did GM1.  The "
            "frames landed on 20..280 s exactly through the misclibs.F:668-680 "
            "ramp-down, and both codes wrote them at the same step numbers "
            "(12, 22, ..., 142), 142 large steps instead of 280.  From 120 s "
            "on, the CM1-GM1 nrmse is no more than 3% above the fixed-step "
            "floor in every field, and as much as 11% below it (0.894 to "
            "1.030 of the floor).  Before that it is at most 4.6 times the "
            "floor (v3d at 40 s: 6.4e-6 against 1.4e-6).  "
            "evidence/oracle/dynamics-cm1-words/adaptive-timestep-32.txt"
        ),
        not_checked=(
            "the diffusion limb of misclibs.F:647-650 did not bind in that run "
            "(ksmax stayed below 1e-6), so only the Courant limb and the "
            "output ramp-down were exercised on a device; the ks arithmetic is "
            "covered against the source by tests, not by a run.  Resuming an "
            "adaptive run from a checkpoint is refused rather than supported, "
            "and so is an adaptive run in a translating frame: "
            "rust/cm1-state/src/domain_location.rs proves cm1.F:3512's running "
            "sum exact by checking it against completed_steps*dt*umove, which "
            "needs one dt for the whole run"
        ),
    ),
    DynamicsCapability(
        key="dynamics.idiff",
        cm1_name="idiff",
        authority="README.namelist:281-288; src/adv.F:962; src/cm1.F:3791",
        status=INERT,
        validation=(
            "admitted at 0 only, which is what all 23 of CM1 r21.1's shipped "
            "decks set; any other value is refused by name because this lane's "
            "advection kernels carry no difforder branch"
        ),
        not_checked="nothing: the capability is refused, not silently ignored",
    ),
    DynamicsCapability(
        key="dynamics.difforder",
        cm1_name="difforder",
        authority="README.namelist:297-307",
        status=INERT,
        validation="admitted and carried; read by nothing while idiff = 0",
        not_checked=(
            "declaring it changes no result and no identity, because "
            "adv.F:962 and cm1.F:3791 both test idiff first"
        ),
    ),
    DynamicsCapability(
        key="dynamics.mdiff",
        cm1_name="mdiff",
        authority="README.namelist:289-296",
        status=INERT,
        validation="admitted and carried; read by nothing while idiff = 0",
        not_checked="as difforder",
    ),
    DynamicsCapability(
        key="dynamics.kdiff2",
        cm1_name="kdiff2",
        authority="README.namelist:743",
        status=INERT,
        validation="admitted and carried; read by nothing while idiff = 0",
        not_checked="as difforder",
    ),
    DynamicsCapability(
        key="dynamics.kdiff6",
        cm1_name="kdiff6",
        authority="README.namelist:744-747",
        status=INERT,
        validation="admitted and carried; read by nothing while idiff = 0",
        not_checked="as difforder",
    ),
)

_BY_KEY = MappingProxyType({item.key: item for item in DYNAMICS_CAPABILITIES})


def dynamics_capability(key: str) -> DynamicsCapability:
    return _BY_KEY[key]


def selected_dynamics_capabilities(dynamics: Any) -> tuple[DynamicsCapability, ...]:
    """The register rows a given ``[dynamics]`` table actually selects.

    A row is selected when the document's value for it can change a result --
    a nonzero selector, or a real the run reads.  The four diffusion reals are
    never selected while ``idiff`` is zero, because nothing reads them.
    """

    def selector(*names: str, default: Any = 0) -> Any:
        """Read the value under whichever spelling the caller carries.

        The configuration document spells these ``coriolis``/``beta_plane``
        and the runtime contract spells them CM1's way, ``icor``/
        ``betaplane``.  Both are legitimate callers of this register, and a
        register that silently reported "no rotation" for one of them would be
        worse than no register.
        """

        for name in names:
            if hasattr(dynamics, name):
                return getattr(dynamics, name)
        return default

    chosen: list[DynamicsCapability] = []
    if selector("coriolis", "icor") != 0:
        chosen.append(_BY_KEY["dynamics.coriolis"])
        chosen.append(_BY_KEY["dynamics.fcor_s_inv"])
        if selector("beta_plane", "betaplane") != 0:
            chosen.append(_BY_KEY["dynamics.beta_plane"])
    # An undeclared lspgrad is None on the document and 0 on the runtime
    # contract; both are CM1's lspgrad=0 and select nothing.
    if (selector("lspgrad") or 0) != 0:
        chosen.append(_BY_KEY["dynamics.lspgrad"])
    chosen.append(_BY_KEY["dynamics.alph"])
    chosen.append(_BY_KEY["dynamics.kdiv"])
    if selector("adaptive_timestep", default=False):
        chosen.append(_BY_KEY["dynamics.adaptive_timestep"])
    if selector("idiff") != 0:
        chosen.append(_BY_KEY["dynamics.idiff"])
    return tuple(chosen)


def selected_value(dynamics: Any, item: DynamicsCapability) -> Any:
    """The value this ``[dynamics]`` table gives one register row, or None."""

    name = item.key.partition(".")[2]
    for spelling in (name, item.cm1_name):
        if hasattr(dynamics, spelling):
            return getattr(dynamics, spelling)
    return None


def unvalidated_dynamics_capabilities(
    dynamics: Any,
) -> tuple[tuple[DynamicsCapability, Any], ...]:
    """The selected rows that are unvalidated at the value this run uses.

    Value-aware: ``kdiv`` at CM1's 0.10, the value every measured run used, is
    not reported; any other ``kdiv`` is.
    """

    return tuple(
        (item, value)
        for item in selected_dynamics_capabilities(dynamics)
        for value in (selected_value(dynamics, item),)
        if item.status_at(value) == UNVALIDATED
    )


def dynamics_validation_record(dynamics: Any) -> dict[str, object]:
    """The receipt block: every selected row, and the unvalidated ones first."""

    selected = selected_dynamics_capabilities(dynamics)
    unvalidated = [item for item, _ in unvalidated_dynamics_capabilities(dynamics)]
    return {
        "schema": DYNAMICS_VALIDATION_SCHEMA,
        "unvalidated": bool(unvalidated),
        "unvalidated_keys": [item.key for item in unvalidated],
        "capabilities": [
            item.as_dict(selected_value(dynamics, item)) for item in selected
        ],
    }


__all__ = [
    "DYNAMICS_CAPABILITIES",
    "DYNAMICS_VALIDATION_SCHEMA",
    "INERT",
    "UNVALIDATED",
    "VALIDATED",
    "DynamicsCapability",
    "dynamics_capability",
    "dynamics_validation_record",
    "selected_dynamics_capabilities",
    "selected_value",
    "unvalidated_dynamics_capabilities",
]
