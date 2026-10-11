"""Capabilities this port exposes and has not measured against CM1.

A capability becomes selectable the moment it is reachable.  Anything exposed
without a measurement carries a row here, and that row is loud in two places a
user actually reads: ``gpuwm-cm1 doctor`` lists every row, and a run receipt
lists the rows the run's own document selected.  Silence is the failure mode
this module exists to prevent -- a capability that is neither refused nor
marked reads as validated, and nothing downstream can tell the difference.

A row is not a refusal.  A refusal names a wrong result and stops the run; a
row here names a result nobody has *checked*, and the run proceeds.  The two
are written differently on purpose: ``not_checked`` says what was not measured
and ``would_show`` says what a wrong answer would look like in the output, so a
reader can decide whether this run's conclusion depends on it.

Retiring a row is the same act as retiring a guard: when the measurement is
made, delete the row and record the measurement.  A row that survives its own
validation is as misleading as a missing one.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


UNVALIDATED_REGISTER_SCHEMA = "gm1.unvalidated-capability-register/v1"

#: The marker string.  Machine-readable, greppable, and the same word in the
#: receipt, in doctor and in this module.
UNVALIDATED = "unvalidated"


@dataclass(frozen=True, slots=True)
class UnvalidatedCapability:
    """One exposed, unmeasured capability."""

    #: The configuration key a user types, dotted exactly as the TOML spells it.
    option: str
    #: The values this row covers.  Empty means every value the key admits.
    values: tuple[Any, ...]
    #: What was not measured.  Not "this is untested" -- the specific check.
    not_checked: str
    #: What a wrong answer would look like in the run's own output.
    would_show: str
    #: CM1 r21.1 file:line the capability is implemented from.
    authority: str

    def covers(self, value: Any) -> bool:
        if not self.values:
            return True
        return any(value == candidate and type(value) is type(candidate) for candidate in self.values)

    def record(self) -> dict[str, Any]:
        return {
            "marker": UNVALIDATED,
            "option": self.option,
            "values": list(self.values),
            "not_checked": self.not_checked,
            "would_show": self.would_show,
            "authority": self.authority,
        }

    def line(self) -> str:
        values = "any value" if not self.values else ", ".join(repr(v) for v in self.values)
        return f"{UNVALIDATED} {self.option} ({values}): {self.not_checked}; a wrong answer would show as {self.would_show}"


#: Grid geometry.  ``grid.stretch_z`` itself is not here: 1, 3 and 4 are each
#: measured against CM1 r21.1 and ``evidence/oracle/stretched-column-32``,
#: ``.../explicit-w-levels-32`` and ``.../explicit-scalar-levels-32`` hold the
#: comparisons.  What those runs did *not* reach is a cloud.
UNVALIDATED_CAPABILITIES: tuple[UnvalidatedCapability, ...] = (
    UnvalidatedCapability(
        option="grid.stretch_z",
        values=(1, 3, 4),
        not_checked=(
            "no stretched run has reached condensation. The three differential "
            "cases are one advective cell crossing long, which is the horizon "
            "the vertical metric's own timescale asks for, and at that horizon "
            "qc, qr, qi, qs and qg are identically zero in both codes. So the "
            "profile-bound scalar family (cuda_gm1_child_stretched_scalar) is "
            "measured on water vapour only, and its final-stage pdef chain has "
            "never repaired a negative condensate on a non-uniform column"
        ),
        would_show=(
            "hydrometeor mass moving through the wrong layer depth once cloud "
            "forms: sedimentation and the pdef repair both divide by rdsf(k), "
            "so the error would appear first as precipitation reaching the "
            "ground early or late, in proportion to how far the near-ground "
            "layers depart from the dz the uniform family would have used"
        ),
        authority="src/adv.F:681-695;src/solve2.F:1430-1533",
    ),
)


def selected(record: Mapping[str, Any]) -> tuple[UnvalidatedCapability, ...]:
    """The rows one configuration record selects.

    ``record`` is a flat mapping of dotted option to value, as a caller builds
    it from the document it admitted.  A key the caller does not carry selects
    nothing: this reports what a run chose, never what it might have chosen.
    """

    if not isinstance(record, Mapping):
        raise TypeError("record must be a mapping of dotted option to value")
    return tuple(
        row
        for row in UNVALIDATED_CAPABILITIES
        if row.option in record and row.covers(record[row.option])
    )


def selected_by_document(config: Any) -> tuple[UnvalidatedCapability, ...]:
    """The rows one admitted ``CM1Config`` selects.

    The document is flattened through its own canonical projection, so the
    keys this reads are exactly the keys the run's ``config_sha256`` covers.
    """

    flat: dict[str, Any] = {}
    for table, fields in config.canonical_dict().items():
        if isinstance(fields, Mapping):
            for name, value in fields.items():
                flat[f"{table}.{name}"] = value
    return selected(flat)


#: CM1 r21.1's own source for each column component's scheme, for the
#: component rows below.  A component with no CM1 counterpart says so.
_COMPONENT_AUTHORITY = {
    # CM1 r21.1 src/param.F: each ptype's setup block, ELSEIF to ELSEIF.
    "morrison-mp10": "src/morrison.F; src/param.F:3400-3631 (ptype=5)",
    "thompson-mp8": "src/thompson.F; src/param.F:3322-3399 (ptype=3)",
    "thompson-aerosol-mp28": (
        "none: src/param.F:3141-4509 numbers no aerosol-aware Thompson scheme"
    ),
    "wsm6-mp6": "src/mp_wsm6.F; src/param.F:3632-3713 (ptype=7)",
}

#: What a wrong answer would look like, for the dynamics rows the value-aware
#: register can report.  The register itself says what was not checked.
_DYNAMICS_WOULD_SHOW = {
    "dynamics.kdiv": (
        "acoustic divergence damping of a strength CM1 would not apply at this "
        "kdiv: pressure and vertical-velocity noise left in the fields, or "
        "divergent motions damped that CM1 keeps"
    ),
    "dynamics.lspgrad": (
        "a near-surface wind and 0-1 km storm-relative helicity that drift from "
        "CM1's below about 700 m while the large-scale pressure gradient acts"
    ),
}


def component_row(component_id: str) -> UnvalidatedCapability | None:
    """The row for a column component with no measurement, else ``None``."""

    from .component_validation import UNVALIDATED_MARKER, component_validation

    state = component_validation(component_id)
    if state.validated:
        return None
    return UnvalidatedCapability(
        option="physics.component",
        values=(component_id,),
        not_checked=(
            f"{UNVALIDATED_MARKER} {component_id} ({state.state}): "
            + "; ".join(state.not_checked)
        ),
        would_show=(
            "a moist trajectory that departs from native CM1's scheme at this "
            "ptype: hydrometeor mass and number, latent heating and so updraft "
            "and downdraft strength, the cold pool and the surface rain"
        ),
        authority=_COMPONENT_AUTHORITY.get(component_id, "not stated"),
    )


def dynamics_rows(dynamics: Any) -> tuple[UnvalidatedCapability, ...]:
    """The ``[dynamics]`` rows unvalidated at the values this run uses."""

    from .dynamics_validation_register import unvalidated_dynamics_capabilities

    return tuple(
        UnvalidatedCapability(
            option=item.key,
            values=(value,) if value is not None else (),
            not_checked=item.not_checked,
            would_show=_DYNAMICS_WOULD_SHOW.get(
                item.key,
                f"results that differ from CM1 r21.1's wherever {item.cm1_name} acts",
            ),
            authority=item.authority,
        )
        for item, value in unvalidated_dynamics_capabilities(dynamics)
    )


def initialization_rows(initialization: Mapping[str, Any] | None) -> tuple[UnvalidatedCapability, ...]:
    """The initialization selectors the run's own validation block flags."""

    if not initialization:
        return ()
    return tuple(
        UnvalidatedCapability(
            option=f"initialization.{item['selector']}",
            values=(item["value"],),
            not_checked=str(item.get("not_checked") or "not stated"),
            would_show=(
                "an initial state that differs from the one CM1 builds for this "
                "selector value: the sounding, the perturbation or the balance "
                "the run starts from"
            ),
            authority=str(item.get("evidence") or item.get("method") or "not stated"),
        )
        for item in initialization.get("items", ())
    )


def selected_for_run(config: Any, *, initialization: Mapping[str, Any] | None = None,
                     ) -> tuple[UnvalidatedCapability, ...]:
    """Every unvalidated capability one run uses: the receipt's top-level rows.

    The capability rows the document selects (:func:`selected_by_document`),
    then the column component when its state is not ``measured``
    (``component_validation``), then the ``[dynamics]`` rows unvalidated at
    the values the document gives them (``kdiv`` at CM1's 0.10 is not one),
    then the initialization selectors ``initialization`` (the run's
    ``unvalidated_initialization`` block) flags.  A reader of the top-level
    register sees every one of them without opening the nested blocks.
    """

    rows = list(selected_by_document(config))
    physics = getattr(config, "physics", None)
    component = getattr(physics, "component", None)
    # "off" is a dry run: no column component, nothing to have measured.
    if isinstance(component, str) and component != "off":
        row = component_row(component)
        if row is not None:
            rows.append(row)
    dynamics = getattr(config, "dynamics", None)
    if dynamics is not None:
        rows.extend(dynamics_rows(dynamics))
    rows.extend(initialization_rows(initialization))
    return tuple(rows)


def receipt(rows: Sequence[UnvalidatedCapability] | None = None) -> dict[str, Any]:
    """The receipt block.  Present even when empty, so absence is a statement."""

    rows = UNVALIDATED_CAPABILITIES if rows is None else tuple(rows)
    return {
        "schema": UNVALIDATED_REGISTER_SCHEMA,
        "marker": UNVALIDATED,
        "count": len(rows),
        "capabilities": [row.record() for row in rows],
    }


__all__ = [
    "UNVALIDATED",
    "UNVALIDATED_CAPABILITIES",
    "UNVALIDATED_REGISTER_SCHEMA",
    "UnvalidatedCapability",
    "component_row",
    "dynamics_rows",
    "initialization_rows",
    "receipt",
    "selected",
    "selected_by_document",
    "selected_for_run",
]
