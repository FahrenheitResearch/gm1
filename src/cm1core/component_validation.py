"""What has actually been measured about each selectable column component.

Every component this port binds is selectable.  Not every one has been
measured, and the two facts must never be confused by a reader of a receipt,
a history frame or ``gpuwm-cm1 doctor``.  This module is the one place that
says, per component, exactly what was checked and exactly what was not, in a
machine-readable form, so that a run that executes an unmeasured scheme says
so out loud rather than looking like every other run.

Importing this module imports neither CuPy nor :mod:`gpuwm`.

The states are deliberately few:

``measured``
    An independent oracle for this component's execution has been captured and
    is cited by sha256.

``unvalidated``
    The binding is exercised by tests from configuration text to the values
    handed to ArWen, and the transcribed constants are checked against the
    authority, but no run of this scheme has been compared against anything.
    A receipt for such a run carries :data:`UNVALIDATED_MARKER`.

``unvalidatable-against-cm1``
    As ``unvalidated``, plus the recorded fact that CM1 r21.1 has no scheme at
    this identity at all, so no CM1 differential can ever exist for it and any
    future validation has to be against WRF or an analytic check instead.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping


COMPONENT_VALIDATION_SCHEMA = "gm1.column-component-validation/v1"

MEASURED = "measured"
UNVALIDATED = "unvalidated"
UNVALIDATABLE_AGAINST_CM1 = "unvalidatable-against-cm1"

#: The string a consumer greps for.  It is one token, uppercase, and it appears
#: verbatim in the receipt and in doctor.
UNVALIDATED_MARKER = "GM1-UNVALIDATED-CAPABILITY"


class ComponentValidation:
    """One component's measured state, and the sentence a reader needs."""

    __slots__ = ("component_id", "state", "checked", "not_checked", "oracle_sha256")

    def __init__(
        self,
        *,
        component_id: str,
        state: str,
        checked: tuple[str, ...],
        not_checked: tuple[str, ...],
        oracle_sha256: str | None = None,
    ) -> None:
        self.component_id = component_id
        self.state = state
        self.checked = checked
        self.not_checked = not_checked
        self.oracle_sha256 = oracle_sha256

    @property
    def validated(self) -> bool:
        return self.state == MEASURED

    def record(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": COMPONENT_VALIDATION_SCHEMA,
            "component_id": self.component_id,
            "state": self.state,
            "validated": self.validated,
            "checked": list(self.checked),
            "not_checked": list(self.not_checked),
        }
        if not self.validated:
            payload["marker"] = UNVALIDATED_MARKER
        if self.oracle_sha256 is not None:
            payload["native_oracle_receipt_sha256"] = self.oracle_sha256
        return payload

    def sentence(self) -> str:
        if self.validated:
            return (
                f"{self.component_id}: measured; oracle receipt "
                f"{self.oracle_sha256}"
            )
        return (
            f"{UNVALIDATED_MARKER} {self.component_id}: {self.state}; "
            f"NOT CHECKED: {'; '.join(self.not_checked)}"
        )


_VALIDATION: Mapping[str, ComponentValidation] = MappingProxyType({
    "wsm6-mp6": ComponentValidation(
        component_id="wsm6-mp6",
        state=MEASURED,
        checked=(
            "public ArWen WSM6 column execution against a captured native oracle",
            "the five ihail-selected constants of src/mp_wsm6.F:88-99 against the "
            "installed gpuwm.core.wsm6_constants.rimed_ice_constants, both branches, "
            "raw f64 equality",
        ),
        not_checked=(
            "a CM1-vs-GM1 differential of the whole moist trajectory: GM1 substitutes "
            "ArWen's WSM6 for CM1's own, so the two are not the same numerical scheme",
        ),
        oracle_sha256="cbecf65c42095031c4d7210ac0c0693daa49e2f2f4363e05efe2b36d7dd2a482",
    ),
    "morrison-mp10": ComponentValidation(
        component_id="morrison-mp10",
        state=UNVALIDATED,
        checked=(
            "the ptype=5 inum=1 scalar table of src/param.F:3414-3473, transcribed "
            "and tested",
            "the ihail constants of src/morrison.F:407-412,441-444, transcribed and "
            "checked live against the authority tree",
            "configuration text to the values handed to ArWen's RunConfig, without "
            "opening a device",
            "that the binding dispatches: one 120 s seeded near-saturated column "
            "on an RTX 5070 Ti returns a finite, non-negative column whose dense "
            "ice falls 1147.9 m",
            "stability of one storm on a GPU: a 512x512x75 stretched (stretch_z=1) "
            "100 m bubble storm, sgs_model=4, bbc=3, ihail=1, integrated 5400 s "
            "(10800 steps at 0.5 s) on an RTX 5090 on 2026-09-16 with every carried "
            "field finite; that tree predates the stretched-grid numq fix (fbd129e), "
            "so its four number rows did not advect and the run proves stability "
            "only, not the trajectory",
            "after that fix, on an RTX 4090 (2026-09-17): the stretched-profile "
            "device test carries all ten Morrison rows as the uniform family does, "
            "and a 256x256x75 360 s stretched storm is word-identical to the pre-fix "
            "tree until number concentrations first exist and finite after",
            "CM1 r21.1's own supercell deck (run/config_files/supercell: 1 km, 2 h) "
            "against native CM1 ptype=5 once the column schemes are heated at CM1's "
            "eqtset=2 constant volume (2026-09-23, evidence/oracle/supercell-fidelity): "
            "the ten mature-phase means over 3600-7200 s at 0.986-1.046 of CM1's "
            "(CM1's own twin 0.984-1.003), the max-w column at 5 km within one cell "
            "of CM1's at every 900 s frame, and max w at 1800 s at 0.997",
        ),
        not_checked=(
            "a differential against native CM1 ptype=5 on any deck but the stock "
            "supercell: stretched columns, a surface layer, Coriolis, or cloud on "
            "grids finer than 1 km",
            "pointwise agreement: ArWen's WRF-lineage Morrison is not CM1's "
            "morrison.F, and after the first rain w departs from CM1's beyond the "
            "twin floor (nrmsd 0.029 at 1800 s against 0.0005) while the storm "
            "statistics stay within 5 %",
            "any Morrison run inside a live nest on a GPU",
            "the cloud-droplet number refresh against CM1's own statement order at "
            "runtime",
        ),
    ),
    "thompson-mp8": ComponentValidation(
        component_id="thompson-mp8",
        state=UNVALIDATED,
        checked=(
            "the ptype=3 scalar table of src/param.F:3322-3398, transcribed and tested",
            "the CM1-name to ArWen-name moment map against ArWen's own required tuple",
            "configuration text to the values handed to ArWen's RunConfig, without "
            "opening a device",
            "that the binding dispatches: one 120 s seeded near-saturated column "
            "on an RTX 5070 Ti returns a finite, non-negative column whose dense "
            "ice falls 1011.9 m",
        ),
        not_checked=(
            "a differential against native CM1 ptype=3",
            "any storm: the only execution is one 120 s seeded column, which says "
            "the binding runs and nothing about a trajectory",
            "the packaged classic Thompson table assets, which this port has not "
            "added to the audited ArWen surface because they are data rather than "
            "code and ArWen byte-validates them itself",
        ),
    ),
    "thompson-aerosol-mp28": ComponentValidation(
        component_id="thompson-aerosol-mp28",
        state=UNVALIDATABLE_AGAINST_CM1,
        checked=(
            "the eleven-scalar axis and the CM1-name to ArWen-name map against "
            "ArWen's own state reads (gpuwm/core/microphysics_aerosol.py)",
            "that the binding dispatches: one 120 s seeded near-saturated column "
            "on an RTX 5070 Ti returns a finite, non-negative column",
            "configuration text to the values handed to ArWen's RunConfig, without "
            "opening a device",
        ),
        not_checked=(
            "a differential against CM1: src/param.F:3141-4509's ptype=3 is the "
            "CLASSIC Thompson scheme and CM1 numbers no aerosol-aware variant, so "
            "one cannot be constructed",
            "the aerosol initial state: this seam reaches the scheme through "
            "gpuwm.core.microphysics.apply without microphysics_init, so nwfa and "
            "nifa take ArWen's own declared fallback profile rather than a WRF "
            "global-emission one, and every run of this component is on that "
            "fallback",
            "the two surface aerosol emission rates nwfa2d and nifa2d, which this "
            "port allocates at zero because CM1's idealized lane models no surface "
            "aerosol source at all",
        ),
    ),
})


def component_validation(component_id: str) -> ComponentValidation:
    """The recorded validation state of one bound component."""

    try:
        return _VALIDATION[component_id]
    except KeyError:
        raise KeyError(
            f"component {component_id!r} has no recorded validation state; a bound "
            "component without one would publish receipts that look measured"
        ) from None


def validation_records() -> list[dict[str, object]]:
    """Every bound component's state, in a stable order."""

    return [value.record() for value in _VALIDATION.values()]


def unvalidated_component_ids() -> tuple[str, ...]:
    return tuple(
        component_id
        for component_id, value in _VALIDATION.items()
        if not value.validated
    )


__all__ = [
    "COMPONENT_VALIDATION_SCHEMA",
    "ComponentValidation",
    "MEASURED",
    "UNVALIDATABLE_AGAINST_CM1",
    "UNVALIDATED",
    "UNVALIDATED_MARKER",
    "component_validation",
    "unvalidated_component_ids",
    "validation_records",
]
