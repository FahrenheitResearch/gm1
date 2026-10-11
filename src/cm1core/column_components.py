"""The one place a CM1 ``ptype`` becomes an executable GM1 column component.

Importing this module imports neither CuPy nor :mod:`gpuwm`.  It exists so
that a lane can go from ``physics.component`` in a TOML document to the exact
:class:`~cm1core.arwen_column_component.ColumnComponentSpec` that owns the
scalar axis, the persistence inventory and the ``ihail`` selector, without any
caller spelling a component id in a branch of its own.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

from .arwen_column_component import ColumnComponentSpec
from .arwen_morrison_component import MORRISON_SPEC
from .arwen_scheme_components import THOMPSON_AEROSOL_SPEC, THOMPSON_SPEC
from .arwen_wsm6_component import WSM6_SPEC
from .column_component_blockers import (
    COLUMN_COMPONENT_BLOCKERS,
    refuse_blocked_component,
)
from .errors import refuse
from .moisture_species import MOISTURE_COMPONENT_IDS


COLUMN_COMPONENT_SPECS: Mapping[str, ColumnComponentSpec] = MappingProxyType({
    WSM6_SPEC.component_id: WSM6_SPEC,
    MORRISON_SPEC.component_id: MORRISON_SPEC,
    THOMPSON_SPEC.component_id: THOMPSON_SPEC,
    THOMPSON_AEROSOL_SPEC.component_id: THOMPSON_AEROSOL_SPEC,
})

#: Column components this distribution does not offer, each with the
#: developer switch that re-admits it.  thompson-aerosol-mp28 reaches WRF
#: Thompson-aerosol routines historically adapted from Numerical Recipes, whose
#: licence does not permit free redistribution, and no CM1 ptype selects it.
#: Until a legal review clears it, the Cartesian door refuses it by value
#: (cartesian_door.require_listed_component), doctor does not list it and the
#: wheel ships no deck naming it.  A developer tree sets the switch to "1" to
#: keep exercising the code path (tests/conftest.py does).
UNLISTED_COMPONENTS: Mapping[str, str] = MappingProxyType({
    THOMPSON_AEROSOL_SPEC.component_id: "GPUWM_CM1_UNLISTED_MP28",
})

#: The tile-local claim this lane's published physics receipts carry.
GM1_TILESTREAM_COLUMN_CLAIM_LIMIT = (
    "tile-local CM1-to-ArWen staging, one public column-microphysics call per "
    "physical column, and publication of the component's transported scalars "
    "and persistence carriers after the pdefq boundary; no qa mutation, CM1 "
    "post-physics EOS, solve3, pressure, halo/open-history mutation, MPI "
    "reduction or coupled-step claim from this operation alone"
)


def column_component_spec(component_id: object) -> ColumnComponentSpec:
    """Return the bound spec for one component id, refusing anything else."""

    if isinstance(component_id, str):
        spec = COLUMN_COMPONENT_SPECS.get(component_id)
        if spec is not None:
            return spec
        if component_id in COLUMN_COMPONENT_BLOCKERS:
            refuse_blocked_component(component_id, option="physics.component")
    refuse(
        code="GM1-COLUMN-COMPONENT-UNSUPPORTED",
        gate="column-component",
        reason=(
            f"component {component_id!r} has no CM1-owned column binding, so "
            "GM1 has no transcribed scalar axis, persistence inventory or "
            "dense-ice selector for it; binding it would allocate a moisture "
            "axis of the wrong length and hand the scheme whichever rows "
            "happened to align"
        ),
        remedy="select one of " + ", ".join(map(repr, COLUMN_COMPONENT_SPECS)),
        option="physics.component",
        value=component_id,
    )


def _check_registry_matches_species_table() -> None:
    """The two registries are one inventory; a drift is an import-time defect."""

    if tuple(COLUMN_COMPONENT_SPECS) != MOISTURE_COMPONENT_IDS:
        raise RuntimeError(
            "the column-component registry and the CM1 moisture-species table "
            f"disagree: {tuple(COLUMN_COMPONENT_SPECS)} vs {MOISTURE_COMPONENT_IDS}"
        )


_check_registry_matches_species_table()


__all__ = [
    "COLUMN_COMPONENT_SPECS",
    "GM1_TILESTREAM_COLUMN_CLAIM_LIMIT",
    "column_component_spec",
]
