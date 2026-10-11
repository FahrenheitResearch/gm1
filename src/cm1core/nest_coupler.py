"""The schedule that drives one refined child inside one parent step.

This is the loop a nested run is.  It is written as a host-side schedule over
an ``advance`` callable rather than as device code, for two reasons: the
schedule is the part a nested run gets wrong (a child forced from the parent's
state at the wrong time looks exactly like a working nest and is being driven
by ``t=0``), and a schedule expressed this way can be executed and checked
without a GPU.

ONE PARENT STEP, IN ORDER
------------------------
Given a parent that has just completed the large step ending at ``t + dtp``
and a child sitting at ``t``:

1. **Build the tables once, from the parent's state at the END of the
   interval.**  ``value`` is the child's own state at ``t``; ``tendency`` is
   ``(interpolated parent at t+dtp - child at t)/dtp``.  This is the only
   place the parent is read.
2. **Take ``q`` child steps.**  Each sub-step first advances
   ``dtbc = dtbc + dtc`` (WRF ``dyn_em/solve_em.F:372``, with ``dtbc`` reset
   to zero when the tables are built, ``share/mediation_force_domain.F:213``),
   so sub-step ``n`` (``n = 1..q``) evaluates its relaxation target at
   ``dtbc = n*dtc``: linearly in time across the parent interval, and equal to
   the interpolated parent at the last sub-step.
3. **Feed back, if two-way.**  After the last sub-step the child's interior
   block means replace the parent's covered cells.  The child's own relaxation
   rows are excluded: those rows carry the parent's values, and feeding them
   back would relax the parent toward itself through the child.

THE FAILURE THIS ORDERING PREVENTS
----------------------------------
A coupler that built the tables from the parent's state at the START of the
interval, or that rebuilt them inside the sub-step loop from a parent that had
not moved, would produce a child that runs, produces no NaN, and is forced by a
parent time that never advances.  :func:`parent_interval_schedule` states the
``dtbc`` sequence by value so a test can assert it rather than trust it, and
:func:`couple_one_parent_step` is the only entry point that builds tables.

CM1 r21.1 has no nesting and therefore no such schedule; see
``records/CM1-NEST-SEAM.md``.  What is CM1's is the relationship between the
two clocks: ``src/misclibs.F:857-868`` and ``:883-885`` are why ``q`` equal to
the space ratio leaves the child's acoustic substep count equal to the
parent's, and ``src/cm1.F:3171``/``:3510`` are why a child is only ever
handed a parent state from a completed large step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from .nest_boundary import (
    BoundaryTable,
    apply_specified_relaxation,
    build_boundary_tables,
    restrict_child_to_parent,
)
from .nest_geometry import FEEDBACK_TWO_WAY, NestGeometry

NEST_COUPLER_SCHEMA = "gm1.cartesian-nest-coupler/v1"

#: What a child advance callable is handed and must return.
#:
#: ``advance(field, tendency, dtbc, sub_step)`` returns the child's field after
#: one child large step.  The runtime's device stepper satisfies this; the
#: tests satisfy it with an explicit-Euler stub, which is enough to prove the
#: schedule rather than the dynamics.
ChildAdvance = Callable[
    [list[list[float]], list[list[float]], float, int], list[list[float]]
]


@dataclass(frozen=True, slots=True)
class SubStep:
    """One child large step inside one parent interval."""

    index: int
    dtbc: float
    model_time_s: float

    def record(self) -> dict[str, object]:
        return {
            "index": self.index,
            "dtbc_s": self.dtbc,
            "model_time_s": self.model_time_s,
        }


def parent_interval_schedule(
    geometry: NestGeometry, *, parent_dt_s: float, parent_time_s: float = 0.0
) -> tuple[SubStep, ...]:
    """The ``dtbc`` the child evaluates its boundary target at, per sub-step.

    WRF's order, not a choice made here: ``med_force_domain`` sets
    ``dtbc = 0`` when it builds the tables (``share/mediation_force_domain.F:213``)
    and ``solve_em`` advances ``dtbc = dtbc + dt`` before anything else in a
    step (``dyn_em/solve_em.F:372``), so the relaxation inside sub-step ``n``
    (``n = 0..q-1`` here) targets ``(n+1)*dtc`` and the last sub-step targets
    the interpolated parent at the end of the interval exactly.
    ``model_time_s`` is the time the sub-step starts from.
    """

    q = geometry.time_refinement_ratio
    dtc = geometry.child_dt_s(parent_dt_s)
    return tuple(
        SubStep(index=n, dtbc=(n + 1) * dtc, model_time_s=parent_time_s + n * dtc)
        for n in range(q)
    )


@dataclass(frozen=True, slots=True)
class CoupledStep:
    """What one coupled parent step produced, for a receipt."""

    parent_time_s: float
    parent_dt_s: float
    sub_steps: tuple[SubStep, ...]
    fed_back_cells: int
    feedback: int

    def record(self) -> dict[str, object]:
        return {
            "schema": NEST_COUPLER_SCHEMA,
            "parent_time_s": self.parent_time_s,
            "parent_dt_s": self.parent_dt_s,
            "child_sub_steps": [step.record() for step in self.sub_steps],
            "feedback": "two-way" if self.feedback else "one-way",
            "fed_back_parent_cells": self.fed_back_cells,
        }


def couple_one_parent_step(
    *,
    geometry: NestGeometry,
    parent_at_end: Sequence[Sequence[float]],
    child_at_start: Sequence[Sequence[float]],
    child_tendency: Sequence[Sequence[float]],
    advance: ChildAdvance,
    parent_dt_s: float,
    parent_time_s: float = 0.0,
    parent_writeback: list[list[float]] | None = None,
) -> tuple[list[list[float]], CoupledStep]:
    """Advance the child across one parent interval and feed back if two-way.

    ``parent_at_end`` is the parent's state at the end of the interval and is
    never written; ``parent_writeback``, when two-way feedback is selected, is
    the array the child's block means are written into.  Handing the same list
    object as both is legitimate and is what a real run does; the tables are
    built before any write, so the forcing is never read from a partly
    fed-back parent.

    ``child_tendency`` is the **base** tendency the lateral terms are added to
    -- what the child's own dynamics produced for this sub-step.  The device
    driver recomputes it every sub-step and hands a different array each time;
    this reference takes one array and reuses it, which is what makes a
    zero-tendency stub isolate the lateral forcing exactly.  The relaxation is
    always *added* to whatever it is given, never substituted for it, except on
    the specified rows, where the parent's tendency replaces it outright.
    """

    width = geometry.spec_zone + geometry.relax_zone
    tables: Mapping[str, BoundaryTable] = build_boundary_tables(
        parent_at_end=parent_at_end,
        child_at_start=child_at_start,
        geometry=geometry,
        parent_dt_s=parent_dt_s,
        width=width,
    )
    child_dt = geometry.child_dt_s(parent_dt_s)
    field = [list(row) for row in child_at_start]
    schedule = parent_interval_schedule(
        geometry, parent_dt_s=parent_dt_s, parent_time_s=parent_time_s
    )
    for step in schedule:
        relaxed = apply_specified_relaxation(
            field=field,
            tendency=child_tendency,
            tables=dict(tables),
            geometry=geometry,
            dtbc=step.dtbc,
            child_dt_s=child_dt,
        )
        field = [list(row) for row in advance(field, relaxed, step.dtbc, step.index)]

    fed_back = 0
    if geometry.feedback == FEEDBACK_TWO_WAY and parent_writeback is not None:
        restricted = restrict_child_to_parent(field, geometry)
        for jp in range(geometry.parent_span_y):
            for ip in range(geometry.parent_span_x):
                value = restricted[jp][ip]
                if value != value:  # NaN marks an excluded relaxation row
                    continue
                parent_writeback[geometry.parent_j0 + jp][
                    geometry.parent_i0 + ip
                ] = value
                fed_back += 1

    return field, CoupledStep(
        parent_time_s=parent_time_s,
        parent_dt_s=parent_dt_s,
        sub_steps=schedule,
        fed_back_cells=fed_back,
        feedback=geometry.feedback,
    )


def euler_advance(child_dt_s: float) -> ChildAdvance:
    """A first-order child stepper, for exercising the schedule on the host.

    This is not a dynamics: it applies the relaxed tendency for one child step
    and nothing else.  It exists so the coupling schedule can be run and
    asserted without a device, and it is named so nothing mistakes it for one.
    """

    def advance(
        field: list[list[float]],
        tendency: list[list[float]],
        dtbc: float,
        index: int,
    ) -> list[list[float]]:
        return [
            [value + child_dt_s * rate for value, rate in zip(frow, trow)]
            for frow, trow in zip(field, tendency)
        ]

    return advance


__all__ = [
    "NEST_COUPLER_SCHEMA",
    "ChildAdvance",
    "CoupledStep",
    "SubStep",
    "couple_one_parent_step",
    "euler_advance",
    "parent_interval_schedule",
]
