"""A live nest's restart: the parent's restart, then the child's, in one directory.

CM1 r21.1 has no nesting (``records/CM1-NEST-SEAM.md`` section 1), so the
authority for restarting a nested run is WRF's, and WRF's rule is per domain:

* Only the outermost domain decides when a restart is written, so every domain
  is written at one model time (``share/mediation_integrate.F:347-354``:
  "Only the parent initiates the restart writing").
* ``med_restart_out`` writes this domain's own restart file and then calls
  itself for each child (``share/mediation_integrate.F:1157-1182``), so each
  domain has a complete restart of its own (``wrfrst_d01``, ``wrfrst_d02``).
* On a restart, ``med_nest_initial`` reads the nest's own restart with
  ``input_restart`` instead of interpolating the nest from its parent
  (``share/mediation_integrate.F:861-892``).
* A moving nest's vortex centre ``vc_i``/``vc_j`` is restart state
  (``Registry/Registry.EM_COMMON:438-441``, "vortex center indices; need for
  restarts of moving nests").  The nest's placement ``i_parent_start``/
  ``j_parent_start`` is not a state row but a namelist (``rconfig``) entry
  flagged for restart and history I/O (``:2318-2319``), which WRF updates on
  every move (``share/mediation_nest_move.F:162-163,176-177``).

GM1 does exactly that.  ``restart-NNNNNN/`` holds the parent's CM1FCP1 restart,
unchanged, and ``restart-NNNNNN/nest-child/`` the child's, written by the same
writer (:func:`cm1core.cartesian_restart.commit_cartesian_restart`) from the
child runtime's own restart profile -- every child prognostic carrier, held
tendency, TKE, microphysics persistence, radiative lateral-boundary history and
rain accumulator the parent's own restart carries for the parent -- plus the
coupling state below.  Both are assembled in one staging directory and renamed
into place together, so a restart that exists always has its child.

WRF restarts a moving nest exactly only on a ``vortex_interval`` boundary and
warns otherwise (``share/mediation_integrate.F:897-913``).  GM1 carries the
tracker's own state and re-derives the tracking cadence from the absolute
parent step, so a resume at any completed parent step continues the tracker
the uninterrupted run would have.

WHAT THE CHILD RESTART CARRIES BEYOND A SINGLE-DOMAIN RESTART
-------------------------------------------------------------
Scheduler scalars (bit-exact in the envelope's scalar table):

* the placement ``nest.parent_i0``/``nest.parent_j0`` (a following nest has
  moved since the document was written) and the fixed geometry it is checked
  against;
* the parent step the child restart belongs to and the time ratio;
* a following nest's tracker (:class:`cm1core.nest_move.TrackerState`): WRF's
  ``vc_i``/``vc_j`` and CM1's pressure centre and held offset;
* the whole run's move and tracker-event counts, and the last track's parent
  step and value, which a move decided from the carried centre reports before
  the resumed run tracks again; a resumed mover adds its own to them;
* under ``nest.follow_on_edge = "hold"`` (:mod:`cm1core.nest_move`, THE HOLD
  POLICY), the tracker's hold state: the consecutive held samples, which set
  the next sample's search reach and decide when it relocates, the
  consecutive corral-cancelled parent steps, which decide when the run stops
  by name, and the whole run's hold-event counts and longest runs, which the
  receipt reports; a resumed mover continues them exactly.  A restart written
  under ``stop`` carries none, and a document that resumes it under ``hold``
  starts with no sample held, which is what the restart's tracker had;
* the coupling, relaxation and parcel counters the receipt reports;
* a declared ``[nest.tornado_criterion]``'s ledger
  (:meth:`cm1core.nest_tornado_criterion.CriterionLedger.checkpoint`,
  ``nest.tornado.*``): the previous centre, the qualifying run in progress,
  the episodes, the stop request and every whole-run count, so the resumed
  run's verdict is the uninterrupted run's.

Carriers: a nest child parcel set's positions and owning grid
(:class:`cm1core.nest_parcels.NestParcelTracker`), when one is declared; and
the child's own run-long surface swaths (``nest.swath.sws`` .. ``shs``), when
the parent keeps them (:func:`cm1core.nest_live.child_swath_selection`).
``restart_write.F:692-757`` writes ``sws``..``shs`` for a CM1 grid and
``restart_read.F`` reads them straight back, so a resumed child's swath is the
uninterrupted child's.  They are carried in the ``nest`` group rather than the
single-domain groups because the child's swaths are opened with the run's step
count (:meth:`cm1core.nest_live.NestDriver.open_child_swaths`), after the
restore: the restored words wait on the driver until then.

WHAT IS REBUILT RATHER THAN CARRIED, AND WHY THAT IS SOUND
---------------------------------------------------------
See :data:`NEST_REBUILT`.  Every one of them is overwritten before any word of
it is read by the next parent step: the boundary tables by ``med_nest_force``
at the top of that step (``NestLateralForcing.refresh``, which also resets
``dtbc``), the specified-zone snapshots at the top of each acoustic small step,
the vertical-refinement windows by the next capture, and the parcel velocity
snapshots by the next ``solve2.F:1290-1301`` capture.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
import struct
from types import MappingProxyType
from typing import Any

from .nest_move import HOLD_EVENTS as _HOLD_EVENTS
from .nest_move import PRESSURE_TRACKERS as _PRESSURE_TRACKERS
from .nest_move import SURFACE_TRACKER as _SURFACE_TRACKER

NEST_RESTART_SCHEMA = "gm1.cartesian-nest-restart/v1"
NEST_CHILD_RESTART_DIRECTORY = "nest-child"
NEST_CHILD_RESTART_PROFILE_ID = "gm1.cartesian-nest-child-restart/v1"

WRF_RESTART_AUTHORITY = MappingProxyType({
    "who_writes": (
        "share/mediation_integrate.F:347-354: only the parent initiates restart "
        "writing, so every domain is written at one model time"
    ),
    "per_domain_files": (
        "share/mediation_integrate.F:1157-1182 med_restart_out: this domain's own "
        "restart, then recursively each child's"
    ),
    "nest_read_on_restart": (
        "share/mediation_integrate.F:861-892 med_nest_initial: the nest reads its "
        "own restart (input_restart) instead of being interpolated from its parent"
    ),
    "moving_nest_state": (
        "Registry/Registry.EM_COMMON:438-441 vc_i, vc_j ('need for restarts of "
        "moving nests'); :2318-2319 i_parent_start, j_parent_start, rconfig "
        "(namelist) entries flagged for restart I/O that WRF updates on a move "
        "(share/mediation_nest_move.F:162-163,176-177), not state rows"
    ),
    "vortex_interval_boundary": (
        "share/mediation_integrate.F:897-913: WRF agrees with a straight-through "
        "run only when restarted on a vortex_interval boundary; GM1 carries the "
        "tracker state and derives the cadence from the absolute parent step, so "
        "it does not need that restriction"
    ),
})

NEST_REBUILT = MappingProxyType({
    "nest.lateral.tables": (
        "the Davies boundary value and tendency tables and dtbc: rebuilt from the "
        "parent's new state and the child's current state at the top of every "
        "parent step (med_nest_force, share/interp_fcn.F:2583-2584, "
        "share/mediation_force_domain.F:213) before any child step reads them"
    ),
    "nest.lateral.spec_snapshots": (
        "the specified-zone gathers: taken at the top of every acoustic small step "
        "before they are read"
    ),
    "nest.vertical.windows": (
        "a refined child deck's parent-resolution windows: refilled by every capture "
        "that reads them; the window placement is the carried parent_i0/parent_j0"
    ),
    "nest.vertical.weights_and_maps": (
        "fixed by the two vertical decks and the child's horizontally uniform "
        "rho0/rf0, rebuilt from the same sounding and grids"
    ),
    "nest.parcels.velocity": (
        "the child and parent advecting-velocity snapshots: captured at the next "
        "nrk == nrkmax (solve2.F:1290-1301) before the parcels are stepped"
    ),
    "nest.parcels.status": (
        "zero at every parent-step boundary (terminations are folded into the "
        "owner before it)"
    ),
})

#: The restart code of each tracker is its position here: the trackers
#: :mod:`cm1core.nest_move` declares, in declaration order.  A tracker added
#: there is appended, never reordered, so a restart written before it still
#: names the tracker it was written with; and a tracker without a code cannot
#: exist, so :func:`nest_restart_scalars` cannot find one at the first restart
#: (:meth:`CudaCartesianRuntime.integrate` prices the table before the first
#: step in any case).
TRACKER_CODES = (_SURFACE_TRACKER, *_PRESSURE_TRACKERS)

# ---------------------------------------------------------------- scalar names
PARENT_STEP = "nest.parent_step"
TIME_RATIO = "nest.time_refinement_ratio"
RATIO = "nest.refinement_ratio"
PARENT_I0 = "nest.parent_i0"
PARENT_J0 = "nest.parent_j0"
SPAN_X = "nest.parent_span_x"
SPAN_Y = "nest.parent_span_y"
SPEC_ZONE = "nest.spec_zone"
RELAX_ZONE = "nest.relax_zone"
FEEDBACK = "nest.feedback"
PARENT_STEPS_COUPLED = "nest.parent_steps_coupled"
CHILD_STEPS = "nest.child_steps"
FEEDBACK_WRITES = "nest.feedback_writes"
LATERAL_COUNTERS = {
    "refreshes": "nest.lateral.table_refreshes",
    "held_relaxations": "nest.lateral.held_relaxations",
    "acoustic_small_steps": "nest.lateral.acoustic_small_steps",
    "scalar_specifications": "nest.lateral.scalar_stage_specifications",
}
FOLLOW_TRACKER = "nest.follow.tracker_code"
FOLLOW_READY = "nest.follow.ready"
FOLLOW_VC_I = "nest.follow.vc_i"
FOLLOW_VC_J = "nest.follow.vc_j"
FOLLOW_PRESSURE = "nest.follow.pressure_centre"
FOLLOW_PRESSURE_I = "nest.follow.pressure_centre_i"
FOLLOW_PRESSURE_J = "nest.follow.pressure_centre_j"
FOLLOW_OFFSET = "nest.follow.held_offset"
FOLLOW_OFFSET_I = "nest.follow.held_offset_i"
FOLLOW_OFFSET_J = "nest.follow.held_offset_j"
FOLLOW_MOVES = "nest.follow.moves"
FOLLOW_TRACKS = "nest.follow.tracker_events"
FOLLOW_LAST_TRACK = "nest.follow.last_track"
FOLLOW_LAST_TRACK_STEP = "nest.follow.last_track.parent_step"
FOLLOW_LAST_TRACK_VALUED = "nest.follow.last_track.valued"
FOLLOW_LAST_TRACK_VALUE = "nest.follow.last_track.value"
FOLLOW_HOLD = "nest.follow.hold"
FOLLOW_HOLD_CONSECUTIVE = "nest.follow.hold.consecutive_held_samples"
FOLLOW_HOLD_CORRAL = "nest.follow.hold.consecutive_corral_parent_steps"
FOLLOW_HOLD_MAX_CONSECUTIVE = "nest.follow.hold.max_consecutive_held_samples"
FOLLOW_HOLD_MAX_CORRAL = "nest.follow.hold.max_consecutive_corral_parent_steps"
FOLLOW_HOLD_COUNT = "nest.follow.hold.counts.{event}"
PARCELS_N = "nest.parcels.nparcels"
PARCELS_CHILD_STEPS = "nest.parcels.child_steps"
PARCELS_PARENT_STEPS = "nest.parcels.parent_steps"
PARCELS_TO_PARENT = "nest.parcels.handoffs.child_to_parent"
PARCELS_TO_CHILD = "nest.parcels.handoffs.parent_to_child"
PARCELS_TERMINATED = "nest.parcels.terminations.status{code}"
PARCELS_SEEDED_STEP = "nest.parcels.seeded.parent_step"
PARCELS_SEEDED_TIME = "nest.parcels.seeded.model_time_s"
PARCELS_SEEDED_PLACED = "nest.parcels.seeded.placed_parcels"
PARCELS_SEEDED_I0 = "nest.parcels.seeded.parent_i0"
PARCELS_SEEDED_J0 = "nest.parcels.seeded.parent_j0"
PARCEL_POSITIONS = "nest.parcels.positions"
PARCEL_OWNER = "nest.parcels.owner"
SWATH_CARRIER = "nest.swath.{name}"

NEST_SCALAR_AUTHORITY = MappingProxyType({
    PARENT_I0: (
        "WRF Registry.EM_COMMON:2318 i_parent_start, an rconfig (namelist) entry WRF "
        "updates on a move (share/mediation_nest_move.F:162-163,176-177): the moved placement"
    ),
    PARENT_J0: (
        "WRF Registry.EM_COMMON:2319 j_parent_start, an rconfig (namelist) entry WRF "
        "updates on a move (share/mediation_nest_move.F:162-163,176-177): the moved placement"
    ),
    FOLLOW_VC_I: "WRF Registry.EM_COMMON:440 vc_i, in the child's corner (face) indices",
    FOLLOW_VC_J: "WRF Registry.EM_COMMON:441 vc_j, in the child's corner (face) indices",
    FOLLOW_PRESSURE: "cm1core.nest_move.TrackerState.pressure_centre (CM1 azimavg.F:4159-4163 face)",
    FOLLOW_OFFSET: "cm1core.nest_move.TrackerState.held_offset",
    FOLLOW_HOLD_CONSECUTIVE: (
        "cm1core.nest_move.MovingNest.consecutive_holds: sets the next sample's search "
        "reach (WRF's max_vortex_speed bound over the elapsed follow intervals) and "
        "whether it relocates (follow_hold_limit)"
    ),
    FOLLOW_HOLD_CORRAL: (
        "cm1core.nest_move.MovingNest.consecutive_corral: parent steps in a row whose "
        "move the corral cancelled (share/mediation_nest_move.F:849-859), against "
        "follow_hold_limit*interval_steps"
    ),
    PARCEL_POSITIONS: "cm1core.nest_parcels positions, ground-frame metres (parcel.F pdata x,y,z)",
    PARCEL_OWNER: "cm1core.nest_parcels owning grid: 0 terminated, 1 child, 2 parent",
})


def _f64_bits(value: float) -> str:
    return f"{struct.unpack('<Q', struct.pack('<d', float(value)))[0]:016x}"


def _from_f64_bits(text: str) -> float:
    return struct.unpack("<d", struct.pack("<Q", int(text, 16)))[0]


def _u64(value: int) -> tuple[str, str]:
    value = int(value)
    if value < 0:
        raise ValueError(f"a u64 restart scalar cannot hold {value}")
    return ("u64", str(value))


def _i64(value: int) -> tuple[str, str]:
    return ("i64", str(int(value)))


def _f64(value: float) -> tuple[str, str]:
    return ("f64-bits", _f64_bits(value))


def _bool(value: bool) -> tuple[str, str]:
    return ("bool", "true" if value else "false")


# ---------------------------------------------------------------- the profile
def child_swath_names(child_config: Any) -> tuple[str, ...]:
    """The swaths a child contract keeps, in ``solve3.F`` order."""

    selection = getattr(child_config, "diagnostics", None)
    return () if selection is None else tuple(selection.swath_names)


def nest_child_restart_profile(child_config: Any, *, nparcels: int | None = None) -> Any:
    """The child's own restart profile, plus its parcel and swath carriers; no CUDA.

    The child is a full CM1 grid (``NestBinding``), so its carried state is
    exactly what :func:`cm1core.cartesian_restart.cartesian_restart_profile`
    prices for that grid.  A nest child parcel set adds its positions and
    owning grid, each a whole float32 carrier; each swath the child keeps adds
    its surface plane (``restart_write.F:692-757``).
    """

    from .cartesian_history import MAX_FORECAST_PAYLOAD_BYTES
    from .cartesian_restart import (
        CartesianRestartCarrier,
        CartesianRestartProfile,
        cartesian_restart_profile,
    )
    from .forecast_arena import forecast_arena_nbytes

    base = cartesian_restart_profile(child_config, profile_id=NEST_CHILD_RESTART_PROFILE_ID)
    carriers = list(base.carriers)
    if nparcels is not None:
        n = int(nparcels)
        if n < 1:
            raise ValueError("a nest child parcel set carries at least one parcel")
        carriers += [
            CartesianRestartCarrier(
                PARCEL_POSITIONS,
                "nest child parcel positions x, y, z in ground-frame metres",
                "m", "parcel", (3, n), "nest", "parcels.positions",
            ),
            CartesianRestartCarrier(
                PARCEL_OWNER,
                "owning grid of each nest parcel: 0 terminated, 1 child, 2 parent",
                "1", "parcel", (n,), "nest", "parcels.owner",
            ),
        ]
    swaths = child_swath_names(child_config)
    if swaths:
        from .tornado_diagnostics import SWATH_MEANINGS, SWATH_UNITS

        surface = tuple(int(extent) for extent in child_config.grid.surface_shape)
        carriers += [
            CartesianRestartCarrier(
                SWATH_CARRIER.format(name=name),
                f"nest child run-long {SWATH_MEANINGS[name]} (restart_write.F:692-757)",
                SWATH_UNITS[name], "surface", surface, "nest", f"swath.{name}",
            )
            for name in swaths
        ]
    payload = forecast_arena_nbytes({c.name: c.shape for c in carriers})
    if payload > MAX_FORECAST_PAYLOAD_BYTES:
        raise ValueError(
            f"the nest child's carried state is {payload} bytes; the CM1FCP1 contract "
            f"caps one forecast payload at {MAX_FORECAST_PAYLOAD_BYTES} bytes, so the "
            "child has no writable restart and the nested run could never be continued"
        )
    return CartesianRestartProfile(
        NEST_CHILD_RESTART_PROFILE_ID, base.geometry, tuple(carriers), payload,
        MappingProxyType({**base.rebuilt, **NEST_REBUILT}),
    )


def driver_nparcels(driver: Any) -> int | None:
    parcels = getattr(driver, "parcels", None)
    return None if parcels is None else int(parcels.settings.nparcels)


# ---------------------------------------------------------------- the scalars
def nest_restart_scalars(driver: Any) -> dict[str, tuple[str, str]]:
    """The coupling state of one driver at a completed parent step, CM1FCP1-typed."""

    g = driver.geometry
    forcing = driver.forcing
    rows: dict[str, tuple[str, str]] = {
        PARENT_STEP: _u64(driver.parent.clock.completed_steps),
        TIME_RATIO: _u64(g.time_refinement_ratio),
        RATIO: _u64(g.refinement_ratio),
        PARENT_I0: _u64(g.parent_i0),
        PARENT_J0: _u64(g.parent_j0),
        SPAN_X: _u64(g.parent_span_x),
        SPAN_Y: _u64(g.parent_span_y),
        SPEC_ZONE: _u64(g.spec_zone),
        RELAX_ZONE: _u64(g.relax_zone),
        FEEDBACK: _u64(g.feedback),
        PARENT_STEPS_COUPLED: _u64(driver.parent_steps),
        CHILD_STEPS: _u64(driver.child_steps),
        FEEDBACK_WRITES: _u64(driver.feedback_writes),
    }
    for attribute, name in LATERAL_COUNTERS.items():
        rows[name] = _u64(getattr(forcing, attribute))
    mover = getattr(driver, "mover", None)
    if mover is not None:
        state = mover.state
        tracker = mover.settings.tracker
        if tracker not in TRACKER_CODES:
            raise ValueError(f"no restart code is assigned to nest tracker {tracker!r}")
        rows[FOLLOW_TRACKER] = _u64(TRACKER_CODES.index(tracker))
        rows[FOLLOW_READY] = _bool(state.ready)
        rows[FOLLOW_VC_I] = _f64(state.vc_i if state.vc_i is not None else 0.0)
        rows[FOLLOW_VC_J] = _f64(state.vc_j if state.vc_j is not None else 0.0)
        if (state.vc_i is None) != (state.vc_j is None):
            raise ValueError("the nest tracker holds one centre coordinate without the other")
        centre = state.pressure_centre
        rows[FOLLOW_PRESSURE] = _bool(centre is not None)
        rows[FOLLOW_PRESSURE_I] = _i64(0 if centre is None else centre[0])
        rows[FOLLOW_PRESSURE_J] = _i64(0 if centre is None else centre[1])
        offset = state.held_offset
        rows[FOLLOW_OFFSET] = _bool(offset is not None)
        rows[FOLLOW_OFFSET_I] = _f64(0.0 if offset is None else offset[0])
        rows[FOLLOW_OFFSET_J] = _f64(0.0 if offset is None else offset[1])
        # The whole run's counts: a resumed mover starts from the restart's.
        rows[FOLLOW_MOVES] = _u64(mover.move_count)
        rows[FOLLOW_TRACKS] = _u64(mover.tracker_event_count)
        # What a move decided from the carried centre reports as its track
        # (the surface tracker's area-mean vorticity; None for CM1's pressure).
        last = mover.last_track()
        value = None if last is None else last["value"]
        rows[FOLLOW_LAST_TRACK] = _bool(last is not None)
        rows[FOLLOW_LAST_TRACK_STEP] = _u64(0 if last is None else last["parent_step"])
        rows[FOLLOW_LAST_TRACK_VALUED] = _bool(value is not None)
        rows[FOLLOW_LAST_TRACK_VALUE] = _f64(0.0 if value is None else value)
        # nest.follow_on_edge = "hold": the state the next sample reads and
        # the whole run's counts.  A stop tracker writes only the flag.
        holds = bool(mover.settings.holds)
        rows[FOLLOW_HOLD] = _bool(holds)
        if holds:
            rows[FOLLOW_HOLD_CONSECUTIVE] = _u64(mover.consecutive_holds)
            rows[FOLLOW_HOLD_CORRAL] = _u64(mover.consecutive_corral)
            rows[FOLLOW_HOLD_MAX_CONSECUTIVE] = _u64(mover.whole_run_max_consecutive_holds)
            rows[FOLLOW_HOLD_MAX_CORRAL] = _u64(mover.whole_run_max_consecutive_corral)
            for event, count in mover.whole_run_hold_counts().items():
                rows[FOLLOW_HOLD_COUNT.format(event=event)] = _u64(count)
    parcels = getattr(driver, "parcels", None)
    if parcels is not None:
        from .nest_parcels import STATUS_REASONS

        rows[PARCELS_N] = _u64(parcels.settings.nparcels)
        rows[PARCELS_CHILD_STEPS] = _u64(parcels.child_steps)
        rows[PARCELS_PARENT_STEPS] = _u64(parcels.parent_steps)
        rows[PARCELS_TO_PARENT] = _u64(parcels.handoffs["child_to_parent"])
        rows[PARCELS_TO_CHILD] = _u64(parcels.handoffs["parent_to_child"])
        for code, reason in sorted(STATUS_REASONS.items()):
            key = reason.split(";")[0]
            rows[PARCELS_TERMINATED.format(code=code)] = _u64(parcels.terminations.get(key, 0))
        seeded = parcels.seeded
        rows[PARCELS_SEEDED_STEP] = _u64(seeded["parent_step"])
        rows[PARCELS_SEEDED_TIME] = _f64(seeded["model_time_s"])
        rows[PARCELS_SEEDED_PLACED] = _u64(seeded["placed_parcels"])
        rows[PARCELS_SEEDED_I0] = _u64(seeded["child_placement"]["parent_i0"])
        rows[PARCELS_SEEDED_J0] = _u64(seeded["child_placement"]["parent_j0"])
    criterion = getattr(driver, "criterion", None)
    if criterion is not None:
        rows.update(criterion.ledger.checkpoint())
    return rows


@dataclass(frozen=True, slots=True)
class NestRestartHold:
    """The ``hold`` tracker's state at the restart (nest_move, THE HOLD POLICY)."""

    consecutive_held_samples: int
    consecutive_corral_parent_steps: int
    max_consecutive_held_samples: int
    max_consecutive_corral_parent_steps: int
    counts: Mapping[str, int]

    def record(self) -> dict[str, Any]:
        return {
            "consecutive_held_samples": self.consecutive_held_samples,
            "consecutive_corral_parent_steps": self.consecutive_corral_parent_steps,
            "max_consecutive_held_samples": self.max_consecutive_held_samples,
            "max_consecutive_corral_parent_steps": self.max_consecutive_corral_parent_steps,
            "counts": dict(self.counts),
        }


@dataclass(frozen=True, slots=True)
class NestRestartFollow:
    tracker: str
    ready: bool
    vc_i: float | None
    vc_j: float | None
    pressure_centre: tuple[int, int] | None
    held_offset: tuple[float, float] | None
    moves: int
    tracker_events: int
    #: ``{"parent_step", "value"}`` of the last track before the restart, or None.
    last_track: Mapping[str, Any] | None = None
    #: The hold state, when the restart was written under ``follow_on_edge = "hold"``.
    hold: NestRestartHold | None = None

    def record(self) -> dict[str, Any]:
        return {
            "tracker": self.tracker, "ready": self.ready, "vc_i": self.vc_i,
            "vc_j": self.vc_j,
            "pressure_centre": None if self.pressure_centre is None else list(self.pressure_centre),
            "held_offset": None if self.held_offset is None else list(self.held_offset),
            "moves_before_restart": self.moves,
            "tracker_events_before_restart": self.tracker_events,
            "last_track": None if self.last_track is None else dict(self.last_track),
            "hold": None if self.hold is None else self.hold.record(),
        }


@dataclass(frozen=True, slots=True)
class NestRestartParcels:
    nparcels: int
    child_steps: int
    parent_steps: int
    child_to_parent: int
    parent_to_child: int
    terminations: Mapping[str, int]
    seeded: Mapping[str, Any]

    def record(self) -> dict[str, Any]:
        return {
            "nparcels": self.nparcels, "child_steps": self.child_steps,
            "parent_steps": self.parent_steps,
            "handoffs": {"child_to_parent": self.child_to_parent,
                         "parent_to_child": self.parent_to_child},
            "terminations": dict(self.terminations), "seeded": dict(self.seeded),
        }


@dataclass(frozen=True, slots=True)
class NestRestartState:
    """The coupling state one child restart carries, decoded on the host."""

    parent_step: int
    time_refinement_ratio: int
    refinement_ratio: int
    parent_i0: int
    parent_j0: int
    parent_span_x: int
    parent_span_y: int
    spec_zone: int
    relax_zone: int
    feedback: int
    parent_steps_coupled: int
    child_steps: int
    feedback_writes: int
    lateral: Mapping[str, int]
    follow: NestRestartFollow | None
    parcels: NestRestartParcels | None
    #: The tornado-criterion ledger's rows (``nest.tornado.*``), kept as
    #: written for :meth:`CriterionLedger.restore`; None when the nest that
    #: wrote the restart declared no criterion.
    criterion: Mapping[str, tuple[str, str]] | None = None

    @classmethod
    def from_scalars(cls, scalars: Mapping[str, tuple[str, str]]) -> "NestRestartState":
        def get(name: str, kind: str) -> str:
            try:
                found, value = scalars[name]
            except KeyError:
                raise ValueError(
                    f"the nest child restart carries no {name}; a nested resume without "
                    "it would continue a child whose coupling state it does not know"
                ) from None
            if found != kind:
                raise ValueError(f"nest restart scalar {name} is {found}, not {kind}")
            return value

        def u64(name: str) -> int:
            return int(get(name, "u64"))

        def i64(name: str) -> int:
            return int(get(name, "i64"))

        def f64(name: str) -> float:
            return _from_f64_bits(get(name, "f64-bits"))

        def boolean(name: str) -> bool:
            value = get(name, "bool")
            if value not in ("true", "false"):
                raise ValueError(f"nest restart scalar {name} is {value!r}, not a bool")
            return value == "true"

        follow = None
        if FOLLOW_TRACKER in scalars:
            code = u64(FOLLOW_TRACKER)
            if code >= len(TRACKER_CODES):
                raise ValueError(f"nest restart tracker code {code} names no tracker")
            ready = boolean(FOLLOW_READY)
            follow = NestRestartFollow(
                tracker=TRACKER_CODES[code],
                ready=ready,
                vc_i=f64(FOLLOW_VC_I) if ready else None,
                vc_j=f64(FOLLOW_VC_J) if ready else None,
                pressure_centre=(
                    (i64(FOLLOW_PRESSURE_I), i64(FOLLOW_PRESSURE_J))
                    if boolean(FOLLOW_PRESSURE) else None
                ),
                held_offset=(
                    (f64(FOLLOW_OFFSET_I), f64(FOLLOW_OFFSET_J))
                    if boolean(FOLLOW_OFFSET) else None
                ),
                moves=u64(FOLLOW_MOVES),
                tracker_events=u64(FOLLOW_TRACKS),
                last_track=(
                    MappingProxyType({
                        "parent_step": u64(FOLLOW_LAST_TRACK_STEP),
                        "value": f64(FOLLOW_LAST_TRACK_VALUE)
                        if boolean(FOLLOW_LAST_TRACK_VALUED) else None,
                    })
                    if boolean(FOLLOW_LAST_TRACK) else None
                ),
                # A restart written before the flag existed carries no hold
                # state, like one written under stop.
                hold=(
                    NestRestartHold(
                        consecutive_held_samples=u64(FOLLOW_HOLD_CONSECUTIVE),
                        consecutive_corral_parent_steps=u64(FOLLOW_HOLD_CORRAL),
                        max_consecutive_held_samples=u64(FOLLOW_HOLD_MAX_CONSECUTIVE),
                        max_consecutive_corral_parent_steps=u64(FOLLOW_HOLD_MAX_CORRAL),
                        counts=MappingProxyType({
                            event: u64(FOLLOW_HOLD_COUNT.format(event=event))
                            for event in _HOLD_EVENTS
                        }),
                    )
                    if FOLLOW_HOLD in scalars and boolean(FOLLOW_HOLD) else None
                ),
            )
        parcels = None
        if PARCELS_N in scalars:
            from .nest_parcels import STATUS_REASONS

            parcels = NestRestartParcels(
                nparcels=u64(PARCELS_N),
                child_steps=u64(PARCELS_CHILD_STEPS),
                parent_steps=u64(PARCELS_PARENT_STEPS),
                child_to_parent=u64(PARCELS_TO_PARENT),
                parent_to_child=u64(PARCELS_TO_CHILD),
                terminations=MappingProxyType({
                    reason.split(";")[0]: u64(PARCELS_TERMINATED.format(code=code))
                    for code, reason in sorted(STATUS_REASONS.items())
                }),
                seeded=MappingProxyType({
                    "model_time_s": f64(PARCELS_SEEDED_TIME),
                    "parent_step": u64(PARCELS_SEEDED_STEP),
                    "placed_parcels": u64(PARCELS_SEEDED_PLACED),
                    "nparcels": u64(PARCELS_N),
                    "child_placement_parent_cell": [u64(PARCELS_SEEDED_I0), u64(PARCELS_SEEDED_J0)],
                }),
            )
        from .nest_tornado_criterion import CHECKPOINT_PREFIX

        criterion = MappingProxyType({
            name: tuple(row) for name, row in scalars.items() if name.startswith(CHECKPOINT_PREFIX)
        })
        return cls(
            parent_step=u64(PARENT_STEP),
            time_refinement_ratio=u64(TIME_RATIO),
            refinement_ratio=u64(RATIO),
            parent_i0=u64(PARENT_I0),
            parent_j0=u64(PARENT_J0),
            parent_span_x=u64(SPAN_X),
            parent_span_y=u64(SPAN_Y),
            spec_zone=u64(SPEC_ZONE),
            relax_zone=u64(RELAX_ZONE),
            feedback=u64(FEEDBACK),
            parent_steps_coupled=u64(PARENT_STEPS_COUPLED),
            child_steps=u64(CHILD_STEPS),
            feedback_writes=u64(FEEDBACK_WRITES),
            lateral=MappingProxyType({key: u64(name) for key, name in LATERAL_COUNTERS.items()}),
            follow=follow,
            parcels=parcels,
            criterion=criterion or None,
        )

    def record(self) -> dict[str, Any]:
        return {
            "parent_step": self.parent_step,
            "time_refinement_ratio": self.time_refinement_ratio,
            "refinement_ratio": self.refinement_ratio,
            "placement": {"parent_i0": self.parent_i0, "parent_j0": self.parent_j0},
            "parent_span": [self.parent_span_x, self.parent_span_y],
            "zones": {"spec_zone": self.spec_zone, "relax_zone": self.relax_zone},
            "feedback": self.feedback,
            "parent_steps_coupled": self.parent_steps_coupled,
            "child_steps": self.child_steps,
            "feedback_writes": self.feedback_writes,
            "lateral_forcing_counters": dict(self.lateral),
            "storm_following": None if self.follow is None else self.follow.record(),
            "child_parcels": None if self.parcels is None else self.parcels.record(),
            "tornado_criterion_ledger_rows": 0 if self.criterion is None else len(self.criterion),
        }

    def geometry(self, declared: Any) -> Any:
        """The document's nest geometry at the restored placement, or a refusal.

        The refinement ratios and spans decide what every child word means --
        the child's grid and clock -- and a document that changed them would
        read the child restart onto another lattice.  The placement is the
        restart's: a following nest has moved since the document was written.
        """

        wrong = {
            name: (restored, declared_value)
            for name, restored, declared_value in (
                ("refinement_ratio", self.refinement_ratio, declared.refinement_ratio),
                ("time_refinement_ratio", self.time_refinement_ratio, declared.time_refinement_ratio),
                ("parent_span_x", self.parent_span_x, declared.parent_span_x),
                ("parent_span_y", self.parent_span_y, declared.parent_span_y),
            )
            if int(restored) != int(declared_value)
        }
        if wrong:
            raise ValueError(
                f"the nest child restart was written with {wrong} (restored, declared); "
                "the child's lattice and clock are those numbers, so the restored child "
                "words would be read onto another grid"
            )
        geometry = replace(declared, parent_i0=self.parent_i0, parent_j0=self.parent_j0)
        defect = geometry.placement_defect()
        if defect is not None:
            raise ValueError(f"the restored nest placement is inadmissible: {defect}")
        return geometry


# ---------------------------------------------------------------- write
def nest_child_declaration(driver: Any, profile: Any) -> dict[str, Any]:
    """What the parent's restart receipt says about the child restart beside it."""

    from .cartesian_restart import RECEIPT_FILENAME

    g = driver.geometry
    return {
        "schema": NEST_RESTART_SCHEMA,
        "directory": NEST_CHILD_RESTART_DIRECTORY,
        "receipt_filename": RECEIPT_FILENAME,
        "profile_id": profile.profile_id,
        "profile_sha256": profile.sha256,
        "parent_step": int(driver.parent.clock.completed_steps),
        "child_step": int(driver.child.clock.completed_steps),
        "time_refinement_ratio": int(g.time_refinement_ratio),
        "placement": {"parent_i0": int(g.parent_i0), "parent_j0": int(g.parent_j0)},
        # Filled in by ``bind_nest_child_declaration`` once the child's
        # envelope is on disk; a declaration without it names no publication.
        "child_identity": None,
        "wrf_authority": dict(WRF_RESTART_AUTHORITY),
    }


#: The child publication's identity in its parent's declaration: the digests
#: of the child's envelope header (Rust's per-carrier SHA-256 of every word it
#: carries, its contract, step and model time), the contract the envelope is
#: read back under, and its scalar table.  Two children at the same step from
#: the same profile -- another run's, or this run's before it diverged -- carry
#: other words, and so another header digest.
CHILD_IDENTITY_KEYS = ("envelope_header_sha256", "contract_sha256", "scalar_table_sha256")


def child_publication_identity(child_receipt: Mapping[str, Any]) -> dict[str, str]:
    """The identity a committed child restart's receipt records."""

    return {
        "envelope_header_sha256": str(child_receipt["arena"]["envelope_verification"]["header_sha256"]),
        "contract_sha256": str(child_receipt["forecast_envelope"]["contract_sha256"]),
        "scalar_table_sha256": str(child_receipt["scalar_table_sha256"]),
    }


def bind_nest_child_declaration(
    declaration: Mapping[str, Any], child_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """The declaration naming the child publication that was just committed."""

    return {**declaration, "child_identity": child_publication_identity(child_receipt)}


def _require_parcel_boundary(parcels: Any) -> None:
    if parcels._child_captured or parcels._parent_captured:
        raise RuntimeError(
            "a nest parcel velocity is captured but not yet stepped; a restart taken "
            "here would lose that step's parcel advance"
        )


def snapshot_nest_child_restart(
    driver: Any, directory: Path | str, *, publisher: Any, published_directory: Path | str,
    restart_number: int, next_history_index: int, host: Any = None,
) -> tuple[Any, dict[str, Any]]:
    """The child's half of a nested restart, off the device; and the parent's declaration.

    Taken on the stepping thread at the parent's quiescent boundary, after the
    parent step's child steps, feedback, move and parcel hand-off.
    """

    from .cartesian_restart import snapshot_cartesian_restart

    parent, child = driver.parent, driver.child
    q = int(driver.geometry.time_refinement_ratio)
    if int(child.clock.completed_steps) != int(parent.clock.completed_steps) * q:
        raise RuntimeError(
            f"the child stands at {int(child.clock.completed_steps)} steps and the parent "
            f"at {int(parent.clock.completed_steps)} with time ratio {q}; the two "
            "restarts would hold two model times"
        )
    profile = nest_child_restart_profile(child.config, nparcels=driver_nparcels(driver))
    extra: dict[str, Any] = {}
    parcels = getattr(driver, "parcels", None)
    if parcels is not None:
        _require_parcel_boundary(parcels)
        cp = driver.cp
        with driver.stream:
            extra[PARCEL_POSITIONS] = cp.ascontiguousarray(parcels.positions, dtype=cp.float32)
            extra[PARCEL_OWNER] = cp.ascontiguousarray(parcels.owner.astype(cp.float32))
    swaths = child_swath_names(child.config)
    if swaths:
        tornado = getattr(child, "tornado", None)
        if tornado is None or tornado.swaths is None:
            raise RuntimeError(
                "the nest child keeps surface swaths and none are open at this "
                "restart; a restart without them would resume the child's run-long "
                "extrema from CM1's sentinels and publish part of the run as the whole"
            )
        for name in swaths:
            extra[SWATH_CARRIER.format(name=name)] = tornado.swaths.frame[name]
    scalars = nest_restart_scalars(driver)
    state = NestRestartState.from_scalars(scalars)
    snapshot = snapshot_cartesian_restart(
        child, directory, publisher=publisher, profile=profile,
        restart_number=restart_number, next_history_index=next_history_index,
        published_directory=published_directory, host=host,
        extra_carriers=extra, extra_scalars=scalars,
    )
    snapshot = replace(snapshot, receipt_extra=MappingProxyType({
        "nest_coupling": {
            "schema": NEST_RESTART_SCHEMA,
            "state": state.record(),
            "scalar_authority": dict(NEST_SCALAR_AUTHORITY),
            "rebuilt_rather_than_carried": dict(NEST_REBUILT),
            "wrf_authority": dict(WRF_RESTART_AUTHORITY),
        },
    }))
    return snapshot, nest_child_declaration(driver, profile)


# ---------------------------------------------------------------- read
def nest_child_directory(parent_receipt: Mapping[str, Any], parent_directory: Path) -> Path | None:
    """The child restart a parent receipt declares, admitted CUDA-cold; ``None`` if none."""

    admitted = nest_child_receipt(parent_receipt, parent_directory)
    return None if admitted is None else admitted[0]


def nest_child_receipt(
    parent_receipt: Mapping[str, Any], parent_directory: Path,
) -> tuple[Path, Mapping[str, Any]] | None:
    """The declared child restart's directory and publication receipt, CUDA-cold.

    Only the receipts are read: a torn envelope is found when Rust decodes it
    (:func:`read_nest_child_restart`), on the host, after the device is open.
    """

    from .cartesian_restart import RECEIPT_FILENAME, require_restart_receipt

    declaration = parent_receipt.get("nest_child")
    if declaration is None:
        return None
    if not isinstance(declaration, Mapping) or declaration.get("schema") != NEST_RESTART_SCHEMA:
        raise ValueError(
            f"the restart declares a nest child under an unknown schema "
            f"{None if not isinstance(declaration, Mapping) else declaration.get('schema')!r}"
        )
    name = str(declaration.get("directory"))
    if name != NEST_CHILD_RESTART_DIRECTORY:
        raise ValueError(
            f"the restart names its nest child directory {name!r}, not "
            f"{NEST_CHILD_RESTART_DIRECTORY!r}; a child read from anywhere else could "
            "be another run's"
        )
    home = Path(parent_directory) / name
    receipt, _ = require_restart_receipt(home / RECEIPT_FILENAME)
    if str(receipt["profile_sha256"]) != str(declaration["profile_sha256"]):
        raise ValueError(
            f"the nest child restart in {home} prices profile {receipt['profile_sha256']} "
            f"and its parent declared {declaration['profile_sha256']}; it is not the "
            "child written with this parent"
        )
    if int(receipt["step"]) != int(declaration["child_step"]):
        raise ValueError(
            f"the nest child restart in {home} is at child step {receipt['step']} and "
            f"its parent declared {declaration['child_step']}"
        )
    if int(parent_receipt["step"]) != int(declaration["parent_step"]):
        raise ValueError("the parent restart's step is not the step its nest declaration names")
    identity = declaration.get("child_identity")
    if not isinstance(identity, Mapping):
        raise ValueError(
            "the parent restart's nest declaration names no child publication identity "
            "(child_identity); nothing then binds the child beside it to this parent"
        )
    recorded = child_publication_identity(receipt)
    wrong = sorted(key for key in CHILD_IDENTITY_KEYS if str(identity.get(key)) != recorded[key])
    if wrong:
        raise ValueError(
            f"the nest child restart in {home} is not the publication its parent declared: "
            f"its {wrong} differ from the declaration; a child of the same profile at the "
            "same step could be another run's, or this run's before it diverged"
        )
    coupling = receipt.get("nest_coupling")
    if not isinstance(coupling, Mapping) or coupling.get("schema") != NEST_RESTART_SCHEMA:
        raise ValueError(
            f"the nest child restart in {home} records no {NEST_RESTART_SCHEMA} coupling "
            "state; nothing then says which lattice, clock and placement its words belong to"
        )
    return home, receipt


#: Coupling numbers a resume may not change: the child's lattice and clock are
#: these numbers, so the restored words would be read onto another grid.
NEST_RESUME_FIXED = ("refinement_ratio", "time_refinement_ratio", "parent_span_x", "parent_span_y")
#: Coupling numbers a resume may change, as a following nest's tracker settings
#: may: they decide how the restored child is forced and fed back from here on,
#: not what its carried words mean.  Each change is recorded in the receipt.
NEST_RESUME_ADMITTED = ("spec_zone", "relax_zone", "feedback")


def _coupling_numbers(state: Mapping[str, Any]) -> dict[str, int]:
    span_x, span_y = state["parent_span"]
    zones = state["zones"]
    return {
        "refinement_ratio": int(state["refinement_ratio"]),
        "time_refinement_ratio": int(state["time_refinement_ratio"]),
        "parent_span_x": int(span_x), "parent_span_y": int(span_y),
        "spec_zone": int(zones["spec_zone"]), "relax_zone": int(zones["relax_zone"]),
        "feedback": int(state["feedback"]),
    }


def nest_document_changes(restart: Mapping[str, int], document: Any) -> dict[str, dict[str, int]]:
    """Every admitted coupling number the document changed, restart and document values."""

    return {
        name: {"restart": int(restart[name]), "document": int(getattr(document, name))}
        for name in NEST_RESUME_ADMITTED
        if int(restart[name]) != int(getattr(document, name))
    }


def nest_resume_differences(child_receipt: Mapping[str, Any], config: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """``(refused, admitted)``: how a child restart's coupling differs from the document.

    Read from the child's publication receipt (``nest_coupling.state``,
    written from the same scalars the envelope carries), so a resume that
    would read the child onto another lattice, drop or add parcels, or place
    it where the document's zones cannot hold it is refused before a device
    opens.  The envelope's own scalars are checked again when it is decoded
    (:meth:`NestRestartState.geometry`, :func:`restore_nest_child`).
    """

    from .nest_geometry import NestGeometry

    nest = config.nest
    state = child_receipt["nest_coupling"]["state"]
    restart = _coupling_numbers(state)
    refused: dict[str, Any] = {
        name: {"restart": restart[name], "document": int(getattr(nest, name))}
        for name in NEST_RESUME_FIXED
        if restart[name] != int(getattr(nest, name))
    }
    carried = state.get("child_parcels")
    lagrangian = getattr(config, "lagrangian", None)
    settings = None if lagrangian is None else getattr(lagrangian, "nest_child", None)
    carried_n = None if carried is None else int(carried["nparcels"])
    declared_n = None if settings is None else int(settings.nparcels)
    if carried_n != declared_n:
        refused["child_parcels"] = {"restart": carried_n, "document": declared_n}
    admitted = nest_document_changes(restart, nest)
    if not refused:
        placement = state["placement"]
        try:
            geometry = NestGeometry(
                parent_nx=config.grid.nx, parent_ny=config.grid.ny,
                parent_dx_m=config.grid.dx_m, parent_dy_m=config.grid.dy_m,
                refinement_ratio=nest.refinement_ratio,
                time_refinement_ratio=nest.time_refinement_ratio,
                parent_i0=int(placement["parent_i0"]), parent_j0=int(placement["parent_j0"]),
                parent_span_x=nest.parent_span_x, parent_span_y=nest.parent_span_y,
                spec_zone=nest.spec_zone, relax_zone=nest.relax_zone, feedback=nest.feedback,
            )
            defect = geometry.placement_defect()
        except (TypeError, ValueError) as error:
            defect = str(error)
        if defect is not None:
            refused["placement"] = {
                "restart": [int(placement["parent_i0"]), int(placement["parent_j0"])],
                "defect": defect,
            }
    return refused, admitted


def read_nest_child_restart(
    parent_receipt: Mapping[str, Any], parent_directory: Path, *, publisher: Any, into: Path,
) -> tuple[Any, NestRestartState] | None:
    """Read the declared child restart through Rust and decode its coupling state."""

    from .cartesian_restart import ENVELOPE_FILENAME, envelope_header_sha256, read_cartesian_restart

    admitted = nest_child_receipt(parent_receipt, parent_directory)
    if admitted is None:
        return None
    home, receipt = admitted
    # The receipts agree (nest_child_receipt); now the envelope on disk: its
    # header carries Rust's digest of every word it holds, and must be the
    # header the parent declared, not one swapped in beside a kept receipt.
    envelope = home / str(receipt.get("envelope_filename", ENVELOPE_FILENAME))
    declared = str(parent_receipt["nest_child"]["child_identity"]["envelope_header_sha256"])
    on_disk = envelope_header_sha256(envelope)
    if on_disk != declared:
        raise ValueError(
            f"the nest child envelope {envelope} has header SHA-256 {on_disk} and its parent "
            f"declared {declared}; it is not the envelope written with this parent"
        )
    restored = read_cartesian_restart(home, publisher=publisher, into=into)
    state = NestRestartState.from_scalars(restored.scalars)
    if state.parent_step != int(parent_receipt["step"]):
        raise ValueError(
            f"the nest child restart belongs to parent step {state.parent_step} and the "
            f"parent restart is at {int(parent_receipt['step'])}"
        )
    if restored.step != state.parent_step * state.time_refinement_ratio:
        raise ValueError(
            f"the nest child restart is at child step {restored.step}, not parent step "
            f"{state.parent_step} times the time ratio {state.time_refinement_ratio}"
        )
    return restored, state


# ---------------------------------------------------------------- restore
def _restore_hold_state(mover: Any, hold: "NestRestartHold | None") -> str:
    """Continue the ``hold`` tracker's state, or say why it is not continued."""

    if mover.settings.holds and hold is not None:
        mover.consecutive_holds = int(hold.consecutive_held_samples)
        mover.consecutive_corral = int(hold.consecutive_corral_parent_steps)
        mover.max_consecutive_holds_before_restart = int(hold.max_consecutive_held_samples)
        mover.max_consecutive_corral_before_restart = int(hold.max_consecutive_corral_parent_steps)
        mover.hold_counts_before_restart = {
            event: int(hold.counts.get(event, 0)) for event in _HOLD_EVENTS
        }
        return (
            f"restored: {hold.consecutive_held_samples} consecutive held sample(s) and "
            f"{hold.consecutive_corral_parent_steps} consecutive corral-cancelled parent "
            "step(s) at the restart; the hold-event counts continue from the restart's"
        )
    if mover.settings.holds:
        return (
            "started with no sample held: the restart was written under "
            "nest.follow_on_edge = \"stop\" (or before the policy existed), whose "
            "tracker never holds"
        )
    if hold is not None:
        return (
            "dropped: the document resumes under nest.follow_on_edge = \"stop\" (or "
            f"none), so the restart's {hold.consecutive_held_samples} consecutive held "
            "sample(s) are not continued and the next out-of-reach sample stops the run "
            "by name"
        )
    return "none: neither the restart nor the document holds"


def restore_nest_child(driver: Any, restored: Any, state: NestRestartState) -> dict[str, Any]:
    """``med_nest_initial`` on a restart: the child continues instead of being rebuilt.

    ``driver`` was built at the restored placement with its parcels unseeded
    (``NestDriver(restoring=True)``); the parent is already restored.  Every
    refusal happens before a device word is written.
    """

    import numpy

    from .cartesian_restart import (
        ASQ_LENGTH,
        ASQ_SCALAR_FORMAT,
        DOMAIN_LOCATION_X_NAME,
        DOMAIN_LOCATION_Y_NAME,
        MASS1_SCALAR_NAME,
        CLOSURE_STEP_NAME,
        PHYSICS_STEP_NAME,
        SOLVE3_STEP_NAME,
        require_cartesian_restart_boundary,
        require_resumable,
        restored_cadence_tokens,
        upload_restored_carriers,
    )

    parent, child, cp = driver.parent, driver.child, driver.cp
    g = driver.geometry
    if (int(g.parent_i0), int(g.parent_j0)) != (state.parent_i0, state.parent_j0):
        raise RuntimeError(
            "the nest driver was built at another placement than the restart's; the "
            "boundary tables and the feedback would read the wrong parent cells"
        )
    if int(parent.clock.completed_steps) != state.parent_step:
        raise ValueError(
            f"the parent stands at step {int(parent.clock.completed_steps)} and the nest "
            f"child restart belongs to parent step {state.parent_step}"
        )
    profile = nest_child_restart_profile(child.config, nparcels=driver_nparcels(driver))
    require_cartesian_restart_boundary(child)
    require_resumable(restored, child, profile)
    if (state.parcels is None) != (driver.parcels is None):
        raise ValueError(
            "this nest child restart "
            + ("carries" if state.parcels is not None else "carries no")
            + " child parcels and this document "
            + ("declares none" if driver.parcels is None else "declares [lagrangian.nest_child]")
            + "; parcels cannot be continued from a restart that never held them, "
            "and dropping them would end their tracks without a termination"
        )
    if state.parcels is not None and state.parcels.nparcels != int(driver.parcels.settings.nparcels):
        raise ValueError(
            f"the nest child restart carries {state.parcels.nparcels} parcels and the "
            f"document declares {int(driver.parcels.settings.nparcels)}"
        )
    tokens = restored_cadence_tokens(restored)
    if restored.step != int(child.clock.completed_steps):
        raise ValueError(
            f"the nest child restart is at child step {restored.step} and the child "
            f"stands at {int(child.clock.completed_steps)} on the restored parent's clock"
        )
    for name, live in (
        (DOMAIN_LOCATION_X_NAME, child.clock.domainlocx_m),
        (DOMAIN_LOCATION_Y_NAME, child.clock.domainlocy_m),
    ):
        persisted = restored.f64(name)
        if persisted != float(live):
            raise ValueError(
                f"the nest child restart holds {name}={persisted!r} and the restored "
                f"parent {float(live)!r}; a child's ground position is its parent's"
            )
    mass1 = restored.f64(MASS1_SCALAR_NAME)
    asq = [restored.f64(ASQ_SCALAR_FORMAT.format(index=index)) for index in range(ASQ_LENGTH)]
    extra: dict[str, Any] = {}
    parcels = driver.parcels
    if parcels is not None:
        n = int(parcels.settings.nparcels)
        with driver.stream:
            extra[PARCEL_POSITIONS] = cp.empty((3, n), dtype=cp.float32)
            extra[PARCEL_OWNER] = cp.empty((n,), dtype=cp.float32)
    swaths = child_swath_names(child.config)
    if swaths:
        surface = tuple(int(extent) for extent in child.config.grid.surface_shape)
        with driver.stream:
            for name in swaths:
                extra[SWATH_CARRIER.format(name=name)] = cp.empty(surface, dtype=cp.float32)
    restored_bytes = upload_restored_carriers(child, restored, profile, extra)
    if swaths:
        # Opened with the run's step count at the start of the integration
        # (NestDriver.open_child_swaths), which copies these in before the
        # first fold.
        driver.restored_swaths = {
            name: extra[SWATH_CARRIER.format(name=name)] for name in swaths
        }
    owner_host = None
    if parcels is not None:
        owner_host = numpy.asarray(cp.asnumpy(extra[PARCEL_OWNER]))
        if not bool(numpy.all(numpy.isin(owner_host, (0.0, 1.0, 2.0)))):
            raise ValueError(
                "the restored nest parcel owner row holds a code that names no grid; "
                "those parcels would be stepped on neither grid or on both"
            )

    child.finish.initial_mass1_f64 = mass1
    child.dynamics.mass1_f64 = mass1
    with driver.stream:
        child.finish.asq[...] = cp.asarray(numpy.asarray(asq, dtype="<f8"))
    child.finish.completed_physics_step = tokens[PHYSICS_STEP_NAME]
    child.finish.completed_solve3_step = tokens[SOLVE3_STEP_NAME]
    child.closure._held_step = tokens[CLOSURE_STEP_NAME]
    child.closure._pending_step = None
    child.closure._tke_stage = 3
    # src/cm1.F:2663 and src/turb.F:1460-1466, as the parent's restore does:
    # both domains resume with CM1's restarted flag.
    two_part = getattr(child.closure, "two_part", None)
    if two_part is not None:
        two_part.mark_resumed()

    driver.parent_steps = state.parent_steps_coupled
    driver.child_steps = state.child_steps
    driver.feedback_writes = state.feedback_writes
    for attribute, value in state.lateral.items():
        setattr(driver.forcing, attribute, int(value))

    follow_record: dict[str, Any] | None = None
    mover = driver.mover
    if mover is not None:
        from .nest_move import TrackerState

        follow = state.follow
        if follow is not None:
            # The nest's own history continues whatever tracks it next: the
            # counts the receipt and the next restart report are the whole run's.
            mover.moves_before_restart = follow.moves
            mover.tracker_events_before_restart = follow.tracker_events
            if follow.moves:
                # What the uninterrupted run's receipt says after its first move.
                mover.vertical_capture = getattr(driver.forcing, "vertical", None) is not None
        if follow is not None and follow.tracker == mover.settings.tracker:
            mover.state = TrackerState(
                vc_i=follow.vc_i, vc_j=follow.vc_j,
                pressure_centre=follow.pressure_centre,
                held_offset=follow.held_offset,
            )
            mover.carried_last_track = (
                None if follow.last_track is None else dict(follow.last_track)
            )
            follow_record = {
                "tracker_state": (
                    "restored" if follow.ready
                    else "restored before its first centre was tracked"
                ),
                **follow.record(),
                "hold_state": _restore_hold_state(mover, follow.hold),
            }
        else:
            # A following nest added at resume, or one whose tracker the document
            # changed: WRF's first time through (mediation_nest_move.F:471-476).
            follow_record = {
                "tracker_state": "started fresh at the restored placement",
                "restart_tracker": None if follow is None else follow.tracker,
                "document_tracker": mover.settings.tracker,
                "moves_before_restart": None if follow is None else follow.moves,
                "tracker_events_before_restart": None if follow is None else follow.tracker_events,
            }
    elif state.follow is not None:
        follow_record = {
            "tracker_state": "not continued: the document declares no following nest; "
                             "the nest stays at the restored placement",
            **state.follow.record(),
        }

    parcels_record: dict[str, Any] | None = None
    if parcels is not None:
        from .nest_parcels import OWNER_CHILD, OWNER_PARENT

        info = state.parcels
        assert info is not None
        with driver.stream:
            parcels.positions = extra[PARCEL_POSITIONS]
            parcels.owner = extra[PARCEL_OWNER].astype(cp.int32)
            parcels.status = cp.zeros((int(info.nparcels),), dtype=cp.int32)
        owner = owner_host
        parcels._owned = {OWNER_CHILD: int((owner == 1.0).sum()),
                          OWNER_PARENT: int((owner == 2.0).sum())}
        parcels.child_steps = info.child_steps
        parcels.parent_steps = info.parent_steps
        parcels.handoffs = {"child_to_parent": info.child_to_parent,
                            "parent_to_child": info.parent_to_child}
        parcels.terminations = {key: value for key, value in info.terminations.items() if value}
        seeded = dict(info.seeded)
        i0, j0 = seeded.pop("child_placement_parent_cell")
        seeded["child_placement"] = {"parent_i0": i0, "parent_j0": j0}
        seeded["restored_from_restart"] = True
        parcels.seeded = seeded
        parcels.relocate_child(g)
        parcels_record = info.record()

    criterion_record: dict[str, Any]
    criterion = getattr(driver, "criterion", None)
    if criterion is not None and state.criterion is not None:
        criterion_record = {
            "ledger": "restored: the verdict, episodes and counts continue the restart's",
            **criterion.restore_from_restart(state.criterion),
        }
    elif criterion is not None:
        criterion_record = {
            "ledger": (
                "started at the resume: the nest that wrote the restart declared no "
                "[nest.tornado_criterion], so no earlier sample is known and the verdict "
                "covers this run's samples only"
            ),
        }
    elif state.criterion is not None:
        criterion_record = {
            "ledger": (
                "not continued: the document declares no [nest.tornado_criterion]; the "
                "restart's ledger rows are kept in the restart and not read"
            ),
        }
    else:
        criterion_record = {"ledger": "none: neither the restart nor the document declares one"}

    driver.stream.synchronize()
    require_cartesian_restart_boundary(child)
    record = restored.record()
    record.update({
        "schema": NEST_RESTART_SCHEMA,
        "status": "RESTORED_NEST_CHILD_RESTART",
        "operator": "the child's own restart read word for word (med_nest_initial on restart)",
        "restored_carrier_bytes": restored_bytes,
        "mass1": mass1,
        "asq": asq,
        "coupling": state.record(),
        # Zones and feedback the document changed at resume (NEST_RESUME_ADMITTED):
        # from here on the child is forced and fed back by the document's.
        "document_changes": nest_document_changes(
            {"spec_zone": state.spec_zone, "relax_zone": state.relax_zone,
             "feedback": state.feedback}, g,
        ),
        "storm_following": follow_record,
        "child_parcels": parcels_record,
        "tornado_criterion": criterion_record,
        "child_swaths": (
            {"restored": list(swaths), "authority": "restart_read.F reads sws..shs back "
             "(restart_write.F:692-757); applied when the child's swaths open"}
            if swaths else None
        ),
        "rebuilt_rather_than_carried": dict(NEST_REBUILT),
        "wrf_authority": dict(WRF_RESTART_AUTHORITY),
    })
    return record


__all__ = [
    "CHILD_IDENTITY_KEYS",
    "NEST_CHILD_RESTART_DIRECTORY",
    "NEST_CHILD_RESTART_PROFILE_ID",
    "NEST_REBUILT",
    "NEST_RESTART_SCHEMA",
    "NestRestartState",
    "TRACKER_CODES",
    "WRF_RESTART_AUTHORITY",
    "bind_nest_child_declaration",
    "child_publication_identity",
    "nest_child_declaration",
    "nest_child_directory",
    "nest_child_receipt",
    "nest_document_changes",
    "nest_resume_differences",
    "nest_child_restart_profile",
    "nest_restart_scalars",
    "read_nest_child_restart",
    "restore_nest_child",
    "snapshot_nest_child_restart",
]
