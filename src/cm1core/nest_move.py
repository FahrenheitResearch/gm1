"""A moving live nest: track the surface vortex, re-centre a running child.

CM1 r21.1 has no nesting (``records/CM1-NEST-SEAM.md`` section 1) and its only
moving mesh moves the whole domain (``src/azimavg.F:4376-4528``).  Moving a
refined child inside a running parent is a declared GM1 extension and its
authority is WRF's moving nest.  Every rule below names the WRF line it
follows; where GM1 departs, the departure is stated with its reason.

WHERE THE MOVE HAPPENS (``frame/module_integrate.F:410-456``)
-----------------------------------------------------------
WRF: ``med_nest_force`` -> ``integrate(nest)`` -> ``med_nest_feedback`` ->
``med_nest_move``.  GM1's :class:`cm1core.nest_live.NestDriver` already does
the first three after each completed parent step; this module is the fourth.
The move therefore happens with parent and child at the same model time,
after the child's interior has been fed back from the **unmoved** footprint,
and the next parent step's ``med_nest_force`` rebuilds the boundary tables
from the moved child (``bdy_xs = nfld``) and the parent at the new placement.

THE TRACKER
-----------
``wrf-surface-circulation`` is WRF's vortex centre (``time_for_move2`` under
``VORTEX_CENTER``, ``share/mediation_nest_move.F:352-704``) run on the
**nest's own** fields, as WRF runs it, with the tracked field changed from a
height minimum at ``track_level`` to the area-averaged vertical vorticity at
the child's lowest scalar level:

* corner vorticity, CM1 ``src/parcel.F:1064-1066``
  (:func:`cm1core.vorticity_budget.corner_vertical_vorticity`);
* the mean over the ``(2h+1)^2`` corners of a box of half-width
  ``h = floor(R/dx)``, i.e. the circulation around the box divided by its
  area (``gm1_nest_zeta_box``).  ``R`` is ``nest.follow_smoothing_radius_m``.
  This is the tracker's robustness to short-lived gust-front sub-vortices: a
  sub-vortex whose circulation is small beside the box's contributes little
  to the box mean, however large its point vorticity.  A single-corner
  maximum is exactly what ``records/CM1-NEST-SEAM.md`` section 5.4 (commit
  dca8ab6) says a nest must never chase, and ``h >= 1`` is enforced;
* the search window is the previous centre plus or minus
  ``follow_max_speed_m_s * follow_interval_s / dx``
  (``mediation_nest_move.F:471-487``: ``vc_i -
  60*vortex_interval*max_vortex_speed/dx``), the whole admissible region the
  first time;
* the first strict maximum in ``DO j / DO i`` order (``:527-542``, mirrored);
* the centroid of ``zbar - vmin`` over the box of half-width ``R/dx`` around
  it (``:556-606`` with WRF's ``a = 0, b = 1``, mirrored from
  ``vmax - height``); if every weight is zero the centre is left where it was
  (``:590``).

Admissible centres keep the whole box off the child's specified and
relaxation rows, whose values are the parent's rather than the child's.

WRF adds ``0.5`` to its mass-point centroid to express it in the staggered
index frame its domain centre ``(ide-ids+1)/2`` is written in (``:602-603,
659-660``).  The corner lattice here already is that frame -- corner ``i``
lies on face ``i`` -- so the centre is compared with ``nx/2`` directly.

``cm1-pressure-held-offset`` and ``cm1-pressure-centre`` stay selectable: CM1's
own ``getpsmth``/``getpcent`` centre on the **parent** (``src/azimavg.F:4071-
4163``, :func:`cm1core.nest_geometry.device_pressure_centre_cell`), turned
into a target by :func:`cm1core.nest_geometry.follow_target_face`.

THE MOVE DECISION (every parent step)
-------------------------------------
``mediation_nest_move.F:659-704``: ``disp = vc - dc``;
``move = int(disp/parent_grid_ratio)`` clamped to ``[-1, 1]``;
``vc = vc - move*parent_grid_ratio``.  ``med_nest_move`` refuses any EM move
of more than one parent cell (``:122-127``), so a nest behind its vortex
catches up one parent cell per parent step.

One departure: WRF first clamps ``disp`` to ``irad = 6`` nest cells
(``:671-680``).  For ``parent_grid_ratio <= 6`` that clamp never changes the
result of ``int(disp/pgr)`` clamped to one cell; for a ratio of 7 or more it
makes ``int(disp/pgr)`` zero for every displacement, i.e. such a nest can
never move.  It is omitted, which is WRF's own answer for every ratio WRF's
clamp permits to move at all.

THE CORRAL (``mediation_nest_move.F:816-859``)
----------------------------------------------
WRF cancels a move that would leave the nest ``corral_dist`` parent cells or
closer to the outermost domain's boundary.  GM1 counts ``corral_cells``, the
minimum number of parent cells that must stay clear (WRF ``corral_dist = c``
is ``corral_cells = c + 1``), from ``nest.follow_corral_m`` or, absent, the
``PARENT_STENCIL_HALO`` every placement already keeps.  Where WRF prints "MOAD
can not move" and keeps integrating a nest that has stopped following, GM1
**stops** with ``CM1-CARTESIAN-NEST-FOLLOW-CORRAL``: a nest that silently
stays put while its vortex leaves is a receipt naming a storm-following nest
that was refining the air the storm left.

THE HOLD POLICY (``nest.follow_on_edge = "hold"``)
-------------------------------------------------
A running nested job cannot be restarted with its child, so a vortex that
leaves the tracker's reach for an interval -- occlusion, cyclic re-formation,
a sub-vortex orbiting the parent-scale circulation faster than
``follow_max_speed_m_s`` -- or a corral hit for a few steps would otherwise
end hours of integration.  ``hold`` is the surface tracker's bounded,
receipted opt-in.  Once a centre exists, every follow-interval sample is
classified before its centroid is accepted, in this order:

* ``lost-hold``: the largest box mean in the window is not positive (no
  cyclonic circulation within reach), or every centroid weight around it is
  zero (flat box means; WRF leaves the centre, ``:590``).  This is tested
  first: such a window's first strict maximum is its ``(is, js)`` corner or
  another arbitrary point, which may lie on an edge without any vortex having
  moved;
* ``edge-hold``: the maximum lies on an unclamped search-window edge, the
  sample ``stop`` refuses with ``CM1-CARTESIAN-NEST-FOLLOW-SPEED``.  WRF does
  not check the edge (``mediation_nest_move.F:482-485``) and takes the
  centroid there;
* ``too-fast-hold``: the maximum is inside the window but the centroid around
  it lies further than the search reach from the previous centre on an axis:
  the circulation the centroid weighs has moved beyond the declared speed.

The search reach is ``follow_max_speed_m_s * follow_interval_s`` times the
follow intervals since the last accepted centre, that is the held samples
since it plus one.  WRF bounds its window by ``max_vortex_speed`` over the
time since the previous centre (``:482-485``, "from previous location over
vortex_interval"), which is one interval only because WRF takes every
centroid.  A held sample keeps an older centre, so the next window, and the
too-fast test, widen by one interval's reach (measured in
``tests/test_nest_move.py`` section 12: with the window left at one
interval's reach, a 15 m/s vortex missed for one sample under a 20 m/s
reach sat on the window edge until a limit of 4 was spent, then relocated
with the tracked centre 898 m off, or with a 1 km relocation radius stopped
the run at 120 s).  Without ``hold`` no sample is held and the window is
WRF's.

Under ``hold`` every search window is one child cell wider than its reach
(``HOLD_WINDOW_SLACK_CELLS``; the too-fast test and the relocation radius
keep the reach itself).  WRF's window edges are ``INT`` of ``vc +- reach``
(``:482-485``) and the maximum is a corner, so on the side the vortex moves
toward the truncated edge lies up to one cell inside the reach and the
corner nearest a vortex within reach can be that edge: without the slack a
vortex at 19 or 19.5 m/s under a 20 m/s, 15 s reach on a 20 m child was
edge-held on 10 and 25 of 41 samples, with the nest parked up to 660 and 970
m off, and ``stop`` (which keeps WRF's edges bit for bit) refuses the 19 m/s
vortex at 45 s.  With the slack none of those samples is held and the
centre stays within 4 m (section 12).  A window the slack widens to the
admissible boundary is clamped there, as WRF clamps (``:482-485``), and a
maximum on a clamped edge is not an edge hit.

What the widened window guarantees, stated exactly: a vortex whose
displacement from the last **accepted** centre is within ``held + 1``
reaches on each axis is found again.  The displacement is measured from
that centre's centroid, not from the vortex, so a centroid that lagged the
vortex when it was accepted -- a box clipped by the admissible boundary, or
the plateau bias of a box many times the core (THE SMOOTHING RADIUS) --
is carried into every widened window, and at or over the declared speed the
window grows no faster than the vortex: expect alternating edge-holds near
the declared speed, and over it a relocation every ``follow_hold_limit + 1``
intervals or the run stopping by name (section 12: at 20.5 m/s, 2.5 per cent
over 20 m/s, the smallest admitted relocation radius of 1520 m stops the run
at 90 s while 2000 m and more survive ten minutes with 7 relocations and the
nest up to 2 km off; a longer hold limit does not help, since the widening
never outpaces the vortex).

A held sample takes no centroid, so ``vc`` keeps the last accepted centre and
the nest holds its position about it: it makes only the one-parent-cell moves
that centre already asked for (none once caught up) and none toward the held
sample.  The sample is listed with its parent step, model time, window,
maximum, value, centroid and a reason sentence, and its tracker row carries
``held = true``.  A sample that classifies clean resets the count: the vortex
is back within reach and following resumes.

When ``follow_hold_limit`` consecutive samples have been held and the next
would be held too, the tracker **relocates** instead: it searches the same
box-mean plane over the kept centre plus or minus
``follow_relocation_radius_m`` (clamped to the admissible centres, as WRF
clamps its window, ``:482-485``), which the configuration requires to reach
one child cell beyond that sample's own ``(follow_hold_limit + 1)``-interval
reach, so a relocation always searches further than the window whose sample
was just held; it takes WRF's first strict maximum and centroid there
(``:527-606``), and lists a ``relocation`` event naming the
triggering sample's reason, the window, the old and new centres and the
displacement; its tracker row carries ``relocated = true``.  The nest then
moves toward the new centre one parent cell per parent step (``:122-127,
659-704``): the child never jumps, and the tracked centre never changes
without a listed tracker row.

The run stops only when holding cannot continue safely:

* a window with no finite box mean stops with ``CM1-CARTESIAN-NEST-FOLLOW-
  LOST`` under ``hold`` exactly as under ``stop``.  The window is never empty
  (``vc`` stays inside the admissible centres and every admissible centre's
  box mean is written) and a finite value always wins the maximum, so nothing
  found means every box mean in reach is NaN: the child's lowest-level winds
  are non-finite, a failed integration no kept centre can recover.  WRF's own
  failure (``:544-548``) is likewise reachable only when no height in its
  window compares inside ``+-99999999.0`` (``:527-542``).  A centroid that is
  not finite -- a finite maximum whose centroid box holds a NaN box mean, so
  ``vmin``, the weight sum and the centroid are NaN (``:556-606``) -- stops
  with the same code under ``hold``, on the sample or on a relocation, before
  the centre is taken: the winds are non-finite one box from the maximum,
  and a NaN centre would end the run one parent step later without a name
  (``fortran_int`` in the move decision).  ``stop`` takes that centroid as
  WRF does, as it did before the key;
* a relocation whose window holds no positive box mean, only flat ones, or
  its strongest circulation on an unclamped edge stops with
  ``CM1-CARTESIAN-NEST-FOLLOW-RELOCATION``: no circulation lies within the
  declared relocation radius after ``follow_hold_limit + 1`` intervals, and
  holding on would publish a storm-following nest refining air its vortex
  left;
* a move into the corral is cancelled **per axis** as WRF cancels it (``:849-
  859``, quoted in ``records/CM1-NEST-SEAM.md`` 5.8: ``move_cd_x = 0`` when
  X moves and ``would_move_x`` is non-zero, and Y independently); the other
  axis still moves.  ``vc`` is not carried on a cancelled axis, which is
  WRF's ``vc_i = vc_i + move_cd_x * pgr`` undo.  More than
  ``follow_hold_limit * interval_steps`` consecutive parent steps with a
  cancelled axis stop the run with ``CM1-CARTESIAN-NEST-FOLLOW-CORRAL``: the
  parent domain ends there, and relocating the centre cannot move the nest
  past it.

Before the first centre there is nothing to hold: the first window is the
whole admissible region (it has no unclamped edge), and its sample is taken as
``stop`` takes it.

Every held sample, relocation, cancelled move and count is in the receipt:
held samples (one per follow interval at most) and corral holds (one per
parent step at most) are each listed up to ``HOLD_EVENT_CAP`` and counted
beyond it, and every relocation is listed.  A move row names the tracker
sample that set the centre it moved toward (``tracked_at_parent_step``,
``tracker_value``), never a held sample after it, and under ``hold`` counts
the held samples since (``tracker_held_samples``).

Absent or ``stop``, every path above is the one this module had before the
key existed, with one disclosed exception shared by both policies: a first
tracker sample whose centroid weights are all zero (a maximum is found but
``zbar - vmin`` is zero over its box, e.g. a child with uniform lowest-level
winds) leaves ``vc`` unset (``:590``), and its tracker row now records
``centre_m = null`` where the code before the key raised an unnamed
``TypeError`` building that row.  The tracker samples again at the next
parent step, as it does whenever no centre exists.

THE SMOOTHING RADIUS
--------------------
WRF's centroid half-width ``rsmooth = 100000.`` m (``:342,556``) is 25/9 of its
default search reach ``60 * vortex_interval * max_vortex_speed`` = 36 km
(``Registry.EM_COMMON:2354-2355``).  That ratio does not carry over: WRF takes
its extremum on the raw height field and only the centroid is wide, while
``R`` here sets both the box whose mean is maximised and the centroid.
Measured on the host (``tests/test_nest_move.py`` section 12, a 20 m child
with a 300 m reach; ``records/CM1-NEST-SEAM.md`` 5.8):

* a 30 m-core sub-vortex orbiting a 1 km-core circulation 350 m out at 21 m/s
  captures a 200 m box on the first sample (266 m from the parent core) and
  ``stop`` ends the run on the window edge at 75 s; ``R`` = 300, 600 and 840 m
  (WRF's ratio) follow the parent circulation within 191, 51 and 33 m;
* a translating 60 m-core vortex is centred within 3 m by ``R`` = 200 m and
  within 87 m by 300 m, while 600 and 840 m stop the run at the first follow
  sample: every box near the vortex holds its whole circulation, the box mean
  is a plateau, and its first strict maximum falls on the plateau's edge.

``R`` belongs between the orbit radius of the sub-vortices the nest must not
chase and a few core radii of the vortex it must centre.  That is a property
of the storm, not of the reach, so ``follow_smoothing_radius_m`` keeps no
default.  ``hold`` does not choose between them: on the reproduction with
``R`` = 200 m it keeps the run alive for ten minutes -- holding no sample,
since the maximum ``stop`` refused at 75 s lay 14.65 cells from the kept
centre, inside the 15-cell reach on the truncated window edge (THE HOLD
POLICY) -- but the tracked centre stays on the sub-vortex's box, 219-272 m
from the parent circulation, on every sample.  ``R`` is what follows the
parent circulation; ``hold`` is what keeps a run alive when a sample still
falls out of reach, including the 600 and 840 m plateau stop above (under
``hold`` the 60 m-core vortex is kept, 1 and 10 samples held, about 300 and
470 m off).

THE SHIFT AND THE FILL
----------------------
WRF shifts every state array (``dyn_em/shift_domain_em.F:52-80`` with
``frame/loop_based_{x,y}_shift_code.h``), marks the exposed strip in
``imask_*`` (``:64-72``), interpolates only that strip from the parent
(``med_interp_domain``, ``mediation_nest_move.F:188``) and copies the new time
level into the old one there (``:282-292``).  GM1, per carrier:

1. snapshot the unmoved words on the device;
2. ``gm1_nest_capture`` the parent onto the whole moved lattice (the same
   staggered operator that built the child's initial condition);
3. ``gm1_nest_shift`` the snapshot back over every point whose source lies
   inside the unmoved lattice.

What remains captured is exactly the exposed strip.  Both time levels (``*a``
and ``*3d``) are shifted from their own words and captured from their own
parent counterparts, which is WRF's level-2-to-level-1 copy.  Halos are then
refilled and ``prs``/``rho``/``rr``/``rf`` rebuilt by ``getset``, as after the
child's initial condition.

The child's run-long surface swaths (``tornado.swath.sws`` ..
``shs``, :func:`cm1core.nest_live.child_swath_selection`) follow the same
rule.  Over ground the child covered before and after the move the shifted
words are the child's own extremum, unchanged, so a swath read at a fixed
ground point never decreases across a move.  The exposed strip is ground the
child never covered: it takes the parent's swath of the same name through the
same capture operator -- the parent's own run-long extremum at the parent's
resolution, which WRF's ``med_interp_domain`` would give any exposed row.
CM1's alternative, ``init3d.F:181-201``'s sentinels (``sps = 200000`` Pa,
``svs = sus = -1000``), would publish those strips as a pressure or a
vorticity no model state ever held.  Both swaths are counted from the same
start: the child's and the parent's are opened together at the start of the
integration.  (After a nested resume the child's swath continues from its
restart while the parent's, which the parent restart does not carry, starts
again at the resume, so an exposed strip then holds the parent's extremum
since the resume.)

The carrier inventory is closed: every registry buffer the child holds is
named below as shifted, rebuilt, invariant, edge history or step workspace,
and a buffer that is none of these stops the run by name before any word
moves (``CM1-CARTESIAN-NEST-FOLLOW-CARRIER``).

FEEDBACK ACROSS A MOVE
----------------------
WRF feeds back before it moves and feeds back again only for
``input_from_hires`` (``:230``).  GM1 feeds back before the move (from the
unmoved footprint) and at the end of the next parent step (from the moved
one).  Parent cells that leave the covered region keep the last child block
means written into them and are integrated by the parent from then on;
parent cells that enter it are replaced, one parent step later, by block
means of a child whose words there began as the capture of those same parent
cells.  Blocks covered before and after the move carry bit-identical child
words (the shift copies them), so their fed-back values are bit-identical.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Callable

from .nest_geometry import (
    PARENT_STENCIL_HALO,
    NestGeometry,
    device_pressure_centre_cell,
    follow_target_face,
)
from .nest_live_kernels import lattice_shape
from .nest_move_kernels import (
    CENTROID_WORDS,
    NEST_MOVE_KERNEL_SHA256,
    fortran_int,
)

NEST_MOVE_SCHEMA = "gm1.cartesian-nest-move/v1"
CARTESIAN_GATE = "cartesian-runtime"

SURFACE_TRACKER = "wrf-surface-circulation"
PRESSURE_TRACKERS = ("cm1-pressure-held-offset", "cm1-pressure-centre")

WRF_MOVE_AUTHORITY = {
    "call_order": (
        "frame/module_integrate.F:410-456: med_nest_force -> integrate(nest) -> "
        "med_nest_feedback -> med_nest_move"
    ),
    "vortex_centre": (
        "share/mediation_nest_move.F:352-606 time_for_move2 (VORTEX_CENTER), run "
        "on the nest's own fields; search window :471-487, extremum :527-542, "
        "centroid :556-606 (a=0, b=1)"
    ),
    "move_decision": (
        "share/mediation_nest_move.F:659-704: move=int(disp/pgr) clamped to one "
        "parent cell, vc=vc-move*pgr; :122-127 refuses EM moves of more than one "
        "parent cell"
    ),
    "corral": "share/mediation_nest_move.F:816-859 (time_for_move, corral_dist)",
    "shift": (
        "dyn_em/shift_domain_em.F:52-80 with frame/loop_based_x_shift_code.h and "
        "frame/loop_based_y_shift_code.h; imask set at :64-72"
    ),
    "fill": (
        "share/mediation_nest_move.F:188 med_interp_domain over imask==1; "
        ":282-292 copies time level 2 into level 1 in the new region"
    ),
    "boundary_tables": (
        "rebuilt by the next med_nest_force (share/interp_fcn.F:2583-2584) from the "
        "moved child and the parent at the new placement"
    ),
    "namelist": (
        "Registry/Registry.EM_COMMON:2354-2358 vortex_interval, max_vortex_speed, "
        "corral_dist, track_level, time_to_move"
    ),
    "not_the_tracker": (
        "share/track_driver.F is WRF's point-track output, not a vortex tracker"
    ),
}

DEPARTURES_FROM_WRF = {
    "tracked_field": (
        "area-averaged lowest-level vertical vorticity (circulation per unit area, "
        "CM1 parcel.F:1064-1066 corner vorticity) instead of track_level "
        "geopotential height; the extremum and centroid are mirrored"
    ),
    "irad": (
        "WRF's hard-coded irad=6 nest-cell clamp (:671-680) is omitted: it never "
        "changes the move for parent_grid_ratio<=6 and makes a ratio>=7 nest "
        "immovable"
    ),
    "centre_frame": (
        "the corner lattice is already WRF's staggered centre frame, so WRF's +0.5 "
        "(:602-603) is not added"
    ),
    "corral": (
        "a corral-cancelled move stops the run by name instead of WRF's message "
        "and a nest that stops following"
    ),
    "accumulation": "centroid sums are float64 rather than WRF's default REAL",
}

#: ``nest.follow_on_edge``.  Absent is ``stop``.
EDGE_STOP = "stop"
EDGE_HOLD = "hold"
EDGE_POLICIES = (EDGE_STOP, EDGE_HOLD)

#: Under ``hold``, child cells every search window (the held search and the
#: relocation) reaches past its declared reach (module docstring, THE HOLD
#: POLICY).  ``stop`` keeps WRF's ``INT``-truncated edges (:482-485) bit for
#: bit.
HOLD_WINDOW_SLACK_CELLS = 1.0

#: Listed events of each capped kind (held samples; corral holds); later ones
#: of that kind are counted, not listed.  Relocations are always listed.
HOLD_EVENT_CAP = 4096
HOLD_EVENT_CAPPED_KINDS = ("held-sample", "corral-hold")

#: What ``hold`` changes in :data:`DEPARTURES_FROM_WRF`; merged into the
#: receipt only when the policy is ``hold``.
HOLD_DEPARTURES_FROM_WRF = {
    "corral": (
        "a corral-cancelled move is cancelled per axis as WRF cancels it "
        "(share/mediation_nest_move.F:849-859) and the nest keeps integrating, "
        "for at most follow_hold_limit*interval_steps consecutive parent steps; "
        "then the run stops by name"
    ),
    "search_window_edge": (
        "a maximum on an unclamped search-window edge, which WRF does not check "
        "(:482-485) and would take the centroid of, leaves the previous centre in "
        "place (edge-hold); the window's half-width is "
        "follow_max_speed_m_s*follow_interval_s times the follow intervals since "
        "the last accepted centre (WRF's max_vortex_speed bound over the time "
        "since its previous centre, which in WRF is always one interval) plus one "
        "child cell, so that a corner maximum within reach is not held on the "
        "INT-truncated edge; the too-fast test keeps the reach itself"
    ),
    "no_circulation": (
        "a window whose largest box mean is not positive, whose centroid WRF would "
        "take, or whose centroid weights are all zero, where WRF also leaves the "
        "centre (:590), is a held sample (lost-hold), classified before the "
        "window edge"
    ),
    "too_fast": (
        "a centroid further than the search reach (follow_max_speed_m_s*"
        "follow_interval_s times the follow intervals since the last accepted "
        "centre) from the previous centre on an axis, which WRF would take, "
        "leaves the previous centre in place (too-fast-hold)"
    ),
    "relocation": (
        "WRF has no relocation: after follow_hold_limit consecutive held samples "
        "the next one that would be held searches the kept centre plus or minus "
        "follow_relocation_radius_m instead (WRF's :482-487 window with a declared "
        "radius at least one child cell beyond that sample's "
        "(follow_hold_limit+1)-interval reach, then :527-606) and the nest moves "
        "toward that centroid one parent cell per parent step"
    ),
}

#: Held-sample events, then the other listed events, in receipt order.
HOLD_SAMPLE_EVENTS = ("edge-hold", "lost-hold", "too-fast-hold")
HOLD_EVENTS = (*HOLD_SAMPLE_EVENTS, "corral-hold", "relocation")


# ---------------------------------------------------------------------------
# Settings.
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class FollowSettings:
    tracker: str
    interval_steps: int
    interval_s: float
    max_speed_m_s: float
    smoothing_radius_m: float | None
    corral_cells_x: int
    corral_cells_y: int
    #: ``nest.follow_on_edge``; ``None`` is undeclared, which is ``stop`` and
    #: is not recorded, so a receipt written before the key keeps its bytes.
    on_edge: str | None = None
    #: ``nest.follow_hold_limit``; declared exactly when ``on_edge`` is ``hold``.
    hold_limit: int | None = None
    #: ``nest.follow_relocation_radius_m``; declared exactly when ``on_edge``
    #: is ``hold``.
    relocation_radius_m: float | None = None

    @property
    def holds(self) -> bool:
        return self.on_edge == EDGE_HOLD

    def record(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "tracker": self.tracker,
            "interval_parent_steps": self.interval_steps,
            "interval_s": self.interval_s,
            "max_speed_m_s": self.max_speed_m_s,
            "smoothing_radius_m": self.smoothing_radius_m,
            "corral_cells_x": self.corral_cells_x,
            "corral_cells_y": self.corral_cells_y,
        }
        if self.on_edge is not None:
            payload["on_edge"] = self.on_edge
            payload["hold_limit"] = self.hold_limit
            payload["relocation_radius_m"] = self.relocation_radius_m
        return payload


def corral_cells(corral_m: float | None, spacing_m: float) -> int:
    """Minimum parent cells kept clear on one axis; absent is the placement halo."""

    if corral_m is None:
        return PARENT_STENCIL_HALO
    return int(math.ceil(float(corral_m) / float(spacing_m)))


def surface_tracker_extent(*, n: int, width: int, h: int) -> tuple[int, int]:
    """Admissible box centres on one child axis, inclusive, in corner indices.

    Corner ``c`` reads the cells ``c-1`` and ``c``; both must lie outside the
    ``width = spec_zone + relax_zone`` boundary rows, and the box of half-width
    ``h`` around the centre must as well.
    """

    return width + 1 + h, n - 1 - width - h


def surface_tracker_defect(*, nx: int, ny: int, width: int, h: int) -> str | None:
    if h < 1:
        return (
            "the smoothing radius is less than one child cell, so the tracker "
            "would follow the vorticity of a single corner -- the instantaneous "
            "maximum that jumps to gust-front filaments (records/CM1-NEST-SEAM.md "
            "section 5.4)"
        )
    for axis, n in (("x", nx), ("y", ny)):
        lo, hi = surface_tracker_extent(n=n, width=width, h=h)
        if lo > hi:
            return (
                f"a box of half-width {h} child cells does not fit between the "
                f"child's {width}-row specified and relaxation zones on the {axis} "
                f"axis of {n} cells, so no admissible centre exists and the "
                "tracker could only read parent-forced rows"
            )
    return None


# ---------------------------------------------------------------------------
# The move decision and the corral.
# ---------------------------------------------------------------------------
def wrf_move_cells(vc: float, n_child: int, ratio: int) -> int:
    """``mediation_nest_move.F:662-688`` without ``irad``: one axis, parent cells."""

    disp = float(vc) - float(n_child) / 2.0
    move = fortran_int(disp / float(ratio))
    return max(-1, min(1, move))


def corral_defect(geometry: NestGeometry, *, move_i: int, move_j: int,
                  corral_x: int, corral_y: int) -> str | None:
    """``mediation_nest_move.F:833-859`` on the post-move footprint, per moving axis."""

    for axis, move, i0, span, n, corral in (
        ("x", move_i, geometry.parent_i0, geometry.parent_span_x, geometry.parent_nx, corral_x),
        ("y", move_j, geometry.parent_j0, geometry.parent_span_y, geometry.parent_ny, corral_y),
    ):
        if move == 0:
            continue
        low = i0 + move
        high = n - (i0 + move + span)
        if low < corral or high < corral:
            return (
                f"the tracked vortex asks the nest to move {move:+d} parent cell on "
                f"the {axis} axis, to parent cells {low}..{low + span - 1} of "
                f"0..{n - 1}, which leaves {min(low, high)} parent cells clear of "
                f"the parent's boundary where the corral keeps {corral}"
            )
    return None


# ---------------------------------------------------------------------------
# The tracker.
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class TrackerState:
    """WRF's ``grid%vc_i``/``vc_j``, in the child's corner (face) indices."""

    vc_i: float | None = None
    vc_j: float | None = None
    pressure_centre: tuple[int, int] | None = None
    held_offset: tuple[float, float] | None = None

    @property
    def ready(self) -> bool:
        return self.vc_i is not None and self.vc_j is not None


@dataclass(frozen=True, slots=True)
class SurfaceTrackerGeometry:
    nx: int
    ny: int
    dx_m: float
    dy_m: float
    width: int
    h: int
    d: float
    search_cells: float
    #: ``follow_relocation_radius_m`` in child cells; ``None`` unless ``hold``.
    relocation_cells: float | None = None
    #: Child cells each window reaches past its reach: :data:`HOLD_WINDOW_SLACK_CELLS`
    #: under ``hold``, zero otherwise (WRF's edges).
    slack_cells: float = 0.0

    @classmethod
    def build(cls, geometry: NestGeometry, settings: FollowSettings) -> "SurfaceTrackerGeometry":
        radius = float(settings.smoothing_radius_m)
        spacing = min(geometry.dx_m, geometry.dy_m)
        relocation = (
            float(settings.relocation_radius_m) / spacing
            if settings.holds and settings.relocation_radius_m is not None else None
        )
        return cls(
            nx=geometry.nx, ny=geometry.ny, dx_m=geometry.dx_m, dy_m=geometry.dy_m,
            width=geometry.spec_zone + geometry.relax_zone,
            h=int(math.floor(radius / spacing)), d=radius / spacing,
            search_cells=settings.max_speed_m_s * settings.interval_s / spacing,
            relocation_cells=relocation,
            slack_cells=float(HOLD_WINDOW_SLACK_CELLS) if settings.holds else 0.0,
        )

    @property
    def box(self) -> tuple[int, int, int, int]:
        """Corners the box may read: ``(lo_j, hi_j, lo_i, hi_i)``."""

        return self.width + 1, self.ny - 1 - self.width, self.width + 1, self.nx - 1 - self.width

    @property
    def centres(self) -> tuple[int, int, int, int]:
        """``(clo_i, chi_i, clo_j, chi_j)``."""

        lo_i, hi_i = surface_tracker_extent(n=self.nx, width=self.width, h=self.h)
        lo_j, hi_j = surface_tracker_extent(n=self.ny, width=self.width, h=self.h)
        return lo_i, hi_i, lo_j, hi_j

    def window(self, state: TrackerState, *, intervals: int = 1) -> tuple[int, int, int, int]:
        """``mediation_nest_move.F:471-487``: ``(is, ie, js, je)``.

        WRF bounds the search by ``max_vortex_speed`` over the time since the
        previous centre, which is one ``vortex_interval`` because WRF takes
        every centroid.  ``intervals`` is that elapsed time in follow intervals:
        one, except under ``hold`` after held samples, when it is the held
        samples plus one; under ``hold`` the window also reaches
        ``slack_cells`` past the reach (module docstring, THE HOLD POLICY).
        """

        clo_i, chi_i, clo_j, chi_j = self.centres
        if not state.ready:
            return clo_i, chi_i, clo_j, chi_j
        return self._reach(state, self.reach_cells(intervals) + self.slack_cells)

    def reach_cells(self, intervals: int = 1) -> float:
        """``follow_max_speed_m_s * follow_interval_s * intervals`` in child cells."""

        if int(intervals) < 1:
            raise ValueError("a search spans at least one follow interval")
        return self.search_cells * int(intervals)

    def relocation_window(self, state: TrackerState) -> tuple[int, int, int, int]:
        """The kept centre plus or minus ``follow_relocation_radius_m`` (plus the slack), clamped as :meth:`window`."""

        if self.relocation_cells is None or not state.ready:
            raise ValueError("a relocation needs a kept centre and a declared relocation radius")
        return self._reach(state, self.relocation_cells + self.slack_cells)

    def _reach(self, state: TrackerState, s: float) -> tuple[int, int, int, int]:
        clo_i, chi_i, clo_j, chi_j = self.centres
        return (
            fortran_int(max(state.vc_i - s, float(clo_i))),
            fortran_int(min(state.vc_i + s, float(chi_i))),
            fortran_int(max(state.vc_j - s, float(clo_j))),
            fortran_int(min(state.vc_j + s, float(chi_j))),
        )


def lowest_level_corner_vorticity(u_padded: Any, v_padded: Any, *, nx: int, ny: int,
                                  dx_m: float, dy_m: float, xp: Any) -> Any:
    """CM1 ``parcel.F:1064-1066`` at the lowest scalar level, ``(ny+1, nx+1)``."""

    from .vorticity_budget import corner_vertical_vorticity

    plane = corner_vertical_vorticity(u_padded, v_padded, nx=nx, ny=ny, nz=1,
                                      dx=dx_m, dy=dy_m, xp=xp)
    return xp.ascontiguousarray(plane[0])


def surface_tracker_reference(u_padded: Any, v_padded: Any, geo: SurfaceTrackerGeometry,
                              state: TrackerState) -> tuple[list[float], tuple[int, int, int, int]]:
    """The whole surface tracker on the host: vorticity, box mean, WRF centroid."""

    import numpy

    from .nest_move_kernels import reference_vortex_centroid, reference_zeta_box

    zeta = lowest_level_corner_vorticity(u_padded, v_padded, nx=geo.nx, ny=geo.ny,
                                         dx_m=geo.dx_m, dy_m=geo.dy_m, xp=numpy)
    lo_j, hi_j, lo_i, hi_i = geo.box
    zbar = reference_zeta_box(zeta, h=geo.h, lo_j=lo_j, hi_j=hi_j, lo_i=lo_i, hi_i=hi_i)
    window = geo.window(state)
    return reference_vortex_centroid(zbar, window=window, centres=geo.centres, d=geo.d), window


def window_edge_defect(words: list[float], window: tuple[int, int, int, int],
                       centres: tuple[int, int, int, int], *,
                       intervals: int = 1) -> str | None:
    """Name a maximum clipped by the search window rather than found inside it.

    WRF bounds the search by ``max_vortex_speed`` (``mediation_nest_move.F:
    482-485``) and does not check what it found.  A maximum on an edge of the
    window that is not also an edge of the admissible region means the
    strongest circulation within reach lies at the limit of reach: the vortex
    moved faster than ``follow_max_speed_m_s`` allows, and the centroid is the
    window edge's, not the vortex's.  ``intervals`` is the follow intervals the
    window spans (:meth:`SurfaceTrackerGeometry.window`); at one the sentence
    is the one ``stop`` has always written.
    """

    if words[0] != 1.0:
        return None
    im, jm = int(words[1]), int(words[2])
    is0, ie0, js0, je0 = window
    clo_i, chi_i, clo_j, chi_j = centres
    clipped = (
        (im == is0 and is0 > clo_i) or (im == ie0 and ie0 < chi_i)
        or (jm == js0 and js0 > clo_j) or (jm == je0 and je0 < chi_j)
    )
    if not clipped:
        return None
    elapsed = (
        "in one follow interval" if int(intervals) == 1
        else f"in the {int(intervals)} follow intervals since the last accepted centre"
    )
    return (
        f"the largest area-averaged vorticity within reach lies on the edge of the "
        f"search window {list(window)} at child face ({im}, {jm}); the vortex has "
        f"moved further {elapsed} than follow_max_speed_m_s permits, "
        "so the tracked centre would be the window's edge rather than the vortex"
    )


def hold_classification(words: list[float], window: tuple[int, int, int, int],
                        centres: tuple[int, int, int, int], state: TrackerState,
                        search_cells: float, *, intervals: int = 1) -> tuple[str, str] | None:
    """Under ``hold``, the event a sample with a kept centre is held for, with its reason.

    ``None`` accepts the centroid.  The caller has refused a window with no
    finite box mean (``words[0] != 1``) before asking.  ``search_cells`` is
    one follow interval's reach and ``intervals`` the follow intervals since
    the last accepted centre, which ``window`` spans.  Order: no circulation,
    then the window edge, then a centroid beyond reach (module docstring, THE
    HOLD POLICY): a window with no positive box mean has its first strict
    maximum at its ``(is, js)`` corner, which is an edge, and is not a vortex
    that moved too fast.
    """

    im, jm = int(words[1]), int(words[2])
    if not words[3] > 0.0:
        return "lost-hold", (
            f"the largest area-averaged vorticity in the search window {list(window)} "
            f"is {words[3]!r} s-1 at child face ({im}, {jm}): no cyclonic circulation "
            "lies within reach, so a centroid would follow no vortex"
        )
    if words[8] == 0.0:
        return "lost-hold", (
            f"every centroid weight around the maximum at child face ({im}, {jm}) in "
            f"the search window {list(window)} is zero: the box means are flat there, "
            "so no centre is defined (WRF leaves it, share/mediation_nest_move.F:590)"
        )
    clipped = window_edge_defect(words, window, centres, intervals=intervals)
    if clipped is not None:
        return "edge-hold", clipped
    assert state.vc_i is not None and state.vc_j is not None
    reach = search_cells * int(intervals)
    di, dj = words[9] - state.vc_i, words[10] - state.vc_j
    if abs(di) > reach or abs(dj) > reach:
        elapsed = (
            "in one interval" if int(intervals) == 1
            else f"in the {int(intervals)} intervals since the last accepted centre"
        )
        return "too-fast-hold", (
            f"the centroid around the maximum at child face ({im}, {jm}) lies at "
            f"({words[9]!r}, {words[10]!r}), ({di!r}, {dj!r}) child cells from the "
            f"previous centre, beyond the {reach!r} cells "
            f"follow_max_speed_m_s*follow_interval_s reaches {elapsed}: the "
            "circulation it weighs has moved further than the declared speed permits"
        )
    return None


def relocation_defect(words: list[float], window: tuple[int, int, int, int],
                      centres: tuple[int, int, int, int]) -> str | None:
    """Name a relocation search that found no circulation to relocate to."""

    if words[0] != 1.0:
        return (
            f"the relocation window {list(window)} holds no finite box-mean vorticity"
        )
    im, jm = int(words[1]), int(words[2])
    if not words[3] > 0.0:
        return (
            f"the largest area-averaged vorticity in the relocation window "
            f"{list(window)} is {words[3]!r} s-1 at child face ({im}, {jm}): no "
            "cyclonic circulation lies within follow_relocation_radius_m"
        )
    if words[8] == 0.0:
        return (
            f"every centroid weight around the maximum at child face ({im}, {jm}) in "
            f"the relocation window {list(window)} is zero, so no centre is defined"
        )
    clipped = window_edge_defect(words, window, centres)
    if clipped is not None:
        return (
            f"the largest area-averaged vorticity within follow_relocation_radius_m "
            f"lies on the edge of the relocation window {list(window)} at child face "
            f"({im}, {jm}): the strongest circulation is at or beyond the declared "
            "radius, so the relocated centre would be the window's edge"
        )
    return None


def centroid_is_finite(words: list[float]) -> bool:
    """Whether a found sample's weight sum and centroid are finite.

    A NaN box mean anywhere in the centroid box makes ``vmin`` or a weight NaN
    and so the sum and the centroid (``mediation_nest_move.F:556-606``);
    :func:`apply_centroid` would then set a NaN centre.  ``hold`` refuses such
    a sample by name instead (module docstring, THE HOLD POLICY); ``stop``
    keeps WRF's path.
    """

    return all(math.isfinite(words[k]) for k in (8, 9, 10))


def apply_centroid(state: TrackerState, words: list[float]) -> bool:
    """``mediation_nest_move.F:590-606``: move ``vc`` only when the weights are not all zero."""

    if words[0] != 1.0:
        return False
    if words[8] != 0.0:
        state.vc_i = float(words[9])
        state.vc_j = float(words[10])
    return True


# ---------------------------------------------------------------------------
# The carrier inventory.
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class CarrierRule:
    """One shifted child carrier: its lattice and its owner layout."""

    name: str
    stagger: str
    origin: tuple[int, int, int]
    plane: bool = False
    species: bool = False


_PADDED_WIND = {"ua": "u", "u3d": "u", "va": "v", "v3d": "v", "wa": "w", "w3d": "w",
                "rru": "u", "rrv": "v", "rrw": "w"}
_PADDED_PRESSURE = ("ppi", "pp3d", "ppx", "ppten")
_FIELD_MASS = ("ppi", "pp3d", "ppx", "tha", "th3d", "qpten", "qtten", "qvten", "qcten")
_FIELD_SPECIES = ("qa", "q3d")
_HELD = {"uten": "u", "vten": "v", "wten": "w", "thten1": "mass"}
_TKE = ("tkea", "tke3d", "tketen")
_SURFACE_STATE = ("znt", "xland", "mavail", "tsk")
_SWATHS = ("sws", "svs", "sps", "srs", "sgs", "sus", "shs")
#: The live producer of the swaths (cartesian_tornado_diagnostics) registers
#: its planes as ``tornado.swath.<name>``; a nest child keeps no ground twin
#: (``tornado.swath.<name>_ground``), so none has a rule and one would stop
#: the move by name.
_TORNADO_SWATHS = tuple(f"tornado.swath.{name}" for name in _SWATHS)

#: Rebuilt after the move, or before their next read, from shifted state.
REBUILT = {
    **{f"field.{n}": "init3d.F getset, relaunched after the move from the shifted pp3d/th3d/q3d"
       for n in ("prs", "rho", "rr", "rf")},
    **{f"output.{n}": "recomputed from the moved winds and scalars before it is published"
       for n in ("zeta", "uh", "dbz", "cref")},
    **{f"physics.reflectivity.{n}": "recomputed on the step whose frame publishes it (cm1.F:3214-3237)"
       for n in ("dbz", "cref", "dbz_level")},
    "tornado.uh": (
        "calcuh's plane, recomputed from the moved winds before shs folds it "
        "(solve3.F:1119) and before a frame publishes it"
    ),
}

#: The child's per-step swath series: one word per completed child step, a
#: record of each step's extremum on the lattice the step ran on, not a
#: horizontal field, so a move leaves every word as it was.
SWATH_SERIES_PREFIX = "tornado.swath.series."

#: Child-edge boundary histories, indexed along the child's own edges.
EDGE_HISTORY = {
    f"padded.radbc{s}": (
        "the child's own open-halo phase speed along its own edge; rebuilt from the "
        "A-level winds before use for irbc=1 (solve1.F:402-409) and irbc=2 "
        "(sound.F:260-268); for irbc=4 re-produced after the next RK pressure stage "
        "(solve2.F:1258-1271), so that step's first-stage halo fill reads the "
        "pre-move edge history on rows the specified zone overwrites"
    )
    for s in ("w", "e", "s", "n")
}

INVARIANT_PREFIXES = {
    "metric.": "horizontally uniform grid metrics (verified on the device before the first move)",
    "metric-profile.": "compact vertical column profiles (zh, zf) the metric map broadcasts; field(profile) proved every horizontal plane equal before upload",
    "base.": "base-state column profiles",
    "closure.vertical.": "vertical grid column",
    "closure.rayleigh.": "Rayleigh-damping column profiles",
    # solve1.F:590-612 on a horizontally uniform base state: one word per
    # level, built from the child's own u0/v0 and the fixed frame.
    "closure.lspgrad.": "lspgrad=1 base-state Coriolis column profiles",
    "initial.": "startup dry-mass scratch",
}
INVARIANT_NAMES = {
    **{f"field.{n}": "base-state volume, horizontally uniform on this lane" for n in ("pi0", "th0", "qv0")},
    **{f"padded.{n}": "CM1 param.F uniform-horizontal grid coefficient, exactly 1" for n in ("uh", "vh", "ruf", "rvf")},
    "physics.asq": "per-species positive-definite repair scalars, not a horizontal field",
    "tornado.prs0": "CM1's base-state pressure column sps differences against (solve3.F:977)",
}

#: Buffers that do not carry a word across a completed large step.  The
#: restart profile (cartesian_restart.cartesian_restart_profile) carries none
#: of them, and restart identity rests on the same fact.
WORKSPACE_PREFIXES = (
    "acoustic.", "scalar.", "bridge.", "boundary.", "lateral.", "source.",
    "closure.held.", "closure.tke-stage.", "physics.block.", "physics.aux.",
    "physics.rate-work", "output.getcape.", "tornado.transient.",
)


def padded_workspace_names() -> frozenset[str]:
    """Padded acoustic/pre-acoustic workspaces: every padded owner that is not persistent.

    ``CartesianDynamics._ensure_arrays`` registers any name of the acoustic
    shape table on demand; the persistent ones are exactly
    ``_PERSISTENT_PADDED``, which is what the restart profile carries.
    """

    from .cuda_cartesian_dynamics import (
        COLUMN_DIAGNOSTIC_NAMES,
        _PERSISTENT_PADDED,
        terrain_open_sound_shapes,
    )

    names = set(terrain_open_sound_shapes(4, 4, 4)) | set(COLUMN_DIAGNOSTIC_NAMES)
    return frozenset(f"padded.{n}" for n in names - set(_PERSISTENT_PADDED))


def shifted_rule(name: str, shape: tuple[int, ...], *, grid: Any) -> CarrierRule | None:
    """The shift rule for one registry name, or ``None`` if it is not shifted."""

    group, _, key = name.partition(".")
    if group == "padded" and key in _PADDED_WIND:
        return CarrierRule(name, _PADDED_WIND[key], (1, 3, 3))
    if group == "padded" and key in _PADDED_PRESSURE:
        return CarrierRule(name, "mass", (1, 1, 1))
    if group == "field" and key in _FIELD_MASS:
        return CarrierRule(name, "mass", (0, 0, 0))
    if group == "field" and key in _FIELD_SPECIES:
        return CarrierRule(name, "mass", (0, 0, 0), species=True)
    if group == "held" and key in _HELD:
        return CarrierRule(name, _HELD[key], (0, 0, 0))
    if group == "held" and key == "qten":
        return CarrierRule(name, "mass", (0, 0, 0), species=True)
    if group == "tke" and key in _TKE:
        return CarrierRule(name, "w", (0, 0, 0))
    if name.startswith("closure.surface.") and name.removeprefix("closure.surface.") in _SURFACE_STATE:
        return CarrierRule(name, "mass", (0, 0, 0), plane=True)
    if group == "output" and key in _SWATHS:
        return CarrierRule(name, "mass", (0, 0, 0), plane=True)
    if name in _TORNADO_SWATHS:
        # The child's run-long extremum moves with the child: the words over
        # ground the child still covers are shifted unchanged, and the exposed
        # strip takes the parent's own swath of the same name, interpolated
        # like every other exposed row (THE SHIFT AND THE FILL).
        return CarrierRule(name, "mass", (0, 0, 0), plane=True)
    if group == "physics" and name not in INVARIANT_NAMES and not name.startswith(
        ("physics.reflectivity.", "physics.block.", "physics.aux.", "physics.rate-work")
    ):
        # The microphysics component's persistence (WSM6: h_diabatic, the
        # effective radii and the surface precipitation accumulators): each is
        # a mass volume or a surface plane.
        if tuple(shape) == tuple(grid.surface_shape):
            return CarrierRule(name, "mass", (0, 0, 0), plane=True)
        if tuple(shape) == tuple(grid.mass_shape):
            return CarrierRule(name, "mass", (0, 0, 0))
    return None


def classify_carriers(names_and_shapes: dict[str, tuple[int, ...]], *, grid: Any
                      ) -> tuple[list[CarrierRule], dict[str, str], list[str]]:
    """Split a child's registry into shifted rules, named non-shifts and unknowns."""

    shifted: list[CarrierRule] = []
    kept: dict[str, str] = {}
    unknown: list[str] = []
    workspace_names = padded_workspace_names()
    for name, shape in sorted(names_and_shapes.items()):
        rule = shifted_rule(name, shape, grid=grid)
        if rule is not None:
            shifted.append(rule)
        elif name in REBUILT:
            kept[name] = "rebuilt: " + REBUILT[name]
        elif name in EDGE_HISTORY:
            kept[name] = "edge history: " + EDGE_HISTORY[name]
        elif name in INVARIANT_NAMES:
            kept[name] = "invariant: " + INVARIANT_NAMES[name]
        elif any(name.startswith(p) for p in INVARIANT_PREFIXES):
            prefix = next(p for p in INVARIANT_PREFIXES if name.startswith(p))
            kept[name] = "invariant: " + INVARIANT_PREFIXES[prefix]
        elif name in workspace_names or name.startswith(WORKSPACE_PREFIXES):
            kept[name] = "step workspace: carries no word across a completed large step"
        elif name.startswith(SWATH_SERIES_PREFIX):
            kept[name] = (
                "per-step series: one word per completed child step, recorded on the "
                "lattice that step ran on"
            )
        else:
            unknown.append(name)
    return shifted, kept, unknown


def rule_lattice(rule: CarrierRule, *, nz: int, ny: int, nx: int) -> tuple[int, int, int]:
    if rule.plane:
        return (1, ny, nx)
    return lattice_shape(rule.stagger, nz, ny, nx)


def rule_owners(rule: CarrierRule, buffer: Any, *, numq: int) -> list[Any]:
    """The contiguous 3-D owners one registry buffer holds (one per species)."""

    if rule.plane:
        return [buffer.reshape(1, *buffer.shape)]
    if rule.species:
        if int(buffer.shape[0]) != int(numq):
            raise ValueError(f"{rule.name} carries {buffer.shape[0]} species where the child has {numq}")
        return [buffer[s] for s in range(int(numq))]
    return [buffer]


# ---------------------------------------------------------------------------
# The executor.
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class MovingNest:
    """Track, decide and move, once per completed parent step."""

    settings: FollowSettings
    cupy: Any
    stream: Any
    refuse: Callable[..., Any]
    state: TrackerState = field(default_factory=TrackerState)
    moves: list[dict[str, Any]] = field(default_factory=list)
    tracks: list[dict[str, Any]] = field(default_factory=list)
    kept_carriers: dict[str, str] = field(default_factory=dict)
    shifted_carriers: list[str] = field(default_factory=list)
    module: Any = None
    tracker_module: Any = None
    surface: SurfaceTrackerGeometry | None = None
    uniform_metrics_checked: bool = False
    vertical_capture: bool = False
    # nest.follow_on_edge = "hold" (module docstring, THE HOLD POLICY).
    holds: list[dict[str, Any]] = field(default_factory=list)
    holds_not_listed: int = 0
    holds_listed: dict[str, int] = field(
        default_factory=lambda: dict.fromkeys(HOLD_EVENT_CAPPED_KINDS, 0))
    hold_counts: dict[str, int] = field(default_factory=lambda: dict.fromkeys(HOLD_EVENTS, 0))
    consecutive_holds: int = 0
    consecutive_corral: int = 0
    max_consecutive_holds: int = 0
    max_consecutive_corral: int = 0
    #: A resumed nest's history before its restart (cm1core.nest_restart):
    #: the moves and tracker events already made, and the last track's parent
    #: step and value, which a move decided from the carried centre reports
    #: before this run tracks again.  Zero and ``None`` for a run that did not
    #: resume a following nest.
    moves_before_restart: int = 0
    tracker_events_before_restart: int = 0
    carried_last_track: dict[str, Any] | None = None
    #: The same for ``hold``: the hold events counted before the restart and
    #: the longest runs seen, whole-run in the receipt; ``consecutive_holds``
    #: and ``consecutive_corral`` themselves are restored in place, since the
    #: search reach, the relocation trigger and the corral limit read them.
    hold_counts_before_restart: dict[str, int] = field(
        default_factory=lambda: dict.fromkeys(HOLD_EVENTS, 0))
    max_consecutive_holds_before_restart: int = 0
    max_consecutive_corral_before_restart: int = 0

    # -- construction ---------------------------------------------------------
    @classmethod
    def build(cls, *, settings: FollowSettings, driver: Any, cupy: Any, stream: Any) -> "MovingNest":
        from .errors import refuse

        mover = cls(settings=settings, cupy=cupy, stream=stream, refuse=refuse)
        if settings.tracker == SURFACE_TRACKER:
            mover.surface = SurfaceTrackerGeometry.build(driver.geometry, settings)
        elif settings.tracker not in PRESSURE_TRACKERS:
            refuse(
                code="CM1-CARTESIAN-NEST-FOLLOW-TRACKER",
                gate=CARTESIAN_GATE,
                reason=f"no live tracker is implemented for {settings.tracker!r}",
                remedy="declare one of " + ", ".join((SURFACE_TRACKER, *PRESSURE_TRACKERS)),
                option="nest.follow_tracker", value=settings.tracker,
            )
        mover.inventory(driver)
        from .nest_move_kernels import compile_nest_move_module

        mover.module = compile_nest_move_module(cupy)
        return mover

    def inventory(self, driver: Any) -> list[CarrierRule]:
        child = driver.child
        registry = child.registry
        shapes = {name: tuple(int(v) for v in value.shape) for name, value in registry.buffers.items()}
        shifted, kept, unknown = classify_carriers(shapes, grid=child.config.grid)
        if unknown:
            self.refuse(
                code="CM1-CARTESIAN-NEST-FOLLOW-CARRIER",
                gate=CARTESIAN_GATE,
                reason=(
                    "the running child holds device buffers the moving nest has no "
                    "rule for: " + ", ".join(unknown) + ".  Moving the child without "
                    "them would leave those words at the old ground position under "
                    "a child that moved"
                ),
                remedy=(
                    "name each buffer in cm1core.nest_move as shifted, rebuilt, "
                    "invariant, edge history or step workspace, or set "
                    "nest.follow_storm = false"
                ),
                option="nest.follow_storm", value=unknown,
            )
        parent_names = set(driver.parent.registry.buffers)
        missing = [rule.name for rule in shifted if rule.name not in parent_names]
        if missing:
            self.refuse(
                code="CM1-CARTESIAN-NEST-FOLLOW-CARRIER",
                gate=CARTESIAN_GATE,
                reason=(
                    "the exposed strip of these child carriers would be filled from "
                    "a parent carrier the parent does not hold: " + ", ".join(missing)
                ),
                remedy="set nest.follow_storm = false",
                option="nest.follow_storm", value=missing,
            )
        self.kept_carriers = kept
        self.shifted_carriers = [rule.name for rule in shifted]
        return shifted

    # -- one parent step --------------------------------------------------------
    def after_parent_step(self, driver: Any) -> dict[str, Any] | None:
        step = int(driver.parent.clock.completed_steps)
        if not self.state.ready or step % self.settings.interval_steps == 0:
            self.track(driver, step)
        if not self.state.ready:
            return None
        g = driver.geometry
        r = g.refinement_ratio
        move_i = wrf_move_cells(self.state.vc_i, g.nx, r)
        move_j = wrf_move_cells(self.state.vc_j, g.ny, r)
        if move_i == 0 and move_j == 0:
            if self.settings.holds:
                self.consecutive_corral = 0
            return None
        cancelled = None
        if self.settings.holds:
            move_i, move_j, cancelled = self._hold_corral(driver, step, move_i, move_j)
            if move_i == 0 and move_j == 0:
                return None
        else:
            defect = corral_defect(g, move_i=move_i, move_j=move_j,
                                   corral_x=self.settings.corral_cells_x,
                                   corral_y=self.settings.corral_cells_y)
            if defect is not None:
                self._refuse_corral(step, g, move_i, move_j, defect)
        tracker_centre = (self.state.vc_i, self.state.vc_j)
        last = self.last_track()
        self.state.vc_i -= move_i * r
        self.state.vc_j -= move_j * r
        moved = self.execute(driver, move_i=move_i, move_j=move_j)
        row = {
            "parent_step": step,
            "model_time_s": float(driver.parent.clock.model_seconds),
            "from_parent_cell": [g.parent_i0, g.parent_j0],
            "to_parent_cell": [moved.parent_i0, moved.parent_j0],
            "shift_parent_cells": [move_i, move_j],
            "tracker": self.settings.tracker,
            "tracker_centre_child_faces": list(tracker_centre),
            "tracker_centre_m": [g.origin_x_m + tracker_centre[0] * g.dx_m,
                                 g.origin_y_m + tracker_centre[1] * g.dy_m],
            "tracker_value": None if last is None else last["value"],
            "tracked_at_parent_step": None if last is None else last["parent_step"],
        }
        if cancelled is not None:
            row["corral_cancelled_parent_cells"] = cancelled
        if self.settings.holds:
            row["tracker_held_samples"] = self.consecutive_holds
        self.moves.append(row)
        return row

    def _centre_sample(self) -> dict[str, Any] | None:
        """The newest tracker row of this run whose centroid rule ran: the sample ``vc`` came from.

        A held row (``held = true``) kept the centre an earlier sample set, so
        it is skipped; without ``hold`` no row is held and this is the newest
        row, as it always was.  Held rows run at most ``follow_hold_limit``
        deep, so the walk is bounded.  ``None`` before this run's first
        accepted sample; :meth:`last_track` then answers from the restart.
        """

        for row in reversed(self.tracks):
            if row.get("held") is not True:
                return row
        return None

    def _refuse_corral(self, step: int, g: NestGeometry, move_i: int, move_j: int,
                       defect: str, *, held: dict[str, int] | None = None) -> None:
        reason = (
            defect + ".  WRF cancels such a move (share/mediation_nest_move.F:"
            "849-859) and keeps integrating a nest that no longer follows; "
            "continuing here would publish a storm-following nest refining "
            "the air its vortex left"
        )
        value = {"parent_step": step, "move": [move_i, move_j],
                 "parent_i0": g.parent_i0, "parent_j0": g.parent_j0}
        if held is not None:
            reason += (
                f"; nest.follow_on_edge = \"hold\" has cancelled it on "
                f"{held['consecutive_corral_parent_steps']} consecutive parent steps, "
                f"more than follow_hold_limit*interval_steps = "
                f"{held['hold_limit_parent_steps']}"
            )
            value.update(held)
        self.refuse(
            code="CM1-CARTESIAN-NEST-FOLLOW-CORRAL",
            gate=CARTESIAN_GATE,
            reason=reason,
            remedy=(
                "enlarge the parent domain around the storm's path, lower "
                "nest.follow_corral_m, or stop the run at this model time"
            ),
            option="nest.follow_corral_m",
            value=value,
        )

    def _hold_corral(self, driver: Any, step: int, move_i: int, move_j: int
                     ) -> tuple[int, int, list[int]]:
        """``mediation_nest_move.F:849-859``: cancel each axis that would enter the corral.

        WRF cancels X when X moves and the moved footprint is within
        ``corral_dist`` of an X boundary (``move_cd_x = 0``, and ``vc_i`` gets
        the ``move_cd_x * pgr`` back), and Y independently; the caller carries
        ``vc`` only on the axes that still move, which is the same undo.
        """

        g = driver.geometry
        s = self.settings
        defects = [
            corral_defect(g, move_i=move_i, move_j=0,
                          corral_x=s.corral_cells_x, corral_y=s.corral_cells_y),
            corral_defect(g, move_i=0, move_j=move_j,
                          corral_x=s.corral_cells_x, corral_y=s.corral_cells_y),
        ]
        blocked_i, blocked_j = defects[0] is not None, defects[1] is not None
        if not (blocked_i or blocked_j):
            self.consecutive_corral = 0
            return move_i, move_j, [0, 0]
        cancelled = [move_i if blocked_i else 0, move_j if blocked_j else 0]
        self.consecutive_corral += 1
        self.max_consecutive_corral = max(self.max_consecutive_corral, self.consecutive_corral)
        limit = int(s.hold_limit) * int(s.interval_steps)
        if self.consecutive_corral > limit:
            defect = "; ".join(d for d in defects if d is not None)
            self._refuse_corral(step, g, move_i, move_j, defect, held={
                "consecutive_corral_parent_steps": self.consecutive_corral,
                "hold_limit_parent_steps": limit,
            })
        sample = self.last_track()
        self._record_hold({
            "parent_step": step,
            "model_time_s": float(driver.parent.clock.model_seconds),
            "event": "corral-hold",
            "reason": "; ".join(d for d in defects if d is not None) + (
                ".  WRF cancels the move on that axis (share/mediation_nest_move.F:"
                "849-859); the other axis still moves"
            ),
            "window": None,
            "maximum_child_face": None,
            "value": sample["value"] if sample is not None else None,
            "tracked_at_parent_step": sample["parent_step"] if sample is not None else None,
            "cancelled_shift_parent_cells": cancelled,
            "tracker_centre_child_faces": [self.state.vc_i, self.state.vc_j],
            "parent_cell": [g.parent_i0, g.parent_j0],
            "consecutive": self.consecutive_corral,
        })
        return (0 if blocked_i else move_i), (0 if blocked_j else move_j), cancelled

    def _record_hold(self, row: dict[str, Any]) -> None:
        """Count every event; list every relocation, and each capped kind up to its cap.

        ``corral-hold`` is one event per parent step and held samples one per
        follow interval, so each has its own :data:`HOLD_EVENT_CAP`: a run of
        corral hits never unlists a held sample.  A relocation is at most one
        per ``follow_hold_limit + 1`` follow intervals and moves the centre, so
        every one is listed with its old and new centre and displacement.
        """

        event = row["event"]
        self.hold_counts[event] += 1
        if event == "relocation":
            self.holds.append(row)
            return
        kind = "corral-hold" if event == "corral-hold" else "held-sample"
        if self.holds_listed[kind] < HOLD_EVENT_CAP:
            self.holds_listed[kind] += 1
            self.holds.append(row)
        else:
            self.holds_not_listed += 1

    # -- the whole run's counts, across restarts --------------------------------
    @property
    def move_count(self) -> int:
        return self.moves_before_restart + len(self.moves)

    @property
    def tracker_event_count(self) -> int:
        return self.tracker_events_before_restart + len(self.tracks)

    def whole_run_hold_counts(self) -> dict[str, int]:
        """Every hold event of the run, the restart's and this run's together."""

        return {
            event: self.hold_counts_before_restart.get(event, 0) + self.hold_counts[event]
            for event in HOLD_EVENTS
        }

    @property
    def whole_run_max_consecutive_holds(self) -> int:
        return max(self.max_consecutive_holds_before_restart, self.max_consecutive_holds)

    @property
    def whole_run_max_consecutive_corral(self) -> int:
        return max(self.max_consecutive_corral_before_restart, self.max_consecutive_corral)

    def last_track(self) -> dict[str, Any] | None:
        """The parent step and value of the sample ``vc`` came from, this run's or the restart's.

        Under ``hold`` that is the newest accepted (unheld) sample
        (:meth:`_centre_sample`); a held sample kept an earlier centre.  Before
        this run's first accepted sample it is the one the restart carried,
        so a move decided from the carried centre reports the track that set
        it.
        """

        sample = self._centre_sample()
        if sample is not None:
            return {"parent_step": sample["parent_step"], "value": sample["value"]}
        return self.carried_last_track

    # -- tracking ---------------------------------------------------------------
    @property
    def search_intervals(self) -> int:
        """Follow intervals since the last accepted centre: held samples plus one.

        Always one without ``hold``, where no sample is held.
        """

        return self.consecutive_holds + 1

    def track(self, driver: Any, step: int) -> None:
        if self.settings.tracker == SURFACE_TRACKER:
            self._track_surface(driver, step)
        else:
            self._track_pressure(driver, step)

    def _track_surface(self, driver: Any, step: int) -> None:
        from .nest_move_kernels import launch_vortex_centroid, launch_zeta_box

        cp, geo = self.cupy, self.surface
        assert geo is not None
        a = driver.child.dynamics.arrays
        with self.stream:
            zeta = lowest_level_corner_vorticity(a["u3d"], a["v3d"], nx=geo.nx, ny=geo.ny,
                                                 dx_m=geo.dx_m, dy_m=geo.dy_m, xp=cp)
            zbar = cp.zeros(zeta.shape, dtype=cp.float64)
            lo_j, hi_j, lo_i, hi_i = geo.box
            launch_zeta_box(cp, self.module, zeta=zeta, out=zbar, h=geo.h, lo_j=lo_j,
                            hi_j=hi_j, lo_i=lo_i, hi_i=hi_i, stream=self.stream)
            window = geo.window(self.state, intervals=self.search_intervals)
            out = cp.zeros((CENTROID_WORDS,), dtype=cp.float64)
            launch_vortex_centroid(cp, self.module, zbar=zbar, out=out, window=window,
                                   centres=geo.centres, d=geo.d, stream=self.stream)
        words = [float(v) for v in out.get(stream=self.stream)]

        def search(other: tuple[int, int, int, int]) -> list[float]:
            # A relocation re-reads the same device plane over a wider window.
            with self.stream:
                found = cp.zeros((CENTROID_WORDS,), dtype=cp.float64)
                launch_vortex_centroid(cp, self.module, zbar=zbar, out=found, window=other,
                                       centres=geo.centres, d=geo.d, stream=self.stream)
            return [float(v) for v in found.get(stream=self.stream)]

        self._accept_surface(driver, step, words, window, search=search)

    def _accept_surface(self, driver: Any, step: int, words: list[float],
                        window: tuple[int, int, int, int], *,
                        search: Callable[[tuple[int, int, int, int]], list[float]] | None = None,
                        ) -> None:
        """Accept, hold or relocate one surface-tracker sample.

        ``search`` runs the centroid kernel over the same box-mean plane with
        another window; only a relocation calls it.
        """

        assert self.surface is not None
        holds = self.settings.holds
        intervals = self.search_intervals
        if holds:
            # The edge test reads the window and the too-fast test the reach;
            # both must span the same follow intervals.
            expected = self.surface.window(self.state, intervals=intervals)
            if tuple(window) != tuple(expected):
                raise ValueError(
                    f"the sample's window {list(window)} is not the tracker's search "
                    f"window {list(expected)} over {intervals} follow interval(s) since "
                    "the last accepted centre, so its edge and reach tests would not "
                    "be the ones the receipt names"
                )
        if holds and words[0] == 1.0 and not centroid_is_finite(words):
            # Never held, before or after the first centre: the winds are
            # non-finite one box from the maximum, and apply_centroid would
            # set a NaN centre (module docstring, THE HOLD POLICY).
            self._refuse_lost(step, window, holds=True, words=words)
        if holds and self.state.ready:
            if words[0] != 1.0:
                # Never held: nothing found means every box mean in reach is NaN.
                self._refuse_lost(step, window, holds=True)
            held = hold_classification(words, window, self.surface.centres, self.state,
                                       self.surface.search_cells, intervals=intervals)
            if held is not None:
                self._hold_sample(driver, step, words, window, event=held[0],
                                  reason=held[1], search=search)
                return
            # Found (checked above) and classified clean: the weights are not all zero.
            apply_centroid(self.state, words)
            self.consecutive_holds = 0
            self._track_row(driver, step, words, window, held=False)
            return
        # Before the first centre, and always without hold: the code before
        # nest.follow_on_edge (module docstring, THE HOLD POLICY).
        clipped = window_edge_defect(words, window, self.surface.centres)
        if clipped is not None:
            self._refuse_speed(step, words, window, clipped)
        if not apply_centroid(self.state, words):
            self._refuse_lost(step, window, holds=holds)
        if holds:
            self.consecutive_holds = 0
        self._track_row(driver, step, words, window, held=False if holds else None)

    def _track_row(self, driver: Any, step: int, words: list[float],
                   window: tuple[int, int, int, int], *, held: bool | None,
                   relocated: bool = False) -> None:
        g = driver.geometry
        row = {
            "parent_step": step,
            "tracker": SURFACE_TRACKER,
            "search_window_child_faces": list(window),
            "maximum_child_face": [int(words[1]), int(words[2])],
            "value": words[3],
            "centroid_weight_sum": words[8],
            "centre_child_faces": [self.state.vc_i, self.state.vc_j],
            # None only after a first sample whose centroid weights are all
            # zero (:590 leaves vc unset); the code before nest.follow_on_edge
            # raised an unnamed TypeError here (module docstring).
            "centre_m": self._centre_m(g),
            "parent_cell": [g.parent_i0, g.parent_j0],
        }
        if held is not None:
            row["held"] = held
            row["relocated"] = relocated
        self.tracks.append(row)

    def _centre_m(self, g: NestGeometry) -> list[float] | None:
        if not self.state.ready:
            return None
        return [g.origin_x_m + self.state.vc_i * g.dx_m, g.origin_y_m + self.state.vc_j * g.dy_m]

    def _hold_sample(self, driver: Any, step: int, words: list[float],
                     window: tuple[int, int, int, int], *, event: str, reason: str,
                     search: Callable[[tuple[int, int, int, int]], list[float]] | None) -> None:
        """Keep ``vc`` for one held sample, or relocate once the limit is spent."""

        assert self.state.ready and self.surface is not None
        limit = int(self.settings.hold_limit)
        if self.consecutive_holds >= limit:
            self._relocate(driver, step, words, window, event=event, reason=reason,
                           search=search)
            return
        self.consecutive_holds += 1
        self.max_consecutive_holds = max(self.max_consecutive_holds, self.consecutive_holds)
        g = driver.geometry
        self._record_hold({
            "parent_step": step,
            "model_time_s": float(driver.parent.clock.model_seconds),
            "event": event,
            "reason": reason,
            "window": list(window),
            "search_reach_child_cells": self.surface.reach_cells(self.consecutive_holds),
            "maximum_child_face": [int(words[1]), int(words[2])],
            "value": words[3],
            "centroid_child_faces": [words[9], words[10]] if words[8] != 0.0 else None,
            "centre_child_faces": [self.state.vc_i, self.state.vc_j],
            "centre_m": self._centre_m(g),
            "consecutive": self.consecutive_holds,
        })
        self._track_row(driver, step, words, window, held=True)

    def _relocate(self, driver: Any, step: int, trigger_words: list[float],
                  trigger_window: tuple[int, int, int, int], *, event: str, reason: str,
                  search: Callable[[tuple[int, int, int, int]], list[float]] | None) -> None:
        """Search ``follow_relocation_radius_m`` around the kept centre and move ``vc`` there."""

        geo = self.surface
        assert geo is not None
        if search is None:
            raise ValueError("a relocation needs the tracker's box-mean plane to search")
        window = geo.relocation_window(self.state)
        words = search(window)
        if words[0] == 1.0 and not centroid_is_finite(words):
            self._refuse_lost(step, window, holds=True, words=words)
        limit = int(self.settings.hold_limit)
        trigger = {
            "event": event,
            "reason": reason,
            "window": list(trigger_window),
            "maximum_child_face": [int(trigger_words[1]), int(trigger_words[2])],
            "value": trigger_words[3],
        }
        defect = relocation_defect(words, window, geo.centres)
        if defect is not None:
            self._refuse_relocation(step, words, window, defect, trigger=trigger, limit=limit)
        g = driver.geometry
        before = [self.state.vc_i, self.state.vc_j]
        before_m = self._centre_m(g)
        # relocation_defect admitted only a found maximum with non-zero weights.
        apply_centroid(self.state, words)
        after_m = self._centre_m(g)
        assert before_m is not None and after_m is not None
        self.consecutive_holds = 0
        self._record_hold({
            "parent_step": step,
            "model_time_s": float(driver.parent.clock.model_seconds),
            "event": "relocation",
            "reason": (
                f"{reason}; {limit} consecutive samples had been held "
                f"(follow_hold_limit = {limit}), so the tracker searched "
                f"follow_relocation_radius_m = {self.settings.relocation_radius_m!r} m "
                "around the kept centre and took the centroid of the strongest "
                "circulation there"
            ),
            "trigger": trigger,
            "window": list(window),
            "relocation_radius_m": self.settings.relocation_radius_m,
            "maximum_child_face": [int(words[1]), int(words[2])],
            "value": words[3],
            "from_centre_child_faces": before,
            "to_centre_child_faces": [self.state.vc_i, self.state.vc_j],
            "from_centre_m": before_m,
            "to_centre_m": after_m,
            "displacement_m": [after_m[0] - before_m[0], after_m[1] - before_m[1]],
            "held_samples": limit,
        })
        self._track_row(driver, step, words, window, held=False, relocated=True)

    def _refuse_relocation(self, step: int, words: list[float],
                           window: tuple[int, int, int, int], defect: str, *,
                           trigger: dict[str, Any], limit: int) -> None:
        self.refuse(
            code="CM1-CARTESIAN-NEST-FOLLOW-RELOCATION",
            gate=CARTESIAN_GATE,
            reason=(
                f"{trigger['reason']}; nest.follow_on_edge = \"hold\" had kept the "
                f"previous centre for {limit} consecutive samples (follow_hold_limit), "
                f"and the relocation search failed: {defect}.  Holding on would "
                "publish a storm-following nest refining air its vortex left"
            ),
            remedy=(
                "raise nest.follow_relocation_radius_m (at least twice the smallest "
                "admitted radius is recommended; records/CM1-NEST-SEAM.md 5.8); if "
                "the held samples were edge-hold or too-fast-hold, raise "
                "nest.follow_max_speed_m_s, since the held window widens at the "
                "declared speed only and a longer nest.follow_hold_limit never "
                "catches a faster vortex; or stop the run at this model time"
            ),
            option="nest.follow_relocation_radius_m",
            value={
                "parent_step": step,
                "relocation_window": list(window),
                "maximum_child_face": [int(words[1]), int(words[2])],
                "value": words[3],
                "trigger_event": trigger["event"],
                "trigger_window": trigger["window"],
                "held_samples": limit,
            },
        )

    def _refuse_speed(self, step: int, words: list[float], window: tuple[int, int, int, int],
                      clipped: str) -> None:
        self.refuse(
            code="CM1-CARTESIAN-NEST-FOLLOW-SPEED",
            gate=CARTESIAN_GATE,
            reason=clipped + (
                "; continuing would move the running child toward that edge "
                "while the receipt named it following the vortex"
            ),
            remedy=(
                "raise nest.follow_max_speed_m_s to at least the storm's "
                "translation speed relative to the frame, or shorten "
                "nest.follow_interval_s"
            ),
            option="nest.follow_max_speed_m_s",
            value={"parent_step": step, "window": list(window),
                   "maximum_child_face": [int(words[1]), int(words[2])]},
        )

    def _refuse_lost(self, step: int, window: tuple[int, int, int, int], *,
                     holds: bool = False, words: list[float] | None = None) -> None:
        value: dict[str, Any] = {"parent_step": step}
        if words is not None:
            # A found maximum whose centroid is not finite (hold only).
            box = [int(words[4]), int(words[5]), int(words[6]), int(words[7])]
            reason = (
                f"the largest area-averaged vorticity in the search window "
                f"{list(window)} is {words[3]!r} s-1 at child face "
                f"({int(words[1])}, {int(words[2])}), but the centroid over the box "
                f"{box} around it is not finite (weight sum {words[8]!r}): a box mean "
                "in that box is NaN, so the child's lowest-level winds are non-finite "
                "within one smoothing radius of the strongest circulation, a failed "
                "integration no kept centre can recover; nest.follow_on_edge = "
                "\"hold\" does not hold this sample, and taking the centroid, as "
                "WRF does (share/mediation_nest_move.F:556-606), would set a NaN "
                "centre that ends the run one parent step later without a name"
            )
            value.update({"window": list(window), "maximum_child_face": [int(words[1]), int(words[2])],
                          "centroid_box": box, "centroid_weight_sum": words[8]})
        else:
            reason = (
                f"the surface-vortex search window {list(window)} holds no finite "
                "box-mean vorticity; WRF stops on the same failure "
                "(share/mediation_nest_move.F:544-548)"
            )
            if holds:
                reason += (
                    "; nest.follow_on_edge = \"hold\" does not hold this sample: the "
                    "window is never empty and any finite box mean would have been "
                    "found, so the child's lowest-level winds are non-finite across "
                    "it, a failed integration no kept centre can recover"
                )
        self.refuse(
            code="CM1-CARTESIAN-NEST-FOLLOW-LOST",
            gate=CARTESIAN_GATE,
            reason=reason,
            remedy="inspect the child's lowest-level winds at this step",
            option="nest.follow_tracker", value=value,
        )

    def _track_pressure(self, driver: Any, step: int) -> None:
        from .nest_geometry import compile_nest_tracker_module

        if self.tracker_module is None:
            self.tracker_module = compile_nest_tracker_module(self.cupy)
        with self.stream:
            surface = self.cupy.ascontiguousarray(driver.parent.dynamics.fields["pp3d"][0])
            ci, cj = device_pressure_centre_cell(self.cupy, surface, module=self.tracker_module)
        self._accept_pressure(driver, step, int(ci), int(cj))

    def _accept_pressure(self, driver: Any, step: int, ci: int, cj: int) -> None:
        g = driver.geometry
        st = self.state
        if st.pressure_centre is not None:
            reach = self.settings.max_speed_m_s * self.settings.interval_s
            jump = max(abs(ci - st.pressure_centre[0]) * g.parent_dx_m,
                       abs(cj - st.pressure_centre[1]) * g.parent_dy_m)
            if jump > reach:
                self.refuse(
                    code="CM1-CARTESIAN-NEST-FOLLOW-JUMP",
                    gate=CARTESIAN_GATE,
                    reason=(
                        f"CM1's pressure centre moved {jump!r} m in one follow "
                        f"interval, beyond follow_max_speed_m_s*follow_interval_s = "
                        f"{reach!r} m; no storm translates that fast, so the tracker "
                        "has jumped to another pressure minimum and following it "
                        "would carry the nest off the storm"
                    ),
                    remedy="use nest.follow_tracker = \"wrf-surface-circulation\", whose "
                           "search window is bounded by the same speed",
                    option="nest.follow_tracker",
                    value={"parent_step": step, "from": list(st.pressure_centre), "to": [ci, cj]},
                )
        if st.held_offset is None:
            st.held_offset = (
                (g.parent_i0 + g.parent_span_x / 2.0) - ci,
                (g.parent_j0 + g.parent_span_y / 2.0) - cj,
            )
        st.pressure_centre = (ci, cj)
        ti, tj = follow_target_face(
            self.settings.tracker, centre_i_face=ci, centre_j_face=cj,
            placement_offset_faces=st.held_offset,
        )
        r = g.refinement_ratio
        st.vc_i = (ti - g.parent_i0) * r
        st.vc_j = (tj - g.parent_j0) * r
        self.tracks.append({
            "parent_step": step,
            "tracker": self.settings.tracker,
            "pressure_centre_parent_face": [ci, cj],
            "target_parent_face": [ti, tj],
            "value": None,
            "centre_child_faces": [st.vc_i, st.vc_j],
            "centre_m": [ti * g.parent_dx_m, tj * g.parent_dy_m],
            "parent_cell": [g.parent_i0, g.parent_j0],
        })

    # -- the move ---------------------------------------------------------------
    def _require_uniform_metrics(self, driver: Any) -> None:
        if self.uniform_metrics_checked:
            return
        cp = self.cupy
        bad = []
        with self.stream:
            for name, value in driver.child.registry.buffers.items():
                if not name.startswith("metric.") or value.ndim < 2:
                    continue
                first = value[(..., slice(0, 1), slice(0, 1))]
                if not bool(cp.all(value == first)):
                    bad.append(name)
        if bad:
            self.refuse(
                code="CM1-CARTESIAN-NEST-FOLLOW-TERRAIN",
                gate=CARTESIAN_GATE,
                reason=(
                    "these child metrics vary horizontally: " + ", ".join(bad) + "; the "
                    "move keeps metrics in place, so a moved child would integrate "
                    "on the metrics of the ground it left"
                ),
                remedy="run the moving nest over a flat, uniform grid",
                option="nest.follow_storm", value=bad,
            )
        self.uniform_metrics_checked = True

    def execute(self, driver: Any, *, move_i: int, move_j: int) -> NestGeometry:
        from .nest_live import _refill_halos_and_thermodynamics
        from .nest_live_kernels import launch_capture
        from .nest_move_kernels import launch_shift

        self._require_uniform_metrics(driver)
        rules = self.inventory(driver)
        cp = self.cupy
        old = driver.geometry
        new = replace(old, parent_i0=old.parent_i0 + move_i, parent_j0=old.parent_j0 + move_j)
        r = old.refinement_ratio
        child, parent = driver.child, driver.parent
        cg, pg = child.config.grid, parent.config.grid
        numq = int(child.dynamics.numq)
        # A child on its own vertical deck (records/CM1-NEST-SEAM.md 5.6) takes
        # the exposed strip through the same vertical window its tables and
        # initial condition read, re-placed over the moved footprint first.
        vertical = getattr(driver.forcing, "vertical", None)
        self.vertical_capture = vertical is not None
        if vertical is not None:
            vertical.relocate(new)
        with self.stream:
            for rule in rules:
                c_buf = child.registry.buffers[rule.name]
                p_buf = parent.registry.buffers[rule.name]
                cl = rule_lattice(rule, nz=cg.nz, ny=cg.ny, nx=cg.nx)
                pl = rule_lattice(rule, nz=pg.nz, ny=pg.ny, nx=pg.nx)
                for c_owner, p_owner in zip(rule_owners(rule, c_buf, numq=numq),
                                            rule_owners(rule, p_buf, numq=numq)):
                    snap = c_owner.copy()
                    target = cp.empty(cl, dtype=cp.float32)
                    if vertical is not None and not rule.plane:
                        source, s_origin, s_lattice, p_i0, p_j0 = vertical.source(
                            p_owner, rule.origin, pl, rule.stagger
                        )
                    else:
                        source, s_origin, s_lattice = p_owner, rule.origin, pl
                        p_i0, p_j0 = new.parent_i0, new.parent_j0
                    launch_capture(
                        cp, driver.module, parent=source, parent_origin=s_origin,
                        parent_lattice=s_lattice, child=None, child_origin=(0, 0, 0),
                        child_lattice=cl, out_value=target, out_tend=None, side="full",
                        width=driver.forcing.width, ratio=r, parent_i0=p_i0,
                        parent_j0=p_j0, stagger=rule.stagger,
                        parent_dt_s=float(parent.config.dt_s), stream=self.stream,
                    )
                    k0, j0, i0 = rule.origin
                    cp.copyto(c_owner[k0:k0 + cl[0], j0:j0 + cl[1], i0:i0 + cl[2]], target)
                    launch_shift(cp, self.module, snap=snap, out=c_owner, origin=rule.origin,
                                 lattice=cl, shift_j=r * move_j, shift_i=r * move_i,
                                 stream=self.stream)
                    del snap, target
        _refill_halos_and_thermodynamics(child.dynamics)
        self.stream.synchronize()
        driver.geometry = new
        driver.forcing.geometry = new
        return new

    def record(self) -> dict[str, Any]:
        payload = self._record()
        if self.settings.on_edge is None:
            return payload
        if self.settings.holds:
            payload["departures_from_wrf"].update(HOLD_DEPARTURES_FROM_WRF)
        payload.update({
            "on_edge": self.settings.on_edge,
            "hold_limit": self.settings.hold_limit,
            "relocation_radius_m": self.settings.relocation_radius_m,
            "corral_hold_limit_parent_steps": (
                None if self.settings.hold_limit is None
                else int(self.settings.hold_limit) * int(self.settings.interval_steps)
            ),
            "holds": list(self.holds),
            "holds_not_listed": self.holds_not_listed,
            "hold_counts": self.whole_run_hold_counts(),
            "max_consecutive_holds": self.whole_run_max_consecutive_holds,
            "max_consecutive_corral_parent_steps": self.whole_run_max_consecutive_corral,
        })
        if self.settings.holds and "carried_from_restart" in payload:
            payload["carried_from_restart"]["hold_counts"] = dict(self.hold_counts_before_restart)
            payload["carried_from_restart"]["max_consecutive_holds"] = (
                self.max_consecutive_holds_before_restart
            )
            payload["carried_from_restart"]["max_consecutive_corral_parent_steps"] = (
                self.max_consecutive_corral_before_restart
            )
        if self.surface is not None and self.surface.relocation_cells is not None:
            payload["surface_tracker"]["relocation_radius_child_cells"] = self.surface.relocation_cells
            payload["surface_tracker"]["search_window_slack_child_cells"] = self.surface.slack_cells
        return payload

    def _record(self) -> dict[str, Any]:
        return {
            "schema": NEST_MOVE_SCHEMA,
            "status": "moving",
            "settings": self.settings.record(),
            "wrf_authority": dict(WRF_MOVE_AUTHORITY),
            "departures_from_wrf": dict(DEPARTURES_FROM_WRF),
            "kernel_sha256": NEST_MOVE_KERNEL_SHA256,
            "surface_tracker": None if self.surface is None else {
                "box_half_width_child_cells": self.surface.h,
                "centroid_half_width_child_cells": self.surface.d,
                "search_radius_child_cells": self.surface.search_cells,
                "admissible_centres_child_faces": list(self.surface.centres),
            },
            "exposed_strip_vertical_window": self.vertical_capture,
            "shifted_carriers": list(self.shifted_carriers),
            "carriers_not_shifted": dict(self.kept_carriers),
            "tracker_events": list(self.tracks),
            "moves": list(self.moves),
            "move_count": self.move_count,
            **({} if not (self.moves_before_restart or self.tracker_events_before_restart
                          or self.carried_last_track is not None) else {
                "carried_from_restart": {
                    "moves": self.moves_before_restart,
                    "tracker_events": self.tracker_events_before_restart,
                    "last_track": None if self.carried_last_track is None
                    else dict(self.carried_last_track),
                },
            }),
        }


__all__ = [
    "DEPARTURES_FROM_WRF",
    "EDGE_HOLD",
    "EDGE_POLICIES",
    "EDGE_STOP",
    "FollowSettings",
    "HOLD_DEPARTURES_FROM_WRF",
    "HOLD_EVENTS",
    "HOLD_EVENT_CAP",
    "HOLD_EVENT_CAPPED_KINDS",
    "HOLD_SAMPLE_EVENTS",
    "HOLD_WINDOW_SLACK_CELLS",
    "MovingNest",
    "NEST_MOVE_SCHEMA",
    "PRESSURE_TRACKERS",
    "SURFACE_TRACKER",
    "SurfaceTrackerGeometry",
    "TrackerState",
    "WRF_MOVE_AUTHORITY",
    "apply_centroid",
    "centroid_is_finite",
    "classify_carriers",
    "corral_cells",
    "corral_defect",
    "hold_classification",
    "lowest_level_corner_vorticity",
    "relocation_defect",
    "rule_lattice",
    "shifted_rule",
    "surface_tracker_defect",
    "surface_tracker_extent",
    "surface_tracker_reference",
    "wrf_move_cells",
]
