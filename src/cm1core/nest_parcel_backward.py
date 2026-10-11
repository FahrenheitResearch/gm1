r"""Backward trajectories from saved nest-child history frames.  OFFLINE DIAGNOSTIC.

This is not the model.  The live child parcels (:mod:`cm1core.nest_parcels`)
integrate forward at every child step with the step's own acoustic average.
This tool starts from a target box at a later model time and integrates
*backward* through the child's **published** history frames, which hold the
completed state ``u3d``/``v3d``/``w3d`` at the frame times only.  Between two
frames the wind is a linear interpolation in time, and that interpolation is
the dominant error of every trajectory it produces.  Every output says so.

TWO WIND SOURCES
----------------
``wind_source = "history-blend"`` (the default, and what this tool has always
done) reads published **history** frames -- completed ``u3d``/``v3d``/``w3d``
tens of seconds apart -- and blends them linearly in time.  That blend is the
dominant error and it is what the estimate below bounds.

``wind_source = "model-step"`` reads a
:mod:`cm1core.nest_child_window` stream instead: the model's own advecting
velocity words (``src/solve2.F:1290-1301``), one record per child step.  It is
refused on a stream written with ``stride_child_steps > 1``, because a record
is the average over **one** child step: across a strided interval this source
would freeze that one step's wind over steps it was never the average of,
while the receipt said the time-interpolation error was zero.  There
the backward step uses **one record and no interpolation at all**, which is
exactly what the forward model did -- ``parcel.F`` holds one wind for both RK2
stages -- so the step is the mirror image of the model's own step in the same
frozen field.  The time-interpolation term is then not estimated but *absent*.
This mode is the reason the module :mod:`cm1core.nest_child_window` exists;
see its docstring for why an 800-1200 m history-blend estimate is not good
enough near a 100 m/s core.

What is left, and how it is published
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Two things remain, and the receipt keeps them apart because they behave
differently:

``round_trip_residual`` (measured).  Every parcel is re-integrated forward
from where it ended, through the same records in the same order, and the
distance back to where it started is published beside every position.  It is a
**lower** bound, and an honest reader must know why: where the flow is locally
linear, RK2 with ``-dt`` and RK2 with ``+dt`` are inverses to
:math:`O(\Delta t^{4})`, so the measurement reads near zero while the
trajectory still carries its own :math:`O(\Delta t^{2})` truncation.  It
catches irreversibility, not truncation.

``sensitivity_amplification`` (measured), and the published
``position_error_estimate`` built from it.  What actually limits a backward
trajectory into a tornado is not the scheme: it is that the *target* position
is not known to better than the grid resolves, and that uncertainty grows
backward at the flow's own local rate.  Dahl, Parker and Wicker (2012) is the
demonstration.  The amplification :math:`\prod(1 + |\Delta t| L_n)` is
accumulated along each trajectory, and ``position_error_estimate`` is the
measured round trip plus half a window cell carried back through it.

METHOD
------
* **Step.**  CM1 r21.1 ``src/parcel.F:137-568``'s RK2, sampler and vertical
  clamp, run with ``-dt`` on the child's own lattice and deck, exactly as the
  live tracker runs it (:func:`cm1core.nest_parcels.reference_stencils`).
  CM1 has no backward mode, so the sign of ``dt`` is GM1's.
* **Time interpolation** (``history-blend`` only).  ``parcel.F`` holds one wind for both RK stages --
  the step's small-step average, a centred-in-time wind.  The offline analogue
  used here holds one wind for both stages too: the frames bracketing the
  step's **midpoint** time ``t - dt/2``, blended linearly,
  ``U = S_a + alpha*(S_b - S_a)`` in binary32 with ``alpha = (t_mid -
  t_a)/(t_b - t_a)``.  ``S_a`` and ``S_b`` are sampled at the parcel on each
  frame's **own** lattice, so a frame pair that straddles a nest move is
  blended correctly -- a pointwise blend of the two lattices would mix winds
  from different ground.  The frozen-field-per-step choice follows CM1's own;
  the linear-in-time wind is the standard offline treatment (Seibert 1993;
  Stohl 1998).
* **Termination.**  A parcel stops, with a recorded reason, when it leaves
  either bracketing frame's free interior (the child less its specified and
  relaxation rows, whose winds are the parent's interpolated), when a sample
  leaves a frame's lattice or deck, or when the requested interval runs out.

INTERPOLATION ERROR BOUND
-------------------------
For a wind with a continuous second time derivative, linear interpolation
between frames ``T`` apart errs by at most

    ``|u(t) - U(t)| <= (T**2 / 8) * max |d2u/dt2|``

(the Lagrange interpolation remainder; e.g. Burden and Faires, *Numerical
Analysis*, section 3.1).  The frames cannot give ``max |d2u/dt2|``, so each
step **estimates** it from the second divided difference of three frames
sampled at the parcel's position (the bracketing pair and the next frame
beyond it), giving ``E_n``.  Position errors are accumulated with the discrete
Gronwall recursion

    ``delta_{n+1} = delta_n * (1 + |dt| * L_n) + |dt| * E_n``,

with ``L_n = |U(x1) - U(x0)| / |x1 - x0|`` the wind's rate of change along the
step's own first stage.  ``delta`` is an **estimate**, not a guarantee: it
omits RK2 truncation and spatial interpolation error, it samples the time
curvature at three instants rather than bounding it, and it measures the
wind's spatial sensitivity along one direction per step.  Dahl, Parker and
Wicker (2012) show that near-surface trajectories in simulated supercells are
sensitive to exactly this history interval, which is why the estimate is
published beside every position.

References
~~~~~~~~~~
* Seibert, P., 1993: Convergence and accuracy of numerical methods for
  trajectory calculations. *J. Appl. Meteor.*, **32**, 558-566.
* Stohl, A., 1998: Computation, accuracy and applications of trajectories --
  a review and bibliography. *Atmos. Environ.*, **32**, 947-966.
* Dahl, J. M. L., M. D. Parker, and L. J. Wicker, 2012: Uncertainties in
  trajectory calculations within near-surface mesocyclones of simulated
  supercells. *Mon. Wea. Rev.*, **140**, 2959-2966.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .nest_parcels import (
    NEST_CHILD_PLACEMENT_DIRECTORY,
    NEST_CHILD_PLACEMENT_SCHEMA,
    SENTINEL_F32,
    STATUS_LEFT_LATTICE,
    STATUS_REASONS,
    ParcelDeck,
    fill_velocity_ghosts,
    fill_velocity_halos,
    reference_sample_uvw,
    seed_box_positions,
    semi_slip_for,
)

#: v2 adds ``wind_source`` and the two rows the ``model-step`` source fills:
#: the measured ``round_trip_residual`` and the ``sensitivity_amplification``
#: an error at the target has grown by when it reaches each earlier time.
BACKWARD_SCHEMA = "gm1.nest-child-backward-trajectories/v2"
BACKWARD_KIND = "offline-diagnostic"
BACKWARD_ARENA_FILENAME = "backward-trajectories.f32"
BACKWARD_TABLE_FILENAME = "backward-trajectory-table.tsv"
BACKWARD_RECEIPT_FILENAME = "backward-trajectories.json"
BACKWARD_ROWS = (
    ("x", "m"), ("y", "m"), ("z", "m"), ("u", "m/s"), ("v", "m/s"), ("w", "m/s"),
    ("mtime", "s"), ("status", "code"),
    ("wind_time_interpolation_error_estimate", "m/s"),
    ("position_error_estimate", "m"),
    ("round_trip_residual", "m"),
    ("sensitivity_amplification", "1"),
)

#: ``history-blend``: published history frames, blended linearly in time.
#: ``model-step``: a :mod:`cm1core.nest_child_window` stream, one record per
#: child step, taken one record at a time with no interpolation.
WIND_SOURCES: tuple[str, ...] = ("history-blend", "model-step")
HISTORY_ARENA = "history-carriers.f32"
HISTORY_TABLE = "history-field-table.tsv"
FRAME_DIRECTORY_FORMAT = "frame-{index:06d}"
NG = 3

STATUS_ACTIVE = 0
STATUS_LEFT_INTERIOR = 4
STATUS_BOUND_NOT_ESTIMABLE = 5
BACKWARD_STATUS = {
    STATUS_ACTIVE: "integrating, or reached the end time",
    **STATUS_REASONS,
    STATUS_LEFT_INTERIOR: "reached a bracketing frame's specified or relaxation rows (parent-interpolated winds)",
    STATUS_BOUND_NOT_ESTIMABLE: (
        "the third frame for the time-curvature estimate does not cover the parcel; the "
        "position is still integrated, the error estimate is not"
    ),
}

CITATIONS = (
    "CM1 r21.1 src/parcel.F:137-568 (RK2 step, sampler, vertical clamp)",
    "Lagrange interpolation remainder |f - p1| <= (h^2/8) max|f''| (e.g. Burden and Faires, Numerical Analysis, section 3.1)",
    "Seibert, P., 1993: Convergence and accuracy of numerical methods for trajectory calculations. J. Appl. Meteor., 32, 558-566",
    "Stohl, A., 1998: Computation, accuracy and applications of trajectories -- a review and bibliography. Atmos. Environ., 32, 947-966",
    "Dahl, J. M. L., M. D. Parker, and L. J. Wicker, 2012: Uncertainties in trajectory calculations within near-surface mesocyclones of simulated supercells. Mon. Wea. Rev., 140, 2959-2966",
)


class BackwardTrajectoryError(ValueError):
    """A request or a frame set that cannot produce the trajectory it names."""


def _np() -> Any:
    import numpy

    return numpy


@dataclass
class ChildFrame:
    """One published child frame with its winds on padded, filled host carriers."""

    index: int
    model_time_s: float
    placement: dict[str, Any]
    deck: ParcelDeck
    u: Any
    v: Any
    w: Any
    arena_sha256: str
    directory: str
    #: nest_parcels.SemiSlip for a bbc=3 frame (parcel.F:470-515), else None.
    surface: Any = None

    def free_interior(self) -> tuple[float, float, float, float]:
        p, d = self.placement, self.deck
        width = int(p["spec_zone"]) + int(p["relax_zone"])
        return (
            float(p["origin_x_m"]) + width * float(p["dx_m"]),
            float(p["origin_x_m"]) + (d.nx - width) * float(p["dx_m"]),
            float(p["origin_y_m"]) + width * float(p["dy_m"]),
            float(p["origin_y_m"]) + (d.ny - width) * float(p["dy_m"]),
        )

    def inside(self, x: Any, y: Any) -> Any:
        np = _np()
        x_lo, x_hi, y_lo, y_hi = self.free_interior()
        xs, ys = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        return (xs >= x_lo) & (xs <= x_hi) & (ys >= y_lo) & (ys <= y_hi)

    def sample(self, x: Any, y: Any, z: Any, *, second_stage: bool) -> tuple[Any, Any, Any, Any]:
        np = _np()
        return reference_sample_uvw(np, self.u, self.v, self.w, self.deck, x, y, z,
                                    second_stage=second_stage, ng=NG, surface=self.surface)

    def record(self) -> dict[str, Any]:
        return {
            "index": self.index, "model_time_s": self.model_time_s, "directory": self.directory,
            "parent_i0": self.placement["parent_i0"], "parent_j0": self.placement["parent_j0"],
            "origin_x_m": self.placement["origin_x_m"], "origin_y_m": self.placement["origin_y_m"],
            "arena_sha256": self.arena_sha256,
            "winds": self.placement.get("winds_published"),
        }


def _read_field_table(path: Path) -> dict[str, tuple[tuple[int, ...], int]]:
    rows: dict[str, tuple[tuple[int, ...], int]] = {}
    for line in path.read_text(encoding="ascii").splitlines():
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) != 8:
            raise BackwardTrajectoryError(f"{path}: a field-table row has {len(parts)} columns, not 8")
        name, _meaning, _units, _stagger, shape_csv, _bounds, _raw, offset = parts
        rows[name] = (tuple(int(v) for v in shape_csv.split(",")), int(offset))
    return rows


def read_child_frame(child_root: Path | str, index: int) -> ChildFrame:
    """One ``nest-child/frame-NNNNNN`` and its ``placements/frame-NNNNNN.json``."""

    np = _np()
    root = Path(child_root)
    placement_path = root / NEST_CHILD_PLACEMENT_DIRECTORY / f"frame-{index:06d}.json"
    if not placement_path.is_file():
        raise BackwardTrajectoryError(
            f"{placement_path} does not exist; a child frame without its placement cannot be "
            "put back on the ground (a following nest moves between frames), so its winds "
            "would be sampled at the wrong place"
        )
    placement = json.loads(placement_path.read_text(encoding="utf-8"))
    if placement.get("schema") != NEST_CHILD_PLACEMENT_SCHEMA:
        raise BackwardTrajectoryError(f"{placement_path} is not a {NEST_CHILD_PLACEMENT_SCHEMA} document")
    frame_dir = root / FRAME_DIRECTORY_FORMAT.format(index=index)
    table = _read_field_table(frame_dir / HISTORY_TABLE)
    nx, ny, nz = int(placement["nx"]), int(placement["ny"]), int(placement["nz"])
    expected = {"u3d": (nz, ny, nx + 1), "v3d": (nz, ny + 1, nx), "w3d": (nz + 1, ny, nx)}
    arena = frame_dir / HISTORY_ARENA
    raw = arena.read_bytes()
    carriers: dict[str, Any] = {}
    for name, shape in expected.items():
        if name not in table or table[name][0] != shape:
            raise BackwardTrajectoryError(
                f"{frame_dir}: {name} is {table.get(name, (None,))[0]}, not {shape}; the frame "
                "does not match its placement's lattice"
            )
        count = shape[0] * shape[1] * shape[2]
        offset = table[name][1]
        carriers[name] = np.frombuffer(raw, dtype="<f4", count=count, offset=offset).reshape(shape)
    deck = ParcelDeck(
        name="child-frame", nx=nx, ny=ny, nz=nz,
        dx_m=float(np.float32(placement["dx_m"])), dy_m=float(np.float32(placement["dy_m"])),
        minx_m=float(np.float32(placement["origin_x_m"])), miny_m=float(np.float32(placement["origin_y_m"])),
        dz_reference_m=float(placement["dz_reference_m"]),
        zf=np.asarray(placement["zf_cm1_indexed"], dtype=np.float32),
        zh=np.asarray(placement["zh_cm1_indexed"], dtype=np.float32),
        mf=np.asarray(placement["mf_cm1_indexed"], dtype=np.float32),
        cgs=tuple(float(v) for v in placement["cgs"]),  # type: ignore[arg-type]
        cgt=tuple(float(v) for v in placement["cgt"]),  # type: ignore[arg-type]
        lateral=(2, 2, 2, 2), periodic=False, vertical_sha256=str(placement["vertical_sha256"]),
    )
    roughness = None
    if int(placement["bbc"]) == 3:
        entry = placement.get("roughness")
        path = None if entry is None else root / NEST_CHILD_PLACEMENT_DIRECTORY / str(entry["file"])
        if path is None or not path.is_file():
            raise BackwardTrajectoryError(
                f"{placement_path} is a bbc=3 frame without its znt; parcel.F:470-515's log layer "
                "below zh(1) cannot be applied, and the trilinear wind there is not CM1's"
            )
        words = path.read_bytes()
        if hashlib.sha256(words).hexdigest() != entry["sha256"]:
            raise BackwardTrajectoryError(f"{path} does not match the SHA-256 its placement records")
        roughness = np.frombuffer(words, dtype="<f4").reshape(ny, nx)
    return frame_from_arrays(
        index=index, model_time_s=float(placement["model_time_s"]), placement=placement, deck=deck,
        u=carriers["u3d"], v=carriers["v3d"], w=carriers["w3d"],
        arena_sha256=hashlib.sha256(raw).hexdigest(), directory=str(frame_dir), roughness=roughness,
    )


def frame_from_arrays(*, index: int, model_time_s: float, placement: dict[str, Any], deck: ParcelDeck,
                      u: Any, v: Any, w: Any, arena_sha256: str = "", directory: str = "",
                      roughness: Any = None) -> ChildFrame:
    """Pad physical ``u3d``/``v3d``/``w3d`` and fill them as ``parcel.F`` reads them.

    ``roughness`` is the frame's ``(ny, nx)`` ``znt``, required for ``bbc = 3``.
    """

    np = _np()
    nx, ny, nz = deck.nx, deck.ny, deck.nz
    up = np.zeros((nz + 2, ny + 2 * NG, nx + 1 + 2 * NG), dtype=np.float32)
    vp = np.zeros((nz + 2, ny + 1 + 2 * NG, nx + 2 * NG), dtype=np.float32)
    wp = np.zeros((nz + 3, ny + 2 * NG, nx + 2 * NG), dtype=np.float32)
    up[1:nz + 1, NG:NG + ny, NG:NG + nx + 1] = u
    vp[1:nz + 1, NG:NG + ny + 1, NG:NG + nx] = v
    wp[1:nz + 2, NG:NG + ny, NG:NG + nx] = w
    fill_velocity_halos(np, up, vp, wp, deck, ng=NG)
    fill_velocity_ghosts(np, up, vp, wp, deck, bbc=int(placement["bbc"]), tbc=int(placement["tbc"]),
                         u_move_m_s=float(placement["u_move_m_s"]),
                         v_move_m_s=float(placement["v_move_m_s"]))
    from types import SimpleNamespace

    boundary = SimpleNamespace(bbc=int(placement["bbc"]), u_move_m_s=float(placement["u_move_m_s"]),
                               v_move_m_s=float(placement["v_move_m_s"]), imove=placement.get("imove"))
    try:
        surface = semi_slip_for(np, boundary, deck, roughness, ng=NG)
    except RuntimeError as error:
        raise BackwardTrajectoryError(f"frame {index}: {error}") from None
    return ChildFrame(index, float(model_time_s), placement, deck, up, vp, wp, arena_sha256, directory,
                      surface)


def read_child_frames(child_root: Path | str) -> list[ChildFrame]:
    root = Path(child_root)
    directory = root / NEST_CHILD_PLACEMENT_DIRECTORY
    if not directory.is_dir():
        raise BackwardTrajectoryError(
            f"{directory} does not exist; frames published before child placements were "
            "written carry no ground position and cannot be read by this tool"
        )
    indices = sorted(int(p.stem.split("-")[1]) for p in directory.glob("frame-*.json"))
    return [read_child_frame(root, index) for index in indices]


@dataclass(frozen=True)
class BackwardRequest:
    """A seed box (parent-grid-origin metres) at ``start_time_s``, integrated back to ``end_time_s``."""

    seeding: Any  # lagrangian_config.NestChildParcelConfig (prclfrq_s is unused here)
    start_time_s: float
    end_time_s: float
    dt_s: float
    wind_source: str = "history-blend"
    #: ``model-step`` only: seed positions supplied directly as a
    #: ``(3, n)`` float32 array instead of built from ``seeding``'s box, which
    #: is how a circuit or a core disk is seeded.  ``seeding`` may then be
    #: ``None``.
    positions: Any = None
    #: ``backward`` (the default) or ``forward``.  ``forward`` is admitted for
    #: ``history-blend`` only: it integrates vortex-targeted forward parcels
    #: offline through the same records, with the same blend and the same
    #: error accounting run the other way in time.  (``model-step`` reverses
    #: one record per step by construction and has no forward reading here.)
    direction: str = "backward"
    #: ``history-blend`` only.  ``"nearest"`` (the default) blends the two
    #: frames that bracket the step in time and stops a parcel that leaves
    #: either.  ``"containing"`` blends, per parcel, the nearest earlier and
    #: later frames whose free interior HOLDS that parcel, looking at most
    #: ``max_bracket_s`` either way: a vortex-following window that jumps
    #: away from a parcel for a few records then costs that parcel a wider
    #: blend -- whose ``(T**2/8)|u_tt|`` estimate grows with it -- instead of
    #: ending its trajectory.
    bracket: str = "nearest"
    max_bracket_s: float = 15.0

    def __post_init__(self) -> None:
        if self.bracket not in ("nearest", "containing"):
            raise BackwardTrajectoryError(
                "bracket must be nearest or containing; any other word names no frame choice"
            )
        if self.bracket == "containing" and self.wind_source != "history-blend":
            raise BackwardTrajectoryError(
                'bracket = "containing" blends frames per parcel and is a history-blend rule; the '
                "model-step source reads exactly one record per step"
            )
        if self.direction not in ("backward", "forward"):
            raise BackwardTrajectoryError(
                "direction must be backward or forward; any other word names no time order"
            )
        if self.direction == "forward" and self.wind_source != "history-blend":
            raise BackwardTrajectoryError(
                'direction = "forward" is admitted with wind_source = "history-blend" only; the '
                "model-step source is defined as the reversal of the model's own step"
            )
        if self.wind_source not in WIND_SOURCES:
            raise BackwardTrajectoryError(
                f"wind_source must be one of {', '.join(WIND_SOURCES)}; an unadmitted source "
                "would leave the history blend running while the receipt named the model's own "
                "step words, and the published error estimate would be the wrong one"
            )
        if self.seeding is None and self.positions is None:
            raise BackwardTrajectoryError(
                "a request needs either a seed box or explicit positions; neither names a parcel"
            )

    @property
    def model_step(self) -> bool:
        return self.wind_source == "model-step"

    def seed_positions(self) -> Any:
        np = _np()
        if self.positions is not None:
            out = np.array(self.positions, dtype=np.float32, copy=True)
            if out.ndim != 2 or out.shape[0] != 3:
                raise BackwardTrajectoryError("explicit seed positions must be a (3, n) array")
            return out
        return seed_box_positions(self.seeding)

    @property
    def sign(self) -> float:
        """``-1`` backward in time, ``+1`` forward."""

        return 1.0 if self.direction == "forward" else -1.0

    def steps(self) -> int:
        span = -self.sign * (float(self.start_time_s) - float(self.end_time_s))
        if not (math.isfinite(span) and span > 0.0):
            raise BackwardTrajectoryError(
                "start_time_s must be later than end_time_s" if self.sign < 0 else
                "a forward request's end_time_s must be later than its start_time_s")
        if not (math.isfinite(self.dt_s) and self.dt_s > 0.0):
            raise BackwardTrajectoryError("dt_s must be positive; the sign of the step is the tool's")
        count = span / float(self.dt_s)
        if abs(count - round(count)) > 1.0e-9 * max(1.0, count):
            raise BackwardTrajectoryError(
                f"(start - end) / dt = {count} is not a whole number of steps; a fractional last "
                "step would be a step size the method does not state"
            )
        return int(round(count))


@dataclass
class BackwardResult:
    rows: Any                 # (steps+1, len(BACKWARD_ROWS), nparcels) float32
    status: Any
    times_s: list[float]
    frames_used: list[int]


def _bracket(times: Sequence[float], t: float) -> tuple[int, int]:
    for q in range(len(times) - 1):
        if times[q] <= t <= times[q + 1]:
            return q, q + 1
    raise BackwardTrajectoryError(f"model time {t} s lies outside the frames' [{times[0]}, {times[-1]}] s")


def integrate_backward(frames: Sequence[ChildFrame], request: BackwardRequest) -> BackwardResult:
    """Integrate backward through ``frames`` with the request's own wind source."""

    if request.model_step:
        return _integrate_model_step(frames, request)
    if request.bracket == "containing":
        return _integrate_blend_containing(frames, request)
    return _integrate_history_blend(frames, request)


def _integrate_history_blend(frames: Sequence[ChildFrame], request: BackwardRequest) -> BackwardResult:
    np = _np()
    frames = sorted(frames, key=lambda f: f.model_time_s)
    if len(frames) < 3:
        raise BackwardTrajectoryError(
            "at least three frames are needed: the time-interpolation error estimate is a second "
            "divided difference, and a trajectory published without it has no stated accuracy"
        )
    times = [f.model_time_s for f in frames]
    if len(set(times)) != len(times):
        raise BackwardTrajectoryError("two frames carry the same model time")
    steps = request.steps()
    lo_t = min(float(request.start_time_s), float(request.end_time_s))
    hi_t = max(float(request.start_time_s), float(request.end_time_s))
    if hi_t > times[-1] or lo_t < times[0]:
        raise BackwardTrajectoryError(
            f"[{request.end_time_s}, {request.start_time_s}] s is not inside the frames' "
            f"[{times[0]}, {times[-1]}] s; winds outside the saved frames would be extrapolated"
        )
    positions = request.seed_positions()
    n = positions.shape[1]
    placed = positions[0] > np.float32(-1.0e29)
    a0, b0 = _bracket(times, float(request.start_time_s))
    for frame in (frames[a0], frames[b0]):
        outside = placed & ~frame.inside(positions[0], positions[1])
        if bool(outside.any()):
            p = int(np.flatnonzero(outside)[0])
            lo_x, hi_x, lo_y, hi_y = frame.free_interior()
            raise BackwardTrajectoryError(
                f"seed {p} at ({float(positions[0, p])}, {float(positions[1, p])}) m from the parent "
                f"grid origin is outside frame {frame.index}'s free interior x [{lo_x}, {hi_x}], "
                f"y [{lo_y}, {hi_y}] at the start time"
            )
    rows = np.full((steps + 1, len(BACKWARD_ROWS), n), np.float32(SENTINEL_F32), dtype=np.float32)
    status = np.where(placed, STATUS_ACTIVE, STATUS_LEFT_LATTICE).astype(np.int32)
    status_note = np.zeros(n, dtype=bool)  # bound-not-estimable, recorded beside status
    delta = np.zeros(n, dtype=np.float64)
    amplification = np.ones(n, dtype=np.float64)
    sign = request.sign
    dt = np.float32(sign * float(request.dt_s))
    dt2 = dt / np.float32(2.0)
    used: set[int] = set()
    out_times: list[float] = []
    for step in range(steps + 1):
        t = float(request.start_time_s) + sign * step * float(request.dt_s)
        out_times.append(t)
        rows[step, 0:3, placed] = positions[:, placed].T
        rows[step, 6, placed] = np.float32(t)
        rows[step, 7] = status.astype(np.float32)
        rows[step, 9] = np.where(placed, delta, SENTINEL_F32).astype(np.float32)
        rows[step, 11] = np.where(placed, amplification, SENTINEL_F32).astype(np.float32)
        if step == steps:
            break
        live = status == STATUS_ACTIVE
        if not bool(live.any()):
            continue
        t_mid = t + sign * 0.5 * float(request.dt_s)
        a, b = _bracket(times, t_mid)
        fa, fb = frames[a], frames[b]
        used.update((fa.index, fb.index))
        alpha = np.float32((t_mid - times[a]) / (times[b] - times[a]))
        x0, y0, z0 = positions[0].copy(), positions[1].copy(), positions[2].copy()

        def blend(xs: Any, ys: Any, zs: Any, second: bool) -> tuple[Any, Any, Any, Any]:
            ua, va, wa, sa = fa.sample(xs, ys, zs, second_stage=second)
            ub, vb, wb, sb = fb.sample(xs, ys, zs, second_stage=second)
            st = np.where(sa != 0, sa, sb)
            return (ua + alpha * (ub - ua), va + alpha * (vb - va), wa + alpha * (wb - wa), st)

        u1, v1, w1, s1 = blend(x0, y0, z0, False)
        rows[step, 3, live] = u1[live]
        rows[step, 4, live] = v1[live]
        rows[step, 5, live] = w1[live]
        x1, y1, z1 = x0 + dt * u1, y0 + dt * v1, z0 + dt * w1
        u2, v2, w2, s2 = blend(x1, y1, z1, True)
        x = x0 + dt2 * (u1 + u2)
        y = y0 + dt2 * (v1 + v2)
        z = z0 + dt2 * (w1 + w2)
        z = np.where(z < np.float32(0.0), np.float32(1.0e-6), z)
        maxz = np.float32(min(fa.deck.maxz_m, fb.deck.maxz_m))
        z = np.where(z <= maxz, z, maxz)
        # The time-curvature estimate: three frames sampled at x0.
        c = b + 1 if b + 1 < len(frames) else a - 1
        fc = frames[c]
        uc, vc, wc, sc = fc.sample(x0, y0, z0, second_stage=False)
        ua_, va_, wa_, _ = fa.sample(x0, y0, z0, second_stage=False)
        ub_, vb_, wb_, _ = fb.sample(x0, y0, z0, second_stage=False)
        trio = sorted(((times[a], (ua_, va_, wa_)), (times[b], (ub_, vb_, wb_)), (times[c], (uc, vc, wc))),
                      key=lambda item: item[0])
        (t0, f0), (t1, f1), (t2, f2) = trio
        curvature = np.zeros(n, dtype=np.float64)
        for axis in range(3):
            g0, g1, g2 = (np.asarray(f[axis], dtype=np.float64) for f in (f0, f1, f2))
            second = 2.0 * ((g2 - g1) / (t2 - t1) - (g1 - g0) / (t1 - t0)) / (t2 - t0)
            curvature = curvature + second * second
        span = times[b] - times[a]
        estimate = (span * span / 8.0) * np.sqrt(curvature)
        step_len = np.sqrt((np.asarray(x1, np.float64) - x0) ** 2 + (np.asarray(y1, np.float64) - y0) ** 2
                           + (np.asarray(z1, np.float64) - z0) ** 2)
        change = np.sqrt((np.asarray(u2, np.float64) - u1) ** 2 + (np.asarray(v2, np.float64) - v1) ** 2
                         + (np.asarray(w2, np.float64) - w1) ** 2)
        rate = np.where(step_len > 0.0, change / np.where(step_len > 0.0, step_len, 1.0), 0.0)
        failed = live & ((s1 != 0) | (s2 != 0))
        status[failed] = np.where(s1 != 0, s1, s2)[failed]
        ok = live & ~failed
        left = ok & ~(fa.inside(x, y) & fb.inside(x, y))
        status[left] = STATUS_LEFT_INTERIOR
        ok &= ~left
        no_bound = ok & (sc != 0)
        status_note |= no_bound
        rows[step, 8, ok] = np.where(no_bound[ok], np.nan, estimate[ok]).astype(np.float32)
        delta = np.where(ok, delta * (1.0 + float(request.dt_s) * rate) + float(request.dt_s) * estimate, delta)
        delta = np.where(no_bound, np.nan, delta)
        amplification = np.where(ok, amplification * (1.0 + float(request.dt_s) * rate), amplification)
        positions[0, ok], positions[1, ok], positions[2, ok] = x[ok], y[ok], z[ok]
    status = np.where(status_note & (status == STATUS_ACTIVE), STATUS_BOUND_NOT_ESTIMABLE, status)
    rows[steps, 7] = status.astype(np.float32)
    return BackwardResult(rows, status, out_times, sorted(used))


def _integrate_blend_containing(frames: Sequence[ChildFrame],
                                request: BackwardRequest) -> BackwardResult:
    """``history-blend`` with a per-parcel bracket of frames that hold the parcel.

    Identical to :func:`_integrate_history_blend` wherever the two frames
    bracketing the step hold the parcel (the same blend, the same RK2 with one
    wind for both stages, the same error accounting).  Where one of them does
    not -- a vortex-following window has jumped away from the parcel for a few
    records -- the nearest earlier (later) frame within ``max_bracket_s`` whose
    free interior holds the parcel's position, with a margin of one step's
    travel for the second RK stage, is used instead, and the blend's
    ``(T**2/8)|u_tt|`` bound is formed over that wider span with a third frame
    that also holds the parcel.  A parcel no frame in reach holds stops as
    ``STATUS_LEFT_INTERIOR``.
    """

    np = _np()
    frames = sorted(frames, key=lambda f: f.model_time_s)
    if len(frames) < 3:
        raise BackwardTrajectoryError(
            "at least three frames are needed: the time-interpolation error estimate is a second "
            "divided difference, and a trajectory published without it has no stated accuracy"
        )
    times = np.asarray([f.model_time_s for f in frames], dtype=np.float64)
    if len(set(times.tolist())) != len(times):
        raise BackwardTrajectoryError("two frames carry the same model time")
    lo_t = min(float(request.start_time_s), float(request.end_time_s))
    hi_t = max(float(request.start_time_s), float(request.end_time_s))
    if hi_t > times[-1] or lo_t < times[0]:
        raise BackwardTrajectoryError(
            f"[{lo_t}, {hi_t}] s is not inside the frames' [{times[0]}, {times[-1]}] s; winds "
            "outside the saved frames would be extrapolated"
        )
    boxes = np.asarray([f.free_interior() for f in frames], dtype=np.float64)   # (F, 4)
    reach = float(request.max_bracket_s)
    steps = request.steps()
    positions = request.seed_positions()
    n = positions.shape[1]
    placed = positions[0] > np.float32(-1.0e29)
    rows = np.full((steps + 1, len(BACKWARD_ROWS), n), np.float32(SENTINEL_F32), dtype=np.float32)
    status = np.where(placed, STATUS_ACTIVE, STATUS_LEFT_LATTICE).astype(np.int32)
    status_note = np.zeros(n, dtype=bool)
    delta = np.zeros(n, dtype=np.float64)
    amplification = np.ones(n, dtype=np.float64)
    sign = request.sign
    dt = np.float32(sign * float(request.dt_s))
    dt2 = dt / np.float32(2.0)
    used: set[int] = set()
    out_times: list[float] = []
    margin = 0.0

    def holds(q: int, x: Any, y: Any, pad: float) -> Any:
        x_lo, x_hi, y_lo, y_hi = boxes[q]
        return (x >= x_lo + pad) & (x <= x_hi - pad) & (y >= y_lo + pad) & (y <= y_hi - pad)

    for step in range(steps + 1):
        t = float(request.start_time_s) + sign * step * float(request.dt_s)
        out_times.append(t)
        rows[step, 0:3, placed] = positions[:, placed].T
        rows[step, 6, placed] = np.float32(t)
        rows[step, 7] = status.astype(np.float32)
        rows[step, 9] = np.where(placed, delta, SENTINEL_F32).astype(np.float32)
        rows[step, 11] = np.where(placed, amplification, SENTINEL_F32).astype(np.float32)
        if step == steps:
            break
        live = status == STATUS_ACTIVE
        if not bool(live.any()):
            continue
        t_mid = t + sign * 0.5 * float(request.dt_s)
        x0, y0, z0 = positions[0].copy(), positions[1].copy(), positions[2].copy()
        xs, ys = np.asarray(x0, np.float64), np.asarray(y0, np.float64)
        # One step's travel at the fastest wind this step could plausibly
        # see, so the second RK stage stays inside the chosen frames.
        margin = float(request.dt_s) * 150.0
        right = int(np.searchsorted(times, t_mid, side="left"))
        earlier = [q for q in range(min(right, len(times) - 1), -1, -1)
                   if times[q] <= t_mid and t_mid - times[q] <= reach]
        later = [q for q in range(max(right - 1, 0), len(times))
                 if times[q] >= t_mid and times[q] - t_mid <= reach]
        a_of = np.full(n, -1, dtype=np.int64)
        b_of = np.full(n, -1, dtype=np.int64)
        for q in earlier:
            take = live & (a_of < 0) & holds(q, xs, ys, margin)
            a_of[take] = q
        for q in later:
            take = live & (b_of < 0) & holds(q, xs, ys, margin)
            b_of[take] = q
        # A step exactly on a frame time brackets with itself on one side;
        # keep a real interval.
        same = live & (a_of >= 0) & (a_of == b_of)
        if bool(same.any()):
            for p in np.flatnonzero(same):
                nxt = [q for q in later if q > a_of[p] and holds(q, xs[p], ys[p], margin)]
                if nxt:
                    b_of[p] = nxt[0]
                else:
                    prv = [q for q in earlier if q < a_of[p] and holds(q, xs[p], ys[p], margin)]
                    a_of[p] = prv[0] if prv else -1
        lost = live & ((a_of < 0) | (b_of < 0))
        status[lost] = STATUS_LEFT_INTERIOR
        go = live & ~lost
        pairs = np.unique(np.stack([a_of[go], b_of[go]]), axis=1) if bool(go.any()) else np.zeros((2, 0), int)
        new_x, new_y, new_z = x0.copy(), y0.copy(), z0.copy()
        for a, b in pairs.T:
            a, b = int(a), int(b)
            sel = go & (a_of == a) & (b_of == b)
            idx = np.flatnonzero(sel)
            fa, fb = frames[a], frames[b]
            used.update((fa.index, fb.index))
            alpha = np.float32((t_mid - times[a]) / (times[b] - times[a]))
            px, py, pz = x0[idx], y0[idx], z0[idx]

            def blend(xx: Any, yy: Any, zz: Any, second: bool) -> tuple[Any, Any, Any, Any]:
                ua, va, wa, sa = fa.sample(xx, yy, zz, second_stage=second)
                ub, vb, wb, sb = fb.sample(xx, yy, zz, second_stage=second)
                st = np.where(sa != 0, sa, sb)
                return (ua + alpha * (ub - ua), va + alpha * (vb - va), wa + alpha * (wb - wa), st)

            u1, v1, w1, s1 = blend(px, py, pz, False)
            rows[step, 3, idx] = u1
            rows[step, 4, idx] = v1
            rows[step, 5, idx] = w1
            x1, y1, z1 = px + dt * u1, py + dt * v1, pz + dt * w1
            u2, v2, w2, s2 = blend(x1, y1, z1, True)
            x = px + dt2 * (u1 + u2)
            y = py + dt2 * (v1 + v2)
            z = pz + dt2 * (w1 + w2)
            z = np.where(z < np.float32(0.0), np.float32(1.0e-6), z)
            maxz = np.float32(min(fa.deck.maxz_m, fb.deck.maxz_m))
            z = np.where(z <= maxz, z, maxz)
            # the third frame for the curvature estimate: the next one past b
            # (or before a) that also holds the parcel
            third = [q for q in range(b + 1, len(frames))
                     if times[q] - t_mid <= reach + (times[b] - times[a])]
            third += [q for q in range(a - 1, -1, -1)
                      if t_mid - times[q] <= reach + (times[b] - times[a])]
            c_of = np.full(idx.size, -1, dtype=np.int64)
            for q in third:
                take = (c_of < 0) & holds(q, np.asarray(px, np.float64), np.asarray(py, np.float64), 0.0)
                c_of[take] = q
            curvature = np.full(idx.size, np.nan, dtype=np.float64)
            ua_, va_, wa_, _ = fa.sample(px, py, pz, second_stage=False)
            ub_, vb_, wb_, _ = fb.sample(px, py, pz, second_stage=False)
            for c in np.unique(c_of[c_of >= 0]):
                m = c_of == c
                uc, vc, wc, sc = frames[int(c)].sample(px[m], py[m], pz[m], second_stage=False)
                trio = sorted(((times[a], (ua_[m], va_[m], wa_[m])), (times[b], (ub_[m], vb_[m], wb_[m])),
                               (times[int(c)], (uc, vc, wc))), key=lambda item: item[0])
                (t0, f0), (t1, f1), (t2, f2) = trio
                acc = np.zeros(int(m.sum()), dtype=np.float64)
                for axis in range(3):
                    g0, g1, g2 = (np.asarray(f[axis], dtype=np.float64) for f in (f0, f1, f2))
                    second_d = 2.0 * ((g2 - g1) / (t2 - t1) - (g1 - g0) / (t1 - t0)) / (t2 - t0)
                    acc = acc + second_d * second_d
                curvature[m] = np.where(sc != 0, np.nan, np.sqrt(acc))
            span = times[b] - times[a]
            estimate = (span * span / 8.0) * curvature
            step_len = np.sqrt((np.asarray(x1, np.float64) - px) ** 2 + (np.asarray(y1, np.float64) - py) ** 2
                               + (np.asarray(z1, np.float64) - pz) ** 2)
            change = np.sqrt((np.asarray(u2, np.float64) - u1) ** 2 + (np.asarray(v2, np.float64) - v1) ** 2
                             + (np.asarray(w2, np.float64) - w1) ** 2)
            rate = np.where(step_len > 0.0, change / np.where(step_len > 0.0, step_len, 1.0), 0.0)
            failed = (s1 != 0) | (s2 != 0)
            status[idx[failed]] = np.where(s1 != 0, s1, s2)[failed]
            ok = ~failed
            no_bound = ok & ~np.isfinite(curvature)
            status_note[idx[no_bound]] = True
            rows[step, 8, idx[ok]] = np.where(no_bound[ok], np.nan, estimate[ok]).astype(np.float32)
            d = delta[idx]
            d = np.where(ok, d * (1.0 + float(request.dt_s) * rate)
                         + float(request.dt_s) * np.where(no_bound, 0.0, estimate), d)
            delta[idx] = np.where(no_bound, np.nan, d)
            amplification[idx] = np.where(ok, amplification[idx] * (1.0 + float(request.dt_s) * rate),
                                          amplification[idx])
            new_x[idx[ok]], new_y[idx[ok]], new_z[idx[ok]] = x[ok], y[ok], z[ok]
        positions[0], positions[1], positions[2] = new_x, new_y, new_z
    status = np.where(status_note & (status == STATUS_ACTIVE), STATUS_BOUND_NOT_ESTIMABLE, status)
    rows[steps, 7] = status.astype(np.float32)
    del margin
    return BackwardResult(rows, status, out_times, sorted(used))


def _records_per_model_step(frames: Sequence[ChildFrame]) -> int | None:
    """How many model steps one record spans, or ``None`` when the frames do not say.

    A :mod:`cm1core.nest_child_window` record carries the child step it
    completes, so consecutive records that differ by one are one model step
    apart.  A stream written with ``stride_child_steps > 1`` skips steps: each
    record is still the average over ONE child step, but the interval between
    two of them is several, and the wind is then frozen across steps it was
    never the average of.
    """

    steps = [f.placement.get("child_step") for f in frames]
    if len(steps) < 2 or any(not isinstance(v, int) for v in steps):
        return None
    gaps = {int(steps[q + 1]) - int(steps[q]) for q in range(len(steps) - 1)}
    return gaps.pop() if len(gaps) == 1 else None


def _uniform_step(times: Sequence[float]) -> float:
    """The frames' own uniform spacing, or a refusal naming the first gap."""

    gaps = [times[q + 1] - times[q] for q in range(len(times) - 1)]
    first = gaps[0]
    for q, gap in enumerate(gaps):
        if abs(gap - first) > 1.0e-6 * max(1.0, abs(first)):
            raise BackwardTrajectoryError(
                f"the stream's records are not evenly spaced: {first} s between records 0 and 1 "
                f"but {gap} s between {q} and {q + 1}; the model-step source reverses one model "
                "step per record, and a missing record would silently reverse two"
            )
    return float(first)


def _rk2_one_field(frame: ChildFrame, x0: Any, y0: Any, z0: Any, dt: Any,
                   live: Any) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any]:
    """``parcel.F:521-568`` with one frozen wind for both stages, as CM1 has it.

    Returns ``(x, y, z, u1, v1, w1, status, rate)``: the new position, the
    first-stage wind that is published beside it, the first non-zero sampler
    status of the two stages, and ``|U(x1) - U(x0)| / |x1 - x0|``, the wind's
    rate of change along this step, which is the local Lyapunov factor the
    sensitivity amplification accumulates.
    """

    np = _np()
    u1, v1, w1, s1 = frame.sample(x0, y0, z0, second_stage=False)
    x1, y1, z1 = x0 + dt * u1, y0 + dt * v1, z0 + dt * w1
    u2, v2, w2, s2 = frame.sample(x1, y1, z1, second_stage=True)
    half = dt / np.float32(2.0)
    x = x0 + half * (u1 + u2)
    y = y0 + half * (v1 + v2)
    z = z0 + half * (w1 + w2)
    z = np.where(z < np.float32(0.0), np.float32(1.0e-6), z)
    maxz = np.float32(frame.deck.maxz_m)
    z = np.where(z <= maxz, z, maxz)
    step_len = np.sqrt((np.asarray(x1, np.float64) - x0) ** 2
                       + (np.asarray(y1, np.float64) - y0) ** 2
                       + (np.asarray(z1, np.float64) - z0) ** 2)
    change = np.sqrt((np.asarray(u2, np.float64) - u1) ** 2
                     + (np.asarray(v2, np.float64) - v1) ** 2
                     + (np.asarray(w2, np.float64) - w1) ** 2)
    rate = np.where(step_len > 0.0, change / np.where(step_len > 0.0, step_len, 1.0), 0.0)
    status = np.where(s1 != 0, s1, s2)
    del live
    return x, y, z, u1, v1, w1, status, rate


def _integrate_model_step(frames: Sequence[ChildFrame], request: BackwardRequest) -> BackwardResult:
    """One record per step, no interpolation, with the round trip measured.

    A record published for child step ``s`` holds the wind that moved the
    model's own parcels from ``t(s-1)`` to ``t(s)``
    (:mod:`cm1core.nest_child_window`), so the backward step **from** ``t(s)``
    reads the record whose model time is ``t(s)``.  One wind serves both RK2
    stages, which is what ``parcel.F`` itself does, so this step is the mirror
    of the model's own step in the same frozen field.
    """

    np = _np()
    frames = sorted(frames, key=lambda f: f.model_time_s)
    if len(frames) < 2:
        raise BackwardTrajectoryError(
            "the model-step source needs at least two records; one record reverses no step"
        )
    times = [f.model_time_s for f in frames]
    if len(set(times)) != len(times):
        raise BackwardTrajectoryError("two records carry the same model time")
    spacing = _uniform_step(times)
    stride = _records_per_model_step(frames)
    if stride is not None and stride != 1:
        raise BackwardTrajectoryError(
            f"this stream was written every {stride} child steps, so a record is the average "
            "over one model step but the interval between two of them is "
            f"{stride}; the model-step source would freeze that one step's wind across "
            f"{stride} of them and its receipt would say the time-interpolation error is zero "
            "when it is first order in the record interval.  Read it with "
            'wind_source = "history-blend", whose (T^2/8) estimate covers exactly that, or '
            "write the stream with stride_child_steps = 1"
        )
    steps = request.steps()
    if abs(float(request.dt_s) - spacing) > 1.0e-6 * max(1.0, spacing):
        raise BackwardTrajectoryError(
            f"dt_s = {request.dt_s} s is not the stream's own record spacing of {spacing} s; the "
            "model-step source reverses exactly one written model step per step, and any other "
            "dt would either skip a record's wind or reuse one"
        )
    index = {round(t / spacing): q for q, t in enumerate(times)}

    def frame_at(t: float) -> ChildFrame:
        key = round(t / spacing)
        if key not in index or abs(times[index[key]] - t) > 1.0e-6 * max(1.0, abs(t)):
            raise BackwardTrajectoryError(
                f"the stream holds no record at model time {t} s; its records run "
                f"[{times[0]}, {times[-1]}] s and a backward step from a time it does not hold "
                "would have no wind at all"
            )
        return frames[index[key]]

    # The first step is taken from the start time itself, the last from
    # end + dt, so those are the records the stream must hold.
    frame_at(float(request.start_time_s))
    frame_at(float(request.end_time_s) + float(request.dt_s))

    positions = request.seed_positions()
    n = positions.shape[1]
    placed = positions[0] > np.float32(-1.0e29)
    seed = np.array(positions, dtype=np.float32, copy=True)
    first = frame_at(float(request.start_time_s))
    outside = placed & ~first.inside(positions[0], positions[1])
    if bool(outside.any()):
        p = int(np.flatnonzero(outside)[0])
        lo_x, hi_x, lo_y, hi_y = first.free_interior()
        raise BackwardTrajectoryError(
            f"seed {p} at ({float(positions[0, p])}, {float(positions[1, p])}) m from the parent "
            f"grid origin is outside the window record at {request.start_time_s} s, which covers "
            f"x [{lo_x}, {hi_x}], y [{lo_y}, {hi_y}]; a trajectory cannot start where the stream "
            "holds no wind"
        )
    rows = np.full((steps + 1, len(BACKWARD_ROWS), n), np.float32(SENTINEL_F32), dtype=np.float32)
    status = np.where(placed, STATUS_ACTIVE, STATUS_LEFT_LATTICE).astype(np.int32)
    amplification = np.ones(n, dtype=np.float64)
    dt = np.float32(-float(request.dt_s))
    used: set[int] = set()
    out_times: list[float] = []
    for step in range(steps + 1):
        t = float(request.start_time_s) - step * float(request.dt_s)
        out_times.append(t)
        rows[step, 0:3, placed] = positions[:, placed].T
        rows[step, 6, placed] = np.float32(t)
        rows[step, 7] = status.astype(np.float32)
        # There is no time interpolation in this mode, so its error is not
        # estimated: it is zero.
        rows[step, 8, placed] = np.float32(0.0)
        rows[step, 11] = np.where(placed, amplification, SENTINEL_F32).astype(np.float32)
        if step == steps:
            break
        live = status == STATUS_ACTIVE
        if not bool(live.any()):
            continue
        frame = frame_at(t)
        used.add(frame.index)
        x, y, z, u1, v1, w1, sampled, rate = _rk2_one_field(
            frame, positions[0].copy(), positions[1].copy(), positions[2].copy(), dt, live)
        rows[step, 3, live] = u1[live]
        rows[step, 4, live] = v1[live]
        rows[step, 5, live] = w1[live]
        failed = live & (sampled != 0)
        status[failed] = sampled[failed]
        ok = live & ~failed
        left = ok & ~frame.inside(x, y)
        status[left] = STATUS_LEFT_INTERIOR
        ok &= ~left
        amplification = np.where(ok, amplification * (1.0 + float(request.dt_s) * rate),
                                 amplification)
        positions[0, ok], positions[1, ok], positions[2, ok] = x[ok], y[ok], z[ok]
    residual = _round_trip_residual(frames, request, rows, status, spacing, index, times)
    rows[:, 10] = residual
    # The published estimate.  The round trip alone is a LOWER bound and is
    # nearly blind where the flow is locally linear -- there RK2 backward and
    # RK2 forward are inverses to O(dt^4) and the measurement reads zero while
    # the trajectory still carries its own truncation.  What actually bounds a
    # backward trajectory into a tornado is the growth of the uncertainty in
    # where it started, which no scheme removes: the target position is not
    # known to better than the grid resolves, and Dahl, Parker and Wicker
    # (2012) is the demonstration that this is the term that matters.  The
    # estimate is therefore the measured irreversibility plus half a window
    # cell carried back through the trajectory's own amplification.
    seed_uncertainty = np.float64(0.5 * float(frames[0].deck.dx_m))
    amp = rows[:, 11]
    known = (residual > np.float32(-1.0e29)) & (amp > np.float32(-1.0e29))
    rows[:, 9] = np.where(
        known,
        np.asarray(residual, np.float64) + seed_uncertainty * np.asarray(amp, np.float64),
        np.float32(SENTINEL_F32),
    ).astype(np.float32)
    return BackwardResult(rows, status, out_times, sorted(used))


def _round_trip_residual(frames: Sequence[ChildFrame], request: BackwardRequest, rows: Any,
                         status: Any, spacing: float, index: Mapping[int, int],
                         times: Sequence[float]) -> Any:
    """Re-integrate forward from every parcel's last position and measure the gap.

    This is the whole error statement of the ``model-step`` source.  The wind
    is the model's own and is not interpolated, so what is left is the RK2
    scheme's own irreversibility and the trilinear spatial interpolation, and
    both of them are *measured* here rather than bounded: the forward pass runs
    the same records in the same order with ``+dt``, and the distance between
    the forward position and the backward position at the same model time is
    published beside every row.  A parcel that stopped early is compared over
    the steps it took.
    """

    np = _np()
    steps = rows.shape[0] - 1
    n = rows.shape[2]
    out = np.full((steps + 1, n), np.float32(SENTINEL_F32), dtype=np.float32)
    # The last output step at which each parcel still had a position: the
    # forward pass starts there, which is that parcel's own far end.
    stopped = np.full(n, -1, dtype=np.int64)
    for step in range(steps + 1):
        alive = (rows[step, 7] == STATUS_ACTIVE) & (rows[step, 0] > np.float32(-1.0e29))
        stopped = np.where(alive, step, stopped)
    started = stopped >= 0
    take = np.where(started, stopped, 0)
    columns = np.arange(n)
    x = rows[take, 0, columns].astype(np.float32)
    y = rows[take, 1, columns].astype(np.float32)
    z = rows[take, 2, columns].astype(np.float32)
    live = started.copy()
    dt = np.float32(float(request.dt_s))
    out[take, columns] = np.where(started, np.float32(0.0), np.float32(SENTINEL_F32))
    for back in range(steps - 1, -1, -1):
        # The record that carried the model from t(back) to t(back + 1) is the
        # one written at t(back + 1), which is the record the backward pass
        # used for that same step.
        key = round((float(request.start_time_s) - back * float(request.dt_s)) / spacing)
        frame = frames[index[key]]
        active = live & (stopped >= back + 1)
        nx, ny, nz, _u, _v, _w, sampled, _rate = _rk2_one_field(frame, x, y, z, dt, active)
        keep = active & (sampled == 0)
        x = np.where(keep, nx, x)
        y = np.where(keep, ny, y)
        z = np.where(keep, nz, z)
        gap = np.sqrt((np.asarray(x, np.float64) - rows[back, 0]) ** 2
                      + (np.asarray(y, np.float64) - rows[back, 1]) ** 2
                      + (np.asarray(z, np.float64) - rows[back, 2]) ** 2)
        out[back] = np.where(keep, gap, out[back]).astype(np.float32)
        live = live & (keep | (stopped < back + 1))
    del times
    return out


def write_backward_trajectories(directory: Path | str, result: BackwardResult,
                                frames: Sequence[ChildFrame], request: BackwardRequest) -> dict[str, Any]:
    np = _np()
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=False)
    raw = np.ascontiguousarray(result.rows, dtype="<f4").tobytes(order="C")
    with (root / BACKWARD_ARENA_FILENAME).open("xb", buffering=0) as handle:
        handle.write(raw)
        os.fsync(handle.fileno())
    table = ["index\tname\tunits"] + [f"{q}\t{name}\t{unit}" for q, (name, unit) in enumerate(BACKWARD_ROWS)]
    with (root / BACKWARD_TABLE_FILENAME).open("xb", buffering=0) as handle:
        handle.write(("\n".join(table) + "\n").encode("utf-8"))
        os.fsync(handle.fileno())
    model_step = request.wind_source == "model-step"
    # history-blend accumulates its estimate backward, so the far end of the
    # trajectory carries it; the model-step source measures the round trip,
    # which closes back at the seed.
    final_delta = result.rows[0 if model_step else -1, 9]
    finite = np.isfinite(final_delta) & (final_delta > -1.0e29)
    by_frame = {f.index: f for f in frames}
    receipt = {
        "schema": BACKWARD_SCHEMA,
        "kind": BACKWARD_KIND,
        "label": (
            "OFFLINE DIAGNOSTIC -- backward trajectories through saved child history frames with "
            "linearly time-interpolated winds; not the model's inline parcels, not a CM1 product"
        ),
        "arena_shape_steps_rows_nparcels": list(result.rows.shape),
        "arena_sha256": hashlib.sha256(raw).hexdigest(),
        "rows": [name for name, _ in BACKWARD_ROWS],
        "velocity_rows": "the step's stage-1 blended wind at the row's position; the last row is the sentinel",
        "model_times_s": result.times_s,
        "request": {
            "seeding": None if request.seeding is None else request.seeding.record(),
            "explicit_seed_positions": None if request.positions is None else int(np.asarray(request.positions).shape[1]),
            "start_time_s": request.start_time_s,
            "end_time_s": request.end_time_s, "dt_s": request.dt_s,
            "wind_source": request.wind_source,
            "direction": request.direction,
        },
        "frames_used": [by_frame[i].record() for i in result.frames_used],
        "status_codes": {str(k): v for k, v in BACKWARD_STATUS.items()},
        "status_counts": {str(int(k)): int(v) for k, v in zip(*np.unique(result.status, return_counts=True))},
        "method": ({
            "step": "CM1 parcel.F:137-568 RK2 and sampler with dt < 0 on the record's own lattice and deck",
            "wind_source": "model-step: one cm1core.nest_child_window record per step, the model's own advecting velocity (solve2.F:1290-1301), one wind for both RK2 stages exactly as parcel.F holds it",
            "time_interpolation": "none; the time-interpolation error is zero, not estimated",
            "error_measurement": "every parcel is re-integrated forward from where it stopped, through the same records in the same order with +dt; round_trip_residual is the distance back to the backward position at the same model time, and position_error_estimate is that same measurement",
            "sensitivity_amplification": "prod(1 + |dt| L_n) with L_n = |U(x1) - U(x0)| / |x1 - x0| along each step's own first stage: the factor by which a displacement at the target has grown by this time (Dahl, Parker and Wicker 2012)",
            "position_error_estimate": "round_trip_residual + (dx/2) * sensitivity_amplification: the measured irreversibility plus half a window cell of target-position uncertainty, carried back through the trajectory's own amplification",
            "round_trip_is_a_lower_bound": "where the flow is locally linear, RK2 with -dt and RK2 with +dt are inverses to O(dt^4), so the round trip reads near zero while the trajectory still carries its own O(dt^2) truncation; it measures irreversibility, not truncation, which is why the published estimate adds the amplified target uncertainty",
            "not_included": "the difference between the model's RK2 trajectory and the true characteristic, and any error already present in the model's own wind",
        } if model_step else {
            "step": "CM1 parcel.F:137-568 RK2 and sampler with dt < 0 on each frame's own lattice and deck",
            "wind_source": "history-blend: published completed-state history frames",
            "time_interpolation": "one wind per step: frames bracketing t - dt/2, U = S_a + alpha*(S_b - S_a), each sampled on its own lattice",
            "error_bound": "|u - U| <= (T^2/8) max|u_tt| (Lagrange remainder); max|u_tt| estimated per step by the second divided difference of three frames at the parcel",
            "accumulation": "delta_{n+1} = delta_n (1 + |dt| L_n) + |dt| E_n, L_n = |U(x1) - U(x0)| / |x1 - x0|",
            "not_included": "RK2 truncation error, spatial interpolation error, the difference between the published completed-state wind and the model's step-averaged advecting wind",
        }),
        "position_error_estimate_m": {
            "measured_at": "the seed time, after the round trip" if model_step else "the far end of the trajectory",
            "max": float(final_delta[finite].max()) if bool(finite.any()) else None,
            "median": float(np.median(final_delta[finite])) if bool(finite.any()) else None,
            "parcels_with_an_estimate": int(finite.sum()),
        },
        "citations": list(CITATIONS),
        "unvalidated": (
            "UNVALIDATED against CM1: CM1 r21.1 has no backward or offline trajectory mode. Checked "
            "analytically: exact reversal of a steady uniform wind across a nest move, an "
            "estimate that bounds the actual error for a wind quadratic in time, and -- for the "
            "model-step source -- a closed-form stretching vortex whose trajectory the backward "
            "integration reproduces to the RK2 scheme's own order."
        ),
    }
    with (root / BACKWARD_RECEIPT_FILENAME).open("xb", buffering=0) as handle:
        handle.write((json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"))
        os.fsync(handle.fileno())
    return receipt


__all__ = [
    "BACKWARD_KIND", "BACKWARD_ROWS", "BACKWARD_SCHEMA", "BACKWARD_STATUS", "WIND_SOURCES",
    "BackwardRequest",
    "BackwardResult", "BackwardTrajectoryError", "CITATIONS", "ChildFrame", "frame_from_arrays",
    "integrate_backward", "read_child_frame", "read_child_frames", "write_backward_trajectories",
]
