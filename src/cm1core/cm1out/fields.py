"""Each CM1 history variable, read from one GM1 frame.

:func:`plan` pairs every variable CM1 would write for the run's settings with
the reader that produces it from a frame, or with the plain-words reason GM1
does not produce it.  A reader returns float32 in CM1's own layout:
``(ny, nx)`` for grid ``2``, ``(nz, ny, nx)`` for ``s``, ``(nz, ny, nx+1)``
for ``u``, ``(nz, ny+1, nx)`` for ``v`` and ``(nz+1, ny, nx)`` for ``w``.

Where a frame carries the variable under CM1's name the words are copied.
Where GM1 keeps it under its own name the words are the same array
(``th3d`` is CM1's ``tha``, ``pp3d`` is ``ppi``, ``q3d``/``nc3d`` are the q3d
axis).  Where CM1 writes a base-state column as a 3-D field, the column is
broadcast, as CM1's own arrays are.  The ground-anchored swaths are sampled
from the canvas at the ground under each cell of the domain at the frame's
time (:func:`ground_window`).  The few derived fields use writeout.F's own
REAL arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Callable, Mapping

from .names import Cm1Variable, not_produced_reason, selected, species_variables, switches_not_produced
from .settings import Cm1Settings, Geometry, Species


@dataclass(frozen=True)
class PlannedVariable:
    variable: Cm1Variable
    read: Callable[[Any], Any]
    #: "copied", "renamed", "broadcast", "sampled" or "derived".
    how: str


@dataclass(frozen=True)
class Plan:
    variables: tuple[PlannedVariable, ...]
    not_produced: Mapping[str, str]
    #: &param9 switches that are on and whose whole family of fields GM1 never writes.
    switches_not_produced: Mapping[str, str] = field(default_factory=dict)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(row.variable.name for row in self.variables)


def _f32(np: Any, array: Any) -> Any:
    return np.ascontiguousarray(array, dtype=np.float32)


def _broadcast(np: Any, column: Any, shape: tuple[int, ...]) -> Any:
    column = np.asarray(column, dtype=np.float32).reshape(-1, *([1] * (len(shape) - 1)))
    return np.ascontiguousarray(np.broadcast_to(column, shape), dtype=np.float32)


#: CM1 r21.1 constants.F:17, :113-114, :137, as binary32.
_P00, _RD, _CP = 1.0e5, 287.04, 1005.7


def pi0_column(frame: Any) -> str | None:
    """The carrier holding the base-state pi0 column, if the frame has one."""

    for name in ("pi0", "base.pi0"):
        if frame.has(name) and len(frame.carriers[name].shape) == 1:
            return name
    return None


def reconstructed_prs(frame: Any) -> Any:
    """``prs = p00*(pi0+ppi)**(cp/rd)`` (solve2.F:1588), REAL, for a frame that omits prs.

    The reduced carrier profile (``--history-carriers reconstructible-omitted``)
    drops prs and keeps pp3d and the base.pi0 column it is formed from.  The
    power is rounded once from binary64, so a word can differ by one unit in
    the last place from the device's own powf.
    """

    import numpy as np

    f32 = np.float32
    name = pi0_column(frame)
    if name is None or not frame.has("pp3d"):
        return None
    total = (np.asarray(frame.array(name), dtype=f32).reshape(-1, 1, 1) + frame.array("pp3d")).astype(f32)
    cpdrd = f32(f32(_CP) / f32(_RD))
    return (f32(_P00) * np.power(total.astype(np.float64), float(cpdrd)).astype(f32)).astype(f32)


def _round_half_away(value: float) -> int:
    # tornado_diagnostics._round_half_away, the rule the canvas was placed with.
    return int(math.floor(value + 0.5)) if value >= 0.0 else -int(math.floor(-value + 0.5))


def ground_window(frame: Any, geometry: Geometry) -> tuple[int, int] | None:
    """Where the domain sits on the ground-anchored swath canvas at this frame.

    The runtime places every swath update on a ground-fixed canvas at the
    domain's exact ground position, ``domainlocx/domainlocy`` rounded to a
    whole cell (``tornado_diagnostics.GroundAnchor.place``), with the canvas
    origin at ``min(0, final offset)`` in each direction.  CM1 writes its
    translated swaths on the domain grid, so the CM1 field at this time is the
    ``ny x nx`` window of the canvas under the domain now.  Returns the
    window's (x, y) offset in canvas cells, or None when the frame does not
    say where the canvas is.
    """

    swaths = ((frame.publication.get("tornado_diagnostics") or {}).get("swaths")) or {}
    anchor = swaths.get("moving_frame") or {}
    moving = frame.moving_frame
    try:
        dt = float(anchor["dt_s"])
        steps = int(anchor["total_steps"])
        umove = float(anchor["u_move_m_s"])
        vmove = float(anchor["v_move_m_s"])
        locx = float(moving["domainlocx"])
        locy = float(moving["domainlocy"])
    except (KeyError, TypeError, ValueError):
        return None
    seconds = dt * steps
    final_x = _round_half_away(umove * seconds / geometry.dx)
    final_y = _round_half_away(vmove * seconds / geometry.dy)
    origin_x, origin_y = min(0, final_x), min(0, final_y)
    return (
        _round_half_away(locx / geometry.dx) - origin_x,
        _round_half_away(locy / geometry.dy) - origin_y,
    )


def plan(frame: Any, settings: Cm1Settings, geometry: Geometry, species: Species,
         *, reflectivity: Any = None) -> Plan:
    """What to write for each selected CM1 variable, from this frame."""

    import numpy as np

    nx, ny, nz = geometry.nx, geometry.ny, geometry.nz
    shapes = {"2": (ny, nx), "s": (nz, ny, nx), "u": (nz, ny, nx + 1),
              "v": (nz, ny + 1, nx), "w": (nz + 1, ny, nx)}
    wanted = selected(settings.values, species_variables(species.names, species.units))
    rows: list[PlannedVariable] = []
    missing: dict[str, str] = {}

    def has(name: str) -> bool:
        return frame.has(name)

    def copy(name: str, carrier: str | None = None, how: str = "copied") -> PlannedVariable:
        source = carrier or name
        return PlannedVariable(CM1[name], lambda f: _f32(np, f.array(source)), how)

    def column(name: str) -> Any:
        return lambda f: np.asarray(f.array(name), dtype=np.float32).reshape(-1)

    q_index = {n: i for i, n in enumerate(species.mass_names)}
    number_names = tuple(n for n in species.names if n not in q_index)
    n_index = {n: i for i, n in enumerate(number_names)}

    CM1 = {row.name: row for row in wanted}
    window = None
    for row in wanted:
        name = row.name
        grid_shape = shapes[row.grid]
        if row.grid == "s" and name in q_index and has("q3d"):
            index = q_index[name]
            rows.append(PlannedVariable(row, (lambda i: lambda f: _f32(np, f.array("q3d")[i]))(index), "renamed"))
            continue
        if row.grid == "s" and name in n_index and has("nc3d"):
            index = n_index[name]
            rows.append(PlannedVariable(row, (lambda i: lambda f: _f32(np, f.array("nc3d")[i]))(index), "renamed"))
            continue
        if name in ("sws2", "svs2", "sps2", "srs2", "sgs2", "sus2", "shs2"):
            if not has(name):
                missing[name] = not_produced_reason(name, uniform=geometry.uniform)
                continue
            if window is None:
                window = ground_window(frame, geometry)
            if window is None:
                missing[name] = "The frame does not record where its ground swath canvas sits."
                continue
            ox, oy = window
            canvas_shape = frame.carriers[name].shape
            if ox < 0 or oy < 0 or oy + ny > canvas_shape[0] or ox + nx > canvas_shape[1]:
                missing[name] = "The domain has moved off the frame's ground swath canvas."
                continue
            rows.append(PlannedVariable(
                row, (lambda carrier, x, y: lambda f: _f32(np, f.array(carrier)[y:y + ny, x:x + nx]))(name, ox, oy),
                "sampled",
            ))
            continue
        if has(name) and tuple(frame.carriers[name].shape) == grid_shape:
            rows.append(copy(name))
            continue
        # GM1 carries a base-state or level column where CM1 writes a 3-D field.
        if has(name) and tuple(frame.carriers[name].shape) in ((nz,),):
            rows.append(PlannedVariable(row, (lambda n, s: lambda f: _broadcast(np, column(n)(f), s))(name, grid_shape), "broadcast"))
            continue
        renamed = {"thpert": "th3d", "pipert": "pp3d", "u": "u3d", "v": "v3d", "w": "w3d",
                   "tke": "closure.tkea", "zs": "terrain_msl"}.get(name)
        if renamed and has(renamed) and tuple(frame.carriers[renamed].shape) == grid_shape:
            rows.append(copy(name, renamed, "renamed"))
            continue
        derived = _derived(np, name, frame, geometry, grid_shape, species, settings)
        if derived is not None:
            rows.append(PlannedVariable(row, derived, "derived"))
            continue
        if name in ("dbz", "cref") and reflectivity is not None:
            reader = reflectivity(frame, name)
            if reader is not None:
                rows.append(PlannedVariable(row, reader, "derived"))
                continue
            # The reflectivity step says why this frame cannot give it.
            from .reflectivity import not_produced_reason as _reflectivity_reason

            missing[name] = _reflectivity_reason(frame)
            continue
        missing[name] = not_produced_reason(name, uniform=geometry.uniform)
    return Plan(tuple(rows), missing, switches_not_produced(settings.values))


def _derived(np: Any, name: str, frame: Any, geometry: Geometry, shape: tuple[int, ...],
             species: Species, settings: Cm1Settings) -> Callable[[Any], Any] | None:
    """The fields writeout.F forms from others, when the frame has the inputs."""

    has = frame.has
    if name == "zhval":
        return lambda f: _broadcast(np, geometry.zh, shape)
    if name == "uinterp" and has("u3d"):
        # writeout.F:4166-4172, REAL arithmetic.
        return lambda f: _f32(np, np.float32(0.5) * (f.array("u3d")[:, :, :-1] + f.array("u3d")[:, :, 1:]))
    if name == "vinterp" and has("v3d"):
        return lambda f: _f32(np, np.float32(0.5) * (f.array("v3d")[:, :-1, :] + f.array("v3d")[:, 1:, :]))
    if name == "winterp" and has("w3d"):
        return lambda f: _f32(np, np.float32(0.5) * (f.array("w3d")[:-1] + f.array("w3d")[1:]))
    column = lambda f, n: np.asarray(f.array(n), dtype=np.float32).reshape(-1, 1, 1)  # noqa: E731
    if name == "th" and has("th3d") and has("th0"):
        return lambda f: _f32(np, column(f, "th0") + f.array("th3d"))
    if name == "prs" and pi0_column(frame) is not None and has("pp3d"):
        return reconstructed_prs
    if name == "pi" and settings.int("psolver", 3) != 6 and pi0_column(frame) is not None and has("pp3d"):
        # writeout.F:3336-3343: pi0+ppi, REAL, for every psolver but 6.
        base = pi0_column(frame)
        return lambda f: _f32(np, column(f, base) + f.array("pp3d"))
    if name == "pi" and has("prs"):
        # writeout.F:3326-3334 (psolver = 6): (prs*rp00)**rovcp, REAL.  For
        # the other solvers it is the inverse of the model's own
        # prs = p00*(pi0+ppi)**cpdrd, so it agrees with pi0+ppi to rounding.
        rovcp = np.float32(np.float32(_RD) / np.float32(_CP))
        rp00 = np.float32(np.float32(1.0) / np.float32(_P00))
        return lambda f: _f32(np, np.power((f.array("prs") * rp00).astype(np.float64), float(rovcp)))
    if name == "prspert" and has("prs") and has("prs0"):
        return lambda f: _f32(np, f.array("prs") - column(f, "prs0"))          # writeout.F:3309
    if name == "qvpert" and has("q3d") and has("qv0"):
        return lambda f: _f32(np, f.array("q3d")[species.nqv - 1] - column(f, "qv0"))  # :3554
    if name == "upert" and has("u3d") and has("u0"):
        return lambda f: _f32(np, f.array("u3d") - column(f, "u0"))            # :6875
    if name == "vpert" and has("v3d") and has("v0"):
        return lambda f: _f32(np, f.array("v3d") - column(f, "v0"))            # :7336
    return None


__all__ = ["Plan", "PlannedVariable", "ground_window", "pi0_column", "plan", "reconstructed_prs"]
