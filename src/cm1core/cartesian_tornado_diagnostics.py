"""Bind CM1's rotation diagnostics and surface swaths to the live runtime.

:mod:`cm1core.tornado_diagnostics` owns the mathematics and knows nothing
about CUDA.  This module is the call site: it reads the selection the TOML
document sealed, prices every array against the runtime's own device
registry, folds one swath update into every completed large step, refreshes
the instantaneous fields at each published history frame, and hands the
history envelope the carriers to publish.

Nothing here runs when the selection is off.  A run that declares no
``[diagnostics]`` table constructs none of this, allocates none of it, and
publishes exactly the twelve carriers it always did.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from .cuda_memory_inventory import shape_bytes
from .tornado_diagnostics import (
    NEAR_SURFACE_VORTEX_S_INV,
    SOURCE_ANCHORS,
    SWATH_GROUND_SUFFIX,
    SWATH_MEANINGS,
    SWATH_UNITS,
    TORNADO_DIAGNOSTIC_CLAIM_LIMIT,
    TORNADO_DIAGNOSTIC_SCHEMA,
    UH_ROTATING_UPDRAFT_M2_S2,
    UH_STRONG_ROTATION_M2_S2,
    GroundAnchor,
    LowerBoundaryState,
    SurfaceSwaths,
    SwathRestartLayout,
    SwathRestartState,
    TornadoDiagnosticSelection,
    boundary_closure_words,
    potential_vorticity,
    tornado_report,
    updraft_helicity,
    vertical_column,
    vertical_vorticity_level,
    vorticity_components,
)


TORNADO_CARRIER_PREFIX = "tornado"
#: The registry name of the base-state pressure column ``sps`` differences
#: against.  It is CM1's own ``prs0`` from the Rust initializer, uploaded
#: once: reconstructing it from ``pi0`` with a Python power would publish a
#: pressure perturbation whose zero is not the model's own.
BASE_PRESSURE_BUFFER = f"{TORNADO_CARRIER_PREFIX}.prs0"


@dataclass(frozen=True, slots=True)
class TornadoCarrier:
    """One published diagnostic field and where this runtime holds it."""

    name: str
    meaning: str
    units: str
    stagger: str
    shape: tuple[int, ...]
    key: str

    @property
    def nbytes(self) -> int:
        return shape_bytes(self.shape)


def tornado_carrier_table(
    config: Any, *, anchor: GroundAnchor | None = None
) -> tuple[TornadoCarrier, ...]:
    """Every carrier this selection adds to a published history frame.

    The names are CM1's own output names (``writeout.F:4339`` ``xvort``,
    ``:4360`` ``yvort``, ``:4381`` ``zvort``, ``:4408`` ``pv``, ``:727``
    ``uh``, ``:412-538`` the seven swaths, ``:575-711`` their translated
    twins under CM1's ``name + "2"`` spelling), so a reader of a frame can
    find each one in CM1's own writer.
    """

    selection = config.diagnostics
    if selection is None or not selection.active:
        return ()
    g = config.grid
    volume = g.mass_shape
    surface = g.surface_shape
    rows: list[TornadoCarrier] = []
    if selection.output_uh:
        rows.append(TornadoCarrier(
            "uh", "integrated (2-5 km AGL) updraft helicity", "m2 s-2",
            "surface", surface, "uh",
        ))
    if selection.output_vort:
        rows.extend((
            TornadoCarrier("xvort", "horizontal vorticity (x)", "s-1", "mass", volume, "xvort"),
            TornadoCarrier("yvort", "horizontal vorticity (y)", "s-1", "mass", volume, "yvort"),
            TornadoCarrier("zvort", "vertical vorticity", "s-1", "mass", volume, "zvort"),
            TornadoCarrier(
                "zvort_sfc", "vertical vorticity at the lowest model level",
                "s-1", "surface", surface, "zvort_sfc",
            ),
        ))
    if selection.output_pv:
        rows.append(TornadoCarrier(
            "pv", "potential vorticity", "K m2 kg-1 s-1", "mass", volume, "pv",
        ))
    for name in selection.swath_names:
        rows.append(TornadoCarrier(
            name, SWATH_MEANINGS[name], SWATH_UNITS[name], "surface", surface, f"swath.{name}",
        ))
    # CM1 writes a second, ground-fixed copy of every swath whenever the
    # domain moves (writeout.F:568-720, nrain=2 at param.F:4698), under its
    # own name with a "2" appended.  Its extent is the run's whole track, not
    # the domain, so it can only be priced once the step count is known; a
    # caller that has not built the swaths yet gets the frame-relative set and
    # the profile says the ground set is absent rather than guessing a shape.
    if anchor is not None and anchor.imove:
        canvas = anchor.canvas_shape(g.ny, g.nx)
        for name in selection.swath_names:
            rows.append(TornadoCarrier(
                f"{name}{SWATH_GROUND_SUFFIX}",
                f"ground-anchored {SWATH_MEANINGS[name]} over the whole run",
                SWATH_UNITS[name], "ground-canvas", canvas, f"ground.{name}",
            ))
    return tuple(rows)


def tornado_diagnostic_bytes(config: Any, *, total_steps: int,
                             ground: bool = True) -> dict[str, int]:
    """Price every device word this selection needs, before allocating one.

    The transient peak is the vorticity routine's own scratch: two staggered
    volumes and three planes, all released before the next is taken.  It is
    listed separately because it is paid once per published frame and not
    once per step.
    """

    selection = config.diagnostics
    if selection is None or not selection.active:
        return {"persistent_bytes": 0, "transient_peak_bytes": 0, "total_peak_bytes": 0}
    g = config.grid
    nx, ny, nz = g.nx, g.ny, g.nz
    volume = shape_bytes((nz, ny, nx))
    plane = shape_bytes((ny, nx))
    persistent = 4 * nz  # the base-state pressure column
    if selection.needs_updraft_helicity:
        persistent += plane
    if selection.output_vort:
        persistent += 3 * volume + plane
    if selection.output_pv:
        persistent += volume
        if not selection.output_vort:
            persistent += 3 * volume
    anchor = GroundAnchor(
        dx_m=g.dx_m, dy_m=g.dy_m,
        u_move_m_s=config.u_move_m_s, v_move_m_s=config.v_move_m_s,
        dt_s=config.dt_s, total_steps=max(int(total_steps), 1),
    )
    swaths = len(selection.swath_names)
    persistent += swaths * plane
    if anchor.imove and ground:
        persistent += swaths * shape_bytes(anchor.canvas_shape(ny, nx))
    persistent += 4 * max(int(total_steps), 1) * 5  # four f32 series and one i32 index
    transient = 0
    if selection.needs_vorticity:
        transient = max(
            shape_bytes((nz + 1, ny + 1, nx)) + shape_bytes((ny - 1, nx))
            + shape_bytes((ny - 2, nx)),
            shape_bytes((nz + 1, ny, nx + 1)) + shape_bytes((ny, nx - 1))
            + shape_bytes((ny, nx - 2)),
        )
    if selection.output_pv:
        transient = max(
            transient,
            shape_bytes((nz, ny, nx + 1)) + volume,
            shape_bytes((nz, ny + 1, nx)) + volume,
            shape_bytes((nz + 1, ny, nx)) + volume,
        )
    return {
        "persistent_bytes": int(persistent),
        "transient_peak_bytes": int(transient),
        "total_peak_bytes": int(persistent + transient),
    }


def swath_restart_layout(
    config: Any, *, total_steps: int, ground: bool = True,
) -> SwathRestartLayout | None:
    """What a restart of this run carries of its swaths, with no runtime; no CUDA.

    The same layout :meth:`CartesianTornadoDiagnostics.restart_layout` gives
    for the live swaths of a run of ``total_steps`` large steps, so a restart
    can be priced (``cm1run.estimate``) before a device is opened.
    """

    selection = getattr(config, "diagnostics", None)
    if selection is None or not selection.active or not selection.swath_names:
        return None
    g = config.grid
    anchor = GroundAnchor(
        dx_m=float(g.dx_m), dy_m=float(g.dy_m),
        u_move_m_s=float(config.u_move_m_s), v_move_m_s=float(config.v_move_m_s),
        dt_s=float(config.dt_s), total_steps=max(int(total_steps), 1),
    )
    return SwathRestartLayout.for_anchor(
        selection.swath_names, ny=int(g.ny), nx=int(g.nx), anchor=anchor, ground=ground,
    )


def _require_plane(name: str, value: Any, shape: tuple[int, ...]) -> None:
    if tuple(value.shape) != tuple(shape):
        raise ValueError(
            f"the restored {name} swath has extent {tuple(value.shape)} and this "
            f"grid's swath is {tuple(shape)}; copying it would put one column's "
            "extremum at another column's ground"
        )


class CartesianTornadoDiagnostics:
    """CM1's rotation diagnostics and swaths bound to one live runtime."""

    def __init__(self, runtime: Any, *, total_steps: int, ground: bool = True) -> None:
        # ``ground=False`` keeps no ground-fixed "2" twin (SurfaceSwaths); a
        # live nest child (cm1core.nest_live.NestDriver.open_child_swaths) is
        # the one runtime that asks for it.
        if type(ground) is not bool:
            raise TypeError("ground must be a boolean")
        self.ground = ground
        selection = runtime.config.diagnostics
        if selection is None or not isinstance(selection, TornadoDiagnosticSelection):
            raise TypeError("the runtime configuration carries no diagnostic selection")
        if not selection.active:
            raise ValueError(
                "an all-off diagnostic selection asks for no field at all; "
                "constructing this would allocate device memory a run that "
                "declared no [diagnostics] table never asked for"
            )
        if type(total_steps) is not int or total_steps < 1:
            raise ValueError("total_steps must be a positive integer")
        self.runtime = runtime
        self.cp = runtime.cp
        self.stream = runtime.stream
        self.config = runtime.config
        self.grid = self.config.grid
        self.selection = selection
        self.registry = runtime.registry
        self.column = vertical_column(self.grid)
        self.lower = LowerBoundaryState(
            bbc=int(self.config.bbc),
            # CM1 run/namelist.input:71 and param.F: this lane's upper
            # boundary is the free-slip rigid lid the configuration door
            # already pins at boundaries.top=1.
            tbc=1,
            u_move_m_s=float(self.config.u_move_m_s),
            v_move_m_s=float(self.config.v_move_m_s),
            cecd=int(self.config.cecd),
            cnstcd=float(self.config.cnstcd),
        )
        self.anchor = GroundAnchor(
            dx_m=float(self.grid.dx_m), dy_m=float(self.grid.dy_m),
            u_move_m_s=float(self.config.u_move_m_s),
            v_move_m_s=float(self.config.v_move_m_s),
            dt_s=float(self.config.dt_s), total_steps=int(total_steps),
        )
        self.plan = tornado_diagnostic_bytes(self.config, total_steps=total_steps, ground=ground)
        self.rdx = self.cp.float32(1.0 / float(self.grid.dx_m))
        self.rdy = self.cp.float32(1.0 / float(self.grid.dy_m))
        self.refreshed_step: int | None = None
        self.restored_from_restart = False
        #: What continue_swaths did with a parent restart's swaths, if any.
        self.continued_from_restart: dict[str, Any] | None = None
        self.carriers: dict[str, Any] = {}
        self._transient: list[str] = []

        g = self.grid
        cp = self.cp
        if selection.needs_updraft_helicity:
            self.carriers["uh"] = self.registry.allocate(
                f"{TORNADO_CARRIER_PREFIX}.uh", g.surface_shape, zero=True
            )
        if selection.needs_vorticity:
            for name in ("xvort", "yvort", "zvort"):
                self.carriers[name] = self.registry.allocate(
                    f"{TORNADO_CARRIER_PREFIX}.{name}", g.mass_shape, zero=True
                )
            self.carriers["zvort_sfc"] = self.registry.allocate(
                f"{TORNADO_CARRIER_PREFIX}.zvort_sfc", g.surface_shape, zero=True
            )
        if selection.output_pv:
            self.carriers["pv"] = self.registry.allocate(
                f"{TORNADO_CARRIER_PREFIX}.pv", g.mass_shape, zero=True
            )
        # solve3.F:977 differences the lowest level against CM1's own prs0.
        self.prs0_plane = None
        if selection.output_sps:
            self.prs0_plane = self._upload_base_pressure()
        # The swaths open at the runtime's own step: zero for a run from its
        # initial condition, the restart step for a resumed one.
        clock = getattr(runtime, "clock", None)
        opened_step = 0 if clock is None else int(clock.completed_steps)
        self.swaths = SurfaceSwaths(
            cp, grid=g, selection=selection, anchor=self.anchor, column=self.column,
            allocate=self._allocate_named, ground=ground, opened_step=opened_step,
        ) if selection.swath_names else None

    # -- allocation --------------------------------------------------------

    def _allocate_named(self, name: str, shape: tuple[int, ...], dtype: Any) -> Any:
        return self.registry.allocate(
            f"{TORNADO_CARRIER_PREFIX}.{name}", shape, dtype=dtype
        )

    def _allocate_transient(self, name: str, shape: tuple[int, ...], dtype: Any) -> Any:
        full = f"{TORNADO_CARRIER_PREFIX}.transient.{name}"
        value = self.registry.allocate(full, shape, dtype=dtype)
        self._transient.append(full)
        return value

    def _release_transient(self, *names: str) -> None:
        wanted = [f"{TORNADO_CARRIER_PREFIX}.transient.{name}" for name in names]
        present = [name for name in wanted if name in self.registry.buffers]
        if present:
            self.registry.release(present)
            for name in present:
                self._transient.remove(name)

    def _release_all_transient(self) -> None:
        if self._transient:
            self.registry.release(list(self._transient))
            self._transient.clear()

    def _upload_base_pressure(self) -> Any:
        """CM1's own base-state pressure column, one upload for the whole run."""

        from .gm1_idealized_residency import SourceRequest

        source = getattr(self.runtime.dynamics, "source", None)
        if source is None:
            raise RuntimeError(
                "output_sps differences the lowest model level against CM1's "
                "prs0 (solve3.F:977) and this runtime retains no initializer "
                "source to read it from; a reconstructed base state would "
                "publish a pressure perturbation whose zero is not the model's"
            )
        uploaded = self.registry.upload(
            source, [SourceRequest(BASE_PRESSURE_BUFFER, "terrain_open.prs0", "profile", True)]
        )
        column = uploaded[BASE_PRESSURE_BUFFER]
        if tuple(column.shape) != (self.grid.nz,):
            raise ValueError(
                f"the initializer's prs0 profile has extent {tuple(column.shape)} "
                f"and this grid has {self.grid.nz} levels; differencing the "
                "lowest level against it would subtract another grid's base state"
            )
        with self.stream:
            plane = self.cp.broadcast_to(column[0], self.grid.surface_shape)
        return plane

    # -- per step ----------------------------------------------------------

    def update(self, step: int) -> None:
        """Fold one completed large step into every selected swath.

        ``solve3.F:855-1136`` runs at the end of every large step and so does
        this.  When ``output_shs`` is on the updraft-helicity field is
        recomputed here, because ``solve3.F:1119`` folds ``calcuh``'s own
        output and nothing else may stand in for it.
        """

        if self.swaths is None:
            return
        d = self.runtime.dynamics
        u, v, w = d.fields["u3d"], d.fields["v3d"], d.fields["w3d"]
        plane = None
        with self.stream:
            if self.selection.output_shs:
                plane = updraft_helicity(
                    self.cp, u=u, v=v, w=w, column=self.column,
                    rdx=self.rdx, rdy=self.rdy, out=self.carriers["uh"],
                )
            self.swaths.update(
                step=step, u=u, v=v, w=w,
                prs=d.fields["prs"],
                prs0_column=self.prs0_plane,
                q3d=d.fields["q3d"],
                updraft_helicity_plane=plane,
                domainlocx_m=float(self.runtime.clock.domainlocx_m),
                domainlocy_m=float(self.runtime.clock.domainlocy_m),
            )

    def restore_swaths(self, planes: Mapping[str, Any]) -> None:
        """Continue every selected swath from words a restart carried.

        ``restart_read.F:692-757`` reads ``sws``..``shs`` straight back into
        the arrays ``init3d.F:181-201`` would otherwise have seeded, so a
        resumed run's run-long extremum is the uninterrupted run's.  The set
        of planes must be exactly the selection's: a swath missing from the
        restart would restart from CM1's sentinel and publish a maximum over
        only the resumed part of the run under the whole run's name.
        """

        if self.swaths is None:
            raise ValueError("this selection keeps no swath to restore")
        wanted, given = set(self.swaths.names), set(planes)
        if wanted != given:
            raise ValueError(
                f"the restart carries swaths {sorted(given)} and this run selects "
                f"{sorted(wanted)}; a swath resumed from CM1's initial sentinel "
                "would report the resumed part of the run as the whole run"
            )
        if self.swaths.updates:
            raise RuntimeError(
                "the swaths have already folded a step of this run; restoring over "
                "them would discard it"
            )
        with self.stream:
            for name in self.swaths.names:
                value = planes[name]
                _require_plane(name, value, self.swaths.frame[name].shape)
                self.cp.copyto(self.swaths.frame[name], value)
                # The nest child restart carries the planes and not the step
                # their lineage began at.
                self.swaths.first_step[name] = None
        self.restored_from_restart = True

    def restart_layout(self) -> SwathRestartLayout | None:
        """What a parent restart written now carries of these swaths."""

        return None if self.swaths is None else self.swaths.restart_layout()

    def restart_words(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """The swath arrays and counters a parent restart at this step carries."""

        if self.swaths is None:
            raise ValueError("this selection keeps no swath for a restart to carry")
        with self.stream:
            return self.swaths.restart_words()

    def continue_swaths(self, state: SwathRestartState) -> dict[str, Any]:
        """Continue a parent restart's swaths (``restart_read.F:1064-1126``).

        Unlike :meth:`restore_swaths` (a nest child's own, whose set always
        matches), a parent resume may select swaths the restart does not carry
        -- they start at the restart step, and the record says so -- or drop
        some it does.
        """

        if self.swaths is None:
            raise ValueError("this selection keeps no swath to continue")
        with self.stream:
            record = self.swaths.continue_from(state)
        self.restored_from_restart = bool(record["continued"])
        self.continued_from_restart = record
        return record

    # -- per published frame ----------------------------------------------

    def refresh(self, step: int) -> None:
        """Recompute the instantaneous fields for one published frame."""

        d = self.runtime.dynamics
        u, v, w = d.fields["u3d"], d.fields["v3d"], d.fields["w3d"]
        try:
            with self.stream:
                if self.selection.output_uh:
                    updraft_helicity(
                        self.cp, u=u, v=v, w=w, column=self.column,
                        rdx=self.rdx, rdy=self.rdy, out=self.carriers["uh"],
                    )
                if self.selection.needs_vorticity:
                    vorticity_components(
                        self.cp, u=u, v=v, w=w, column=self.column,
                        rdx=self.rdx, rdy=self.rdy, lower=self.lower,
                        allocate=self._allocate_transient,
                        release=self._release_transient,
                        out={name: self.carriers[name] for name in ("xvort", "yvort", "zvort")},
                    )
                    vertical_vorticity_level(
                        self.cp, u=u, v=v, level=0, rdx=self.rdx, rdy=self.rdy,
                        out=self.carriers["zvort_sfc"],
                    )
                if self.selection.output_pv:
                    theta = self._allocate_transient("theta", self.grid.mass_shape, self.cp.float32)
                    # misclibs.F:1596: the full potential temperature.
                    self.cp.add(d.fields["th0"], d.fields["th3d"], out=theta)
                    potential_vorticity(
                        self.cp, theta=theta, rr=d.fields["rr"],
                        vorticity={name: self.carriers[name] for name in ("xvort", "yvort", "zvort")},
                        column=self.column, rdx=self.rdx, rdy=self.rdy,
                        allocate=self._allocate_transient,
                        release=self._release_transient,
                        out=self.carriers["pv"],
                    )
                    self._release_transient("theta")
        finally:
            self._release_all_transient()
        self.refreshed_step = int(step)

    # -- publication -------------------------------------------------------

    def frame_row(self, step: int) -> dict[str, Any]:
        """The scalars one diagnostic frame adds to a per-step record.

        Two numbers per selected swath cross to the host here: this step's own
        extremum and the whole-run one.  That is the only host traffic this
        module produces per frame, and none at all per step.
        """

        if self.swaths is None:
            return {}
        cp = self.cp
        to_host = getattr(cp, "asnumpy", None)
        slot = max(int(step), 1) - 1
        row: dict[str, Any] = {}
        names = {
            "svs_max": "near_surface_vertical_vorticity_max_s_inv",
            "sws_max": "near_surface_wind_max_m_s",
            "sps_min": "surface_pressure_perturbation_min_pa",
            "shs_max": "updraft_helicity_max_m2_s2",
        }
        for series, label in names.items():
            words = self.swaths.series[series]
            if slot >= words.shape[0]:
                continue
            value = words[slot]
            value = float(to_host(value) if to_host is not None else value)
            row[label] = value if value == value else None
        row["swath_extrema"] = self.swaths.extrema()
        row["near_surface_vortex_threshold_s_inv"] = NEAR_SURFACE_VORTEX_S_INV
        row["updraft_helicity_rotating_threshold_m2_s2"] = UH_ROTATING_UPDRAFT_M2_S2
        row["updraft_helicity_strong_threshold_m2_s2"] = UH_STRONG_ROTATION_M2_S2
        return row

    def carrier_table(self) -> tuple[TornadoCarrier, ...]:
        """The carriers this selection publishes, ground canvases included."""

        return tornado_carrier_table(self.config, anchor=self.anchor if self.ground else None)

    def published_carriers(self) -> Mapping[str, Any]:
        """Every device array this selection publishes, by carrier key."""

        out = dict(self.carriers)
        if self.swaths is not None:
            for name, plane in self.swaths.frame.items():
                out[f"swath.{name}"] = plane
            for name, canvas in self.swaths.ground.items():
                out[f"ground.{name}"] = canvas
        return MappingProxyType(out)

    def record(self) -> dict[str, Any]:
        """One host-side record of everything this selection produced."""

        g = self.grid
        row: dict[str, Any] = {
            "schema": TORNADO_DIAGNOSTIC_SCHEMA,
            "selection": self.selection.record(),
            "device_bytes": dict(self.plan),
            "refreshed_step": self.refreshed_step,
            "vertical_layer_m": [
                self.column.zf_m[0], self.column.zf_m[len(self.column.zf_m) - 1]
            ],
            "boundary_closure": {
                "zvort": boundary_closure_words(g.nx, g.ny, edges="both"),
                "xvort": boundary_closure_words(g.nx, g.ny, edges="y"),
                "yvort": boundary_closure_words(g.nx, g.ny, edges="x"),
            },
            "authority": dict(SOURCE_ANCHORS),
            "claim_limit": TORNADO_DIAGNOSTIC_CLAIM_LIMIT,
        }
        if self.swaths is not None:
            row["swaths"] = self.swaths.record()
            row["swaths"]["restored_from_restart"] = self.restored_from_restart
            if self.continued_from_restart is not None:
                row["swaths"]["continued_from_restart"] = dict(self.continued_from_restart)
            if not self.ground:
                row["swaths"]["ground_slot_omitted"] = (
                    "a live nest child's lattice moves inside its parent as well as "
                    "with the frame, so no ground canvas sized from the frame "
                    "velocity can hold it; each child frame's placement "
                    "(nest-child placement document: origin_x_m/origin_y_m and "
                    "domainlocx/domainlocy) puts the frame-relative swath on the ground"
                )
            if self.selection.tornado_report:
                row["tornado"] = tornado_report(
                    swaths=self.swaths, grid=g, column=self.column, lower=self.lower
                )
        return row


__all__ = (
    "BASE_PRESSURE_BUFFER",
    "TORNADO_CARRIER_PREFIX",
    "CartesianTornadoDiagnostics",
    "TornadoCarrier",
    "swath_restart_layout",
    "tornado_carrier_table",
    "tornado_diagnostic_bytes",
)
