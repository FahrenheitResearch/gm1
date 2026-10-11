"""The inline Lagrangian stage: what runs after ``solve3``, every large step.

CM1 calls ``parcel_driver`` from ``src/solve3.F:768-769``, at the end of the
large step, with the small-step-averaged velocities it saved at
``src/solve2.F:1290-1301``.  This stage is that call site.

It owns:

* the parcel positions and their sampled attributes (:mod:`cm1core.lagrangian_parcels`)
* the Eulerian vorticity budget (:mod:`cm1core.vorticity_budget`)
* the material circuits and their circulation (:mod:`cm1core.circulation_circuits`)
* the ``pdcomp`` decomposition (:mod:`cm1core.pressure_decomposition`)
* the high-frequency window (:mod:`cm1core.high_frequency_output`)

and nothing else.  It reads the runtime's live carriers and never writes to
them: turning any of this on cannot change the simulated storm, which is the
property that makes it safe to leave on.

Array module
------------
Everything here is written against an array module passed in, so the same
code path runs on CuPy device arrays in a run and on NumPy host arrays in a
test.  The parcel advance uses :mod:`cm1core.nest_parcels`' packaged kernel
when the module is CuPy and its transcribed host reference otherwise; the two
are compared raw-word, on uniform and stretched decks and with the semi-slip
log layer, in :mod:`tests.test_nest_parcels`.
"""

from __future__ import annotations

import importlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .circulation_circuits import CircuitDefinition, circulation_report
from .high_frequency_output import (
    HighFrequencyWindow,
    full_domain_window,
    high_frequency_cadence,
    write_high_frequency_window,
)
from .lagrangian_config import LagrangianConfig
from .cuda_psolver3_terrain_open import CM1_OPEN_NGXY
from .lagrangian_parcels import (
    ParcelAttributeLayout,
    ParcelGrid,
    ParcelSwitchContext,
    ParcelVelocityField,
    PARCEL_UNFILLED_ATTRIBUTES,
    parcel_circuit_positions,
    seed_parcel_positions,
)
from .nest_parcels import (
    NEST_PARCEL_KERNEL_ABI,
    NEST_PARCEL_KERNEL_SHA256,
    STATUS_REASONS,
    ParcelCarriers,
    ParcelDeck,
    SemiSlip,
    fill_velocity_ghosts,
    fill_velocity_halos,
    reference_rk2,
    reference_stencils,
    sample_parcel_attributes,
    semi_slip_for,
)
from .pressure_decomposition import PDCOMP_UNVALIDATED
from .vorticity_budget import VORTICITY_BUDGET_FIELD_NAMES, cpu_vorticity_budget

LAGRANGIAN_STAGE_SCHEMA = "gpuwm-cm1.lagrangian-stage/v1"
PARCEL_ARENA_FILENAME = "parcel-attributes.f32"
PARCEL_TABLE_FILENAME = "parcel-attribute-table.tsv"
PARCEL_RECEIPT_FILENAME = "parcel-track.json"
CIRCULATION_FILENAME = "circulation-circuits.json"
BUDGET_ARENA_FILENAME = "vorticity-budget.f32"
BUDGET_TABLE_FILENAME = "vorticity-budget-table.tsv"

#: What the receipt and ``doctor`` say about every capability this stage
#: exposes that has not been compared against CM1 in this tree.  A reader
#: never has to work out which is which.
#: Budget terms this lane holds no usable input for, and why.  A zero column
#: that is not named reads as "no friction", which for a tornado study is the
#: opposite of the truth.
BUDGET_UNFILLED_TERMS: Mapping[str, str] = {
    "friction": (
        "GM1's retained momentum tendency (held.uten/vten/wten) carries CM1's "
        "complete held RK tendency -- subgrid turbulence, Rayleigh damping and "
        "the large-scale terms together (src/solve1.F) -- not the frictional "
        "part alone, so publishing its curl under the name 'friction' would "
        "attribute damping and forcing to friction. The slot is zero unless "
        "the budget stage (lagrangian.vorticity_budget_history or "
        "parcel_vorticity_budget) ran on the frame's step, in which case it is "
        "calcvort of CM1's own ub/vb/wb_hturb + _vturb split and nothing else."
    ),
}

UNVALIDATED_MARKERS: Mapping[str, str] = {
    "vorticity_budget": (
        "UNVALIDATED against CM1: CM1 r21.1 has no vorticity budget, so no "
        "differential is possible. Validated analytically instead -- every "
        "term is exact to binary32 rounding for a velocity field linear in "
        "(x,y,z), and identically zero for solid-body rotation."
    ),
    "circulation": (
        "UNVALIDATED against CM1: CM1 r21.1 has no circuit diagnostic. "
        "Validated analytically instead -- the circulation of a regular "
        "N-gon in solid-body rotation is 2*omega times its area, in closed "
        "form."
    ),
    "pdcomp": PDCOMP_UNVALIDATED,
    "high_frequency_window": (
        "UNVALIDATED against CM1: hifrq.F writes the whole domain and is "
        "unreachable from any r21.1 namelist (cm1.F:2645 hard-codes "
        "dohifrq=.false.), so there is no CM1 run to compare a window to. "
        "The window's contents are a slice of the same carriers the full "
        "history frame publishes."
    ),
    "backward_trajectories": (
        "UNVALIDATED: the backward integrator runs the same RK2 with -dt. "
        "CM1 has no backward mode, so there is no differential; a "
        "forward-then-backward round trip is checked analytically only."
    ),
    "vorticity_threshold_seeding": (
        "UNVALIDATED against CM1: CM1 seeds one hard-coded grid "
        "(init3d.F:381-387) and has no threshold mode."
    ),
}


def _is_cupy(xp: Any) -> bool:
    return getattr(xp, "__name__", "") == "cupy"


@dataclass(slots=True)
class LagrangianStage:
    """One run's inline Lagrangian and budget state.

    The parcel step is :mod:`cm1core.nest_parcels`' transcription of
    ``parcel.F:137-619`` on this grid's own deck (its ``zf``/``zh`` faces, its
    ``cgs``/``cgt``, its ``bc.F`` laterals, ``xf(1)`` in its ``iorigin``
    frame) -- the same host reference and kernel text the nest child's
    parcels take, so there is one integrator and one proof of it.
    """

    config: LagrangianConfig
    grid: ParcelGrid
    dt_s: float
    xp: Any
    layout: ParcelAttributeLayout
    bbc: int = 1
    tbc: int = 1
    imove: int = 0
    u_move_m_s: float = 0.0
    v_move_m_s: float = 0.0
    lateral: tuple[int, int, int, int] = (2, 2, 2, 2)
    positions: Any = None
    status: Any = None
    velocities: Any = None
    circuits: tuple[CircuitDefinition, ...] = ()
    window: HighFrequencyWindow | None = None
    steps_taken: int = 0
    frames: list[dict[str, Any]] = field(default_factory=list)
    circulation_rows: list[dict[str, Any]] = field(default_factory=list)
    _module: Any = None
    _device_deck: Any = None
    _owner: Any = None
    _seeded: bool = False
    _velocity: ParcelVelocityField | None = None
    _surface: SemiSlip | None = None
    #: GM1 extension columns after CM1's attributes: ``nest_parcels.ParcelAttributeHook``s.
    hooks: list[Any] = field(default_factory=list)
    _frames_taken: bool = False

    @classmethod
    def build(
        cls,
        config: LagrangianConfig,
        *,
        nx: int, ny: int, nz: int,
        dx_m: float, dy_m: float, dz_m: float,
        dt_s: float,
        xp: Any,
        iorigin: int,
        levels: Any = None,
        bbc: int = 1,
        tbc: int = 1,
        u_move_m_s: float = 0.0,
        v_move_m_s: float = 0.0,
        imove: int | None = None,
        lateral: tuple[int, int, int, int] = (2, 2, 2, 2),
        switch_context: ParcelSwitchContext,
    ) -> "LagrangianStage":
        """``iorigin`` has no default: the door checks extents in it and every
        published position is measured in it, so both must be handed the one
        ``[initialization].iorigin`` rather than each assuming a frame.

        ``switch_context`` has none either: it is the run's own words
        (:func:`cm1core.lagrangian_parcels.parcel_switch_context`) that
        ``param.F:6257-6282`` resolves the ``prcl_*`` switches against, and its
        ``bbc`` must be the ``bbc`` this stage integrates with."""

        if int(switch_context.bbc) != int(bbc):
            raise ValueError(
                f"the parcel switches were resolved for bbc={switch_context.bbc} but this stage "
                f"integrates bbc={bbc}; param.F:6280-6282 decides prcl_sfc from the bbc the run uses"
            )
        grid = config.parcel_grid(
            nx=nx, ny=ny, nz=nz, dx_m=dx_m, dy_m=dy_m, dz_m=dz_m, iorigin=iorigin,
            levels=levels,
        )
        if imove is None:
            imove = int(float(u_move_m_s) != 0.0 or float(v_move_m_s) != 0.0)
        stage = cls(
            config=config, grid=grid, dt_s=float(dt_s), xp=xp,
            layout=config.attribute_layout(switch_context),
            bbc=int(bbc), tbc=int(tbc), imove=int(imove),
            u_move_m_s=float(u_move_m_s), v_move_m_s=float(v_move_m_s),
            lateral=tuple(int(v) for v in lateral),  # type: ignore[arg-type]
        )
        if stage.bbc not in (1, 2, 3):
            raise ValueError(f"bbc={stage.bbc} has no parcel.F lower-boundary branch")
        stage.grid.deck(stage.lateral)  # refuses a lateral rule bc.F's halo fill is not transcribed for
        if config.hifrq_enabled:
            v = config.values
            if v["hifrq_i1"] == 0 and v["hifrq_j1"] == 0 and v["hifrq_k1"] == 0:
                stage.window = full_domain_window(nx=nx, ny=ny, nz=nz)
            else:
                stage.window = HighFrequencyWindow(
                    int(v["hifrq_i0"]), int(v["hifrq_i1"]),
                    int(v["hifrq_j0"]), int(v["hifrq_j1"]),
                    int(v["hifrq_k0"]), int(v["hifrq_k1"]),
                )
            stage.window.require_within(nx=nx, ny=ny, nz=nz)
        if config.circuits_enabled:
            count = int(config.values["circuit_parcels"])
            stage.circuits = tuple(
                CircuitDefinition(f"circuit{index:02d}", index * count, count)
                for index in range(int(config.values["circuits"]))
            )
        return stage

    @property
    def deck(self) -> ParcelDeck:
        return self.grid.deck(self.lateral)

    def _boundary(self) -> Any:
        from types import SimpleNamespace

        return SimpleNamespace(bbc=self.bbc, imove=self.imove,
                               u_move_m_s=self.u_move_m_s, v_move_m_s=self.v_move_m_s)

    # ------------------------------------------- the advecting velocity
    @property
    def advecting_velocity(self) -> ParcelVelocityField | None:
        """CM1's ``uten1``/``vten1``/``wten1`` for this step, or ``None``."""

        return self._velocity

    def capture_advecting_velocity(self, arrays: Mapping[str, Any],
                                   roughness: Any = None) -> ParcelVelocityField:
        """Snapshot ``rru``/``rrv``/``rrw`` at the end of the final RK acoustics.

        This is ``src/solve2.F:1290-1301`` and it is not optional.  CM1 copies
        the acoustic small-step time average into ``uten1``/``vten1``/
        ``wten1`` at ``nrk == nrkmax`` *specifically because* the very next
        thing the large step does is refill ``rru``/``rrv``/``rrw`` with the
        mass flux for the scalar advection -- ``src/adv.F`` wants
        ``rho0*u``, not ``u``.  GM1 does the same refill in
        ``cuda_advection5.py:330``, so a parcel driver that read
        ``arrays["rru"]`` at ``solve3`` time would advect every parcel with
        ``rho0*u`` instead of ``u``: a height-dependent error that is +11 %
        near the ground and -55 % at 9 km, and which looks like a plausible
        trajectory rather than a broken one.

        The lateral halos are ``bc.F``'s ``bcu``/``bcv``/``bcw`` for this
        grid's ``wbc..nbc`` (``solve3.F:498-508`` calls them on these arrays
        before ``parcel_driver``).  ``roughness`` is the grid's
        ``closure.surface.znt``, which ``bbc = 3`` needs: ``parcel.F:470-515``
        reads that step's ``zntmp``.
        """

        xp = self.xp
        if self._velocity is None:
            self._velocity = ParcelVelocityField(
                xp.empty_like(arrays["rru"]),
                xp.empty_like(arrays["rrv"]),
                xp.empty_like(arrays["rrw"]),
            )
        velocity = self._velocity
        for name, target in (("rru", velocity.u), ("rrv", velocity.v), ("rrw", velocity.w)):
            xp.copyto(target, arrays[name])
        deck = self.deck
        fill_velocity_halos(xp, velocity.u, velocity.v, velocity.w, deck, ng=CM1_OPEN_NGXY)
        self._surface = semi_slip_for(xp, self._boundary(), deck, roughness, ng=CM1_OPEN_NGXY)
        return velocity

    # ---------------------------------------------------------------- seeding
    def seed(self, *, vertical_vorticity: Any = None) -> dict[str, Any]:
        """Place every parcel.  Circuits occupy the front of the array.

        A circuit is a *named* subset of the parcels; putting the circuits
        first and the free parcels after them is what lets one integrator run
        both.  ``circulation_circuits.CircuitDefinition`` carries the offsets,
        so a reader of the published arena can find each ring.
        """

        np = importlib.import_module("numpy")
        seeding = self.config.seeding()
        host = seed_parcel_positions(
            seeding, self.grid,
            vertical_vorticity=None if vertical_vorticity is None
            else np.asarray(_to_host(vertical_vorticity, self.xp)),
        )
        v = self.config.values
        for index, circuit in enumerate(self.circuits):
            ring = parcel_circuit_positions(
                center_x_m=float(v["circuit_center_x_m"]),
                center_y_m=float(v["circuit_center_y_m"]),
                center_z_m=float(v["circuit_center_z_m"])
                + float(v["circuit_spacing_z_m"]) * index,
                radius_m=float(v["circuit_radius_m"]),
                count=circuit.count,
            )
            host[:, circuit.first_parcel:circuit.stop] = ring
        n = int(host.shape[1])
        self.positions = self.xp.asarray(host)
        self.status = self.xp.zeros((n,), dtype=self.xp.int32)
        self._owner = self.xp.ones((n,), dtype=self.xp.int32)
        self.velocities = np.zeros((3, n), dtype=np.float32)
        self._seeded = True
        return {
            "schema": LAGRANGIAN_STAGE_SCHEMA,
            "seeding": seeding.record(),
            "circuits": [c.record() for c in self.circuits],
            "placed_parcels": int((host[0] > -1.0e29).sum()),
            "nparcels": n,
        }

    # ------------------------------------------------------------------- step
    def advance(self, velocity: ParcelVelocityField) -> None:
        """One ``parcel.F:137-619`` RK2 step against this step's advecting velocity.

        The ``parcel.F:59-130`` vertical ghost fill is applied first, with this
        deck's weights -- CM1 does exactly that inside ``parcel_driver``, every
        step.  A parcel the step cannot take (``nest_parcels.STATUS_REASONS``)
        keeps its position and is not stepped again; ``record()`` counts them.
        """

        if not self.config.parcels_enabled:
            return
        if not self._seeded:
            raise RuntimeError("the Lagrangian stage was advanced before it was seeded")
        xp, deck = self.xp, self.deck
        fill_velocity_ghosts(
            xp, velocity.u, velocity.v, velocity.w, deck, bbc=self.bbc, tbc=self.tbc,
            u_move_m_s=self.u_move_m_s, v_move_m_s=self.v_move_m_s,
        )
        surface = self._surface if self.bbc == 3 else None
        if self.bbc == 3 and surface is None:
            raise RuntimeError(
                "bbc=3 but no zntmp was captured with this step's advecting velocity; "
                "parcel.F:470-515 cannot take a parcel below zh(1) without it"
            )
        dt_s = -self.dt_s if self.config.backward else self.dt_s
        if _is_cupy(xp):
            from .nest_parcels import compile_nest_parcel_module, launch_nest_parcel_rk2, upload_deck

            if self._module is None:
                self._module = compile_nest_parcel_module(xp)
            if self._device_deck is None:
                self._device_deck = upload_deck(xp, deck)
            launch_nest_parcel_rk2(
                xp, self._module, positions=self.positions, owner=self._owner, status=self.status,
                code=1, u=velocity.u, v=velocity.v, w=velocity.w, deck=deck,
                device_deck=self._device_deck, dt_s=dt_s, ng=CM1_OPEN_NGXY,
                stream=xp.cuda.get_current_stream(), surface=surface,
            )
        else:
            self.positions, self.status = reference_rk2(
                xp, self.positions, self._owner, self.status, code=1,
                u=velocity.u, v=velocity.v, w=velocity.w, deck=deck, dt_s=dt_s,
                ng=CM1_OPEN_NGXY, surface=surface,
            )
        self.steps_taken += 1

    # ----------------------------------------------------------------- frames
    # ------------------------------------------------------------ extension columns
    def attribute_names(self) -> tuple[str, ...]:
        """CM1's attributes in ``parcel.F`` order, then every hook's columns."""

        names = list(self.layout.names)
        for hook in self.hooks:
            names.extend(hook.names)
        return tuple(names)

    def attribute_units(self) -> tuple[str, ...]:
        from .lagrangian_parcels import parcel_attribute_units

        units = [parcel_attribute_units(name) for name in self.layout.names]
        for hook in self.hooks:
            units.extend(hook.units)
        return tuple(units)

    def register_hook(self, hook: Any) -> None:
        """Append a :class:`cm1core.nest_parcels.ParcelAttributeHook`'s columns.

        The same contract as ``NestParcelTracker.register_hook``: fixed before
        the first frame (a column added mid-run would change the arena's shape
        between frames), one unit per name, and no name CM1 already publishes.
        """

        if self._frames_taken:
            raise RuntimeError("attribute hooks must be registered before the first parcel frame")
        if len(hook.names) != len(hook.units):
            raise ValueError("a parcel attribute hook needs one unit per name")
        taken = set(self.attribute_names())
        clash = [name for name in hook.names if name in taken]
        if clash:
            raise ValueError(f"parcel attribute hook column {clash[0]!r} is already published")
        self.hooks.append(hook)

    def parcel_frame(self, *, step: int, model_time_s: float,
                     carriers: ParcelCarriers | None = None) -> dict[str, Any]:
        """One ``prclfrq`` row: every attribute at every parcel, plus its record.

        ``carriers`` is this grid's :class:`cm1core.nest_parcels.ParcelCarriers`.
        Every attribute but the position is ``parcel_interp``'s sample of the
        *model state* at the frame (``cm1.F:4010-4020`` passes ``u3d``/``v3d``/
        ``w3d``, not the advecting average), on the same stencils, and on a
        semi-slip grid with ``parcel.F:1515-1583``'s log layer.  A parcel the
        integrator stopped keeps its position and publishes the sentinel for
        the rest.
        """

        np = importlib.import_module("numpy")
        host_pos = np.asarray(_to_host(self.positions, self.xp), dtype=np.float32)
        n = int(host_pos.shape[1])
        self._frames_taken = True
        names = self.attribute_names()
        rows = np.full((len(names), n), np.float32(-1.0e30), dtype=np.float32)
        index = {name: q for q, name in enumerate(names)}
        for axis, name in enumerate(("x", "y", "z")):
            rows[index[name]] = host_pos[axis]
        rows[index["mtime"]] = np.float32(model_time_s)
        wanted = [name for name in self.layout.names
                  if name not in ("x", "y", "z", "mtime") and name not in PARCEL_UNFILLED_ATTRIBUTES]
        velocities = np.full((3, n), np.float32(-1.0e30), dtype=np.float32)
        if carriers is not None and wanted:
            host_status = np.asarray(_to_host(self.status, self.xp), dtype=np.int32)
            placed = host_pos[0] > np.float32(-1.0e29)
            live = placed & (host_status == 0)
            stencils = reference_stencils(np, carriers.deck, host_pos[0], host_pos[1], host_pos[2],
                                          second_stage=False)
            live &= stencils.status == 0
            chosen = np.flatnonzero(live)
            if chosen.size:
                values = sample_parcel_attributes(self.xp, carriers, host_pos[:, chosen], wanted,
                                                  dict(self.layout.number_rows))
                for name, column in values.items():
                    rows[index[name], chosen] = column
                for axis, name in enumerate(("u", "v", "w")):
                    velocities[axis, chosen] = rows[index[name], chosen]
            if self.hooks:
                from .nest_parcels import ParcelSampleContext

                # This stage's parcels all belong to its one grid (code 1, the
                # owner advance() launches with); a stopped or off-deck parcel
                # is in no index and keeps the sentinel.
                context = ParcelSampleContext(
                    step=int(step), model_time_s=float(model_time_s), positions=host_pos,
                    owner=np.asarray(_to_host(self._owner, self.xp), dtype=np.int32),
                    indices={1: chosen}, carriers={1: carriers}, xp=self.xp,
                    sample=sample_parcel_attributes,
                )
                for hook in self.hooks:
                    columns = hook.sample(context)
                    for name in hook.names:
                        column = np.asarray(columns[name], dtype=np.float32)
                        if column.shape != (n,):
                            raise ValueError(f"parcel attribute hook column {name!r} is not ({n},)")
                        rows[index[name]] = column
        self.velocities = velocities
        return {
            "step": int(step),
            "model_time_s": float(model_time_s),
            "attributes": rows,
            "unfilled": {
                name: PARCEL_UNFILLED_ATTRIBUTES[name]
                for name in self.layout.unfilled
            },
        }

    def circulation_frame(self, *, model_time_s: float) -> dict[str, Any] | None:
        """The circuits' circulation from the winds the last parcel frame recorded."""

        if not self.circuits:
            return None
        np = importlib.import_module("numpy")
        report = circulation_report(
            np.asarray(_to_host(self.positions, self.xp), dtype=np.float64),
            np.asarray(self.velocities, dtype=np.float64),
            self.circuits, model_time_s=model_time_s,
        )
        report["unvalidated"] = UNVALIDATED_MARKERS["circulation"]
        self.circulation_rows.append(report)
        return report

    def write_budget_frame(
        self, directory: Path | str, fields: Mapping[str, Any], *,
        step: int, model_time_s: float,
    ) -> dict[str, Any]:
        """One Eulerian vorticity-budget frame: arena, table, receipt.

        ``fields`` are host ``(nz, ny, nx)`` rectangles keyed by
        :data:`cm1core.vorticity_budget.VORTICITY_COMPONENT_NAMES` and
        :data:`VORTICITY_BUDGET_FIELD_NAMES`.
        """

        from .vorticity_budget import VORTICITY_COMPONENT_NAMES

        order = [*VORTICITY_COMPONENT_NAMES, *VORTICITY_BUDGET_FIELD_NAMES]
        window = HighFrequencyWindow(0, self.grid.nx, 0, self.grid.ny, 0, self.grid.nz)
        return write_high_frequency_window(
            directory, fields, window, order=order,
            nx=self.grid.nx, ny=self.grid.ny, nz=self.grid.nz,
            model_time_s=model_time_s, step=step,
            extra={
                "kind": "vorticity-budget",
                "unvalidated": UNVALIDATED_MARKERS["vorticity_budget"],
                "unfilled_terms": dict(BUDGET_UNFILLED_TERMS),
            },
        )

    def unvalidated(self) -> dict[str, str]:
        """Exactly which exposed capabilities carry no CM1 comparison."""

        out: dict[str, str] = {}
        if self.config.budget_enabled:
            out["vorticity_budget"] = UNVALIDATED_MARKERS["vorticity_budget"]
        if self.circuits:
            out["circulation"] = UNVALIDATED_MARKERS["circulation"]
        if self.config.pdcomp_enabled:
            out["pdcomp"] = UNVALIDATED_MARKERS["pdcomp"]
        if self.config.hifrq_enabled:
            out["high_frequency_window"] = UNVALIDATED_MARKERS["high_frequency_window"]
        if self.config.backward:
            out["backward_trajectories"] = UNVALIDATED_MARKERS["backward_trajectories"]
        if self.config.values["seeding"] == "vorticity-threshold":
            out["vorticity_threshold_seeding"] = (
                UNVALIDATED_MARKERS["vorticity_threshold_seeding"]
            )
        return out

    def stopped_parcels(self) -> dict[str, int]:
        """How many parcels the integrator could not step, by ``STATUS_REASONS``."""

        if self.status is None:
            return {}
        np = importlib.import_module("numpy")
        status = np.asarray(_to_host(self.status, self.xp), dtype=np.int32)
        return {STATUS_REASONS[int(code)]: int((status == code).sum())
                for code in sorted(set(status.tolist()) - {0})}

    def record(self) -> dict[str, Any]:
        return {
            "schema": LAGRANGIAN_STAGE_SCHEMA,
            # param.F:6239-6500: the switches and columns this stage's layout carries.
            "configuration": self.config.record(self.layout.context),
            "grid": self.grid.record(),
            "integrator": {
                "transcription": "cm1core.nest_parcels (parcel.F:137-619 on this grid's deck)",
                "kernel_abi": NEST_PARCEL_KERNEL_ABI,
                "kernel_sha256": NEST_PARCEL_KERNEL_SHA256,
                "bbc": self.bbc, "tbc": self.tbc,
                "lateral_wbc_ebc_sbc_nbc": list(self.lateral),
                "below_lowest_level": {
                    1: "parcel.F:63-72 free-slip ghost extrapolation",
                    2: "parcel.F:73-93 no-slip ghost",
                    3: "parcel.F:470-515 neutral log layer over zntmp (closure.surface.znt, bc2d-padded)",
                }[self.bbc],
                "recorded_winds": "u3d/v3d/w3d at the frame through parcel_interp (parcel.F:1134-1343,1515-1589)",
            },
            "stopped_parcels": self.stopped_parcels(),
            "steps_taken": int(self.steps_taken),
            "circuits": [c.record() for c in self.circuits],
            "window": None if self.window is None else self.window.record(),
            "unvalidated": self.unvalidated(),
            "changes_the_simulated_storm": False,
            **({"attribute_hooks": [{"names": list(h.names), "marker": h.marker} for h in self.hooks]}
               if self.hooks else {}),
        }


def _to_host(value: Any, xp: Any) -> Any:
    if value is None:
        return None
    if _is_cupy(xp):
        return xp.asnumpy(value)
    return value


def write_parcel_track(
    directory: Path | str,
    stage: LagrangianStage,
    frames: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Write the whole track: one raw arena, one table, one receipt.

    The arena is ``(frames, npvals, nparcels)`` little-endian float32 in that
    order.  CM1 writes ``(npvals, nparcels)`` per time to ``cm1out_pdata.dat``
    (``parcel.F:1626``); the variable order is the same, so the same reader
    works with one transposition stated in the table.
    """

    import numpy as np

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    stack = np.stack([np.asarray(f["attributes"], dtype=np.float32) for f in frames]) \
        if frames else np.zeros((0, len(stage.attribute_names()), 0), dtype=np.float32)
    raw = np.ascontiguousarray(stack).tobytes(order="C")
    arena = root / PARCEL_ARENA_FILENAME
    with arena.open("xb", buffering=0) as handle:
        handle.write(raw)
        os.fsync(handle.fileno())
    table_rows = ["index\tname\tunits\tfilled"]
    for index, (name, units) in enumerate(zip(stage.attribute_names(), stage.attribute_units())):
        filled = "no" if name in stage.layout.unfilled else "yes"
        table_rows.append(f"{index}\t{name}\t{units}\t{filled}")
    with (root / PARCEL_TABLE_FILENAME).open("xb", buffering=0) as handle:
        handle.write(("\n".join(table_rows) + "\n").encode("utf-8"))
        os.fsync(handle.fileno())
    import hashlib

    receipt = {
        "schema": LAGRANGIAN_STAGE_SCHEMA,
        "arena_shape_frames_npvals_nparcels": list(stack.shape),
        "arena_bytes": len(raw),
        "arena_sha256": hashlib.sha256(raw).hexdigest(),
        "model_times_s": [float(f["model_time_s"]) for f in frames],
        "stage": stage.record(),
        "cm1_authority": {
            "write": "src/parcel.F:1626-1690 (writepdata / cm1out_pdata.dat)",
            "descriptor": "src/parcel.F:2145-2210 (write_prclctl)",
        },
    }
    with (root / PARCEL_RECEIPT_FILENAME).open("xb", buffering=0) as handle:
        handle.write((json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8"))
        os.fsync(handle.fileno())
    if stage.circulation_rows:
        with (root / CIRCULATION_FILENAME).open("xb", buffering=0) as handle:
            handle.write(
                (json.dumps(stage.circulation_rows, indent=2, sort_keys=True) + "\n")
                .encode("utf-8")
            )
            os.fsync(handle.fileno())
    fd = os.open(root, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return receipt


__all__ = [
    "BUDGET_UNFILLED_TERMS", "BUDGET_ARENA_FILENAME", "BUDGET_TABLE_FILENAME", "CIRCULATION_FILENAME",
    "LAGRANGIAN_STAGE_SCHEMA", "LagrangianStage", "PARCEL_ARENA_FILENAME",
    "PARCEL_RECEIPT_FILENAME", "PARCEL_TABLE_FILENAME", "UNVALIDATED_MARKERS",
    "cpu_vorticity_budget",
    "high_frequency_cadence", "write_high_frequency_window",
    "write_parcel_track", "VORTICITY_BUDGET_FIELD_NAMES",
]
