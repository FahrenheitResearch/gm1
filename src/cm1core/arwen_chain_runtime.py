"""The device side of the ArWen -> GM1 chain.

The chain domain is a standalone GM1 Cartesian runtime whose lateral forcing
is the live nest's :class:`~cm1core.nest_live.NestLateralForcing` -- the same
device tables, the same Davies kernel, the same specified-zone hooks in the
acoustic loop and after ``scalar_moist`` -- with one difference: the tables
are filled from a prepared ArWen history (:mod:`cm1core.arwen_chain`) instead
of being captured from a running parent.

WRF's specified lane decides the table pair and its clock
(``arwen_chain.AUTHORITY``): at every boundary time the tables are loaded
with ``value = driving state at the interval start`` and
``tend = (next - this)/interval`` and ``dtbc`` is reset to zero
(``share/input_wrf.F:1833-1835``); before every GM1 large step ``dtbc`` is
advanced by ``dt`` (``dyn_em/solve_em.F:377``).  The step itself, and every
placement inside it, is ``records/CM1-NEST-SEAM.md`` section 5.2 unchanged.

At the end of every step, after microphysics and before ``solve3`` copies the
new state into the time-level-``n`` arrays, the specified rows of u, v, pi',
theta and every transferred species are set to ``value + dtbc*tend`` -- WRF's
``spec_bdy_final`` in its specified-lane form (:data:`SPEC_BDY_FINAL`), run by
the live nest's own :meth:`NestLateralForcing.finalize_specified`.  Without it
the specified rows keep whatever microphysics did to them, step after step, and
drift away from the driving run (measured: theta by up to 6.4 K and rain
water by 14 g/kg on the outer row after 30 min of the ArWen WK82 box).

``chain.moisture = "vapour-only"`` is WRF's ``have_bcs_moist = .false.``
(``arwen_chain.MOISTURE_LANES``): only water vapour has tables; every other
species starts at zero and, after each RK scalar stage, takes WRF's
``flow_dep_bdy`` on the specified rows (:data:`FLOW_DEP_BDY`).

As for a nest child, CM1's dry-mass correction (``apmasscon``) does not run:
the domain takes its mass through the specified zone, and holding it to its
initial dry mass would add a domain-uniform pressure offset that the
boundary rows contradict (``nest_live`` module docstring).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .arwen_chain import (
    AUTHORITY,
    CHAIN_SCHEMA,
    GM1_CHOICES,
    MOISTURE_LANES,
    SIDES,
    PreparedChain,
    host_interval_tables,
    perturbation,
    transferred_species,
)
from .nest_live import (
    NestLateralForcing,
    _baseline,
    _carriers,
    _current,
    _grid_lattice,
    _refill_halos_and_thermodynamics,
    make_child_runtime_class,
)

CHAIN_RUNTIME_SCHEMA = "gm1.arwen-chain-execution/v1"

#: WRF's end-of-step reset of the specified zone (the chain's offline-parent lane).
SPEC_BDY_FINAL = (
    "WRF v4.6.1 dyn_em/solve_em.F:4531-4716, the end of solve_em, after microphysics: "
    "spec_bdy_final (share/module_bc.F:2066-2216) sets the specified rows to "
    "bdy + dtbc*bdy_tend.  A specified (real-data) domain does it for u, v, t, ph, mu "
    "(:4540-4620) and for moist only when im == P_QV or have_bcs_moist (:4622-4623); w "
    "only when nested (:4566).  GM1 carries pi' for t/ph/mu, so it resets u3d, v3d, pp3d, "
    "th3d and each transferred q3d species on the specified rows (gm1_nest_spec_apply "
    "modes 2 and 3; loop ownership spec_bdyupdate's, the y sides owning the corners), "
    "before solve3's copy into ua/va/ppi/tha/qa.  A species target is held at zero or "
    "above (GM1's choice: the reset follows pdefq, and value + dtbc*tend rounds a few "
    "1e-12 below zero where both saved times hold none)."
)

#: WRF's boundary rule for the moist species it has no boundary data for (vapour-only lane).
FLOW_DEP_BDY = (
    "WRF v4.6.1 dyn_em/solve_em.F:2337-2349: with specified boundaries and have_bcs_moist = "
    ".false., every moist species but P_QV takes flow_dep_bdy after rk_update_scalar in each "
    "RK stage, and gets no relaxation (:2256-2258) and no spec_bdy_final (:4622-4623).  "
    "share/module_bc.F:2335-2456 flow_dep_bdy: on the specified rows, zero where the normal "
    "velocity at the row's outer face flows in, and the first interior row's value (the "
    "along-boundary index clamped to the interior) where it flows out; the y sides own the "
    "corners.  WRF reads the sign of the stage's mass-coupled ru_m/rv_m; GM1 reads the sign "
    "of the stage's rru/rrv, CM1's small-step-averaged flux that advected the scalars (sound.F:814-827)."
)


@dataclass(frozen=True)
class _Zones:
    """The two geometry words :class:`NestLateralForcing` reads outside ``refresh``."""

    spec_zone: int
    relax_zone: int


class ChainLateralForcing(NestLateralForcing):
    """The live nest's forcing, its tables loaded from an offline ArWen history."""

    #: WRF's specified (real-data) lane of spec_bdy_final (:data:`SPEC_BDY_FINAL`).
    SPEC_BDY_FINAL_LANE = "specified"

    def __init__(self, *, cupy: Any, stream: Any, child: Any, prepared: PreparedChain,
                 spec_zone: int, relax_zone: int, species: tuple[str, ...],
                 th0: Any, pi0: Any, transferred: tuple[str, ...] | None = None) -> None:
        super().__init__(
            cupy=cupy, stream=stream, geometry=_Zones(int(spec_zone), int(relax_zone)),
            child=child, parent_dt_s=prepared.interval_s,
        )
        if self.width != prepared.width:
            raise ValueError(
                f"the prepared strips are {prepared.width} rows wide and the forcing reads "
                f"{self.width}; the Davies zone would index past the strips"
            )
        self.prepared = prepared
        self.species = tuple(species)
        self.transferred = tuple(self.species if transferred is None else transferred)
        unknown = [name for name in self.transferred if name not in self.species]
        if unknown:
            raise ValueError(f"transferred species {unknown} are not on the runtime's moisture axis")
        self.th0 = th0
        self.pi0 = pi0
        driven = {f"q{s}" for s, name in enumerate(self.species) if name in self.transferred}
        self.carriers = tuple(c for c in self.all_carriers if c.species is None or c.name in driven)
        self.flow_dependent = tuple(c for c in self.all_carriers
                                    if c.species is not None and c.name not in driven)
        for carrier in self.flow_dependent:
            del self.tables[carrier.name]
        self.loaded_intervals: list[int] = []
        self.flow_dependent_stages = 0

    def refresh(self, parent: Any, child: Any) -> None:  # pragma: no cover - guard
        raise RuntimeError("an ArWen chain has no running parent; its tables come from load_interval")

    def load_interval(self, interval: int) -> None:
        """``med_latbound_in``: this interval's WRF table pair, ``dtbc = 0``."""

        import numpy

        tables = host_interval_tables(
            self.prepared, interval, th0=self.th0, pi0=self.pi0, species=self.species,
            transferred=self.transferred,
        )
        with self.stream:
            for carrier in self.carriers:
                for side in SIDES:
                    value, tend = tables[carrier.name][side]
                    device_value, device_tend = self.tables[carrier.name][side]
                    if tuple(value.shape) != tuple(device_value.shape):
                        raise ValueError(
                            f"{carrier.name} {side} strip is {value.shape} and the device table "
                            f"{tuple(device_value.shape)}; the prepared chain is for another grid"
                        )
                    self.cp.copyto(device_value, self.cp.asarray(value))
                    self.cp.copyto(device_tend, self.cp.asarray(tend))
        self._dtbc = numpy.float32(0.0)
        self.refreshes += 1
        self.tables_ready = True
        self.loaded_intervals.append(int(interval))

    def spec_bdy_final_carriers(self) -> tuple[Any, ...]:
        """The specified lane: u, v, pi', theta and each transferred species (not w)."""

        return tuple(c for c in self.carriers if c.name != "w")

    def after_scalar_stage(self, dynamics: Any, nrk: int) -> None:
        """WRF ``flow_dep_bdy`` on every species the history does not drive (:data:`FLOW_DEP_BDY`)."""

        if not self.flow_dependent:
            return
        cp = self.cp
        g = dynamics.geometry
        nz, ny, nx = int(g.nz), int(g.ny), int(g.nx)
        sz = self.spec_zone
        rru, rrv = dynamics.arrays["rru"], dynamics.arrays["rrv"]
        if tuple(rru.shape) != tuple(dynamics.arrays["u3d"].shape) or \
                tuple(rrv.shape) != tuple(dynamics.arrays["v3d"].shape):
            raise RuntimeError("rru/rrv do not share u3d/v3d's padded layout; the inflow sign would be "
                               "read from other faces")
        # The physical u (nz, ny, nx+1) and v (nz, ny+1, nx) faces (origin (1, 3, 3), as u3d/v3d).
        u = rru[1:1 + nz, 3:3 + ny, 3:3 + nx + 1]
        v = rrv[1:1 + nz, 3:3 + ny + 1, 3:3 + nx]
        zero = cp.float32(0.0)
        with self.stream:
            for carrier in self.flow_dependent:
                field, origin = _current(dynamics, carrier)
                if tuple(origin) != (0, 0, 0):
                    raise RuntimeError(f"{carrier.name}: flow_dep_bdy needs a physical carrier")
                for d in range(sz):
                    # y sides own the corners (b_limit = b_dist): i = d .. nx-1-d.
                    i = cp.arange(d, nx - d)
                    inner = cp.clip(i, sz, nx - 1 - sz)
                    south = field[:, sz, :][:, inner]
                    field[:, d, d:nx - d] = cp.where(v[:, d, d:nx - d] < zero, south, zero)
                    north = field[:, ny - 1 - sz, :][:, inner]
                    field[:, ny - 1 - d, d:nx - d] = cp.where(v[:, ny - d, d:nx - d] > zero, north, zero)
                    # x sides: j = d+1 .. ny-2-d.
                    j = cp.arange(d + 1, ny - 1 - d)
                    if int(j.size) == 0:
                        continue
                    jinner = cp.clip(j, sz, ny - 1 - sz)
                    west = field[:, :, sz][:, jinner]
                    field[:, d + 1:ny - 1 - d, d] = cp.where(u[:, d + 1:ny - 1 - d, d] < zero, west, zero)
                    east = field[:, :, nx - 1 - sz][:, jinner]
                    field[:, d + 1:ny - 1 - d, nx - 1 - d] = cp.where(
                        u[:, d + 1:ny - 1 - d, nx - d] > zero, east, zero)
        from . import cuda_graphs

        cuda_graphs.bookkeeping(self._count_flow_dependent_stage, 0)

    def _count_flow_dependent_stage(self, _step: int) -> None:
        self.flow_dependent_stages += 1

    def record(self) -> dict[str, Any]:
        out = super().record()
        out["spec_bdy_final"]["authority"] = SPEC_BDY_FINAL
        out["moisture"] = {
            "transferred": list(self.transferred),
            "flow_dependent": [c.name for c in self.flow_dependent],
            "flow_dependent_species": [self.species[c.species] for c in self.flow_dependent],
            "flow_dep_bdy_authority": FLOW_DEP_BDY if self.flow_dependent else None,
            "flow_dependent_stages": self.flow_dependent_stages,
        }
        return out


class ChainDriver:
    """Initial state, then per step: tables at boundary times, ``dtbc += dt``."""

    def __init__(self, *, runtime: Any, prepared: PreparedChain, spec_zone: int,
                 relax_zone: int, cupy: Any, stream: Any) -> None:
        from .moisture_species import moisture_species_profile

        self.runtime, self.prepared, self.cp, self.stream = runtime, prepared, cupy, stream
        d = runtime.dynamics
        species = tuple(moisture_species_profile(runtime.config.microphysics).names)
        if len(species) != int(d.numq):
            raise ValueError("the runtime's moisture axis is not its component's")
        th0 = cupy.asnumpy(d.fields["th0"])
        pi0 = cupy.asnumpy(d.fields["pi0"])
        for name, base in (("th0", th0), ("pi0", pi0)):
            if not bool((base == base[:, :1, :1]).all()):
                raise RuntimeError(
                    f"{name} is not horizontally uniform, so one base column would be "
                    "subtracted from columns with another base state"
                )
        self.th0 = th0[:, 0, 0].copy()
        self.pi0 = pi0[:, 0, 0].copy()
        self.species = species
        self.moisture = prepared.moisture
        self.transferred = transferred_species(runtime.config.microphysics, self.moisture)
        self.steps_per_interval = int(prepared.steps_per_interval)
        if float(runtime.config.dt_s) * self.steps_per_interval != prepared.interval_s:
            raise ValueError("the runtime step does not divide the prepared boundary interval")
        self.forcing = ChainLateralForcing(
            cupy=cupy, stream=stream, child=d, prepared=prepared, spec_zone=spec_zone,
            relax_zone=relax_zone, species=species, th0=self.th0, pi0=self.pi0,
            transferred=self.transferred,
        )
        d.nest_forcing = self.forcing
        self.initialized = False
        self.steps = 0
        self.initial_record: dict[str, Any] = {}

    def _host_carrier(self, name: str) -> Any:
        if name in ("u", "v", "w"):
            return self.prepared.initial(name)
        if name == "pp":
            return perturbation(self.prepared.initial("pi"), self.pi0)
        if name == "th":
            return perturbation(self.prepared.initial("theta"), self.th0)
        species = self.species[int(name[1:])]
        if species not in self.transferred:
            # The vapour-only lane: a hydrometeor starts at zero, as a real.exe start from an
            # analysis without hydrometeors does (arwen_chain.MOISTURE_LANES).
            import numpy

            return numpy.zeros_like(self.prepared.initial("theta"))
        return self.prepared.initial(species)

    def initialize(self) -> dict[str, Any]:
        """``ndown``'s initial condition: every prognostic carrier from the t0 frame.

        TKE is left as GM1's initializer set it (``arwen_chain.GM1_CHOICES``).
        """

        import numpy

        cp = self.cp
        d = self.runtime.dynamics
        if int(self.runtime.clock.completed_steps) != 0:
            raise RuntimeError("the chain's initial state is step 0's; this runtime has stepped")
        carriers = list(_carriers(int(d.numq)))
        with self.stream:
            for carrier in carriers:
                lattice = _grid_lattice(d, carrier.stagger)
                host = self._host_carrier(carrier.name)
                if tuple(host.shape) != tuple(lattice):
                    raise ValueError(
                        f"the prepared {carrier.name} is {host.shape} and the GM1 lattice {lattice}"
                    )
                target = cp.asarray(numpy.ascontiguousarray(host, dtype=numpy.float32))
                for owner_of in (_current, _baseline):
                    owner, (k0, j0, i0) = owner_of(d, carrier)
                    cp.copyto(owner[k0:k0 + lattice[0], j0:j0 + lattice[1], i0:i0 + lattice[2]], target)
                if carrier.name == "pp":
                    for name in ("pp3d", "ppi"):
                        cp.copyto(d.arrays[name][1:1 + lattice[0], 1:1 + lattice[1], 1:1 + lattice[2]], target)
        _refill_halos_and_thermodynamics(d)
        self.stream.synchronize()
        # The chain's initial state is interpolated onto this grid (CM1's
        # interp_on_restart rule, arwen_chain.AUTHORITY), and for such a start
        # CM1 diagnoses the first step's TKE with turbsmag (src/init3d.F:
        # 2688-2760); a fresh iinit start does not.
        self.runtime.closure.interpolated_tke_start = True
        finish = self.runtime.finish
        mass = finish._initial_mass()
        finish.initial_mass1_f64 = mass
        d.mass1_f64 = mass
        pp = self._host_carrier("pp").astype(numpy.float64)
        self.initial_record = {
            "carriers": [c.name for c in carriers],
            "tke": "GM1 initializer's own (not transferred)",
            "moisture": {"lane": self.moisture, "meaning": MOISTURE_LANES[self.moisture],
                         "transferred": list(self.transferred),
                         "started_at_zero": [n for n in self.species if n not in self.transferred]},
            "domain_mean_pi_perturbation_max_abs": float(numpy.max(numpy.abs(pp.mean(axis=(1, 2))))),
            "initial_dry_mass_kg": mass,
        }
        self.initialized = True
        return dict(self.initial_record)

    def before_step(self) -> None:
        if not self.initialized:
            raise RuntimeError("the chain domain is being stepped before its initial state was written")
        step = int(self.runtime.clock.completed_steps)
        if step % self.steps_per_interval == 0:
            self.forcing.load_interval(step // self.steps_per_interval)
        self.forcing.advance_clock()
        self.steps += 1

    def record(self) -> dict[str, Any]:
        receipt = self.prepared.receipt
        return {
            "schema": CHAIN_RUNTIME_SCHEMA,
            "chain_schema": CHAIN_SCHEMA,
            "prepared_directory": str(self.prepared.root),
            "admission": receipt["admission"],
            "vertical_support": receipt.get("vertical_support"),
            "interval_s": self.prepared.interval_s,
            "steps_per_interval": self.steps_per_interval,
            "intervals_loaded": list(self.forcing.loaded_intervals),
            "chain_steps": self.steps,
            "initial_state": dict(self.initial_record),
            "lateral_forcing": self.forcing.record(),
            "skipped_phases": ["dry-mass-reduction", "dry-mass-correction"],
            "authority": dict(AUTHORITY),
            "gm1_choices": dict(GM1_CHOICES),
        }


def make_chain_runtime_class(base: type) -> type:
    """A runtime that skips ``apmasscon`` and drives the chain before each step."""

    child = make_child_runtime_class(base)

    class ArwenChainRuntime(child):  # type: ignore[misc,valid-type]
        chain_driver: Any = None

        # WRF's spec_bdy_final runs in the child runtime's own execute (nest_live), before solve3.

        def step(self) -> None:
            if self.chain_driver is None:
                raise RuntimeError("an ArWen chain runtime was stepped with no chain attached")
            self.chain_driver.before_step()
            super().step()

    return ArwenChainRuntime


def attach_arwen_chain(runtime: Any, binding: Any, *, cupy: Any, stream: Any) -> dict[str, Any]:
    """Bind the driver to a freshly built chain runtime and write its initial state."""

    with stream:
        driver = ChainDriver(
            runtime=runtime, prepared=binding.prepared, spec_zone=binding.spec_zone,
            relax_zone=binding.relax_zone, cupy=cupy, stream=stream,
        )
        initialization = driver.initialize()
    runtime.chain_driver = driver
    return initialization


__all__ = [
    "CHAIN_RUNTIME_SCHEMA",
    "ChainDriver",
    "ChainLateralForcing",
    "attach_arwen_chain",
    "make_chain_runtime_class",
]
