"""Final-RK mass correction, column physics and solve3 for Cartesian GM1.

The operators here are the same CM1-owned numerical implementations used by
the independent validation gates.  This runtime binds them to a variable
grid and executes each physical column once, without constructing a second
reference simulation.  It makes no native-CM1 trajectory identity claim.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any

from . import cuda_graphs
from .component_validation import component_validation
from .column_components import (
    GM1_TILESTREAM_COLUMN_CLAIM_LIMIT,
    column_component_spec,
)
from .cuda_apmasscon import compile_cuda_apmasscon
from .cuda_composite_reflectivity import (
    CM1_CREF_SEED_DBZ,
    compile_cuda_composite_reflectivity,
    launch_cuda_composite_reflectivity,
)
from .cuda_gm1_forecast import compile_cuda_gm1_forecast
from .cuda_gm1_rk3_apmasscon_split import (
    begin_cuda_gm1_rk3_apmasscon_split,
    finish_cuda_gm1_rk3_apmasscon_split,
)
from .cuda_gm1_tilestream_rk3_eqtset2_remove_eos import (
    compile_cuda_gm1_rk3_eqtset2_remove_eos,
    launch_cuda_gm1_rk3_eqtset2_remove_eos_local,
)
from .cuda_gm1_tilestream_post_wsm6 import (
    compile_cuda_gm1_tiled_post_wsm6,
    launch_cuda_gm1_tiled_post_wsm6_eos_qa,
    prepare_cuda_gm1_tiled_post_wsm6_snapshot,
    tiled_post_wsm6_aux_device_shapes,
)
from .cuda_gm1_tilestream_post_column import launch_cuda_gm1_tiled_post_column_eos_qa
from .cuda_cartesian_moist_heating import (
    compile_cuda_cartesian_moist_heating,
    enqueue_moist_heating,
    enqueue_rcvm_snapshot,
    moist_heating_selection,
    refuse_nonconvergent_satadj2,
)
from .cuda_gm1_tilestream_column import (
    launch_cuda_gm1_tiled_column_block,
    tiled_column_device_buffer_shapes,
)
from .cuda_gm1_tilestream_wsm6 import tiled_wsm6_device_buffer_shapes
from .cuda_wsm6_pdefq import (
    compile_cuda_wsm6_pdefq,
    create_cuda_wsm6_pdefq_workspace,
    launch_cuda_wsm6_pdefq_pdscheme1,
)


def _hash_record(record: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(
        record, sort_keys=True, separators=(",", ":"), allow_nan=False,
        default=lambda value: dict(value) if isinstance(value, Mapping) else str(value),
    ).encode()).hexdigest()


#: Mass-volume scratch slots ArWen's own launcher allocates per component,
#: outside GM1's registry.  WSM6 prepares theta/rho/pii/dz plus one reused
#: temporary (gpuwm/core/wsm6.py); Morrison prepares the same four plus its
#: ice-to-snow mask (gpuwm/core/morrison.py), and its column sedimentation
#: needs no extra volume.
#: An extra read-only column carrier, and the dycore field it is staged from.
#: ``w`` is WRF's ``grid%w_2``; GM1's ``w3d`` is the same quantity on the same
#: ``(nz+1, ny, nx)`` stagger, so the staging is a copy and nothing is derived.
_EXTRA_CARRIER_SOURCE: dict[str, str] = {"w": "w3d"}

_COMPONENT_MASS_SCRATCH_VOLUMES: dict[str, int] = {
    "wsm6-mp6": 5,
    "morrison-mp10": 6,
    # Classic Thompson prepares theta/Exner/temperature/dz plus six named
    # reference and marker volumes and a graupel-number shadow, and takes one
    # more for the due reflectivity: fourteen distinct mass slots named in the
    # translation unit that carries the adapter (gpuwm/core/microphysics.py).
    # Two of those fourteen (mp_rho, mp_z) belong to the Kessler adapter that
    # shares the file, so this reserve is conservative by at most two volumes,
    # which is the safe direction for a budget.
    "thompson-mp8": 14,
    # Aerosol-aware Thompson reuses every classic slot and adds fifteen of its
    # own (entry states, tendencies and reference densities) for twenty-nine
    # distinct mass slots named across the two translation units that carry it.
    "thompson-aerosol-mp28": 29,
}


def cartesian_wsm6_scratch_reserve(config: Any, *, alignment: int = 512) -> dict[str, int]:
    """Price component-owned storage outside the registry for the largest block.

    The existing WSM6 inventory reserves five mass volumes and one interface
    volume.  Two further interface-sized slots cover the largest chained
    CuPy expression (geopotential addition followed by division).  This is
    separate from the measured CUDA driver/local-frame backing and the
    registry-owned block arrays, persistence and post-physics snapshots.

    ``output_dbz`` adds one further component-owned mass volume per block:
    the scheme writes its 10 cm reflectivity into its own transient scratch
    on an output step.  It is priced here even though it is live only on
    output steps, because the peak is what the device budget has to survive.

    It is priced against the WSM6 block, which is what a WSM6 call actually
    stages.  Where the configuration does not declare one, that is still the
    closure tile, so this returns exactly what it returned before.
    The WSM6 inventory reserves five mass volumes and one interface volume.
    Two further interface-sized slots cover the largest chained CuPy
    expression (geopotential addition followed by division).  This is separate
    from the measured CUDA driver/local-frame backing and the registry-owned
    block arrays, persistence and post-physics snapshots.
    """
    if type(alignment) is not int or alignment < 1:
        raise ValueError("scratch allocator alignment must be a positive integer")
    g = config.grid
    nx, ny = min(config.physics_block_nx, g.nx), min(config.physics_block_ny, g.ny)
    mass = 4 * g.nz * ny * nx
    face = 4 * (g.nz + 1) * ny * nx
    rounded_mass = ((mass + alignment - 1) // alignment) * alignment
    rounded_face = ((face + alignment - 1) // alignment) * alignment
    reflectivity = rounded_mass if bool(getattr(config, "output_dbz", False)) else 0
    volumes = _COMPONENT_MASS_SCRATCH_VOLUMES[config.microphysics]
    component = volumes * rounded_mass + rounded_face + reflectivity
    temporary = 2 * rounded_face
    record = {
        "block_nx": nx, "block_ny": ny, "levels": g.nz,
        "component_id": config.microphysics,
        "component_mass_scratch_volumes": volumes,
        "component_scratch_reserve_bytes": component,
        "expression_temporary_reserve_bytes": temporary,
        "additional_reserve_bytes": component + temporary,
        "allocator_alignment_bytes": alignment,
    }
    # A run without the diagnostic prices exactly the words it always has, so
    # its memory plan and every receipt that carries it are unchanged.
    if reflectivity:
        record["component_reflectivity_scratch_bytes"] = reflectivity
    return record


def cartesian_reflectivity_allocation_plan(config: Any) -> dict[str, Any]:
    """Price the resident CM1 ``output_dbz`` carriers before any device opens.

    They are registry residents rather than per-frame scratch: the tiled
    scatter writes the volume one block at a time, so a per-frame allocation
    would have to survive the whole tile loop anyway.
    """

    g = config.grid
    output_dbz = bool(getattr(config, "output_dbz", False))
    level = getattr(config, "reflectivity_level_index", None)
    volume = 4 * g.nz * g.ny * g.nx
    plane = 4 * g.ny * g.nx
    planes = 2 if level is not None else 1
    return {
        "output_dbz": output_dbz,
        "reflectivity_level_index": level,
        "resident_volume_bytes": volume if output_dbz else 0,
        "resident_plane_bytes": plane * planes if output_dbz else 0,
        "resident_total_bytes": (volume + plane * planes) if output_dbz else 0,
        "words_per_gridpoint": (
            1.0 + planes / float(g.nz) if output_dbz else 0.0
        ),
        "authority": "CM1 r21.1 README.namelist:1366-1369 output_dbz",
    }


def cartesian_physics_block_payload_bytes(config: Any) -> int:
    """Registry bytes one full-size physics block stages, CUDA-cold.

    The column buffers and the post-physics auxiliaries
    :meth:`CartesianFinish.microphysics` allocates for its largest block, each
    rounded to the registry's 512-byte alignment.  A run with no column
    component (``microphysics="off"``) stages none.
    """

    if config.microphysics == "off":
        return 0
    g = config.grid
    nx, ny = min(config.physics_block_nx, g.nx), min(config.physics_block_ny, g.ny)
    spec = column_component_spec(config.microphysics)
    shapes = list(tiled_column_device_buffer_shapes(spec, tile_nx=nx, tile_ny=ny, nk=g.nz).values())
    shapes += list(tiled_post_wsm6_aux_device_shapes(tile_nx=nx, tile_ny=ny, nk=g.nz).values())
    # CM1's cvm of the scheme's entry state (cuda_cartesian_moist_heating).
    shapes.append((g.nz, ny, nx))
    return sum(((4 * math.prod(shape) + 511) // 512) * 512 for shape in shapes)


def cartesian_physics_blocks(config: Any) -> tuple[tuple[int, int, int, int], ...]:
    """The rectangles one large step's WSM6 calls cover, south-to-north.

    WSM6 is column physics, so this decomposition decides only how many times
    the per-call staging, launch and drain are paid -- not what any column
    computes.  Returned as data so the decomposition a configuration selects
    can be read without a device.
    """

    g = config.grid
    nx = min(config.physics_block_nx, g.nx)
    ny = min(config.physics_block_ny, g.ny)
    if nx < 1 or ny < 1:
        raise ValueError("the WSM6 block must cover at least one column")
    return tuple(
        (j0, min(g.ny, j0 + ny), i0, min(g.nx, i0 + nx))
        for j0 in range(0, g.ny, ny)
        for i0 in range(0, g.nx, nx)
    )



def _refuse_invalid_eos(value: Any) -> None:
    invalid = int(value[0])
    if invalid:
        raise ValueError(f"final-RK EOS rejected {invalid} physical words")

class CartesianFinish:
    # Class defaults, so a shell built with ``object.__new__`` -- and any
    # instance from before the reflectivity carriers existed -- reads "this
    # run has no reflectivity" rather than raising on the first attribute a
    # publisher or a diagnostic asks for.
    output_dbz: bool = False
    reflectivity_level: int | None = None
    completed_reflectivity_step: int = 0
    reflectivity: Mapping[str, Any] = MappingProxyType({})
    # The constant-pressure-to-CM1-eqtset=2 heating step (compiled in
    # __init__; a shell built with object.__new__ compiles it on first use).
    moist_heating_compiled: Any = None

    def __init__(
        self,
        dynamics: Any,
        *,
        source_identity_sha256: str,
        capacity_profile: str | None = None,
    ) -> None:
        self.dynamics = dynamics
        self.config = dynamics.config
        self.grid = self.config.grid
        self.cp = dynamics.cupy
        self.stream = dynamics.stream
        self.registry = dynamics.registry
        self.source_identity_sha256 = source_identity_sha256
        self.capacity_profile = capacity_profile
        self.spec = column_component_spec(self.config.microphysics)
        self.profile = self.config.profile
        self.numq = self.profile.numq
        # src/mp_driver.F:505-520 and :1469-1479: the pre-microphysics
        # positive-definite repair runs on the mass prefix only.  The ptype=5
        # asq(7) call is commented out in the authority, so a double-moment
        # axis must not repair its number concentrations here.
        self.repaired_scalars = self.profile.repaired_scalars
        self.apmasscon = compile_cuda_apmasscon()
        # The image the dynamics published the start-up rr/rf with: the sealed one, or
        # its level variant on a column whose end layers are not equally deep.
        from .cuda_gm1_level_density import compile_cuda_gm1_forecast_for_levels

        self.forecast = compile_cuda_gm1_forecast_for_levels(
            self.grid.levels, compile_sealed=compile_cuda_gm1_forecast
        )
        self.eos = compile_cuda_gm1_rk3_eqtset2_remove_eos()
        self.pdefq_compiled = compile_cuda_wsm6_pdefq()
        # One workspace for the life of the runtime.  It was being allocated
        # again on every large step, which is a device allocation and a pool
        # lookup inside the step for a buffer whose geometry never changes.
        self.pdefq_workspace = create_cuda_wsm6_pdefq_workspace(
            ni=self.grid.nx,
            nj=self.grid.ny,
            nk=self.grid.nz,
            layout=self.config.pdefq_launch,
        )
        self.post_compiled = compile_cuda_gm1_tiled_post_wsm6()
        # Every ArWen component heats at constant pressure; CM1's eqtset=2
        # rebuild (src/mp_driver.F:603-617, the sealed finish) needs the
        # constant-volume heating CM1's own schemes produce.  See
        # cuda_cartesian_moist_heating for the authority lines.
        self.moist_heating_compiled = compile_cuda_cartesian_moist_heating()
        self.reflectivity_sink: Any | None = None
        self.mass_reduction: Any | None = None
        self.eos_invalid: Any | None = None
        self.asq = self.registry.allocate("physics.asq", (self.repaired_scalars,), zero=True, dtype=self.cp.float64)
        self.persistence: dict[str, Any] = {}
        self.step_receipts: dict[str, Any] = {}
        self.completed_physics_step = 0
        self.completed_solve3_step = 0
        # README.namelist:1366-1369.  The reflectivity carriers exist only for
        # a run that asked for them, so a run that did not is byte-for-byte
        # the run it has always been -- same registry inventory, same memory
        # plan, same published frame.
        self.output_dbz = bool(getattr(self.config, "output_dbz", False))
        self.reflectivity_level = (
            self.config.reflectivity_level_index
            if hasattr(self.config, "reflectivity_level_index") else None
        )
        self.reflectivity: dict[str, Any] = {}
        self.reflectivity_compiled: Any | None = None
        self.completed_reflectivity_step = 0
        self._allocate_persistence()
        if self.output_dbz:
            self._allocate_reflectivity()
        self.initial_mass1_f64 = self._initial_mass()
        dynamics.mass1_f64 = self.initial_mass1_f64

    def _allocate_persistence(self) -> None:
        for name in self.spec.component_persistence:
            # surface_persistence is the surface scratch plus any component-owned
            # surface carrier (aerosol-aware Thompson's nwfa2d/nifa2d); keying on
            # surface_scratch alone gave those two a mass volume that no tile
            # buffer of shape (ny, nx) could receive.
            shape = self.grid.surface_shape if name in self.spec.surface_persistence else self.grid.mass_shape
            value = self.registry.allocate(f"physics.{name}", shape, zero=True)
            self.persistence[name] = value
        with self.stream:
            for name, radius in self.spec.effective_radius_initial_microns:
                self.persistence[name].fill(self.cp.float32(radius))

    def _allocate_reflectivity(self) -> None:
        """Hold the published reflectivity carriers for the whole run.

        ``cm1.F:3214-3237`` computes the diagnostic on the step whose state is
        about to be written, so these are live only between that step's
        microphysics and that step's frame.  They are registry-budgeted
        residents rather than per-frame scratch because the tiled scatter
        below writes them one block at a time and a per-frame allocation
        would have to survive the whole tile loop anyway.
        """

        g = self.grid
        self.reflectivity["dbz"] = self.registry.allocate(
            "physics.reflectivity.dbz", g.mass_shape, zero=True
        )
        self.reflectivity["cref"] = self.registry.allocate(
            "physics.reflectivity.cref", g.surface_shape, zero=True
        )
        if self.reflectivity_level is not None:
            self.reflectivity["dbz_level"] = self.registry.allocate(
                "physics.reflectivity.dbz_level", g.surface_shape, zero=True
            )
        with self.stream:
            # misclibs.F:2267-2271 seeds every composite column at -1000.0.
            # Seeding the resident carriers with it too means a frame taken
            # before any output step could only publish a value no
            # meteorological floor produces, instead of a plausible 0 dBZ.
            self.reflectivity["cref"].fill(self.cp.float32(CM1_CREF_SEED_DBZ))
            self.reflectivity["dbz"].fill(self.cp.float32(CM1_CREF_SEED_DBZ))
            if "dbz_level" in self.reflectivity:
                self.reflectivity["dbz_level"].fill(self.cp.float32(CM1_CREF_SEED_DBZ))

    def reflectivity_allocation_plan(self) -> dict[str, Any]:
        """What the reflectivity carriers cost this geometry, in bytes."""

        return cartesian_reflectivity_allocation_plan(self.config)

    def _mass_arrays(self) -> dict[str, Any]:
        d, g = self.dynamics, self.grid
        return {
            "rho": d.fields["rho"],
            "ruh": d.arrays["uh"][3:3 + g.nx],
            "rvh": d.arrays["vh"][3:3 + g.ny],
            "rmh": d.metrics["rmh"],
            "pi0": d.fields["pi0"],
            "th0": d.fields["th0"],
            "th3d": d.fields["th3d"],
            "qv": d.fields["q3d"][0],
            "pp3d": d.fields["pp3d"],
            "prs": d.fields["prs"],
        }

    def _initial_mass(self) -> float:
        """Read one scalar after CM1's fixed GPU layer/summation order."""
        cp, g, a = self.cp, self.grid, self._mass_arrays()
        with self.stream:
            layer_mass = cp.empty((g.nz,), dtype=cp.float64)
            layer_pressure = cp.empty_like(layer_mass)
            result = cp.empty((1,), dtype=cp.float64)
            self.apmasscon.kernels[0](
                (g.nz, 1, 1), (256, 1, 1),
                (a["rho"], a["ruh"], a["rvh"], a["rmh"], a["pi0"],
                 a["pp3d"], layer_mass, layer_pressure,
                 cp.uint32(g.nx), cp.uint32(g.ny), cp.uint32(g.nz)),
                stream=self.stream,
            )
            self.forecast.kernels[2](
                (1, 1, 1), (1, 1, 1), (layer_mass, result, cp.uint32(g.nz)),
                stream=self.stream,
            )
        value = float(result.get(stream=self.stream)[0])
        if not math.isfinite(value) or value <= 0:
            raise ValueError("source-initialized dry mass is not finite and positive")
        return value

    def remove_held_eos(self) -> None:
        self.step_receipts = {}
        f = self.dynamics.fields
        arrays = {
            name: f["q3d"][0] if name == "qv" else f["q3d"][1] if name == "qc" else f[name]
            for name in ("pi0", "th0", "pp3d", "th3d", "qv", "qc", "qpten", "qtten", "qvten", "qcten", "prs", "rho")
        }
        launch = launch_cuda_gm1_rk3_eqtset2_remove_eos_local(
            arrays, active_nx=self.grid.nx, active_ny=self.grid.ny,
            dt_seconds=self.config.dt_s, stream=self.stream, compiled=self.eos,
        )
        self.eos_invalid = launch.invalid_words
        self.step_receipts["remove_held_eos"] = dict(launch.token.receipt)

    def mass_begin(self) -> None:
        if self.mass_reduction is not None:
            raise RuntimeError("a previous dry-mass reduction has not finished")
        self.mass_reduction = begin_cuda_gm1_rk3_apmasscon_split(
            self._mass_arrays(), ni=self.grid.nx, nj=self.grid.ny, nk=self.grid.nz,
            mass1=self.initial_mass1_f64, stream=self.stream, compiled=self.apmasscon,
        )

    def mass_finish(self) -> None:
        if self.mass_reduction is None:
            raise RuntimeError("dry-mass correction has no matching reduction")
        finish = finish_cuda_gm1_rk3_apmasscon_split(self.mass_reduction, stream=self.stream)
        self.step_receipts["mass_finish"] = dict(finish.token.receipt)
        self.mass_reduction = None

    def pdefq(self) -> None:
        if self.eos_invalid is None:
            raise RuntimeError("final-RK EOS did not execute")
        cuda_graphs.read_device(self.eos_invalid, _refuse_invalid_eos, stream=self.stream)
        self.eos_invalid = None
        d, g = self.dynamics, self.grid
        # The launch below uses the one workspace built in __init__; a second
        # one was being allocated here on every large step and never read.
        # src/mp_driver.F:505-520 (ptype=5) and :1469-1479 (ptype=7): the
        # pre-microphysics positive-definite repair runs on the six mass
        # scalars only.  The ptype=5 asq(7) call is commented out in the
        # authority, so the contiguous mass prefix is what is repaired and a
        # number concentration is never clipped here.
        launch = launch_cuda_wsm6_pdefq_pdscheme1(
            {"q3d": d.fields["q3d"][: self.repaired_scalars], "rho": d.fields["rho"],
             "ruh": d.arrays["uh"][3:3 + g.nx], "rvh": d.arrays["vh"][3:3 + g.ny],
             "rmh": d.metrics["rmh"], "asq": self.asq},
            ni=g.nx, nj=g.ny, nk=g.nz, dx=g.dx_m, dy=g.dy_m, dz=g.dz_m,
            layout=self.config.pdefq_launch,
            stream=self.stream, compiled=self.pdefq_compiled,
            workspace=self.pdefq_workspace,
        )
        self.step_receipts["pdefq"] = dict(launch.receipt)
        cuda_graphs.synchronize(launch.ready_event)

    def microphysics(self, *, step: int, reflectivity_due: bool = False) -> None:
        """Complete independent vertical columns, including their EOS drain.

        Post-physics EOS has no horizontal dependencies.  Draining it before
        reusing a column block preserves its source order and avoids keeping
        whole-domain physics scratch alongside the atmospheric state.

        ``reflectivity_due`` is CM1's ``getdbz`` (``cm1.F:3214-3237``): on the
        step whose completed state is about to be written, the scheme also
        produces its own 10 cm reflectivity for these columns, and the tiled
        blocks are scattered into the domain carrier here.

        ``reflectivity_due`` is CM1's ``output_dbz`` cadence
        (``run/namelist.input:243``).  The audited ArWen WSM6 component forms
        its 10 cm reflectivity inside the same call that advances the
        microphysics and hands it back exactly once, so the only place this
        lane can take it is here, on the step whose state the next history
        frame will carry.  The per-block array is handed to
        ``self.reflectivity_sink`` with its physical rectangle; there is no
        second reflectivity operator anywhere in this runtime.
        """
        if type(reflectivity_due) is not bool:
            raise TypeError("reflectivity_due must be a strict boolean schedule decision")
        if reflectivity_due and self.reflectivity_sink is None:
            raise RuntimeError(
                "output_dbz asked WSM6 for reflectivity but no sink is bound; "
                "the component would compute a field and drop it, and the frame "
                "would carry an all-sentinel dbz volume that reads as clear air"
            )
        if step != self.completed_physics_step + 1:
            raise ValueError("physics calls must follow consecutive completed steps")
        if "pdefq" not in self.step_receipts:
            raise RuntimeError("WSM6 requires this step's positive-definite repair")
        if type(reflectivity_due) is not bool:
            raise TypeError("reflectivity_due must be a strict boolean schedule decision")
        if reflectivity_due and not self.output_dbz:
            raise RuntimeError(
                "reflectivity was scheduled for this step but this run declared "
                "output_dbz off, so there is no carrier to scatter the scheme's "
                "dBZ columns into and the diagnostic would be computed and dropped"
            )
        d, g, cp = self.dynamics, self.grid, self.cp
        d.release_phase_workspace("acoustic")
        source_identity = _hash_record({
            "source": self.source_identity_sha256, "step": step,
            "grid": g.record(), "dt_s": self.config.dt_s,
        })
        pdefq_sha = _hash_record(self.step_receipts["pdefq"])
        boundary_sha = _hash_record({"source": source_identity, "pdefq": pdefq_sha})
        heating = moist_heating_selection(self.spec.component_id)
        if self.moist_heating_compiled is None:
            self.moist_heating_compiled = compile_cuda_cartesian_moist_heating()
        nonconvergent = self.registry.allocate(
            "physics.satadj2-nonconvergent", (1,), zero=True, dtype=cp.uint32
        )
        blocks = []
        # mp_driver.F:1483-1559 calls WSM6 once per physical row over an
        # i-slab; GM1 covers the domain with rectangles the document sizes
        # independently of the closure tile.  An undeclared block follows the
        # tile, so this is the decomposition every document already ran.
        rectangles = cartesian_physics_blocks(self.config)
        for block, (j0, j1, i0, i1) in enumerate(rectangles):
            nx, ny = i1 - i0, j1 - j0
            shapes = tiled_column_device_buffer_shapes(self.spec, tile_nx=nx, tile_ny=ny, nk=g.nz)
            aux_shapes = tiled_post_wsm6_aux_device_shapes(tile_nx=nx, tile_ny=ny, nk=g.nz)
            values = {name: self.registry.allocate(f"physics.block.{name}", shape)
                      for name, shape in shapes.items()}
            aux = {name: self.registry.allocate(f"physics.aux.{name}", shape)
                   for name, shape in aux_shapes.items()}
            rcvm_before = self.registry.allocate("physics.aux.rcvm_before", (g.nz, ny, nx))
            with self.stream:
                for name in ("pi0", "th0", "pp3d", "th3d", "rho", "q3d"):
                    cp.copyto(values[name], d.fields[name][..., j0:j1, i0:i1])
                # A scheme that reads WRF's grid%w_2 takes GM1's own w3d on
                # the same (nz+1, ny, nx) stagger, read-only and unchanged.
                for name in self.spec.extra_interface_carriers:
                    cp.copyto(
                        values[name],
                        d.fields[_EXTRA_CARRIER_SOURCE[name]][:, j0:j1, i0:i1],
                    )
                cp.copyto(values["p"], d.fields["prs"][:, j0:j1, i0:i1])
                cp.multiply(d.metrics["zf"][:, j0:j1, i0:i1], cp.float32(9.81), out=values["phb"])
                for name, state in self.persistence.items():
                    cp.copyto(values[name], state[..., j0:j1, i0:i1])
            # mp_driver.F:603-617 reads the vapour row and rebuilds pressure,
            # Exner and theta over the mass prefix, which is a contiguous
            # leading-axis window of any admitted q3d axis, so the sealed
            # post-physics boundary keeps its exact six-species inventory
            # whatever the component transports.  Its kernel also copies
            # those six rows into aux["qa"]; that block scratch is released
            # below and never reaches the domain qa (see the drain).
            post_values = self._post_physics_arrays(values, nx=nx, ny=ny)
            snapshot = prepare_cuda_gm1_tiled_post_wsm6_snapshot(
                post_values, aux, nx=nx, ny=ny, nk=g.nz,
                source_identity_sha256=source_identity, wsm6_receipt_sha256=boundary_sha,
                block_index=block, active_nx=nx, active_ny=ny, stream=self.stream,
                compiled=self.post_compiled,
            )
            # src/morrison.F:1641-1642, src/thompson.F:1875-1876: CM1's cvm of
            # the state the scheme is handed.
            enqueue_rcvm_snapshot(
                self.moist_heating_compiled, q3d=post_values["q3d"],
                rcvm_before=rcvm_before, stream=self.stream,
            )
            capacity = {} if self.capacity_profile is None else {"capacity_profile": self.capacity_profile}
            launch = launch_cuda_gm1_tiled_column_block(
                self.spec,
                values, nx=nx, ny=ny, nk=g.nz, dx=g.dx_m, dy=g.dy_m, dz=g.dz_m,
                dt=self.config.dt_s, source_identity_sha256=source_identity,
                pdefq_receipt_sha256=pdefq_sha, completed_calls=step - 1,
                block_index=block, active_nx=nx, active_ny=ny, stream=self.stream,
                ihail=self.config.ihail,
                refl_10cm_due=reflectivity_due,
                claim_limit=GM1_TILESTREAM_COLUMN_CLAIM_LIMIT,
                **capacity,
            )
            # The component heated at constant pressure; the sealed rebuild
            # below holds rho fixed, which is CM1's eqtset=2 pairing only for
            # constant-volume heating (src/morrison.F:1640-1650, :6402-6504;
            # src/thompson.F:2865-2920).  Before this step the lane kept about
            # 0.78 of every phase change's latent heat.
            heated = enqueue_moist_heating(
                self.moist_heating_compiled, heating, values=post_values,
                aux=aux, rcvm_before=rcvm_before, nonconvergent=nonconvergent,
                stream=self.stream,
            )
            if self.spec.component_id == "wsm6-mp6":
                finished = launch_cuda_gm1_tiled_post_wsm6_eos_qa(
                    post_values, aux, snapshot=snapshot, wsm6_launch=launch,
                    nx=nx, ny=ny, nk=g.nz, source_identity_sha256=source_identity,
                    wsm6_receipt_sha256=boundary_sha, block_index=block,
                    active_nx=nx, active_ny=ny, stream=self.stream, compiled=self.post_compiled,
                )
            else:
                # The sealed WSM6 launcher admits only a WSM6 launch record;
                # every other component takes the same kernel and the same
                # mp_driver.F:603-617 arithmetic through a launcher that checks
                # that component's own record instead.
                finished = launch_cuda_gm1_tiled_post_column_eos_qa(
                    post_values, aux, snapshot=snapshot, column_launch=launch,
                    nx=nx, ny=ny, nk=g.nz, source_identity_sha256=source_identity,
                    column_receipt_sha256=boundary_sha, block_index=block,
                    active_nx=nx, active_ny=ny, stream=self.stream, compiled=self.post_compiled,
                )
            with self.stream:
                for name in ("q3d", "pp3d", "th3d"):
                    cp.copyto(d.fields[name][..., j0:j1, i0:i1], values[name])
                cp.copyto(d.fields["prs"][:, j0:j1, i0:i1], values["p"])
                # qa is deliberately not published here.  CM1 calls mp_driver
                # before solve3 (cm1.F:3416,3432), and only solve3.F:624-643
                # equates qa to q3d, for every n=1,numq -- which
                # CartesianFinish.solve3 does over the whole domain.  Between
                # the two, qa is still the step's starting mixing ratio, which
                # is what the qv budget's actual tendency (q3d - qa)/dt reads
                # (CartesianBudgetStage.after_microphysics).  This drain used
                # to copy aux["qa"] (and the number rows) into qa as well, so
                # that tendency was identically zero; nothing else reads qa
                # before solve3 overwrites it with the same words.
                for name, state in self.persistence.items():
                    cp.copyto(state[..., j0:j1, i0:i1], values[name])
                if reflectivity_due:
                    # The component's reflectivity scratch is consumed exactly
                    # once and is not restored, so it has to be drained into
                    # the domain carrier before the next block rebinds the
                    # component.
                    cp.copyto(
                        self.reflectivity["dbz"][:, j0:j1, i0:i1],
                        launch.reflectivity_dbz,
                    )
            blocks.append({
                "index": block, "physical_yx": [j0, j1, i0, i1],
                "wsm6": launch.receipt["operation_token_sha256"],
                "moist_heating": heated["operation_token_sha256"],
                "post_eos": finished.receipt["operation_token_sha256"],
            })
            names = [f"physics.block.{name}" for name in values]
            names += [f"physics.aux.{name}" for name in aux]
            names.append("physics.aux.rcvm_before")
            del snapshot, launch, finished, values, aux, rcvm_before
            # ``registry.release`` drains the stream before it drops a buffer,
            # which is the same barrier this loop used to take twice per block.
            self.registry.release(names)
        if heating.mode == 2:
            # src/morrison.F:6488-6501: CM1 stops when satadj2 does not converge.
            cuda_graphs.read_device(nonconvergent, refuse_nonconvergent_satadj2, stream=self.stream)
        del nonconvergent
        self.registry.release(["physics.satadj2-nonconvergent"])
        cuda_graphs.bookkeeping(self._set_physics_step, step)
        self.step_receipts["physics"] = {
            "step": step, "blocks": blocks, "vertical_columns_split": False,
            "reflectivity_10cm_due": reflectivity_due,
            "post_eos_drained_before_block_reuse": True,
            "capacity_profile": self.capacity_profile,
            "block_shape_yx": [
                min(self.config.physics_block_ny, g.ny),
                min(self.config.physics_block_nx, g.nx),
            ],
            "block_count": len(rectangles),
            "component_id": self.spec.component_id,
            "cm1_ptype": self.spec.cm1_ptype,
            "numq": self.numq,
            "ihail": self.config.ihail,
            "dense_ice_category": self.config.dense_ice.category,
            # Loud and machine-readable: a run of an unmeasured component says
            # so every step, naming exactly what was not checked.
            "validation": component_validation(self.spec.component_id).record(),
            # How the component's constant-pressure heating became CM1's
            # eqtset=2 heating before the fixed-rho rebuild.
            "moist_heating": heating.record(),
        }
        if reflectivity_due:
            self.step_receipts["reflectivity"] = self.composite_reflectivity(step=step)

    def composite_reflectivity(self, *, step: int) -> dict[str, Any]:
        """``misclibs.F:2259-2292`` and the chosen-level plane, on the device.

        CM1 forms both in ``writeout.F`` from the same ``qd_dbz`` volume the
        scheme just filled (``writeout.F:1081-1091`` and ``:4089-4107``).  The
        fold runs here, immediately after the scattered volume is complete, so
        the plan-view products a frame publishes are always the ones belonging
        to the model time the frame carries.
        """

        if not self.output_dbz:
            raise RuntimeError(
                "this run declared output_dbz off, so it holds no reflectivity "
                "volume to reduce and a composite formed here would be the "
                "column maximum of an unwritten carrier"
            )
        g = self.grid
        if self.reflectivity_compiled is None:
            self.reflectivity_compiled = compile_cuda_composite_reflectivity()
        arrays = {"dbz": self.reflectivity["dbz"], "cref": self.reflectivity["cref"]}
        if self.reflectivity_level is not None:
            arrays["plane"] = self.reflectivity["dbz_level"]
        launch = launch_cuda_composite_reflectivity(
            arrays, ni=g.nx, nj=g.ny, nk=g.nz, level=self.reflectivity_level,
            stream=self.stream, compiled=self.reflectivity_compiled,
        )
        self.completed_reflectivity_step = step
        return dict(launch.receipt)

    def _post_physics_arrays(
        self, values: Mapping[str, Any], *, nx: int, ny: int
    ) -> dict[str, Any]:
        """The sealed six-species post-physics inventory, from any component.

        The post-physics EOS boundary is pinned to CM1's ptype-5 eqtset-2
        rebuild over the mass prefix (``mp_driver.F:603-617``).  Its kernel
        also writes a six-row block ``qa`` scratch, which this lane discards:
        the domain ``qa`` is published only by :meth:`solve3`
        (``solve3.F:624-643``).  Selecting the inventory out of a wider
        component keeps that operator's bytes and its exact array contract
        unchanged; nothing is copied, because ``q3d[:6]`` is a leading-axis
        window of the same allocation.
        """

        expected = tiled_wsm6_device_buffer_shapes(
            tile_nx=nx, tile_ny=ny, nk=self.grid.nz
        )
        selected: dict[str, Any] = {}
        for name, shape in expected.items():
            value = values[name]
            if name == "q3d":
                value = value[: shape[0]]
            selected[name] = value
        return selected

    def post_physics(self) -> None:
        # src/solve3.F:319-337 recaptures exactly four held rates: pp3d, th3d,
        # q3d(nqv) and q3d(nqc).  Both admitted ptypes have nqv=1 and nqc=2 on
        # CM1's one-based axis (src/param.F:3425-3426, :3645-3646), so the two
        # scalar rows are the same two here whatever numq is.
        cp, f = self.cp, self.dynamics.fields
        work = self.registry.allocate("physics.rate-work", self.grid.mass_shape)
        with self.stream:
            for current, rate in (("pp3d", "qpten"), ("th3d", "qtten"),
                                  ("qv", "qvten"), ("qc", "qcten")):
                value = f["q3d"][0] if current == "qv" else f["q3d"][1] if current == "qc" else f[current]
                cp.subtract(value, f[rate], out=work)
                cp.divide(work, cp.float32(self.config.dt_s), out=f[rate])
        del work
        self.registry.release(["physics.rate-work"])

    def solve3(self, *, step: int) -> None:
        if step != self.completed_physics_step or step != self.completed_solve3_step + 1:
            raise RuntimeError("solve3 requires the matching completed physics step")
        d, g, cp = self.dynamics, self.grid, self.cp
        f, a = d.fields, d.arrays
        pressure = g.physical_slices("mass", halo_xy=1, halo_z=1)
        with self.stream:
            cp.add(f["pp3d"], a["ppx"][pressure], out=f["ppx"])
            for current, baseline in (("u3d", "ua"), ("v3d", "va"), ("w3d", "wa"),
                                      ("pp3d", "ppi"), ("th3d", "tha"), ("q3d", "qa")):
                cp.copyto(f[baseline], f[current])
            cp.copyto(d.tke["tkea"], d.tke["tke3d"])
            words = g.nx * g.ny * (g.nz + 1)
            self.forecast.kernels[1](
                ((words + 127) // 128, 1, 1), (128, 1, 1),
                (f["rho"], d.metrics["c1"], d.metrics["c2"], f["rr"], f["rf"],
                 cp.uint32(g.nx), cp.uint32(g.ny), cp.uint32(g.nz)),
                stream=self.stream,
            )
        d.sync_padded_from_physical()
        cuda_graphs.bookkeeping(self._set_solve3_step, step)

    def _set_physics_step(self, step: int) -> None:
        self.completed_physics_step = step

    def _set_solve3_step(self, step: int) -> None:
        self.completed_solve3_step = step
