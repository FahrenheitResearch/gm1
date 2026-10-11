"""Bounded per-frame scalar diagnostics for a sustained Cartesian integration.

Eleven large steps need no diagnostics; fourteen thousand do.  This module
turns one completed post-solve3 device state into a small row of scalars that
say whether the integration is still physical, whether dry mass has drifted
from the ledger the correction is anchored to, and whether anything has gone
non-finite.  Every reduction runs on the device; only the scalars cross to
the host.  Nothing here advances, damps, repairs or renders a field.

Each scalar is transcribed from the CM1 r21.1 statistics package rather than
invented, and the transcription is only as exact as the runtime's own
carriers allow:

* ``calccfl`` and ``calcmass``/``totq`` read CM1's ``uh``/``vh`` and
  ``ruh``/``rvh`` separately.  This runtime fills the one stored pair with
  1.0 from CM1 ``param.F``'s uniform-horizontal branch
  (``cuda_cartesian_dynamics.py`` lines 184-186) and hands the same words to
  ``apmasscon`` as ``ruh``/``rvh``.  The two are therefore the same words
  here, and a stretched horizontal grid would need both.
  ``CartesianGridGeometry`` has no stretching at all, so that grid cannot
  reach this module.
* ``vertvort`` carries its own note that it does not account for terrain, and
  its Cartesian branch is the one transcribed.  The Cartesian runtime has no
  terrain either, so no branch is skipped silently.
* CM1 accumulates each layer sum serially over ``i`` then ``j`` into a
  float64 scalar.  The whole-domain dry-mass ledger here is read through the
  pinned ``apmasscon`` layer kernel, which already owns CM1's fixed GPU layer
  and summation order, so it is directly comparable to ``mass1``.  The
  moisture totals use a device reduction whose within-layer order is the
  reduction's, not CM1's serial order; the claim limit says so.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import math
from typing import Any


FRAME_DIAGNOSTIC_SCHEMA = "gm1.cartesian-frame-diagnostics/v1"
FRAME_DIAGNOSTIC_CLAIM_LIMIT = (
    "one completed post-solve3 device state reduced to CM1 r21.1 statistics-package "
    "scalars on one uniform flat Cartesian grid; the dry-mass ledger uses the pinned "
    "apmasscon layer and summation order and is comparable to mass1, while the moisture "
    "totals and the vorticity and Courant maxima use device reductions whose within-layer "
    "order is the reduction's and not CM1's serial i,j order; not a CM1 statistics-file "
    "reproduction, a trajectory identity, a convective-mode classification, or forecast skill"
)

SOURCE_ANCHORS = {
    # calccfl: the Courant number CM1 itself stops on.  The stop is
    # misclibs.F:2714, reached from statpack.F:469 at statistics cadence with
    # stat_cfl=1; cm1.F:3517-3520 then writes this frame and cm1.F:4385-4392
    # stops the model.
    "courant_number": "src/misclibs.F:2584-2718",
    "courant_stop_threshold": "src/misclibs.F:2714",
    "courant_stop_path": "src/statpack.F:469;src/cm1.F:3517-3520;src/cm1.F:4385-4392",
    # vertvort: signed maximum vertical vorticity at fixed report heights.
    "vertical_vorticity_no_terrain": "src/misclibs.F:2523-2529",
    "vorticity_report_levels": "src/misclibs.F:2454-2466",
    # calcmass / totq: volume-weighted float64 layer ledgers.
    "total_dry_mass": "src/misclibs.F:3134-3191",
    "total_species_mass": "src/misclibs.F:3270-3334",
    # statpack.F guards totq to nqv, nql1..nql2 and nqs1..nqs2: a number
    # concentration is never integrated as a mass.
    "species_mass_selection": "src/statpack.F:507-517",
    "artificial_moisture_selection": "src/mp_driver.F:505-520;1469-1479",
    "liquid_ice_partition": "src/misclibs.F:923-967",
    "wsm6_species_order": "src/param.F:3632-3663",
    # mass1 is read once at t=0 and is the correction's fixed set point.
    "dry_mass_reference": "src/init3d.F:2485-2528",
    "dry_mass_correction": "src/solve2.F:1616-1656",
    "dry_mass_correction_apply": "src/solve2.F:1972-2030",
    # asq is zeroed once and accumulates for the whole run.
    "artificial_moisture_ledger": "src/cm1.F:1677-1678",
    "artificial_moisture_report": "src/statpack.F:588-597",
    # output_dbz: the three-dimensional field, its composite, and the flag
    # that decides which steps carry one.
    "reflectivity_namelist_word": "README.namelist:1366-1369",
    "reflectivity_output_due": "src/cm1.F:3214-3237",
    "composite_reflectivity": "src/misclibs.F:2259-2292",
    "reflectivity_writer": "src/writeout.F:1081-1091;src/writeout.F:4089-4107",
}

# misclibs.F:2714 and solve2.F:2082 both set ``stopit`` at this value.  It is
# CM1's own stop, not a GM1 preference, and it is not configurable here.
CM1_COURANT_STOP = 1.5

# misclibs.F:2454-2466.  VORSFC is level one; the other five are the lowest
# model level whose ``zh`` reaches the named height.  A height the column does
# not reach is reported as absent rather than clamped to the model top.
VORTICITY_REPORT_HEIGHTS_M = (1000.0, 2000.0, 3000.0, 4000.0, 5000.0)
VORTICITY_REPORT_NAMES = ("vorsfc", "vor1km", "vor2km", "vor3km", "vor4km", "vor5km")

# param.F:3642-3663, ptype=7.  nqv=1, nql1..nql2=2..3, nqs1..nqs2=4..6 on
# CM1's one-based species axis; this runtime's ``q3d`` axis is the same order
# from zero.  The live inventory now comes from the bound component's
# transcribed profile (cm1core.moisture_species); these three keep the WSM6
# answer available by name for readers of this module.
WSM6_SPECIES_NAMES = ("qv", "qc", "qr", "qi", "qs", "qg")
WSM6_VAPOUR_INDEX = 0
WSM6_CONDENSATE_INDICES = (1, 2, 3, 4, 5)

_FRAME_PREFIX = "frame-diagnostic"

#: Plan-view coverage thresholds, in dBZ.  These are this port's reporting
#: choice, not CM1's: CM1 writes the field and the composite and applies no
#: threshold to either.  They are the values a radar operator reads a cell by
#: -- 20 dBZ is the precipitating edge, 40 dBZ the convective core, 50 dBZ the
#: heavy-precipitation/hail-bearing core -- and the row says they are the
#: caller's, so nothing here classifies a convective mode.
REFLECTIVITY_COVERAGE_THRESHOLDS_DBZ: tuple[float, ...] = (20.0, 40.0, 50.0)


def frame_diagnostic_scratch_bytes(config: Any) -> dict[str, int]:
    """Price this frame's device scratch before a single word is allocated.

    Two volume planes carry the Courant accumulator and the reusable
    integrand; two corner planes carry the vorticity difference and its
    subtrahend.  Nothing is held between frames, so this is a peak, not a
    residency.
    """

    g = config.grid
    volume = 4 * g.nz * g.ny * g.nx
    corner = 4 * max(g.ny - 1, 0) * max(g.nx - 1, 0)
    record = {
        "courant_accumulator_bytes": volume,
        "integrand_bytes": volume,
        "vorticity_plane_bytes": corner,
        "vorticity_term_bytes": corner,
        "frame_scratch_peak_bytes": 2 * volume + 2 * corner,
    }
    # The composite coverage counts form one boolean plane per threshold, in
    # turn.  It is a surface, not a volume, and it is priced rather than left
    # to appear as an untracked allocation on the frame the echo first fills.
    # A run without output_dbz prices exactly the words it always has.
    if bool(getattr(config, "output_dbz", False)):
        record["reflectivity_coverage_mask_bytes"] = g.ny * g.nx
        record["frame_scratch_peak_bytes"] += g.ny * g.nx
    return record


@dataclass(slots=True)
class _FrameStorage:
    """Registry-budgeted frame scratch released on every path, refusals included.

    This mirrors ``cuda_cartesian_closure._PhaseStorage``.  A frame that
    refuses must not leave a volume-sized allocation behind: at one frame
    every few hundred steps, a leaked plane would outlive the run.
    """

    registry: Any
    names: list[str] = field(default_factory=list)

    def allocate(self, name: str, shape: tuple[int, ...], *, dtype: Any = None) -> Any:
        full = f"{_FRAME_PREFIX}.{name}"
        value = self.registry.allocate(full, shape, dtype=dtype)
        self.names.append(full)
        return value

    def release(self) -> None:
        if self.names:
            self.registry.release(self.names)
            self.names.clear()


def vorticity_report_levels(heights_m: Any, nz: int) -> dict[str, int | None]:
    """Transcribe ``misclibs.F:2454-2466``'s descending level scan.

    ``heights_m`` is the model's own ``zh`` column.  CM1 scans ``k=nk,1,-1``
    and keeps the last assignment, which is the lowest level at or above each
    height; a height the column never reaches leaves ``k=nk+1``, which the
    ``kcheck`` guard then skips.  That skip is reported as ``None`` rather
    than substituted with the model top.
    """

    if type(nz) is not int or nz < 1:
        raise ValueError("nz must be a positive integer")
    if len(heights_m) != nz:
        raise ValueError("the report-level scan needs exactly one zh word per model level")
    found = {name: nz + 1 for name in VORTICITY_REPORT_NAMES[1:]}
    for k in range(nz, 0, -1):
        z = float(heights_m[k - 1])
        for name, threshold in zip(VORTICITY_REPORT_NAMES[1:], VORTICITY_REPORT_HEIGHTS_M):
            if z >= threshold:
                found[name] = k
    levels: dict[str, int | None] = {VORTICITY_REPORT_NAMES[0]: 1}
    for name in VORTICITY_REPORT_NAMES[1:]:
        levels[name] = None if found[name] > nz else found[name]
    return levels


def _reduce_courant(
    *,
    cp: Any,
    grid: Any,
    dynamics: Any,
    stream: Any,
    dtdx: Any,
    dtdy: Any,
    dtdz: Any,
    accumulator: Any,
    integrand: Any,
    k0: int = 0,
    k1: int | None = None,
) -> float:
    """``misclibs.F:2612-2631`` and its ``calccflquick`` twin at 2747-2765.

    CM1 forms ``dtdx=0.5*dt*rdx`` and multiplies it by ``ua(i)+ua(i+1)``, the
    sum of the two faces, so each cell contributes ``u*dt/dx``.  The two
    horizontal squares are summed before the vertical one is added; that
    grouping is CM1's and is kept.  ``uh``/``vh`` and ``mh`` are CM1's own
    stretching factors and are 1.0 on this runtime's uniform grid.

    ``k0``/``k1`` select a contiguous band of model levels, so a caller with
    one plane of scratch can walk the column the way CM1's own ``cfl(k)``
    array does (misclibs.F:2752-2770) and take the maximum over the layers
    afterwards.  The default is the whole volume, which needs volume scratch.
    """

    g, d = grid, dynamics
    k1 = g.nz if k1 is None else k1
    u, v, w = d.fields["u3d"], d.fields["v3d"], d.fields["w3d"]
    uh = d.arrays["uh"][3:3 + g.nx]
    vh = d.arrays["vh"][3:3 + g.ny]
    mh = d.metrics["mh"][k0:k1]
    acc, term = accumulator, integrand
    with stream:
        cp.add(u[k0:k1, :, :g.nx], u[k0:k1, :, 1:], out=term)
        cp.multiply(term, dtdx, out=term)
        cp.multiply(term, uh[None, None, :], out=term)
        cp.multiply(term, term, out=acc)
        cp.add(v[k0:k1, :g.ny, :], v[k0:k1, 1:, :], out=term)
        cp.multiply(term, dtdy, out=term)
        cp.multiply(term, vh[None, :, None], out=term)
        cp.multiply(term, term, out=term)
        cp.add(acc, term, out=acc)
        cp.add(w[k0:k1], w[k0 + 1:k1 + 1], out=term)
        cp.multiply(term, dtdz, out=term)
        cp.multiply(term, mh, out=term)
        cp.multiply(term, term, out=term)
        cp.add(acc, term, out=acc)
        cp.sqrt(acc, out=acc)
        return float(cp.asnumpy(cp.amax(acc)))


def advective_courant_number(runtime: Any) -> float:
    """CM1's ``calccflquick`` (misclibs.F:2726-2814) on one live runtime.

    ``solve2.F:1254-1256`` forms this norm on the final RK substep of every
    large step.  The stop a serial ``adapt_dt=0`` build reaches is
    ``misclibs.F:2714`` inside ``calccfl``, which ``statpack.F:469`` calls at
    statistics cadence whenever ``stat_cfl=1`` (CM1's own shipped default,
    run/namelist.input:320); ``solve2.F:2082`` applies the same test to
    ``calccflquick``'s result but is inside ``#ifdef MPI``.

    This walks the column one model level at a time and keeps the largest, as
    CM1's own ``cfl(k)`` array does, so the scratch is two horizontal planes
    rather than two volumes.  A caller that only wants the scalar therefore
    costs a fraction of a per-frame reducer's peak and needs no registry
    budget of its own.
    """

    cp, g = runtime.cp, runtime.config.grid
    f32 = cp.float32
    dt_s = runtime.config.dt_s
    with runtime.stream:
        accumulator = cp.empty((1, g.ny, g.nx), dtype=cp.float32)
        integrand = cp.empty((1, g.ny, g.nx), dtype=cp.float32)
    try:
        # misclibs.F:2747-2749.  dt and the reciprocal spacings are CM1
        # default reals, multiplied in CM1's own order.
        common = {
            "cp": cp,
            "grid": g,
            "dynamics": runtime.dynamics,
            "stream": runtime.stream,
            "dtdx": f32(f32(0.5) * f32(dt_s) * f32(1.0 / g.dx_m)),
            "dtdy": f32(f32(0.5) * f32(dt_s) * f32(1.0 / g.dy_m)),
            "dtdz": f32(f32(0.5) * f32(dt_s) * f32(1.0 / g.dz_m)),
            "accumulator": accumulator,
            "integrand": integrand,
        }
        largest = 0.0
        for k in range(g.nz):
            largest = max(largest, _reduce_courant(**common, k0=k, k1=k + 1))
        return largest
    finally:
        del accumulator, integrand


class FrameDiagnostics:
    """One reusable reducer bound to one live runtime.

    Construction reads the vertical column once and prices the scratch.  Each
    frame allocates that scratch, reduces, and releases it, so a long run's
    diagnostic cost is flat in the number of completed steps and nothing is
    retained between frames.
    """

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime
        self.cp = runtime.cp
        self.stream = runtime.stream
        self.config = runtime.config
        self.grid = self.config.grid
        self.dynamics = runtime.dynamics
        self.finish = runtime.finish
        g = self.grid
        if g.nx < 2 or g.ny < 2:
            raise ValueError(
                "CM1 vertvort differences neighbouring vorticity points; a single "
                "cell row has no interior corner and would report an empty maximum"
            )
        self.scratch_plan = frame_diagnostic_scratch_bytes(self.config)
        column = self.cp.asnumpy(self.dynamics.metrics["zh"][:, 0, 0])
        self.zh_column_m = [float(value) for value in column]
        self.report_levels = vorticity_report_levels(self.zh_column_m, g.nz)
        # misclibs.F:2612.  dt and the reciprocal spacings are CM1 reals.
        f32 = self.cp.float32
        self.dtdx = f32(f32(0.5) * f32(self.config.dt_s) * f32(1.0 / g.dx_m))
        self.dtdy = f32(f32(0.5) * f32(self.config.dt_s) * f32(1.0 / g.dy_m))
        self.dtdz = f32(f32(0.5) * f32(self.config.dt_s) * f32(1.0 / g.dz_m))
        self.rdx = f32(1.0 / g.dx_m)
        self.rdy = f32(1.0 / g.dy_m)
        self.frames = 0

    # -- CM1 transcriptions -------------------------------------------------

    def _courant(self, scratch: Mapping[str, Any]) -> float:
        """``misclibs.F:2612-2631``: the wind Courant number CM1 stops on."""

        return _reduce_courant(
            cp=self.cp,
            grid=self.grid,
            dynamics=self.dynamics,
            stream=self.stream,
            dtdx=self.dtdx,
            dtdy=self.dtdy,
            dtdz=self.dtdz,
            accumulator=scratch["courant"],
            integrand=scratch["integrand"],
        )

    def _vertical_vorticity(self, scratch: Mapping[str, Any]) -> dict[str, float | None]:
        """``misclibs.F:2523-2529``: signed maximum vorticity at report levels.

        CM1 maximizes the signed value, not its magnitude, so a run whose only
        strong rotation is anticyclonic reports a small number here; the
        minimum is reported beside it so the sign is never lost.  The interior
        bound is CM1's own ``1+ibw .. ni+1-ibe``, which is ``2..ni`` for the
        non-periodic lateral boundaries this runtime uses (``param.F:6678``).
        """

        cp, g, d = self.cp, self.grid, self.dynamics
        u, v = d.fields["u3d"], d.fields["v3d"]
        plane, other = scratch["vorticity"], scratch["vorticity-term"]
        out: dict[str, float | None] = {}
        with self.stream:
            for name, level in self.report_levels.items():
                if level is None:
                    out[f"{name}_max_s_inv"] = None
                    out[f"{name}_min_s_inv"] = None
                    continue
                k = level - 1
                cp.subtract(v[k, 1:g.ny, 1:g.nx], v[k, 1:g.ny, 0:g.nx - 1], out=plane)
                cp.multiply(plane, self.rdx, out=plane)
                cp.subtract(u[k, 1:g.ny, 1:g.nx], u[k, 0:g.ny - 1, 1:g.nx], out=other)
                cp.multiply(other, self.rdy, out=other)
                cp.subtract(plane, other, out=plane)
                out[f"{name}_max_s_inv"] = float(cp.asnumpy(cp.amax(plane)))
                out[f"{name}_min_s_inv"] = float(cp.asnumpy(cp.amin(plane)))
        return out

    def _species_mass(self, scratch: Mapping[str, Any]) -> dict[str, float]:
        """``misclibs.F:3299-3310``: one ``totq`` volume integral per species.

        Fortran multiplies left to right, so the integrand is
        ``(((rho*q)*ruh)*rvh)*rmh`` in float32, and the layer sum is float64.
        The host multiplies by ``dx*dy*dz`` exactly as ``misclibs.F:3318``.
        """

        cp, g, d = self.cp, self.grid, self.dynamics
        profile = self.config.profile
        rho, q3d = d.fields["rho"], d.fields["q3d"]
        ruh = d.arrays["uh"][3:3 + g.nx]
        rvh = d.arrays["vh"][3:3 + g.ny]
        rmh = d.metrics["rmh"]
        term = scratch["integrand"]
        cell_volume = float(g.dx_m) * float(g.dy_m) * float(g.dz_m)
        out: dict[str, float] = {}
        with self.stream:
            # statpack.F:507-517 calls totq for the vapour and the liquid/solid
            # ranges only; a kg-1 number concentration has no mass integral.
            for index in profile.mass_indices:
                name = profile.names[index]
                cp.multiply(rho, q3d[index], out=term)
                cp.multiply(term, ruh[None, None, :], out=term)
                cp.multiply(term, rvh[None, :, None], out=term)
                cp.multiply(term, rmh, out=term)
                layers = cp.sum(term, axis=(1, 2), dtype=cp.float64)
                out[name] = float(cp.asnumpy(cp.sum(layers, dtype=cp.float64))) * cell_volume
        return out

    def _reflectivity(self, step: int) -> dict[str, Any]:
        """The plan-view radar products for this completed frame.

        ``cref`` is CM1's composite (``misclibs.F:2259-2292``), already folded
        on the device by ``CartesianFinish.composite_reflectivity``; this only
        reduces it to the scalars a record row carries.  The coverage counts
        are column counts above a declared dBZ threshold and the area they
        cover, which is what a reader needs to tell a 60 dBZ pixel from a
        60 dBZ core.

        Nothing is derived here: every word read was produced by the
        microphysics scheme that owns the size distributions it depends on.
        """

        finish = self.finish
        if not getattr(finish, "output_dbz", False):
            return {}
        computed = int(getattr(finish, "completed_reflectivity_step", 0))
        if computed != step:
            # cm1.F:3214-3237 raises getdbz only on output steps.  A row that
            # reported the previous output step's dBZ beside this step's winds
            # would read as a storm that stopped evolving.
            return {
                "reflectivity_available": False,
                "reflectivity_computed_at_step": computed,
            }
        cp, g = self.cp, self.grid
        dbz = finish.reflectivity["dbz"]
        cref = finish.reflectivity["cref"]
        cell_area_km2 = float(g.dx_m) * float(g.dy_m) / 1.0e6
        row: dict[str, Any] = {
            "reflectivity_available": True,
            "reflectivity_computed_at_step": computed,
        }
        with self.stream:
            row["reflectivity_max_dbz"] = float(cp.asnumpy(cp.amax(dbz)))
            row["composite_reflectivity_max_dbz"] = float(cp.asnumpy(cp.amax(cref)))
            coverage: dict[str, Any] = {}
            for threshold in REFLECTIVITY_COVERAGE_THRESHOLDS_DBZ:
                columns = int(
                    cp.asnumpy(cp.count_nonzero(cref >= cp.float32(threshold)))
                )
                coverage[f"columns_ge_{threshold:g}_dbz"] = columns
                coverage[f"area_km2_ge_{threshold:g}_dbz"] = columns * cell_area_km2
            level = getattr(finish, "reflectivity_level", None)
            if level is not None:
                plane = finish.reflectivity["dbz_level"]
                row["reflectivity_level_index"] = int(level)
                row["reflectivity_level_m"] = float(self.zh_column_m[int(level)])
                row["reflectivity_at_level_max_dbz"] = float(
                    cp.asnumpy(cp.amax(plane))
                )
        row["composite_reflectivity_coverage"] = coverage
        row["reflectivity_coverage_thresholds_dbz"] = list(
            REFLECTIVITY_COVERAGE_THRESHOLDS_DBZ
        )
        row["reflectivity_coverage_authority"] = (
            "caller-selected reporting thresholds; CM1 writes dbz and cref and "
            "applies no threshold to either"
        )
        row["reflectivity_authority"] = (
            "CM1 r21.1 README.namelist:1366-1369 output_dbz; misclibs.F:2259-2292 "
            "calccref; the volume is the microphysics scheme's own diagnostic"
        )
        return row

    def _dry_mass_ledger(self) -> tuple[float, float]:
        """``solve2.F:1616-1656``: the same layer ledgers the correction reads.

        This reuses ``CartesianFinish``'s pinned apmasscon layer kernel and
        the serial float64 reduce, so the returned ``mass2`` is the number the
        correction would divide ``mass1`` by, and the returned Exner sum is
        the ``p2`` it would scale.  Nothing is written back.
        """

        cp, g, finish = self.cp, self.grid, self.finish
        arrays = finish._mass_arrays()
        with self.stream:
            layer_mass = cp.empty((g.nz,), dtype=cp.float64)
            layer_pressure = cp.empty((g.nz,), dtype=cp.float64)
            totals = cp.empty((2,), dtype=cp.float64)
            finish.apmasscon.kernels[0](
                (g.nz, 1, 1), (256, 1, 1),
                (arrays["rho"], arrays["ruh"], arrays["rvh"], arrays["rmh"],
                 arrays["pi0"], arrays["pp3d"], layer_mass, layer_pressure,
                 cp.uint32(g.nx), cp.uint32(g.ny), cp.uint32(g.nz)),
                stream=self.stream,
            )
            for source, slot in ((layer_mass, totals[0:1]), (layer_pressure, totals[1:2])):
                finish.forecast.kernels[2](
                    (1, 1, 1), (1, 1, 1), (source, slot, cp.uint32(g.nz)),
                    stream=self.stream,
                )
            host = cp.asnumpy(totals)
        return float(host[0]), float(host[1])

    # -- frame --------------------------------------------------------------

    def frame(self, *, step: int, wall_seconds: float | None = None) -> dict[str, Any]:
        """Reduce one completed state; allocate, reduce and release the scratch."""

        if type(step) is not int or step < 1:
            raise ValueError("step must be a positive integer")
        cp, g, d = self.cp, self.grid, self.dynamics
        profile = self.config.profile
        corner = (g.ny - 1, g.nx - 1)
        storage = _FrameStorage(self.runtime.registry)
        try:
            scratch = {
                "courant": storage.allocate("courant", g.mass_shape),
                "integrand": storage.allocate("integrand", g.mass_shape),
                "vorticity": storage.allocate("vorticity", corner),
                "vorticity-term": storage.allocate("vorticity-term", corner),
            }
            courant = self._courant(scratch)
            vorticity = self._vertical_vorticity(scratch)
            species = self._species_mass(scratch)
        finally:
            scratch = None
            storage.release()
        mass2, exner_sum = self._dry_mass_ledger()
        mass1 = float(self.finish.initial_mass1_f64)
        with self.stream:
            w_max = float(cp.asnumpy(cp.amax(d.fields["w3d"])))
            w_min = float(cp.asnumpy(cp.amin(d.fields["w3d"])))
            th_max = float(cp.asnumpy(cp.amax(d.fields["th3d"])))
            th_min = float(cp.asnumpy(cp.amin(d.fields["th3d"])))
            rho_min = float(cp.asnumpy(cp.amin(d.fields["rho"])))
            prs_min = float(cp.asnumpy(cp.amin(d.fields["prs"])))
            mass_rows = len(profile.mass_names)
            q_min = float(cp.asnumpy(cp.amin(d.fields["q3d"][:mass_rows])))
            number_min = (
                float(cp.asnumpy(cp.amin(d.fields["q3d"][mass_rows:])))
                if profile.number_names
                else None
            )
            tke_max = float(cp.asnumpy(cp.amax(d.tke["tkea"])))
            asq = [float(value) for value in cp.asnumpy(self.finish.asq)]
        condensate = sum(
            species[profile.names[index]] for index in profile.condensate_indices
        )
        vapour = species[profile.names[profile.vapour_index]]
        row: dict[str, Any] = {
            "schema": FRAME_DIAGNOSTIC_SCHEMA,
            "step": step,
            "model_seconds": step * self.config.dt_s,
            "w_max_m_s": w_max,
            "w_min_m_s": w_min,
            "theta_perturbation_max_k": th_max,
            "theta_perturbation_min_k": th_min,
            "density_min_kg_m3": rho_min,
            "pressure_min_pa": prs_min,
            "moisture_min_kg_kg": q_min,
            "number_concentration_min_kg_inv": number_min,
            "tke_max_m2_s2": tke_max,
            "courant_number": courant,
            "courant_stop_threshold": CM1_COURANT_STOP,
            "species_mass_kg": species,
            "total_condensate_kg": condensate,
            "total_water_kg": condensate + vapour,
            "dry_mass_ledger": mass2,
            "dry_mass_reference": mass1,
            "dry_mass_relative_drift": (mass2 - mass1) / mass1 if mass1 else None,
            "dry_mass_total_kg": mass2 * float(g.dx_m) * float(g.dy_m) * float(g.dz_m),
            "exner_perturbation_sum": exner_sum,
            "component_id": self.config.microphysics,
            "cm1_ptype": profile.cm1_ptype,
            "ihail": self.config.ihail,
            "dense_ice_category": self.config.dense_ice.category,
            # mp_driver.F repairs only the mass prefix, so asq has exactly
            # that many entries and they are named for exactly those scalars.
            "artificial_moisture_source": dict(
                zip(profile.names[: len(asq)], asq, strict=True)
            ),
            "registered_model_bytes": int(self.runtime.registry.registered_allocation_bytes),
            "frame_scratch_peak_bytes": self.scratch_plan["frame_scratch_peak_bytes"],
            "step_wall_seconds": wall_seconds,
        }
        row.update(vorticity)
        row.update(self._reflectivity(step))
        # CM1's output_* diagnostics, when this run's document selected any.
        # A run that declared no [diagnostics] table adds no key here, so its
        # per-frame record is exactly the row it has always written.
        tornado = getattr(self.runtime, "tornado", None)
        if tornado is not None:
            row.update(tornado.frame_row(step))
        row["stop_reason"] = frame_stop_reason(row)
        self.frames += 1
        return row


def _numeric_words(row: Mapping[str, Any]) -> list[tuple[str, float]]:
    words: list[tuple[str, float]] = []
    for name, value in row.items():
        if isinstance(value, bool) or value is None:
            continue
        if isinstance(value, (int, float)):
            words.append((name, float(value)))
        elif isinstance(value, Mapping):
            for key, item in value.items():
                if isinstance(item, (int, float)) and not isinstance(item, bool):
                    words.append((f"{name}.{key}", float(item)))
    return words


def frame_stop_reason(row: Mapping[str, Any]) -> str | None:
    """Name the concrete defect that makes continuing pointless, or None.

    Two conditions only, both of them CM1's own: a non-finite diagnostic, and
    the Courant number at or above the value at which ``misclibs.F:2714`` sets
    ``stopit``.  Mass drift is reported but is not a stop here, because CM1
    has no threshold for it and this module does not invent one; the driver
    applies its own named operational limit instead.
    """

    for name, value in _numeric_words(row):
        if not math.isfinite(value):
            return f"non-finite diagnostic {name}={value!r}"
    courant = row.get("courant_number")
    if isinstance(courant, (int, float)) and not isinstance(courant, bool):
        if float(courant) >= CM1_COURANT_STOP:
            return (
                f"Courant number {float(courant):.4f} reached CM1's own stop value "
                f"{CM1_COURANT_STOP} (misclibs.F:2714)"
            )
    return None


class BoundedSampleLog(list):
    """A per-step sample list that cannot grow without bound, and says so.

    ``CudaCartesianRuntime._sample`` appends one dictionary per scheduled
    operation, twenty-nine per large step, and never trims.  Fourteen thousand
    steps would retain roughly four hundred thousand dictionaries and then try
    to serialize every one of them into a single receipt.  A driver that
    intends a long run installs this in place of the plain list: it keeps the
    first ``window`` samples and the last ``window`` samples and counts what it
    dropped, so a receipt reports a sampled record and never claims to hold
    every sample.

    This is a driver-side mitigation for a defect in the runtime, not a fix.
    The fix belongs in ``cuda_cartesian_runtime.py``.
    """

    def __init__(self, *, window: int = 256) -> None:
        super().__init__()
        if type(window) is not int or window < 1:
            raise ValueError("window must be a positive integer")
        self.window = window
        self.dropped = 0
        self.tail: list[Any] = []

    def append(self, value: Any) -> None:
        if len(self) < self.window:
            super().append(value)
            return
        self.tail.append(value)
        if len(self.tail) > self.window:
            del self.tail[0]
            self.dropped += 1

    def extend(self, values: Any) -> None:
        # list.extend is a C-level bulk copy that would bypass the window.
        for value in values:
            self.append(value)

    def record(self) -> dict[str, Any]:
        return {
            "schema": "gm1.bounded-sample-log/v1",
            "window": self.window,
            "retained_head": len(self),
            "retained_tail": len(self.tail),
            "dropped": self.dropped,
            "head": list(self),
            "tail": list(self.tail),
            "claim_limit": (
                "the first and last window samples of a per-operation sample stream; "
                "the dropped middle is counted, never reconstructed"
            ),
        }


__all__ = (
    "BoundedSampleLog",
    "CM1_COURANT_STOP",
    "FRAME_DIAGNOSTIC_CLAIM_LIMIT",
    "FRAME_DIAGNOSTIC_SCHEMA",
    "FrameDiagnostics",
    "SOURCE_ANCHORS",
    "VORTICITY_REPORT_HEIGHTS_M",
    "VORTICITY_REPORT_NAMES",
    "WSM6_CONDENSATE_INDICES",
    "WSM6_SPECIES_NAMES",
    "WSM6_VAPOUR_INDEX",
    "frame_diagnostic_scratch_bytes",
    "frame_stop_reason",
    "vorticity_report_levels",
)
