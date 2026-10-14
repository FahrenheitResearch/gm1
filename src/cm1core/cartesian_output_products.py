"""Bind CM1 ``&param9`` to the Cartesian runtime: allocate, step, publish.

Three cadences, and they are not interchangeable:

*per step*
    ``solve3.F:857-1130`` folds the seven run-long surface extrema on every
    large step.  Sampling them at frame times instead would turn a run-long
    maximum into a ten-minute sample, and a tornado-scale vortex that forms
    and dies between two frames would leave no trace at all.

*per frame, inside the microphysics call*
    ``output_dbz``.  The ArWen WSM6 component computes its audited 10 cm
    reflectivity inside the same call that advances the microphysics, so the
    runtime raises ``reflectivity_due`` on the large step whose state the next
    frame will carry.

*per frame, after solve3*
    Everything else -- ``calcuh``, ``calcvort``, ``getcape``, the derived
    perturbation/absolute pairs and the column integrals -- is a pure function
    of the completed state and is computed once, immediately before staging.

Nothing here computes physics in Python.  Every product is either a launch of
:mod:`cm1core.cuda_cm1_output_diagnostics` / :mod:`cm1core.cuda_cm1_getcape`
or a CuPy elementwise expression enqueued on the runtime's own stream, which
is the CUDA driver, not a host loop.
"""

from __future__ import annotations

from typing import Any

from . import cuda_graphs
from .cm1_output_selection import (
    OutputSelection,
    PER_STEP_PRODUCTS,
    requires_per_step_work,
    requires_updraft_helicity,
)
from .cuda_cm1_getcape import (
    CM1_GETCAPE_ADIABAT,
    CM1_GETCAPE_ML_DEPTH_M,
    CM1_GETCAPE_PINC_PA,
    GETCAPE_SCRATCH_NAMES,
    compile_cuda_cm1_getcape,
    getcape_receipt,
    getcape_scratch_bytes,
)
from .cuda_cm1_output_diagnostics import (
    CM1_CREF_INITIAL,
    CM1_SWATH_INITIAL,
    CM1_UH_LAYER_BOTTOM_M,
    CM1_UH_LAYER_TOP_M,
    CUDA_CM1_OUTPUT_DIAGNOSTICS_THREADS,
    compile_cuda_cm1_output_diagnostics,
    diagnostic_receipt,
    require_uniform_diagnostic_grid,
)

CARTESIAN_OUTPUT_PRODUCTS_SCHEMA = "gm1.cartesian-output-products/v1"

#: CM1's own constants block, ``src/constants.F:112-132``.
CM1_P00 = 100000.0
CM1_RD = 287.04
CM1_CP = 1005.7

#: ``src/getcape.F:3`` mixed-layer parcel, which ``writeout.F:2921`` uses for
#: the 2-D CAPE/CIN/LCL/LFC fields.
CM1_GETCAPE_FIELD_SOURCE = 3

#: WSM6 accumulations are millimetres; CM1's ``rain`` output is centimetres
#: (``src/writeout.F:365``).
MM_TO_CM = 0.1


def _blocks(words: int) -> tuple[int, int, int]:
    threads = CUDA_CM1_OUTPUT_DIAGNOSTICS_THREADS
    return ((max(1, words) + threads - 1) // threads, 1, 1)


class CartesianOutputProducts:
    """One run's ``&param9`` products: allocation, cadence and staging."""

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime
        self.config = runtime.config
        self.grid = self.config.grid
        self.cp = runtime.cp
        self.stream = runtime.stream
        self.registry = runtime.registry
        self.selection: OutputSelection = self.config.output
        self.planes: dict[str, Any] = {}
        self.volumes: dict[str, Any] = {}
        self.profiles: dict[str, Any] = {}
        self.compiled: Any = None
        self.getcape_compiled: Any = None
        self.completed_swath_step = 0
        self._threads = (CUDA_CM1_OUTPUT_DIAGNOSTICS_THREADS, 1, 1)
        self.needs_uh = requires_updraft_helicity(self.selection)
        # Only xvort/yvort carry CM1's vertical metric; calcuh's zeta and the
        # svs corner vorticity are horizontal derivatives at fixed k, and
        # calcuh takes the column's real zh/zf (see config.py's
        # CM1-OUTPUT-STRETCHED-METRIC note).
        self.needs_vorticity_metric = any(
            self.selection.on(key) for key in ("output_vort", "output_pv")
        )
        if self.needs_vorticity_metric:
            require_uniform_diagnostic_grid(self.grid)
        self._allocate()

    # -------------------------------------------------------------- pricing

    @property
    def reflectivity_wanted(self) -> bool:
        return self.selection.on("output_dbz")

    @property
    def per_step_wanted(self) -> bool:
        return requires_per_step_work(self.selection)

    def _allocate(self) -> None:
        g, cp = self.grid, self.cp
        for key, name in (
            ("output_sws", "sws"), ("output_svs", "svs"), ("output_sps", "sps"),
            ("output_srs", "srs"), ("output_sgs", "sgs"), ("output_sus", "sus"),
            ("output_shs", "shs"),
        ):
            if not self.selection.on(key):
                continue
            plane = self.registry.allocate(f"output.{name}", g.surface_shape, zero=True)
            with self.stream:
                plane.fill(cp.float32(CM1_SWATH_INITIAL[name]))
            self.planes[name] = plane
        if self.needs_uh:
            # calcuh's corner vorticity volume and the plane it integrates to.
            self.volumes["zeta"] = self.registry.allocate(
                "output.zeta", (g.nz, g.ny + 1, g.nx + 1), zero=True
            )
            self.planes["uh"] = self.registry.allocate(
                "output.uh", g.surface_shape, zero=True
            )
        if self.reflectivity_wanted:
            self.volumes["dbz"] = self.registry.allocate(
                "output.dbz", g.mass_shape, zero=True
            )
            with self.stream:
                self.volumes["dbz"].fill(cp.float32(CM1_CREF_INITIAL))
            self.planes["cref"] = self.registry.allocate(
                "output.cref", g.surface_shape, zero=True
            )

    # ------------------------------------------------------- base profiles

    def _build_profiles(self) -> None:
        """Derive the nz-long base-state column once and retain it.

        ``src/writeout.F:4430-4520`` writes ``pi0``/``th0``/``prs0``/``qv0``
        as 3-D fields.  This lane's base state is horizontally uniform by
        construction (one sounding, flat terrain), so it is published as the
        nz-long column it actually is; the Rust GrADS exporter broadcasts it
        back to CM1's 3-D record when a CM1 tool asks for one.
        """

        if self.profiles or not (
            self.selection.on("output_basestate") or self.selection.on("output_zh")
        ):
            return
        cp, g = self.cp, self.grid
        dynamics = self.runtime.dynamics
        with self.stream:
            if self.selection.on("output_zh"):
                self.profiles["zhval"] = cp.ascontiguousarray(
                    dynamics.metrics["zh"][:, 0, 0]
                )
            if self.selection.on("output_basestate"):
                for name in ("pi0", "th0", "qv0", "u0", "v0"):
                    self.profiles[name] = cp.ascontiguousarray(
                        dynamics.base_profiles[name]
                    )
                # prs0 = p00 * pi0**(cp/rd), the inverse of param.F's Exner
                # definition; CM1 carries prs0 as its own array and this lane
                # does not, so it is derived from the pi0 it does carry.
                self.profiles["prs0"] = (
                    cp.float32(CM1_P00)
                    * cp.power(self.profiles["pi0"], cp.float32(CM1_CP / CM1_RD))
                ).astype(cp.float32)

    # ------------------------------------------------------------ per step

    def _compiled_kernels(self) -> Any:
        if self.compiled is None:
            self.compiled = compile_cuda_cm1_output_diagnostics()
        return self.compiled

    def _sus_face(self) -> int:
        """``solve3.F:1076-1082``: the first w face at or above 5 km AGL.

        CM1 walks the column per point; on this lane's flat, horizontally
        uniform grid the answer is the same in every column, so it is settled
        once from the level heights rather than re-walked per thread.
        """

        levels = self.runtime.dynamics.metrics["zf"][:, 0, 0].get()
        k = 2
        while float(levels[k - 1]) < 5000.0 and k < self.grid.nz:
            k += 1
        return k - 1

    def per_step(self, step: int) -> None:
        """Fold every run-long surface extremum, exactly as solve3.F does."""

        if not self.per_step_wanted:
            cuda_graphs.bookkeeping(self._set_swath_step, step)
            return
        if step != self.completed_swath_step + 1:
            raise RuntimeError(
                f"the surface swaths stand at step {self.completed_swath_step} and "
                f"were asked to fold step {step}; CM1 folds them on every large "
                "step (solve3.F:857-1130), and a skipped step is a run-long "
                "maximum that silently never saw part of the run"
            )
        cp, g = self.cp, self.grid
        dynamics = self.runtime.dynamics
        compiled = self._compiled_kernels()
        zero = cp.zeros(1, dtype=cp.float32)
        self._build_profiles()
        prs0 = self.profiles.get("prs0")
        if prs0 is None:
            with self.stream:
                prs0 = cp.ascontiguousarray(
                    cp.float32(CM1_P00)
                    * cp.power(dynamics.base_profiles["pi0"], cp.float32(CM1_CP / CM1_RD))
                ).astype(cp.float32)
            self.profiles["prs0"] = prs0
        if self.needs_uh:
            self._launch_uh(compiled)
        grid_blocks = _blocks(g.nx * g.ny)
        with self.stream:
            compiled.by_name["cm1_out_swath_update_v1"](
                grid_blocks, self._threads,
                (
                    dynamics.arrays["u3d"], dynamics.arrays["v3d"], dynamics.arrays["w3d"],
                    dynamics.fields["prs"][0], prs0,
                    dynamics.fields["q3d"][2][0], dynamics.fields["q3d"][5][0],
                    self.planes.get("uh", zero),
                    cp.ascontiguousarray(dynamics.metrics["zf"][:, 0, 0]),
                    self.planes.get("sws", zero), self.planes.get("svs", zero),
                    self.planes.get("sps", zero), self.planes.get("srs", zero),
                    self.planes.get("sgs", zero), self.planes.get("sus", zero),
                    self.planes.get("shs", zero),
                    cp.int32(g.nx), cp.int32(g.ny), cp.int32(g.nz),
                    cp.float32(1.0 / g.dx_m), cp.float32(1.0 / g.dy_m),
                    cp.float32(self.config.u_move_m_s),
                    cp.float32(self.config.v_move_m_s),
                    cp.int32(self._sus_face()),
                ),
                stream=self.stream,
            )
        cuda_graphs.bookkeeping(self._set_swath_step, step)

    def _set_swath_step(self, step: int) -> None:
        self.completed_swath_step = step

    def _launch_uh(self, compiled: Any) -> None:
        cp, g = self.cp, self.grid
        dynamics = self.runtime.dynamics
        with self.stream:
            compiled.by_name["cm1_out_zeta_corner_v1"](
                _blocks(g.nz * (g.ny + 1) * (g.nx + 1)), self._threads,
                (dynamics.arrays["u3d"], dynamics.arrays["v3d"], self.volumes["zeta"],
                 cp.int32(g.nx), cp.int32(g.ny), cp.int32(g.nz),
                 cp.float32(1.0 / g.dx_m), cp.float32(1.0 / g.dy_m)),
                stream=self.stream,
            )
            compiled.by_name["cm1_out_calcuh_v1"](
                _blocks(g.nx * g.ny), self._threads,
                (self.volumes["zeta"], dynamics.arrays["w3d"],
                 cp.ascontiguousarray(dynamics.metrics["zh"][:, 0, 0]),
                 cp.ascontiguousarray(dynamics.metrics["zf"][:, 0, 0]),
                 self.planes["uh"],
                 cp.int32(g.nx), cp.int32(g.ny), cp.int32(g.nz),
                 cp.float32(CM1_UH_LAYER_BOTTOM_M), cp.float32(CM1_UH_LAYER_TOP_M)),
                stream=self.stream,
            )

    # -------------------------------------------------------- reflectivity

    def accept_reflectivity(self, *, array: Any, j0: int, j1: int, i0: int, i1: int) -> None:
        """Copy one WSM6 block's audited 10 cm reflectivity into the volume."""

        if not self.reflectivity_wanted:
            raise RuntimeError(
                "a WSM6 block produced reflectivity that output_dbz did not ask "
                "for; the frame would carry a field the selection does not name"
            )
        with self.stream:
            self.cp.copyto(self.volumes["dbz"][:, j0:j1, i0:i1], array)

    # ----------------------------------------------------------- per frame

    def materialize(self) -> dict[str, Any]:
        """Compute every publication-time product and return the staging map."""

        cp, g = self.cp, self.grid
        dynamics = self.runtime.dynamics
        selection = self.selection
        out: dict[str, Any] = {}
        self._build_profiles()
        out.update(self.profiles)
        for name, plane in self.planes.items():
            if name != "uh" or selection.on("output_uh"):
                out[name] = plane
        if selection.on("output_dbz"):
            compiled = self._compiled_kernels()
            with self.stream:
                compiled.by_name["cm1_out_calccref_v1"](
                    _blocks(g.nx * g.ny), self._threads,
                    (self.volumes["dbz"], self.planes["cref"],
                     cp.int32(g.nx), cp.int32(g.ny), cp.int32(g.nz),
                     cp.float32(CM1_CREF_INITIAL)),
                    stream=self.stream,
                )
            out["dbz"] = self.volumes["dbz"]
        if selection.on("output_rain"):
            out.update(self._surface_precipitation())
        if any(selection.on(key) for key in ("output_vort", "output_pv")):
            out.update(self._vorticity())
        if selection.on("output_uh") and not self.per_step_wanted:
            self._launch_uh(self._compiled_kernels())
            out["uh"] = self.planes["uh"]
        if any(selection.on(key) for key in
               ("output_cape", "output_cin", "output_lcl", "output_lfc")):
            out.update(self._getcape())
        out.update(self._derived())
        missing = [name for name, value in out.items() if value is None]
        if missing:
            raise RuntimeError(f"output products left {', '.join(missing)} unfilled")
        self.runtime.output_fields = out
        return out

    def _surface_precipitation(self) -> dict[str, Any]:
        """``writeout.F:361-404``, from the WSM6 surface accumulators.

        The component's accumulators are millimetres and per-step millimetres;
        CM1's ``rain`` output is centimetres and its ``prate`` is kg m-2 s-1,
        which for water is millimetres per second.  Both conversions are named
        here rather than left to the reader of a field table.
        """

        cp = self.cp
        persistence = self.runtime.finish.persistence
        dt = float(self.config.dt_s)
        with self.stream:
            rain = (persistence["mp_rainnc"] * cp.float32(MM_TO_CM)).astype(cp.float32)
            snow = (persistence["mp_snownc"] * cp.float32(MM_TO_CM)).astype(cp.float32)
            graupel = (persistence["mp_graupelnc"] * cp.float32(MM_TO_CM)).astype(cp.float32)
            prate = (persistence["mp_rainncv"] * cp.float32(1.0 / dt)).astype(cp.float32)
            ratio = cp.ascontiguousarray(persistence["mp_sr"])
        return {"rain": rain, "prate": prate, "snownc": snow,
                "graupelnc": graupel, "sr": ratio}

    def _vorticity(self) -> dict[str, Any]:
        cp, g = self.cp, self.grid
        dynamics = self.runtime.dynamics
        compiled = self._compiled_kernels()
        selection = self.selection
        words = _blocks(g.nz * g.ny * g.nx)
        rdx, rdy, rdz = (cp.float32(1.0 / value)
                         for value in (g.dx_m, g.dy_m, g.dz_m))
        bbc, tbc = cp.int32(self.config.bbc), cp.int32(self.config.tbc)
        umove = cp.float32(self.config.u_move_m_s)
        vmove = cp.float32(self.config.v_move_m_s)
        out: dict[str, Any] = {}
        dgs, dgt = self._fornberg_derivative_weights()
        with self.stream:
            if selection.on("output_vort"):
                for name, kernel, arguments in (
                    ("xvort", "cm1_out_xvort_v1",
                     (dynamics.arrays["v3d"], dynamics.arrays["w3d"])),
                    ("yvort", "cm1_out_yvort_v1",
                     (dynamics.arrays["u3d"], dynamics.arrays["w3d"])),
                ):
                    target = cp.empty(g.mass_shape, dtype=cp.float32)
                    compiled.by_name[kernel](
                        words, self._threads,
                        (*arguments, target, cp.int32(g.nx), cp.int32(g.ny),
                         cp.int32(g.nz), rdx, rdy, rdz, bbc, tbc,
                         vmove if name == "xvort" else umove),
                        stream=self.stream,
                    )
                    out[name] = target
                target = cp.empty(g.mass_shape, dtype=cp.float32)
                compiled.by_name["cm1_out_zvort_v1"](
                    words, self._threads,
                    (dynamics.arrays["u3d"], dynamics.arrays["v3d"], target,
                     cp.int32(g.nx), cp.int32(g.ny), cp.int32(g.nz), rdx, rdy),
                    stream=self.stream,
                )
                out["zvort"] = target
            if selection.on("output_pv"):
                target = cp.empty(g.mass_shape, dtype=cp.float32)
                compiled.by_name["cm1_out_pv_v1"](
                    words, self._threads,
                    (dynamics.arrays["u3d"], dynamics.arrays["v3d"],
                     dynamics.arrays["w3d"], dynamics.fields["th0"],
                     dynamics.fields["th3d"], dynamics.fields["rho"], target,
                     cp.int32(g.nx), cp.int32(g.ny), cp.int32(g.nz),
                     rdx, rdy, rdz, bbc, tbc, umove, vmove,
                     *(cp.float32(value) for value in (*dgs, *dgt))),
                    stream=self.stream,
                )
                out["pv"] = target
        return out

    def _fornberg_derivative_weights(self) -> tuple[tuple[float, float, float],
                                                    tuple[float, float, float]]:
        from .cm1_vertical_grid import fornberg_derivative_weights

        levels = self.runtime.dynamics.metrics
        zh = levels["zh"][:, 0, 0].get()
        zf = levels["zf"][:, 0, 0].get()
        sigma = [0.0] + [float(value) for value in zh]
        sigmaf = [0.0] + [float(value) for value in zf]
        nk = self.grid.nz
        return (
            fornberg_derivative_weights(sigma, sigmaf, nk=nk, top=False),
            fornberg_derivative_weights(sigma, sigmaf, nk=nk, top=True),
        )

    def getcape_controls(self) -> dict[str, Any]:
        """The parcel this run lifts for cape/cin/lcl/lfc.

        This engine is the one producer of those four names.  ``writeout.F:2921``
        fixes the mixed-layer parcel and ``getcape.F:60-76`` the increment,
        depth and adiabat; a ``[diagnostics]`` table may name another parcel or
        adiabat (the kernel transcribes every ``getcape.F`` branch), and absent
        one the values are CM1's own.
        """

        environment = getattr(self.config, "environment_field_selection", None)
        if environment is None:
            return {
                "parcel_source": CM1_GETCAPE_FIELD_SOURCE,
                "pressure_increment_pa": CM1_GETCAPE_PINC_PA,
                "mixed_layer_depth_m": CM1_GETCAPE_ML_DEPTH_M,
                "adiabat": CM1_GETCAPE_ADIABAT,
            }
        return {
            "parcel_source": int(environment.parcel_source),
            "pressure_increment_pa": float(environment.pressure_increment_pa),
            "mixed_layer_depth_m": float(environment.mixed_layer_depth_m),
            "adiabat": int(environment.adiabat),
        }

    def _getcape(self) -> dict[str, Any]:
        """``writeout.F:2894-2930``: one parcel ascent per column."""

        cp, g = self.cp, self.grid
        controls = self.getcape_controls()
        dynamics = self.runtime.dynamics
        if self.getcape_compiled is None:
            self.getcape_compiled = compile_cuda_cm1_getcape()
        from .cm1_vertical_grid import fornberg_extrapolation_weights

        zh = dynamics.metrics["zh"][:, 0, 0].get()
        zf = dynamics.metrics["zf"][:, 0, 0].get()
        cgs = fornberg_extrapolation_weights(
            [0.0] + [float(value) for value in zh],
            [0.0] + [float(value) for value in zf],
            nk=g.nz, top=False,
        )
        columns = g.ny * g.nx
        scratch = {
            name: self.registry.allocate(
                f"output.getcape.{name}", (g.nz + 1, g.ny, g.nx), zero=True
            )
            for name in GETCAPE_SCRATCH_NAMES
        }
        planes = {
            name: cp.empty(g.surface_shape, dtype=cp.float32)
            for name in ("cape", "cin", "lcl", "lfc")
        }
        with self.stream:
            self.getcape_compiled.by_name["cm1_getcape_columns_v1"](
                _blocks(columns), self._threads,
                (
                    dynamics.fields["prs"], dynamics.fields["th0"],
                    dynamics.fields["th3d"], dynamics.fields["pi0"],
                    dynamics.fields["pp3d"], dynamics.fields["q3d"][0],
                    *(scratch[name] for name in GETCAPE_SCRATCH_NAMES),
                    planes["cape"], planes["cin"], planes["lcl"], planes["lfc"],
                    cp.int32(g.nx), cp.int32(g.ny), cp.int32(g.nz),
                    cp.int32(controls["parcel_source"]),
                    cp.float32(controls["pressure_increment_pa"]),
                    cp.float32(controls["mixed_layer_depth_m"]),
                    cp.int32(controls["adiabat"]),
                    *(cp.float32(value) for value in cgs),
                ),
                stream=self.stream,
            )
        self.stream.synchronize()
        self.registry.release([f"output.getcape.{name}" for name in GETCAPE_SCRATCH_NAMES])
        selection = self.selection
        return {
            name: planes[name]
            for name, key in (("cape", "output_cape"), ("cin", "output_cin"),
                              ("lcl", "output_lcl"), ("lfc", "output_lfc"))
            if selection.on(key)
        }

    def _derived(self) -> dict[str, Any]:
        """The perturbation/absolute pairs and the column integrals.

        Every one is an exact CuPy expression over carriers this runtime
        already holds, enqueued on its stream.  Nothing crosses to the host.
        """

        cp, g = self.cp, self.grid
        dynamics = self.runtime.dynamics
        selection, f = self.selection, self.runtime.dynamics.fields
        out: dict[str, Any] = {}
        dz = cp.float32(g.dz_m)

        def column_sum(layer: Any) -> Any:
            """The sum over levels of ``layer*dz*rmh(k)``, writeout.F:2799,2836,2874.

            ``dz*rmh(k)`` is the layer's own depth.  On even levels ``rmh`` is
            one and the product is ``layer*dz``, the expression this always
            was; on stretched levels ``dz`` alone is the metric reference and
            is no layer's depth, so the column's own ``rmh`` multiplies it.
            """

            layer = layer * dz
            if int(g.stretch_z) != 0:
                import numpy

                rmh = numpy.asarray(g.levels.rmh[1 : g.nz + 1], dtype=numpy.float32)
                layer = layer * cp.asarray(rmh)[:, None, None]
            return layer.sum(axis=0)

        with self.stream:
            if selection.on("output_th"):
                out["th"] = cp.add(f["th0"], f["th3d"])
            if selection.on("output_pi"):
                out["pi"] = cp.add(f["pi0"], f["pp3d"])
            if selection.on("output_prspert"):
                base = cp.float32(CM1_P00) * cp.power(
                    f["pi0"], cp.float32(CM1_CP / CM1_RD)
                )
                out["prspert"] = cp.subtract(f["prs"], base.astype(cp.float32))
            if selection.on("output_rhopert"):
                out["rhopert"] = cp.subtract(f["rho"], f["rho0"])
            if selection.on("output_qvpert"):
                out["qvpert"] = cp.subtract(f["q3d"][0], f["qv0"])
            if selection.on("output_upert"):
                out["upert"] = cp.ascontiguousarray(
                    f["u3d"] - dynamics.base_profiles["u0"][:, None, None]
                )
            if selection.on("output_vpert"):
                out["vpert"] = cp.ascontiguousarray(
                    f["v3d"] - dynamics.base_profiles["v0"][:, None, None]
                )
            if selection.on("output_uinterp"):
                out["uinterp"] = cp.ascontiguousarray(
                    cp.float32(0.5) * (f["u3d"][:, :, :-1] + f["u3d"][:, :, 1:])
                )
            if selection.on("output_vinterp"):
                out["vinterp"] = cp.ascontiguousarray(
                    cp.float32(0.5) * (f["v3d"][:, :-1, :] + f["v3d"][:, 1:, :])
                )
            if selection.on("output_winterp"):
                out["winterp"] = cp.ascontiguousarray(
                    cp.float32(0.5) * (f["w3d"][:-1] + f["w3d"][1:])
                )
            if selection.on("output_pwat"):
                out["pwat"] = cp.ascontiguousarray(
                    column_sum(f["rho"] * f["q3d"][0]) * cp.float32(1.0e-3)
                ).astype(cp.float32)
            if selection.on("output_lwp"):
                out["lwp"] = cp.ascontiguousarray(
                    column_sum(f["rho"] * (f["q3d"][1] + f["q3d"][2]))
                ).astype(cp.float32)
                out["cwp"] = cp.ascontiguousarray(
                    column_sum(f["rho"] * f["q3d"][1])
                ).astype(cp.float32)
        return out

    # ------------------------------------------------------------- receipt

    def record(self) -> dict[str, Any]:
        g = self.grid
        return {
            "schema": CARTESIAN_OUTPUT_PRODUCTS_SCHEMA,
            "selection": self.selection.record(),
            "per_step_products": {
                key: PER_STEP_PRODUCTS[key]
                for key in PER_STEP_PRODUCTS if self.selection.on(key)
            },
            "completed_swath_step": self.completed_swath_step,
            "reflectivity_scheduled_in_microphysics": self.reflectivity_wanted,
            "diagnostic_kernels": diagnostic_receipt(),
            "getcape": (
                getcape_receipt() | {
                    "applied_controls": self.getcape_controls(),
                    "cm1_writeout_controls": {
                        "parcel_source": CM1_GETCAPE_FIELD_SOURCE,
                        "pressure_increment_pa": CM1_GETCAPE_PINC_PA,
                        "mixed_layer_depth_m": CM1_GETCAPE_ML_DEPTH_M,
                        "adiabat": CM1_GETCAPE_ADIABAT,
                    },
                }
                if any(self.selection.on(key) for key in
                       ("output_cape", "output_cin", "output_lcl", "output_lfc"))
                else None
            ),
            "getcape_transient_scratch_bytes": (
                getcape_scratch_bytes(g.nx, g.ny, g.nz)
                if any(self.selection.on(key) for key in
                       ("output_cape", "output_cin", "output_lcl", "output_lfc"))
                else 0
            ),
            "retained_plane_names": sorted(self.planes),
            "retained_volume_names": sorted(self.volumes),
            "unvalidated": [dict(row) for row in self.unvalidated_markers()],
        }

    def unvalidated_markers(self) -> tuple[dict[str, Any], ...]:
        """Every selected product this run cannot claim to have checked."""

        rows = [dict(row) for row in self.selection.unvalidated_markers()]
        moving = (self.config.u_move_m_s, self.config.v_move_m_s) != (0.0, 0.0)
        swaths = [key for key in PER_STEP_PRODUCTS if self.selection.on(key)]
        if moving and swaths:
            rows.append({
                "marker": "UNVALIDATED",
                "option": ", ".join(sorted(swaths)),
                "cm1_output_names": ["sws2", "svs2", "sps2", "srs2", "sgs2",
                                     "sus2", "shs2"],
                "not_checked": (
                    "this run translates at "
                    f"({self.config.u_move_m_s}, {self.config.v_move_m_s}) m/s, "
                    "and CM1 keeps a SECOND copy of every surface swath that it "
                    "advects back against the domain motion each step "
                    "(src/adv_routines.F:8884-9178 movesfc, called from "
                    "src/solve3.F under imove=1), so its '2' twins are the "
                    "damage track ON THE GROUND.  This runtime publishes only "
                    "CM1's n=1 grid-relative array: a run-long extremum in a "
                    "moving frame is a maximum over a patch of ground that "
                    "moved, and the published domainlocx/domainlocy pair maps "
                    "the FRAME onto the ground exactly but cannot unsmear the "
                    "accumulation.  movesfc's three-stage advection of the "
                    "surface field is not transcribed."
                ),
            })
        getcape_on = [
            key for key in ("output_cape", "output_cin", "output_lcl", "output_lfc")
            if self.selection.on(key)
        ]
        controls = self.getcape_controls()
        moved = {
            key: controls[key]
            for key, cm1 in (
                ("pressure_increment_pa", CM1_GETCAPE_PINC_PA),
                ("mixed_layer_depth_m", CM1_GETCAPE_ML_DEPTH_M),
                ("adiabat", CM1_GETCAPE_ADIABAT),
            )
            if controls[key] != cm1
        }
        if getcape_on and moved:
            rows.append({
                "marker": "UNVALIDATED",
                "option": ", ".join(getcape_on),
                "cm1_output_names": [key.removeprefix("output_") for key in getcape_on],
                "not_checked": (
                    f"this run lifts the parcel with {moved}; the getcape "
                    "differential (tools/oracle/cm1_getcape_oracle.py) compiles "
                    "CM1's own getcape.F, whose pinc, ml_depth and adiabat are "
                    "compile-time constants (getcape.F:60-76), so only CM1's "
                    "values and the three parcel sources have been differenced"
                ),
            })
        if self.config.bbc == 3 and (
            self.selection.on("output_vort") or self.selection.on("output_pv")
        ):
            rows.append({
                "marker": "UNVALIDATED",
                "option": "output_vort",
                "cm1_output_names": ["xvort", "yvort", "pv"],
                "not_checked": (
                    "with closures.bbc=3 CM1 overwrites xvort and yvort at the "
                    "lowest model level with the log-layer form "
                    "(src/misclibs.F:2160-2168) using ust/znt/u1/v1/s1; this "
                    "runtime does not expose those surface-layer arrays to the "
                    "diagnostic, so the lowest level carries CM1's literal "
                    "pre-overwrite value of zero"
                ),
            })
        return tuple(rows)


__all__ = [
    "CARTESIAN_OUTPUT_PRODUCTS_SCHEMA",
    "CM1_GETCAPE_FIELD_SOURCE",
    "CM1_P00",
    "CM1_CP",
    "CM1_RD",
    "MM_TO_CM",
    "CartesianOutputProducts",
]
