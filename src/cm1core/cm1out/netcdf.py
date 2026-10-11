"""CM1 r21.1 netCDF history files (``output_format = 2``) from GM1 frames.

The layout is ``writeout_nc.F``'s ``netcdf_prelim`` (:24-470) as CM1 r21.1
builds it with ``-DNETCDF -DNCFPLUS``, which is netCDF-4 classic model:

* dimensions ``xh, yh, zh, xf, yf, zf, time (unlimited), one``, in that order;
* ``f_cor`` (icor = 1), ``ztop``, ``time``, the six coordinates in ``aunit``
  (km unless ``outunits = 2``) with ``long_name``/``units``/``axis``, and
  ``umove``/``vmove`` per time when the domain moves;
* the history variables in writeout.F's order with CM1's ``long_name`` and
  ``units``, ``(time, zh, yh, xh)`` and its staggered twins;
* CM1's 71 global attributes, in CM1's order, then ``gm1_*`` provenance.

File names follow ``output_filetype``: 1 is one ``cm1out.nc`` holding every
time, 2 is ``cm1out_NNNNNN.nc`` per time, and 3 (one file per MPI process)
is CM1's ``cm1out_000000_NNNNNN.nc``, GM1 being one process.  ``NNNNNN``
is CM1's ``nwrite``, which is 1 for the initial time.

The coordinates repeat CM1's REAL arithmetic, so they are CM1's words:
``xf = outunitconv*xfref`` and ``xh = outunitconv*0.5*(xfref(i)+xfref(i+1))``
evaluated left to right, with ``xfref`` from the origin rule of
``param.F:6646-6670``.

CM1 compresses each variable (deflate level 1 with shuffle).  GM1 writes
uncompressed by default, because deflating a 5-million-cell frame costs
several seconds of one core; ``deflate`` > 0 writes CM1's compression.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from .fields import Plan
from .settings import Cm1Settings, Geometry, Species

#: writeout_nc.F:365, constants.F:62.
MISSING_VALUE = -999999.9
#: CM1 writes its version string; a GM1 file keeps the r21.1 prefix a script may
#: test, and says it is GM1's.
CM1_VERSION = "cm1r21.1 (GM1 GPU port; see gm1_note)"


def filename(filetype: int, nwrite: int) -> str:
    """writeout_nc.F:103-121 (fnum = 51)."""

    if filetype == 1:
        return "cm1out.nc"
    if filetype == 2:
        return f"cm1out_{nwrite:06d}.nc"
    if filetype == 3:
        return f"cm1out_000000_{nwrite:06d}.nc"
    raise ValueError(f"output_filetype must be 1, 2 or 3, not {filetype}")


def _sgs_constants() -> tuple[float, float, float, float]:
    """constants.F:50-52 and :168-170, REAL."""

    import numpy as np

    f = np.float32
    c_m, c_l, ri_c, pi = f(0.10), f(0.8165), f(0.25), f(3.1415926535897932384626433)
    c_e1 = f(f(f(c_m * c_l) * c_l) * f(f(f(1.0) / ri_c) - f(1.0)))
    c_e2 = f(max(f(0.0), f(f(f(c_m * pi) * pi) - c_e1)))
    c_s = f(np.float64(f(f(f(c_m * c_m) * c_m) / f(c_e1 + c_e2))) ** 0.25)
    return float(c_m), float(c_e1), float(c_e2), float(c_s)


def global_attributes(settings: Cm1Settings, geometry: Geometry, species: Species,
                      fornberg: Mapping[str, Sequence[float]]) -> list[tuple[str, Any]]:
    """CM1's 71 global attributes (writeout_nc.F:363-434), in CM1's order and types."""

    import numpy as np

    i32, f32 = np.int32, np.float32
    s = settings
    aunit, _conv = settings.distance_units
    c_m, c_e1, c_e2, c_s = _sgs_constants()
    r = species.ranges
    rows: list[tuple[str, Any]] = [
        ("CM1 version", CM1_VERSION),
        ("Conventions", "CF-1.7"),
        ("missing_value", f32(MISSING_VALUE)),
        ("x_units", aunit), ("x_label", "x"),
        ("y_units", aunit), ("y_label", "y"),
        ("z_units", aunit), ("z_label", "z"),
        ("nx", i32(geometry.nx)), ("ny", i32(geometry.ny)), ("nz", i32(geometry.nz)),
    ]
    for key in ("imoist", "sgsmodel", "tconfig", "bcturbs", "ptype", "wbc", "ebc", "sbc", "nbc",
                "bbc", "tbc", "iorigin", "axisymm", "iptra", "npt"):
        rows.append((key, i32(s.int(key, 0))))
    rows.append(("fcor", f32(s.float("fcor", 0.0))))
    rows.append(("radopt", i32(s.int("radopt", 0))))
    rows.append(("dtrad", f32(s.float("dtrad", 0.0))))
    rows.append(("ctrlat", f32(s.float("ctrlat", 0.0))))
    rows.append(("ctrlon", f32(s.float("ctrlon", 0.0))))
    for key in ("year", "month", "day", "hour", "minute", "second", "sfcmodel", "oceanmodel", "ipbl"):
        rows.append((key, i32(s.int(key, 0))))
    rows += [
        ("iice", i32(species.iice)), ("idm", i32(species.idm)), ("idmplus", i32(species.idmplus)),
        ("numq", i32(species.numq)),
    ]
    for key in ("nql1", "nql2", "nqs1", "nqs2", "nnc1", "nnc2", "nzl1", "nzl2", "nvl1", "nvl2"):
        rows.append((key, i32(r.get(key, -2))))
    rows += [("c_m", f32(c_m)), ("c_e1", f32(c_e1)), ("c_e2", f32(c_e2)), ("c_s", f32(c_s))]
    for prefix in ("cgs", "dgs", "cgt", "dgt"):
        for index, value in enumerate(fornberg[prefix], 1):
            rows.append((f"{prefix}{index}", f32(value)))
    return rows


def coordinates(settings: Cm1Settings, geometry: Geometry) -> dict[str, Any]:
    """xh, xf, yh, yf, zh, zf and ztop in CM1's output units, as CM1's REAL words."""

    import numpy as np

    f32 = np.float32
    _aunit, conv = settings.distance_units
    conv = f32(conv)
    half_conv = f32(conv * f32(0.5))

    def faces(n: int, spacing: float) -> Any:
        # param.F:6646-6664 in DOUBLE, then stored as REAL (xfref); dx and dy
        # are REAL, so dble(dx) is the binary32 value widened.
        spacing = float(f32(spacing))
        index = np.arange(n + 1, dtype=np.float64)
        if settings.int("iorigin", 1) == 2:
            values = spacing * index - 0.5 * spacing * n
        else:
            values = spacing * index
        return values.astype(f32)

    xfref = faces(geometry.nx, geometry.dx)
    yfref = faces(geometry.ny, geometry.dy)
    return {
        "xh": (half_conv * (xfref[:-1] + xfref[1:])).astype(f32),
        "xf": (conv * xfref).astype(f32),
        "yh": (half_conv * (yfref[:-1] + yfref[1:])).astype(f32),
        "yf": (conv * yfref).astype(f32),
        "zh": (conv * geometry.zh).astype(f32),
        "zf": (conv * geometry.zf).astype(f32),
        "ztop": f32(conv * f32(geometry.maxz)),
    }


_DIMS = {"2": ("time", "yh", "xh"), "s": ("time", "zh", "yh", "xh"), "u": ("time", "zh", "yh", "xf"),
         "v": ("time", "zh", "yf", "xh"), "w": ("time", "zf", "yh", "xh")}


class HistoryFile:
    """One CM1 history file being written (a single time, or all times)."""

    def __init__(self, path: Path, settings: Cm1Settings, geometry: Geometry, species: Species,
                 plan: Plan, *, provenance: Mapping[str, Any], fornberg: Mapping[str, Sequence[float]],
                 deflate: int = 0, append: bool = False) -> None:
        import netCDF4

        self.path = Path(path)
        self.settings = settings
        self.geometry = geometry
        self.deflate = int(deflate)
        self.variables: dict[str, Any] = {}
        if append and self.path.exists():
            self.ds = netCDF4.Dataset(self.path, "a")
            self.ds.set_fill_off()
            self.variables = {name: self.ds.variables[name] for name in self.ds.variables}
            return
        self.ds = netCDF4.Dataset(self.path, "w", format="NETCDF4_CLASSIC")
        ds = self.ds
        ds.set_fill_off()
        nx, ny, nz = geometry.nx, geometry.ny, geometry.nz
        for name, size in (("xh", nx), ("yh", ny), ("zh", nz), ("xf", nx + 1), ("yf", ny + 1),
                           ("zf", nz + 1), ("time", None), ("one", 1)):
            ds.createDimension(name, size)
        aunit, _conv = settings.distance_units
        if settings.int("icor", 0) == 1:
            var = ds.createVariable("f_cor", "f4", ("one",))
            var.long_name, var.units = "Coriolis parameter", "1/s"
        var = ds.createVariable("ztop", "f4", ("one",))
        var.long_name, var.units = "height at top of model", aunit
        var = ds.createVariable("time", "f4", ("time",))
        var.long_name, var.units, var.axis = "time since beginning of simulation", "seconds", "T"
        for name, long_name, axis in (
            ("xh", "west-east location of scalar grid points", "X"),
            ("xf", "west-east location of staggered u grid points", "X"),
            ("yh", "south-north location of scalar grid points", "Y"),
            ("yf", "south-north location of staggered v grid points", "Y"),
            ("zh", "nominal height of scalar grid points", "Z"),
            ("zf", "nominal height of staggered w grid points", "Z"),
        ):
            var = ds.createVariable(name, "f4", (name,))
            var.long_name, var.units, var.axis = long_name, aunit, axis
        if settings.int("imove", 0) >= 1:
            for name in ("umove", "vmove"):
                var = ds.createVariable(name, "f4", ("time",))
                var.long_name, var.units = name, "m/s"
        for row in plan.variables:
            self._define(row.variable)
        for key, value in global_attributes(settings, geometry, species, fornberg):
            ds.setncattr(key, value)
        for key, value in provenance.items():
            ds.setncattr(key, value)
        import numpy as np

        coords = coordinates(settings, geometry)
        if settings.int("icor", 0) == 1:
            ds["f_cor"][:] = np.float32(settings.float("fcor", 0.0))
        ds["ztop"][:] = coords["ztop"]
        for name in ("xh", "xf", "yh", "yf", "zh", "zf"):
            ds[name][:] = coords[name]
        self.variables = {name: ds.variables[name] for name in ds.variables}

    def _define(self, variable: Any) -> Any:
        dims = _DIMS[variable.grid]
        g = self.geometry
        horizontal = {"2": (g.ny, g.nx), "s": (g.ny, g.nx), "u": (g.ny, g.nx + 1),
                      "v": (g.ny + 1, g.nx), "w": (g.ny, g.nx)}[variable.grid]
        chunks = (1, *horizontal) if variable.grid == "2" else (1, 1, *horizontal)
        var = self.ds.createVariable(
            variable.name, "f4", dims, chunksizes=chunks,
            zlib=self.deflate > 0, complevel=self.deflate or 1, shuffle=self.deflate > 0,
        )
        var.long_name = variable.long_name
        var.units = variable.units
        self.variables[variable.name] = var
        return var

    def write_time(self, index: int, frame: Any, plan: Plan, *, umove: float, vmove: float) -> list[str]:
        """Write one frame at time index ``index``; returns the variables written."""

        import numpy as np

        self.variables["time"][index] = np.float32(frame.model_time_s)
        if "umove" in self.variables:
            self.variables["umove"][index] = np.float32(umove)
            self.variables["vmove"][index] = np.float32(vmove)
        written = []
        for row in plan.variables:
            var = self.variables.get(row.variable.name)
            if var is None:
                var = self._define(row.variable)
                # A variable a later frame brings into one cm1out.nc (filetype 1):
                # the times already written hold CM1's missing value, not
                # whatever the unfilled file space held (the file is fill-off).
                for earlier in range(len(self.ds.dimensions["time"])):
                    if earlier != index:
                        var[earlier] = np.full(var.shape[1:], np.float32(MISSING_VALUE), dtype=np.float32)
            var[index] = row.read(frame)
            written.append(row.variable.name)
        # A variable this file already holds that this frame cannot supply
        # (filetype 1 only) is written as CM1's missing value at this time.
        planned = set(written)
        for name, var in self.variables.items():
            if len(var.dimensions) >= 3 and var.dimensions[0] == "time" and name not in planned:
                var[index] = np.full(var.shape[1:], np.float32(MISSING_VALUE), dtype=np.float32)
        return written

    def set_attribute(self, key: str, value: Any) -> None:
        self.ds.setncattr(key, value)

    def get_attribute(self, key: str, default: Any = None) -> Any:
        return self.ds.getncattr(key) if key in self.ds.ncattrs() else default

    def close(self) -> None:
        self.ds.close()


def provenance(*, producer: str, settings: Cm1Settings, frame: Any, plan: Plan,
               run_root: Path, frames: Sequence[Any]) -> dict[str, Any]:
    """The ``gm1_*`` attributes that say what this file is and what it is not."""

    config = frame.publication.get("config") or {}
    validation = config.get("component_validation") or {}
    component = config.get("microphysics") or validation.get("component_id") or "unknown"
    state = validation.get("state") or ("validated" if validation.get("validated") else "unknown")
    how = {row.variable.name: row.how for row in plan.variables if row.how not in ("copied", "renamed")}
    rows = {
        "gm1_producer": producer,
        "gm1_note": (
            "Simulated by GM1 (a GPU port of CM1 r21.1), not by CM1. Names, units, dimensions "
            "and layout follow CM1 r21.1 writeout_nc.F; the values are GM1's."
        ),
        "gm1_microphysics": f"{component} ({state} against CM1)",
        "gm1_settings": f"{settings.source} {settings.source_path or ''}".strip(),
        "gm1_frames": " ".join(_relative(f.directory, run_root) for f in frames),
        "gm1_config_sha256": str(frame.publication.get("config_sha256") or ""),
        "gm1_not_produced": " ".join(sorted(plan.not_produced)) or "none",
        "gm1_not_produced_reasons": json.dumps(dict(sorted(plan.not_produced.items())), sort_keys=True),
        "gm1_formed": json.dumps(how, sort_keys=True),
    }
    switches = dict(sorted(getattr(plan, "switches_not_produced", {}).items()))
    if switches:
        rows["gm1_switches_not_produced"] = " ".join(switches)
        rows["gm1_switches_not_produced_reasons"] = json.dumps(switches, sort_keys=True)
    if len(frames) == 1 and frames[0].arena_sha256:
        rows["gm1_arena_sha256"] = frames[0].arena_sha256
    if any(row.how == "sampled" for row in plan.variables):
        rows["gm1_translated_swaths"] = (
            "sws2..shs2 are GM1's ground-anchored swaths (each update placed at the domain's exact "
            "ground position), sampled under the domain at this time. CM1 advects its translated "
            "swaths with WENO instead (adv_routines.F movesfc), which spreads the peaks; the fields "
            "mean the same thing but are not formed the same way."
        )
    return rows


def _relative(path: Path, root: Path) -> str:
    try:
        return os.fspath(Path(path).resolve().relative_to(Path(root).resolve()))
    except ValueError:
        return os.fspath(path)


__all__ = [
    "CM1_VERSION",
    "HistoryFile",
    "MISSING_VALUE",
    "coordinates",
    "filename",
    "global_attributes",
    "provenance",
]
