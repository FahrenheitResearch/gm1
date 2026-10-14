"""One published history frame, as the quick-looks read it.

The bytes come through :mod:`cm1core.cm1out.frames` (the reader ``gpuwm-cm1 cm1out`` uses: published frames only,
never ``.incomplete/``, the arena always the one inside the frame directory) and the grid through
:func:`cm1core.cm1out.settings.geometry`, so a quick-look and a cm1out file of the same frame read the same words.

Conventions:

* arrays are CM1-indexed ``(k, j, i)``: ``(nz, ny, nx)`` at scalar points, ``(nz+1, ny, nx)`` on w faces,
  ``(nz, ny, nx+1)`` for u and ``(nz, ny+1, nx)`` for v;
* winds in a frame are relative to the moving domain; the ground-relative wind is ``u + umove, v + vmove``
  (CM1 init3d.F:32-33), and ``domainlocx/domainlocy`` place the domain on the ground;
* heights are above ground: every lane a CM1 user runs in v0.1 is flat, so ``zf`` above sea level is above ground.

A saved time of a CM1 netCDF history file opens the same way (:mod:`.ncread`): its rows carry an
:class:`.ncread.NcTime` in place of the frame directory, and its frame names the fields it lacks in plain words.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np

from ..cm1out import frames as _frames
from ..cm1out import settings as _settings


class FrameUnreadable(ValueError):
    """The frame lacks what a quick-look needs; the message says what."""


class FieldMissing(FrameUnreadable):
    """A CM1 netCDF file does not hold a field a look needs; the message names it and the CM1 switch that writes
    it.  The look is not drawn, and says why (a file without it is not a broken file)."""


@dataclass(eq=False)
class QuickFrame:
    frame: _frames.Frame
    geometry: _settings.Geometry
    key: str = ""               # the frame's path relative to the run (segments/NNN/frames/frame-NNNNNN)
    segment: int = 0

    @classmethod
    def open(cls, directory: Path | str, *, key: str = "", segment: int = 0) -> "QuickFrame":
        opener = getattr(directory, "open_frame", None)
        if opener is not None:
            # one saved time of a CM1 netCDF file (.ncread.NcTime): its grid is the file's own
            frame = opener()
            return cls(frame, frame.geometry, key or frame.key, segment)
        frame = _frames.open_frame(directory)
        try:
            geometry = _settings.geometry(frame)
        except _settings.SettingsError as error:
            frame.close()
            raise FrameUnreadable(str(error)) from error
        return cls(frame, geometry, key or Path(directory).name, segment)

    def close(self) -> None:
        self.frame.close()

    # -- identity ---------------------------------------------------------------
    @property
    def directory(self) -> Path:
        return self.frame.directory

    @property
    def t(self) -> float:
        return self.frame.model_time_s

    @property
    def publication(self) -> dict[str, Any]:
        return dict(self.frame.publication)

    @property
    def config(self) -> dict[str, Any]:
        return dict(self.frame.publication.get("config") or {})

    # -- grid -------------------------------------------------------------------
    @property
    def nx(self) -> int:
        return self.geometry.nx

    @property
    def ny(self) -> int:
        return self.geometry.ny

    @property
    def nz(self) -> int:
        return self.geometry.nz

    @property
    def dx(self) -> float:
        return float(self.geometry.dx)

    @property
    def dy(self) -> float:
        return float(self.geometry.dy)

    @property
    def cells(self) -> int:
        return self.nx * self.ny * self.nz

    @property
    def extent(self) -> tuple[float, float]:
        """Domain size (m) in x and y."""

        return self.nx * self.dx, self.ny * self.dy

    @cached_property
    def zf(self) -> np.ndarray:
        return np.asarray(self.geometry.zf, dtype=np.float64)

    @cached_property
    def zh(self) -> np.ndarray:
        return np.asarray(self.geometry.zh, dtype=np.float64)

    @property
    def stretched(self) -> bool:
        return not self.geometry.uniform

    def level_near(self, z_m: float, *, faces: bool = False) -> int:
        """The scalar level (or w face) nearest ``z_m`` AGL."""

        z = self.zf if faces else self.zh
        return int(np.argmin(np.abs(z - z_m)))

    # -- the moving domain ------------------------------------------------------
    @property
    def moving(self) -> dict[str, Any]:
        return dict(self.frame.moving_frame)

    @property
    def umove(self) -> float:
        return float(self.moving.get("umove", 0.0) or 0.0)

    @property
    def vmove(self) -> float:
        return float(self.moving.get("vmove", 0.0) or 0.0)

    @property
    def domainlocx(self) -> float:
        return float(self.moving.get("domainlocx", 0.0) or 0.0)

    @property
    def domainlocy(self) -> float:
        return float(self.moving.get("domainlocy", 0.0) or 0.0)

    @property
    def domain_moves(self) -> bool:
        return self.umove != 0.0 or self.vmove != 0.0

    # -- raw access -------------------------------------------------------------
    def has(self, name: str) -> bool:
        return self.frame.has(name)

    def view(self, name: str) -> np.ndarray:
        if not self.frame.has(name):
            missing = getattr(self.frame, "missing", None)
            if missing is not None:
                raise FieldMissing(missing(name))
            raise FrameUnreadable(f"{self.directory.name} has no {name} carrier")
        return self.frame.array(name)

    def get(self, name: str, index: Any = None) -> np.ndarray:
        v = self.view(name)
        return np.array(v if index is None else v[index], dtype=np.float32)

    def species_names(self, carrier: str = "q3d") -> tuple[str, ...]:
        """The rows of a species carrier: from the frame's moisture-species record (the cm1out writer's source),
        else from the carrier's meaning text ('current qv,qc,qr,... mixing ratios')."""

        rows = self.frame.carriers[carrier].shape[0] if carrier in self.frame.carriers else 0
        try:
            species = _settings.species(self.frame)
        except (_settings.SettingsError, KeyError, TypeError, ValueError):
            species = None
        if species is not None:
            mass = [n for n, u in zip(species.names, species.units) if u == "kg/kg"]
            number = [n for n, u in zip(species.names, species.units) if u != "kg/kg"]
            names = tuple(mass if carrier == "q3d" else number)
            if len(names) == rows:
                return names
        meaning = self.frame.carriers[carrier].meaning
        for word in meaning.split():
            if "," in word:
                return tuple(w.strip() for w in word.split(","))
        raise FrameUnreadable(f"{carrier}: no species list in the frame's record or in {meaning!r}")

    def has_species(self, name: str, carrier: str = "q3d") -> bool:
        return self.has(carrier) and name in self.species_names(carrier)

    def species(self, name: str, index: Any = None, carrier: str = "q3d") -> np.ndarray:
        missing = getattr(self.frame, "missing", None)
        if missing is not None and not (self.has(carrier) and name in self.species_names(carrier)):
            raise FieldMissing(missing(name))
        row = self.species_names(carrier).index(name)
        v = self.view(carrier)[row]
        return np.array(v if index is None else v[index], dtype=np.float32)

    # -- derived fields ---------------------------------------------------------
    def uv_level(self, k: int, *, ground: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """u and v at scalar points of level k, relative to the domain (or to the ground with ``ground``)."""

        u = self.get("u3d", k)
        v = self.get("v3d", k)
        uc = 0.5 * (u[:, 1:] + u[:, :-1])
        vc = 0.5 * (v[1:, :] + v[:-1, :])
        if ground:
            uc += np.float32(self.umove)
            vc += np.float32(self.vmove)
        return uc, vc

    def zeta_level(self, k: int) -> tuple[np.ndarray, str]:
        """Vertical vorticity (1/s) at scalar points of level k, and where it came from.

        CM1's own diagnostic when the frame carries it (``zvort``, or ``zvort_sfc`` at the lowest level);
        otherwise dv/dx - du/dy at the cell corners, averaged to the centres.
        """

        if k == 0 and self.has("zvort_sfc"):
            return self.get("zvort_sfc"), "CM1 zvort_sfc"
        if self.has("zvort"):
            return self.get("zvort", k), "CM1 zvort"
        u = self.get("u3d", k)
        v = self.get("v3d", k)
        dvdx = (v[:, 1:] - v[:, :-1]) / np.float32(self.dx)
        dudy = (u[1:, :] - u[:-1, :]) / np.float32(self.dy)
        zc = dvdx[1:-1, :] - dudy[:, 1:-1]
        z = np.pad(zc, 1, mode="edge")
        return (0.25 * (z[1:, 1:] + z[1:, :-1] + z[:-1, 1:] + z[:-1, :-1])).astype(np.float32), \
            "dv/dx \u2212 du/dy"

    def conv_level(self, k: int) -> np.ndarray:
        """Horizontal convergence -(du/dx + dv/dy) (1/s) at level k, exact on the C grid."""

        u = self.get("u3d", k)
        v = self.get("v3d", k)
        return (-((u[:, 1:] - u[:, :-1]) / np.float32(self.dx)
                  + (v[1:, :] - v[:-1, :]) / np.float32(self.dy))).astype(np.float32)

    def colmax(self, name: str, k0: int = 0, k1: int | None = None) -> np.ndarray:
        """Column maximum over levels k0..k1-1, streamed level by level."""

        v = self.view(name)
        k1 = v.shape[0] if k1 is None else k1
        best = np.array(v[k0], dtype=np.float32)
        for k in range(k0 + 1, k1):
            np.maximum(best, v[k], out=best)
        return best

    def colmin(self, name: str) -> np.ndarray:
        v = self.view(name)
        best = np.array(v[0], dtype=np.float32)
        for k in range(1, v.shape[0]):
            np.minimum(best, v[k], out=best)
        return best

    def uh(self) -> tuple[np.ndarray, str]:
        """2-5 km updraft helicity (m2/s2): CM1's uh when the frame has it, otherwise CM1's own sum.

        The sum is CM1 r21.1 misclibs.F:1513-1528 (calcuh, no terrain): over the scalar levels with
        2000 <= zh <= 5000 m AGL, ``(min(zf(k+1), 5000) - max(zf(k), 2000)) * wbar * zbar`` with
        ``wbar = max(0, w averaged to the level)`` and ``zbar = max(0, vertical vorticity averaged from the four
        cell corners)``: only cyclonically rotating updrafts count, as in CM1.
        """

        if self.has("uh"):
            return self.get("uh"), "CM1 uh"
        zz0, zzt = 2000.0, 5000.0
        zf = self.zf
        out = np.zeros((self.ny, self.nx), np.float32)
        w = self.view("w3d")
        for k, z in enumerate(self.zh):
            if zz0 <= z <= zzt:
                wbar = np.maximum(np.float32(0.0), np.float32(0.5) * (np.asarray(w[k], np.float32)
                                                                       + np.asarray(w[k + 1], np.float32)))
                zbar = np.maximum(np.float32(0.0), self.zeta_level(k)[0])
                depth = np.float32(min(float(zf[k + 1]), zzt) - max(float(zf[k]), zz0))
                out += depth * wbar * zbar
        return out, "CM1's calcuh sum"

    def theta_prime(self, k: int) -> np.ndarray:
        """θ′ at level k: CM1's tha (the frame's th3d)."""

        return self.get("th3d", k)

    def pressure_level(self, k: int) -> np.ndarray | None:
        """Pressure (Pa) at scalar level k: the frame's ``prs``, else CM1's ``p00*(pi0+ppi)**(cp/rd)``
        (solve2.F:1588) from the frame's pi0 column and ``pp3d``; None when the frame carries neither."""

        if self.has("prs"):
            return self.get("prs", k)
        pi0 = self.base_column("pi0")
        if pi0 is None or not self.has("pp3d"):
            return None
        from ..cm1out.fields import _CP, _P00, _RD

        total = float(pi0[k]) + self.get("pp3d", k).astype(np.float64)
        return (_P00 * np.power(total, _CP / _RD)).astype(np.float32)

    def base_column(self, name: str) -> np.ndarray | None:
        """A base-state column (th0, pi0, prs0, qv0, u0, v0) when the frame carries it."""

        for carrier in (name, f"base.{name}"):
            if self.has(carrier) and len(self.frame.carriers[carrier].shape) == 1:
                return np.asarray(self.view(carrier), dtype=np.float64)
        return None


def lineage(target: Path | str) -> tuple[_frames.RunLayout, list[tuple[float, str, Path, int]]]:
    """The run's frames, oldest first: (model time, key, directory, segment).

    A run directory's frames are its newest segment's lineage; a later segment's frame at a time already seen
    replaces the earlier one (as ``gpuwm-cm1 cm1out`` does)."""

    from .ncread import discover

    layout = discover(target)
    return layout, frames_of(layout)


def frame_key(path: Path | str, root: Path | str) -> str:
    """A saved frame's key: its folder under the run (``segments/NNN/frames/frame-XXXXXX``), the same whether the
    run's folder was named by a relative or an absolute path; the folder alone when it is not under the run."""

    import os

    try:
        return Path(path).relative_to(root).as_posix()
    except ValueError:
        pass
    try:
        return Path(os.path.abspath(path)).relative_to(os.path.abspath(root)).as_posix()
    except ValueError:
        return Path(path).name


def same_frame(stored: Any, key: str, stored_segment: Any = None, segment: int | None = None) -> bool:
    """Whether a drawn time's record names the saved frame ``key``: the same key, or the frame's folder alone, as
    a run named by a relative path recorded it before keys were made the same either way (of the same segment, when
    the record says which)."""

    if not isinstance(stored, str) or not stored:
        return False
    if stored == key:
        return True
    if "/" in stored or Path(key).name != stored:
        return False
    recorded = stored_segment if isinstance(stored_segment, int) and not isinstance(stored_segment, bool) else None
    return recorded is None or segment is None or recorded == segment


def frames_of(layout: _frames.RunLayout) -> list[tuple[float, str, Path, int]]:
    import json

    own = getattr(layout, "lineage", None)
    if own is not None:
        # CM1 netCDF files (.ncread.NcLayout): each saved time, oldest first
        return own()

    latest: dict[float, tuple[str, Path, int]] = {}
    for segment, path in layout.frames():
        try:
            time_s = float(json.loads((path / _frames.LOCATION).read_text(encoding="utf-8"))["model_time_s"])
        except (OSError, ValueError, KeyError):
            try:
                time_s = _frames.open_frame(path).model_time_s
            except (OSError, ValueError, KeyError):
                continue
        # (a run directory's frames come as absolute paths: the key must not depend on how its folder was named)
        latest[time_s] = (frame_key(path, layout.root), path, segment.number)
    return [(t, key, path, number) for t, (key, path, number) in sorted(latest.items())]


__all__ = ["FieldMissing", "FrameUnreadable", "QuickFrame", "frame_key", "frames_of", "lineage", "same_frame"]
