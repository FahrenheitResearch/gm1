"""Build GM1 history frames on disk for the cm1out tests.

A frame is what ``gpuwm-cm1 cartesian --integrate`` publishes: the raw arena
``history-carriers.f32``, the eight-column ``history-field-table.tsv``, the
``history-frame-location.json`` clock record and the parts of
``history-publication.json`` the converter reads (the grid, the moisture
species, the time step and the swath canvas anchor).  :func:`frame_from_cm1`
fills one with the words of a CM1 history file, so converting it must give
CM1's own file back.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "tests" / "data" / "cm1-ref-24x24"

WSM6_SPECIES = {
    "names": ["qv", "qc", "qr", "qi", "qs", "qg"],
    "cm1_index_ranges": {"nqv": 1, "nql1": 2, "nql2": 3, "nqs1": 4, "nqs2": 6},
    "iice": 1, "idm": 0, "numq": 6, "component_id": "wsm6-mp6", "cm1_ptype": 7,
}

MORRISON_SPECIES = {
    "names": ["qv", "qc", "qr", "qi", "qs", "qg", "nci", "ncs", "ncr", "ncg"],
    "cm1_index_ranges": {"nqv": 1, "nql1": 2, "nql2": 3, "nqs1": 4, "nqs2": 6, "nnc1": 7, "nnc2": 10},
    "iice": 1, "idm": 1, "numq": 10, "component_id": "morrison-mp10", "cm1_ptype": 5,
}


def species_record(spec: Mapping[str, Any] = MORRISON_SPECIES) -> dict[str, Any]:
    names = list(spec["names"])
    return {
        "schema": "gm1.moisture-species-profile/v1",
        "names": names,
        "mass_names": [n for n in names if not n.startswith("nc")],
        "number_names": [n for n in names if n.startswith("nc")],
        "numq": spec["numq"], "iice": spec["iice"], "idm": spec["idm"],
        "cm1_index_ranges": dict(spec["cm1_index_ranges"]),
        "component_id": spec["component_id"], "cm1_ptype": spec["cm1_ptype"],
        "scalars": [
            {"name": n, "unit": "#/kg" if n.startswith("nc") else "kg/kg",
             "cloudvar": n in ("qc", "qi"), "kind": "number-concentration" if n.startswith("nc") else "mass-mixing-ratio"}
            for n in names
        ],
    }


def write_frame(directory: Path, carriers: Mapping[str, Any], *, time_s: float, step: int,
                grid: Mapping[str, Any], dt_s: float, umove: float = 0.0, vmove: float = 0.0,
                domainloc: tuple[float, float] = (0.0, 0.0), swath_anchor: Mapping[str, Any] | None = None,
                species: Mapping[str, Any] | None = None, staged_path_in_table: bool = False,
                microphysics: str = "morrison-mp10", ihail: int | None = 1,
                stagger: Mapping[str, str] | None = None, meanings: Mapping[str, str] | None = None,
                reflectivity: Mapping[str, Any] | None = None) -> Path:
    """Write one frame directory; ``carriers`` maps name -> float32 array (declared shape).

    ``reflectivity`` is the publication's ``reflectivity`` record, which the
    runtime writes when the scheme computes reflectivity in the model
    (``output_dbz``).
    """

    directory.mkdir(parents=True)
    arena = directory / "history-carriers.f32"
    rows = []
    offset = 0
    with arena.open("wb") as stream:
        for name, array in carriers.items():
            array = np.ascontiguousarray(array, dtype="<f4")
            stream.write(array.tobytes())
            shape = ",".join(str(n) for n in array.shape)
            bounds = ",".join(f"1,{n}" for n in array.shape)
            column7 = (os.fspath(directory.parent / ".incomplete" / directory.name / arena.name)
                       if staged_path_in_table else arena.name)
            where = (stagger or {}).get(name, "mass")
            meaning = (meanings or {}).get(name, f"{name} (test)")
            rows.append("\t".join((name, meaning, "1", where, shape, bounds, column7, str(offset))))
            offset += array.nbytes
    (directory / "history-field-table.tsv").write_text("\n".join(rows) + "\n", encoding="ascii")
    moving = {"umove": umove, "vmove": vmove, "domainlocx": domainloc[0], "domainlocy": domainloc[1], "imove": int(bool(umove or vmove))}
    (directory / "history-frame-location.json").write_text(json.dumps({
        "schema": "gm1.cartesian-history-frame-location/v1", "model_time_s": time_s, "step": step,
        "index": int(directory.name.split("-")[1]), "arena_sha256": f"test-{directory.name}-{time_s}",
        "moving_frame": moving,
    }), encoding="utf-8")
    publication = {
        "schema": "gm1.cartesian-history-publication/v1", "model_time_s": time_s, "step": step, "dt_s": dt_s,
        "config_sha256": "test",
        "config": {"grid": dict(grid), "dt_s": dt_s, "microphysics": microphysics,
                   "moisture_species": species_record(species or MORRISON_SPECIES),
                   "dense_ice": {"ihail": ihail},
                   "component_validation": {"component_id": microphysics, "state": "unvalidated"}},
        "moving_frame": moving,
    }
    if swath_anchor is not None:
        publication["tornado_diagnostics"] = {"swaths": {"moving_frame": dict(swath_anchor)}}
    if reflectivity is not None:
        publication["reflectivity"] = dict(reflectivity)
    (directory / "history-publication.json").write_text(json.dumps(publication), encoding="utf-8")
    return directory


def publish(root: Path, index: int, carriers: Mapping[str, Any], **kwargs: Any) -> Path:
    """Stage a frame under ``.incomplete/`` and rename it into place, as the runtime does."""

    name = f"frame-{index:06d}"
    staged = write_frame(root / ".incomplete" / name, carriers, **kwargs)
    os.replace(staged, root / name)
    return root / name


def end_event(events: Path, *, reason: str = "completed", frames: int = 0) -> None:
    with events.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"schema": "gm1.run-events/v1", "kind": "end", "reason": reason, "rc": 0,
                                 "frames_published": frames}) + "\n")


REF_GRID = {"nx": 24, "ny": 24, "nz": 40, "dx_m": 1000.0, "dy_m": 1000.0, "dz_m": 500.0, "ztop_m": 20000.0}


def canvas_anchor(*, dt_s: float, total_steps: int, umove: float, vmove: float) -> dict[str, Any]:
    return {"dt_s": dt_s, "total_steps": total_steps, "u_move_m_s": umove, "v_move_m_s": vmove}


def round_half_away(value: float) -> int:
    import math

    return int(math.floor(value + 0.5)) if value >= 0.0 else -int(math.floor(-value + 0.5))


def frame_from_cm1(path: Path, *, fill_canvas: float = -7777.0, total_steps: int = 20) -> tuple[dict[str, Any], dict[str, Any]]:
    """GM1 carriers holding the words of one time of a CM1 history file.

    The translated swaths go on a ground canvas one cell larger than the
    domain in each moved direction, at the window the anchor names for this
    frame, with ``fill_canvas`` everywhere else.
    """

    import netCDF4

    ds = netCDF4.Dataset(path)
    try:
        get = lambda name: np.asarray(ds[name][0], dtype=np.float32)  # noqa: E731
        nz, ny, nx = 40, 24, 24
        carriers: dict[str, Any] = {
            "closure.tkea": get("tke"),
            "prs": get("prs"),
            "q3d": np.stack([get(n) for n in ("qv", "qc", "qr", "qi", "qs", "qg")]),
            "nc3d": np.stack([get(n) for n in ("nci", "ncs", "ncr", "ncg")]),
            "u3d": get("u"), "v3d": get("v"), "w3d": get("w"),
            "zf": np.broadcast_to((np.arange(nz + 1, dtype=np.float32) * np.float32(500.0)).reshape(-1, 1, 1),
                                  (nz + 1, ny, nx)).copy(),
        }
        for name in ("rain", "prate", "th", "uinterp", "vinterp", "winterp", "uh", "xvort", "yvort", "zvort",
                     "sws", "svs", "sps", "sus", "shs"):
            carriers[name] = get(name)
        umove, vmove = float(ds["umove"][0]), float(ds["vmove"][0])
        time_s = float(ds["time"][0])
        dt_s = 6.0
        anchor = canvas_anchor(dt_s=dt_s, total_steps=total_steps, umove=umove, vmove=vmove)
        locx, locy = umove * time_s, vmove * time_s
        final_x = round_half_away(umove * dt_s * total_steps / 1000.0)
        final_y = round_half_away(vmove * dt_s * total_steps / 1000.0)
        ox = round_half_away(locx / 1000.0) - min(0, final_x)
        oy = round_half_away(locy / 1000.0) - min(0, final_y)
        shape = (ny + abs(final_y) + 1, nx + abs(final_x) + 1)
        for name in ("sws2", "svs2", "sps2", "sus2", "shs2"):
            canvas = np.full(shape, np.float32(fill_canvas), dtype=np.float32)
            canvas[oy:oy + ny, ox:ox + nx] = get(name)
            carriers[name] = canvas
        frame = dict(time_s=time_s, step=int(round(time_s / dt_s)), grid=REF_GRID, dt_s=dt_s, umove=umove,
                     vmove=vmove, domainloc=(locx, locy), swath_anchor=anchor)
        return carriers, frame
    finally:
        ds.close()


#: The field-table meanings the runtime writes for the species carriers
#: (tools/morrison_dbz.py reads the row order from them).
SPECIES_MEANINGS = {
    "q3d": "current qv,qc,qr,qi,qs,qg mixing ratios",
    "nc3d": "current nci,ncs,ncr,ncg number concentrations",
}


def morrison_state(nz: int = 12, ny: int = 5, nx: int = 7, seed: int = 3) -> dict[str, Any]:
    """A Morrison state whose columns take calc_refl10cm's melting branch (warm rain under snow and graupel)."""

    rng = np.random.default_rng(seed)
    f32 = np.float32
    zl = np.linspace(0, 1, nz)[:, None, None]
    th0 = (300.0 + 40.0 * np.linspace(0, 1, nz)).astype(f32)
    pi0 = (1.0 - 0.45 * np.linspace(0, 1, nz)).astype(f32)
    th3d = rng.normal(0, 2, (nz, ny, nx)).astype(f32)
    pp3d = rng.normal(0, 1e-3, (nz, ny, nx)).astype(f32)
    rho = np.broadcast_to(1.2 * np.exp(-1.1 * zl), (nz, ny, nx)).astype(f32)
    q = np.zeros((6, nz, ny, nx), f32)
    q[0] = 1e-2
    q[2] = (rng.uniform(0, 4e-3, (nz, ny, nx)) * (zl < 0.35)).astype(f32)
    q[4] = (rng.uniform(0, 2e-3, (nz, ny, nx)) * (zl > 0.25)).astype(f32)
    q[5] = (rng.uniform(0, 6e-3, (nz, ny, nx)) * (zl > 0.2)).astype(f32)
    nc = np.stack([rng.uniform(1e3, 1e6, (nz, ny, nx)), rng.uniform(1e3, 1e6, (nz, ny, nx)),
                   rng.uniform(1e2, 1e5, (nz, ny, nx)), rng.uniform(1e0, 1e4, (nz, ny, nx))]).astype(f32)
    return {"th0": th0, "pi0": pi0, "th3d": th3d, "pp3d": pp3d, "rho": rho, "q3d": q, "nc3d": nc}


def morrison_frame(root: Path, index: int, *, time_s: float, step: int, state: Mapping[str, Any] | None = None,
                   **kwargs: Any) -> Path:
    """Publish a Morrison frame of :func:`morrison_state` (base state as columns, as GM1 frames carry it)."""

    state = dict(state or morrison_state())
    nz, ny, nx = state["rho"].shape
    carriers = {name: state[name] for name in ("q3d", "nc3d", "rho", "th3d", "pp3d", "th0", "pi0")}
    carriers["u3d"] = np.zeros((nz, ny, nx + 1), np.float32)
    carriers["v3d"] = np.zeros((nz, ny + 1, nx), np.float32)
    carriers["w3d"] = np.zeros((nz + 1, ny, nx), np.float32)
    carriers["zf"] = np.broadcast_to((np.arange(nz + 1, dtype=np.float32) * np.float32(500.0)).reshape(-1, 1, 1),
                                     (nz + 1, ny, nx)).copy()
    grid = {"nx": nx, "ny": ny, "nz": nz, "dx_m": 1000.0, "dy_m": 1000.0, "dz_m": 500.0, "ztop_m": 500.0 * nz}
    return publish(root, index, carriers, time_s=time_s, step=step, grid=grid, dt_s=6.0,
                   meanings=dict(SPECIES_MEANINGS), stagger={"th0": "profile", "pi0": "profile"}, **kwargs)


def write_track(directory: Path, names: list[str], units: list[str], frames: list[Any], times: list[float],
                unfilled: tuple[str, ...] = ()) -> Path:
    """A parcel track as the runtime publishes it, through the runtime's own writer.

    ``frames`` are ``(npvals, nparcels)`` float32 arrays; the writer is
    ``lagrangian_stage.write_parcel_track``, given a stand-in for the stage.
    """

    from types import SimpleNamespace

    from cm1core.lagrangian_stage import write_parcel_track

    stage = SimpleNamespace(
        attribute_names=lambda: tuple(names), attribute_units=lambda: tuple(units),
        layout=SimpleNamespace(unfilled=tuple(unfilled)), record=lambda: {"stand_in": True},
        circulation_rows=[],
    )
    rows = [{"attributes": np.asarray(f, dtype=np.float32), "model_time_s": t} for f, t in zip(frames, times)]
    write_parcel_track(directory, stage, rows)
    return directory


__all__ = [
    "REF", "REF_GRID", "SPECIES_MEANINGS", "WSM6_SPECIES", "morrison_frame", "morrison_state", "write_track", "round_half_away", "canvas_anchor", "end_event", "frame_from_cm1", "publish", "species_record", "write_frame",
]
