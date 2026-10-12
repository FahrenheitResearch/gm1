"""How fast a GM1 step runs on a GPU, from receipts.

Two sources, in order of preference:

1. this machine's own measurements, ``~/.cache/gpuwm-cm1/rates.json``
   (``$XDG_CACHE_HOME`` respected; ``$GPUWM_CM1_CACHE_DIR`` overrides both).
   Every finished run's receipt, and every "Measure speed", adds a point;
2. ``records/GM1-GPU-RATES.json``, where every rate names its receipt and its
   label (measured, documented, estimated, rough).

The model is the record's: ``s/step = max(floor_s + (tiles - 1) x tile_s,
cells x ns_per_cell_step)``, where ``tiles`` is the number of closure tiles a
step launches.  A measured point on this GPU replaces whichever of the two
terms it measured:
a point whose cells put it on the floor measures the floor, one well above the
crossover measures the per-cell rate.  A deck with the same number of cells as
a measured point is priced at that point's own step time, whatever the
regime: a run of that very grid on this card is the best estimate there is,
and the crossover band, where a point measures neither term cleanly, is
priced that way too.  When the grid has been run more than once, the fastest
of its recent runs prices it: other work on the card during a run can only
slow it, so a run made beside other work would otherwise make every later
estimate of that storm too long (measured: an 8 km hurricane grid at 0.038 s
a step alone and 0.067 to 0.099 s beside other work, whose median shortened
the next fit of the same storm).
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import re
import tempfile
import time
from dataclasses import dataclass, replace
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any, Mapping, Sequence

RATES_RECORD = "GM1-GPU-RATES.json"
CACHE_SCHEMA = "gpuwm-cm1.measured-rates/v1"
CACHE_ENV = "GPUWM_CM1_CACHE_DIR"
#: Points kept per (card, component); older ones are dropped.
CACHE_POINTS_PER_KEY = 20
#: A point measures the per-cell rate when its cells put the table's per-cell
#: term at least this far above the floor, and the floor when at least this
#: far below it.  Between the two it says little about either and is kept
#: but not used.
REGIME_MARGIN = 1.5
#: A measured point prices a deck directly when their cell counts agree to
#: this fraction (the same grid, or one of the same size).
SAME_GRID_REL = 0.005
#: How many of the most recent same-grid points are looked at; the fastest of them prices the grid.
SAME_GRID_POINTS = 5
#: A receipt with fewer measured steps than this is not learned: its mean
#: would carry the first steps' one-off costs (lazy module loads, the first
#: pool growth) at a weight the rate cannot absorb.
MIN_LEARN_STEPS = 20


def normalize_gpu_name(name: str) -> str:
    """``NVIDIA GeForce RTX 3080`` -> ``rtx 3080``; ``NVIDIA A100-SXM4-80GB`` -> ``a100 sxm4 80gb``."""

    text = name.lower().replace("-", " ").replace("_", " ")
    for word in ("nvidia", "geforce", "tesla"):
        text = re.sub(rf"\b{word}\b", " ", text)
    text = re.sub(r"\bgeneration\b|\bworkstation edition\b|\bserver edition\b", " ", text)
    return " ".join(text.split())


@lru_cache(maxsize=1)
def load_record() -> Mapping[str, Any]:
    """``records/GM1-GPU-RATES.json``, parsed once."""

    text = resources.files("cm1core.records").joinpath(RATES_RECORD).read_text(encoding="utf-8")
    return json.loads(text)


def cache_dir() -> Path:
    explicit = os.environ.get(CACHE_ENV)
    if explicit:
        return Path(explicit).expanduser()
    base = os.environ.get("XDG_CACHE_HOME")
    return (Path(base).expanduser() if base else Path.home() / ".cache") / "gpuwm-cm1"


def cache_path() -> Path:
    return cache_dir() / "rates.json"


@dataclass(frozen=True)
class StepModel:
    """One card and component's step time.

    ``s/step = max(host, device)``: the device term is ``cells x
    ns_per_cell_step``; the host term is ``floor_s`` for one closure tile plus
    ``tile_s`` for each further tile, since every tile launches the closure's
    operations again (a grid that is not a multiple of the tile gets ragged
    edge tiles, each costing that much however few cells it holds).
    """

    card: str
    component: str
    ns_per_cell_step: float
    floor_s: float
    label: str
    floor_label: str
    band: float
    basis: str
    floor_basis: str
    receipts: tuple[str, ...] = ()
    floor_receipts: tuple[str, ...] = ()
    measured_here: bool = False
    measured_points: tuple[Mapping[str, Any], ...] = ()
    #: Every point this machine measured on this card and component, oldest
    #: first (``MeasuredPoint.record()`` rows); a deck of the same size and
    #: closure tiling as one of them is priced at its step time
    #: (:meth:`same_grid`).
    grid_points: tuple[Mapping[str, Any], ...] = ()
    #: Host seconds each closure tile after the first adds to a step.
    tile_s: float = 0.0
    tile_label: str = "rough"
    tile_basis: str = ""
    tile_receipts: tuple[str, ...] = ()

    def same_grid(self, cells: int, tiles: int = 1) -> list[Mapping[str, Any]]:
        """The most recent measured points of this deck's size and tiling on this card."""

        same = [
            point for point in self.grid_points
            if abs(int(point["cells"]) - cells) <= SAME_GRID_REL * max(cells, 1)
            and point.get("tiles") in (None, tiles)
        ]
        return same[-SAME_GRID_POINTS:]

    def host_s(self, tiles: int = 1) -> float:
        return self.floor_s + max(0, int(tiles) - 1) * self.tile_s

    def device_s(self, cells: int) -> float:
        return cells * self.ns_per_cell_step * 1e-9

    def seconds_per_step(self, cells: int, tiles: int = 1) -> float:
        same = self.same_grid(cells, tiles)
        if same:
            # The fastest recent run of this grid: a run that shared the card is slower, never faster.
            return min(float(point["seconds_per_step"]) for point in same)
        return max(self.host_s(tiles), self.device_s(cells))

    def regime(self, cells: int, tiles: int = 1) -> str:
        if self.same_grid(cells, tiles):
            return "this-grid"
        if self.host_s(tiles) >= self.device_s(cells):
            return "floor" if tiles <= 1 else "tiles"
        return "per-cell"

    def label_for(self, cells: int, tiles: int = 1) -> str:
        regime = self.regime(cells, tiles)
        if regime == "this-grid":
            return "measured-here"
        if regime == "tiles":
            return self.tile_label
        return self.floor_label if regime == "floor" else self.label

    def basis_for(self, cells: int, tiles: int = 1) -> str:
        """What the step time at ``cells`` and ``tiles`` rests on, in one sentence."""

        regime = self.regime(cells, tiles)
        if regime == "this-grid":
            same = self.same_grid(cells, tiles)
            return (
                f"measured on this machine at this grid's size: fastest of {len(same)} run(s), "
                + ", ".join(
                    f"{float(point['seconds_per_step']):.4f} s/step at {int(point['cells']) / 1e6:.2f} M cells"
                    for point in same
                )
            )
        if regime == "tiles":
            return (
                f"{tiles} closure tiles: the one-tile floor {self.floor_s:.4f} s + {tiles - 1} x "
                f"{self.tile_s:.4f} s ({self.tile_basis})"
            )
        return self.basis if regime == "per-cell" else self.floor_basis

    def band_for(self, cells: int, tiles: int = 1) -> float:
        return band_for_label(self.label_for(cells, tiles))

    def record(self, cells: int | None = None, tiles: int = 1) -> dict[str, Any]:
        out: dict[str, Any] = {
            "card": self.card,
            "component": self.component,
            "ns_per_cell_step": self.ns_per_cell_step,
            "floor_s": self.floor_s,
            "tile_s": self.tile_s,
            "label": self.label,
            "floor_label": self.floor_label,
            "tile_label": self.tile_label,
            "basis": self.basis,
            "floor_basis": self.floor_basis,
            "tile_basis": self.tile_basis,
            "receipts": list(self.receipts),
            "floor_receipts": list(self.floor_receipts),
            "tile_receipts": list(self.tile_receipts),
            "measured_here": self.measured_here,
            "measured_points": [dict(point) for point in self.measured_points],
        }
        if cells is not None:
            same = self.same_grid(cells, tiles)
            out.update(
                cells=cells,
                closure_tiles=tiles,
                seconds_per_step=self.seconds_per_step(cells, tiles),
                regime=self.regime(cells, tiles),
                label_used=self.label_for(cells, tiles),
                basis_used=self.basis_for(cells, tiles),
                band=self.band_for(cells, tiles),
            )
            if same:
                out["same_grid_points"] = [dict(point) for point in same]
                out["receipts"] = [str(point.get("receipt", "")) for point in same]
        return out


def _card_key(name: str | None, record: Mapping[str, Any]) -> str | None:
    """The record's measured card for ``name``, matched exactly after normalizing."""

    if not name:
        return None
    normalized = normalize_gpu_name(name)
    return normalized if normalized in record["cards"] else None


def memory_bandwidth_gb_s(name: str | None) -> tuple[float | None, str | None]:
    """Vendor bandwidth for ``name`` from the record's table, and the key matched."""

    if not name:
        return None, None
    table = {k: v for k, v in load_record()["memory_bandwidth_gb_s"].items() if not k.startswith("_")}
    normalized = normalize_gpu_name(name)
    if normalized in table:
        return float(table[normalized]), normalized
    if "laptop" in normalized or "max q" in normalized or "mobile" in normalized:
        return None, None
    best = None
    for key in table:
        if normalized.startswith(key + " ") and (best is None or len(key) > len(best)):
            best = key
    return (float(table[best]), best) if best else (None, None)


def _component_row(card: Mapping[str, Any], component: str, record: Mapping[str, Any]) -> tuple[Mapping[str, Any] | None, str, str | None]:
    rates = card.get("rates", {})
    if component in rates:
        return rates[component], component, None
    fallback = record.get("component_fallback", {}).get(component)
    if fallback and fallback["use"] in rates:
        return rates[fallback["use"]], fallback["use"], fallback["why"]
    return None, component, None


def table_model(gpu_name: str | None, component: str) -> StepModel:
    """The record's model for one card and component, before local measurements."""

    record = load_record()
    reference_key = record["reference_card"]
    reference = record["cards"][reference_key]
    key = _card_key(gpu_name, record)
    if key is not None:
        row, used, why = _component_row(record["cards"][key], component, record)
        if row is not None:
            label = row["label"] if used == component else "rough"
            basis = row["basis"] if used == component else f"{used}'s rate on this card ({why})"
            return StepModel(
                card=key, component=component,
                ns_per_cell_step=float(row["ns_per_cell_step"]), floor_s=float(row["floor_s"]),
                label=label, floor_label=row.get("floor_label", label) if used == component else "rough",
                band=float(record["step_model"].get(f"band_{label}", 0.5)),
                basis=basis, floor_basis=row.get("floor_basis", ""),
                receipts=tuple(row.get("receipts", ())), floor_receipts=tuple(row.get("floor_receipts", ())),
                **_tile_terms(row, reference, component, record, own=True),
            )
    # An unmeasured card: the RTX 3080's measured rate scaled by bandwidth.
    row, used, why = _component_row(reference, component, record)
    if row is None:
        row, used, why = reference["rates"]["morrison-mp10"], "morrison-mp10", "no rate for this component"
    bandwidth, matched = memory_bandwidth_gb_s(gpu_name)
    reference_bandwidth = float(reference["memory_bandwidth_gb_s"])
    ns = float(row["ns_per_cell_step"])
    if bandwidth:
        ns = ns * reference_bandwidth / bandwidth
        basis = (
            f"no GM1 run on this card: the RTX 3080's measured {used} rate "
            f"({row['ns_per_cell_step']} ns) scaled by memory bandwidth, "
            f"{reference_bandwidth:.0f} / {bandwidth:.0f} GB/s ({matched})"
        )
    else:
        basis = (
            "no GM1 run on this card and its memory bandwidth is not in the record: "
            f"the RTX 3080's measured {used} rate ({row['ns_per_cell_step']} ns) as it is"
        )
    return StepModel(
        card=normalize_gpu_name(gpu_name) if gpu_name else "unknown",
        component=component, ns_per_cell_step=ns, floor_s=float(row["floor_s"]),
        label="rough", floor_label="rough",
        band=float(record["step_model"]["band_rough"]),
        basis=basis, floor_basis="the RTX 3080 reference card's floor: the floor is set by the host, not the card",
        receipts=tuple(row.get("receipts", ())), floor_receipts=tuple(row.get("floor_receipts", ())),
        **_tile_terms(row, reference, component, record, own=False),
    )


def _tile_terms(
    row: Mapping[str, Any], reference: Mapping[str, Any], component: str, record: Mapping[str, Any],
    *, own: bool,
) -> dict[str, Any]:
    """The per-tile host cost for a card's row: its own when recorded, else
    the reference card's (the RTX 3080 system's), labelled rough."""

    if own and row.get("tile_s") is not None:
        return {
            "tile_s": float(row["tile_s"]),
            "tile_label": str(row.get("tile_label", "rough")),
            "tile_basis": str(row.get("tile_basis", "")),
            "tile_receipts": tuple(row.get("tile_receipts", ())),
        }
    reference_row, _, _ = _component_row(reference, component, record)
    if reference_row is None:
        reference_row = reference["rates"]["morrison-mp10"]
    return {
        "tile_s": float(reference_row.get("tile_s", 0.0)),
        "tile_label": "rough",
        "tile_basis": (
            "no tiled run on this card: the reference RTX 3080 system's per-tile host cost "
            "(the cost is the host's launches, not the card's)"
        ),
        "tile_receipts": tuple(reference_row.get("tile_receipts", ())),
    }


# ------------------------------------------------------------ local points


@dataclass(frozen=True)
class MeasuredPoint:
    """One run's mean step time on one GPU, from its receipt."""

    gpu_name: str
    gpu_uuid: str | None
    component: str
    cells: int
    nx: int
    ny: int
    nz: int
    acoustic_substeps: int | None
    seconds_per_step: float
    steps: int
    receipt: str
    receipt_sha256: str
    source: str
    recorded_utc: str
    #: Closure tiles per step (``None``: a point recorded before tiles were).
    tiles: int | None = None

    def record(self) -> dict[str, Any]:
        return {
            "gpu_name": self.gpu_name, "gpu_uuid": self.gpu_uuid, "component": self.component,
            "cells": self.cells, "nx": self.nx, "ny": self.ny, "nz": self.nz,
            "acoustic_substeps": self.acoustic_substeps,
            "seconds_per_step": self.seconds_per_step, "steps": self.steps,
            "receipt": self.receipt, "receipt_sha256": self.receipt_sha256,
            "source": self.source, "recorded_utc": self.recorded_utc,
            "tiles": self.tiles,
        }

    @classmethod
    def from_record(cls, row: Mapping[str, Any]) -> "MeasuredPoint":
        return cls(
            gpu_name=str(row["gpu_name"]), gpu_uuid=row.get("gpu_uuid"),
            component=str(row["component"]), cells=int(row["cells"]),
            nx=int(row.get("nx", 0)), ny=int(row.get("ny", 0)), nz=int(row.get("nz", 0)),
            acoustic_substeps=row.get("acoustic_substeps"),
            seconds_per_step=float(row["seconds_per_step"]), steps=int(row.get("steps", 0)),
            receipt=str(row.get("receipt", "")), receipt_sha256=str(row.get("receipt_sha256", "")),
            source=str(row.get("source", "run")), recorded_utc=str(row.get("recorded_utc", "")),
            tiles=None if row.get("tiles") is None else int(row["tiles"]),
        )


class ReceiptError(ValueError):
    """A receipt that carries no usable step time."""


def _read_receipt(path: Path) -> tuple[Mapping[str, Any], bytes]:
    raw = path.read_bytes()
    if path.suffix == ".gz":
        raw = gzip.decompress(raw)
    return json.loads(raw.decode("utf-8")), raw


def point_from_receipt(
    receipt: str | os.PathLike[str] | Mapping[str, Any],
    *,
    source: str = "run",
    gpu_uuid: str | None = None,
) -> MeasuredPoint:
    """The mean step time a Cartesian receipt measured.

    Integration receipts (``--integrate``) carry ``step_wall_seconds.mean``;
    the timing harness carries ``mean_step_wall_seconds``.  Both sit under
    ``runtime_benchmark``.
    """

    if isinstance(receipt, Mapping):
        payload = receipt
        text = json.dumps(receipt, sort_keys=True).encode("utf-8")
        label = "in-memory receipt"
    else:
        path = Path(receipt)
        payload, text = _read_receipt(path)
        label = str(path.resolve())
    bench = payload.get("runtime_benchmark", payload)
    if not isinstance(bench, Mapping):
        raise ReceiptError("the receipt has no runtime_benchmark block")
    if payload.get("nest") or bench.get("nest"):
        # Every parent step of a nested run carries its nests' substeps, so
        # its mean prices the parent grid at the nests' cost.
        raise ReceiptError("the run carried a nest, so its step time measures no single-grid rate")
    steps_block = bench.get("step_wall_seconds")
    mean = None
    steps = 0
    if isinstance(steps_block, Mapping) and steps_block.get("mean"):
        mean = float(steps_block["mean"])
        steps = int(steps_block.get("count", 0))
    elif bench.get("mean_step_wall_seconds"):
        mean = float(bench["mean_step_wall_seconds"])
        steps = int(bench.get("measured_steps", 0) or 0)
    if not mean or not math.isfinite(mean) or mean <= 0:
        raise ReceiptError("the receipt records no mean step time")
    if 0 < steps < MIN_LEARN_STEPS:
        raise ReceiptError(
            f"the receipt measured {steps} steps; at least {MIN_LEARN_STEPS} are needed for a rate"
        )
    config = bench.get("config", {})
    grid = config.get("grid", {}) if isinstance(config, Mapping) else {}
    try:
        nx, ny, nz = int(grid["nx"]), int(grid["ny"]), int(grid["nz"])
    except (KeyError, TypeError, ValueError):
        raise ReceiptError("the receipt records no grid") from None
    name = bench.get("gpu_name")
    if not name:
        raise ReceiptError("the receipt does not name its GPU")
    component = str(config.get("microphysics") or "unknown")
    tiles = None
    shape = config.get("tile_shape_yx")
    if isinstance(shape, (list, tuple)) and len(shape) == 2 and all(int(v) > 0 for v in shape):
        tiles = closure_tile_count(nx, ny, (int(shape[1]), int(shape[0])))
    return MeasuredPoint(
        gpu_name=str(name), gpu_uuid=gpu_uuid, component=component,
        cells=nx * ny * nz, nx=nx, ny=ny, nz=nz,
        acoustic_substeps=config.get("acoustic_substeps"),
        seconds_per_step=mean, steps=steps, receipt=label,
        receipt_sha256=hashlib.sha256(text).hexdigest(), source=source,
        recorded_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        tiles=tiles,
    )


def closure_tile_count(nx: int, ny: int, tile_xy: tuple[int, int] | None) -> int:
    """Closure tiles per step: one for the whole domain, else the grid cut by the tile."""

    if tile_xy is None:
        return 1
    tx, ty = tile_xy
    return math.ceil(nx / min(tx, nx)) * math.ceil(ny / min(ty, ny))


def read_cache(path: Path | None = None) -> list[MeasuredPoint]:
    path = cache_path() if path is None else path
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(payload, Mapping) or payload.get("schema") != CACHE_SCHEMA:
        return []
    points = []
    for row in payload.get("points", ()):
        try:
            points.append(MeasuredPoint.from_record(row))
        except (KeyError, TypeError, ValueError):
            continue
    return points


def _write_cache(points: Sequence[MeasuredPoint], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema": CACHE_SCHEMA, "points": [point.record() for point in points]}
    handle = tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=".rates-", suffix=".tmp", delete=False,
    )
    try:
        with handle:
            json.dump(payload, handle, indent=1, sort_keys=True)
            handle.write("\n")
        os.replace(handle.name, path)
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise


def learn(point: MeasuredPoint, *, path: Path | None = None) -> bool:
    """Add one point to this machine's cache; False when it was already there."""

    path = cache_path() if path is None else path
    points = read_cache(path)
    if any(existing.receipt_sha256 == point.receipt_sha256 for existing in points):
        return False
    points.append(point)
    key = (normalize_gpu_name(point.gpu_name), point.component)
    same = [p for p in points if (normalize_gpu_name(p.gpu_name), p.component) == key]
    if len(same) > CACHE_POINTS_PER_KEY:
        drop = {id(p) for p in same[: len(same) - CACHE_POINTS_PER_KEY]}
        points = [p for p in points if id(p) not in drop]
    _write_cache(points, path)
    return True


def learn_receipt(
    receipt: str | os.PathLike[str] | Mapping[str, Any], *, source: str = "run",
    gpu_uuid: str | None = None, path: Path | None = None,
) -> tuple[MeasuredPoint, bool]:
    point = point_from_receipt(receipt, source=source, gpu_uuid=gpu_uuid)
    return point, learn(point, path=path)


def step_model(
    gpu_name: str | None, component: str, *, cache: Sequence[MeasuredPoint] | None = None,
) -> StepModel:
    """The table's model for this card, with this machine's measurements applied."""

    base = table_model(gpu_name, component)
    if not gpu_name:
        return base
    points = read_cache() if cache is None else list(cache)
    key = normalize_gpu_name(gpu_name)
    mine = [p for p in points if normalize_gpu_name(p.gpu_name) == key and p.component == component]
    if not mine:
        return base
    grid_points = tuple(p.record() for p in mine)
    floor_points, rate_points = [], []
    for point in mine:
        per_cell = base.device_s(point.cells)
        tiles = 1 if point.tiles is None else point.tiles
        # A point is a clean measure of the per-cell rate only when its
        # device work outweighs its host work, tiles included: a ragged tiled
        # run is launch-bound and would inflate the rate for every grid.
        if per_cell >= REGIME_MARGIN * base.host_s(tiles):
            rate_points.append(point)
        elif tiles <= 1 and per_cell * REGIME_MARGIN <= base.floor_s:
            floor_points.append(point)
    ns, floor = base.ns_per_cell_step, base.floor_s
    label, floor_label = base.label, base.floor_label
    basis, floor_basis = base.basis, base.floor_basis
    receipts, floor_receipts = base.receipts, base.floor_receipts
    used: list[MeasuredPoint] = []
    if rate_points:
        recent = rate_points[-5:]
        values = sorted(p.seconds_per_step / p.cells * 1e9 for p in recent)
        ns = values[len(values) // 2]
        label = "measured-here"
        basis = (
            f"measured on this machine's {gpu_name}: median of {len(recent)} run(s), "
            + ", ".join(f"{p.seconds_per_step:.4f} s/step at {p.cells / 1e6:.2f} M cells" for p in recent)
        )
        receipts = tuple(p.receipt for p in recent)
        used += recent
    if floor_points:
        recent = floor_points[-5:]
        values = sorted(p.seconds_per_step for p in recent)
        floor = values[len(values) // 2]
        floor_label = "measured-here"
        floor_basis = (
            f"measured on this machine's {gpu_name}: median of {len(recent)} small run(s), "
            + ", ".join(f"{p.seconds_per_step:.4f} s/step at {p.cells / 1e6:.2f} M cells" for p in recent)
        )
        floor_receipts = tuple(p.receipt for p in recent)
        used += recent
    if not used:
        # Only crossover-band points: neither term moves, but a deck of one
        # of their sizes is still priced at its own measured step.
        return replace(base, measured_here=True, grid_points=grid_points)
    bands = load_record()["step_model"]
    return StepModel(
        card=base.card, component=component, ns_per_cell_step=ns, floor_s=floor,
        label=label, floor_label=floor_label,
        band=float(bands["band_measured"]) if label == "measured-here" else base.band,
        basis=basis, floor_basis=floor_basis, receipts=receipts, floor_receipts=floor_receipts,
        measured_here=True, measured_points=tuple(p.record() for p in used),
        grid_points=grid_points,
        tile_s=base.tile_s, tile_label=base.tile_label, tile_basis=base.tile_basis,
        tile_receipts=base.tile_receipts,
    )


def band_for_label(label: str) -> float:
    bands = load_record()["step_model"]
    if label == "measured-here":
        return float(bands["band_measured"])
    return float(bands.get(f"band_{label}", bands["band_rough"]))


__all__ = [
    "CACHE_ENV",
    "CACHE_SCHEMA",
    "MIN_LEARN_STEPS",
    "MeasuredPoint",
    "RATES_RECORD",
    "ReceiptError",
    "StepModel",
    "band_for_label",
    "cache_dir",
    "closure_tile_count",
    "cache_path",
    "learn",
    "learn_receipt",
    "load_record",
    "memory_bandwidth_gb_s",
    "normalize_gpu_name",
    "point_from_receipt",
    "read_cache",
    "step_model",
    "table_model",
]
