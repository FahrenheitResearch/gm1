"""Atomic streamed closure artifact for the stretched-grid GM1 child.

The historical ``gm1-t300-child-closure-v2`` artifact predates the live
stretched-``z`` deformation correction.  This module deliberately uses a new
schema and never rewrites or relabels those bytes.  It provides only the
bounded publication/inspection boundary shared by the new 300/50 resident
oracle and the 3000/5 tile producer; CUDA source-order execution is supplied
by their runners.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import gc
from dataclasses import dataclass
from pathlib import Path
import shutil
from types import MappingProxyType
from typing import Any, Mapping
import uuid

from cm1core.cuda_gm1_tilestream import Gm1TilePlan, Gm1TileSpec, build_gm1_open_tile_plan


SCHEMA = "gm1.source-coordinate-child-stretched-closure/v3"
MANIFEST_SCHEMA = "gm1.source-coordinate-child-stretched-closure-manifest/v3"
RECEIPT_SCHEMA = "gm1.source-coordinate-child-stretched-closure-receipt/v3"
ABI = "gm1.source-coordinate-child-stretched-closure-f32le-c-order/v3"
STATUS = "PASS_SOURCE_BACKED_STRETCHED_CHILD_FRESH_CLOSURE"
CLAIM_LIMIT = (
    "fresh source-ordered CM1 r21.1 live calcnm, stretched terrain/four-open "
    "calcdef, and glibc-2.39-exact turbsmag/TKE initialization for one admitted "
    "GM1 child grid; no RK stage, completed timestep, checkpoint, forecast "
    "skill, storm, tornado, or synthetic meteorological-detail claim"
)
STREAM_CHUNK_BYTES = 1 << 20
DEFAULT_TILE = 192
SOURCE_HALO = 3
AUTHORITY_TARBALL_SHA256 = (
    "dc49fe84531056d1ae6249b37a5e3ee453fd96861c3b6bafd63828d92e64edf7"
)
VERTICAL_GRID_SHA256 = (
    "f66aa7937e0f4942dfbc55001641f221b8a84352961058e81247ee4e3ca8db56"
)
STATIC_METRIC_KERNEL_SHA256 = (
    "905bec56020e2bf1a73c0c854a9d5f462ccc372334ab3ceee2ad50f041be2f18"
)
CALCNM_KERNEL_SHA256 = (
    "40ddf9aa767b997f8e6d592ffcc8e7f0007f91e81bc122f0af192511b0295aac"
)
STRETCHED_CALCDEF_KERNEL_SHA256 = (
    "3114457a038061913bf760be0a3a6b24c1e56d753f249d77d224af59a7a987fd"
)
GENERIC_TURBSMAG_KERNEL_SHA256 = (
    "5264deec03cc61fe3e6e96874cd215b6e62e44fa7adc071b4808b9d2b26470ea"
)

CARRIER_ORDER = (
    "nm",
    "defv",
    "defh",
    "grdscl",
    "lenscl",
    "dissten",
    "kmh",
    "kmv",
    "khh",
    "khv",
    "tkea",
    "tke3d",
    "tketen",
)


@dataclass(frozen=True, slots=True)
class ClosureGrid:
    name: str
    nx: int
    ny: int
    nz: int
    dx_m: float
    dy_m: float
    reference_dz_m: float
    ztop_m: float

    def record(self) -> dict[str, object]:
        return {
            "name": self.name,
            "nx": self.nx,
            "ny": self.ny,
            "nz": self.nz,
            "dx_m": self.dx_m,
            "dy_m": self.dy_m,
            "reference_dz_m": self.reference_dz_m,
            "ztop_m": self.ztop_m,
        }


GRID_50M = ClosureGrid("accepted-300x300x75-50m", 300, 300, 75, 50.0, 50.0, 100.0, 15000.0)
GRID_5M = ClosureGrid("hard-3000x3000x75-5m", 3000, 3000, 75, 5.0, 5.0, 100.0, 15000.0)
GRID_PRESETS = MappingProxyType({"50m": GRID_50M, "5m": GRID_5M})


@dataclass(frozen=True, slots=True)
class ClosureCarrier:
    name: str
    shape: tuple[int, int, int]
    offset_nbytes: int
    nbytes: int

    def record(self) -> dict[str, object]:
        return {
            "name": self.name,
            "dtype": "little-endian-f32",
            "order": "C",
            "shape": list(self.shape),
            "offset_nbytes": self.offset_nbytes,
            "nbytes": self.nbytes,
        }


def require_grid(value: ClosureGrid) -> ClosureGrid:
    if value not in (GRID_50M, GRID_5M):
        raise ValueError("closure grid must be one of the two frozen presets")
    return value


def closure_layout(grid: ClosureGrid) -> tuple[ClosureCarrier, ...]:
    grid = require_grid(grid)
    result: list[ClosureCarrier] = []
    offset = 0
    for name in CARRIER_ORDER:
        levels = grid.nz if name == "grdscl" else grid.nz + 1
        shape = (levels, grid.ny, grid.nx)
        nbytes = math.prod(shape) * 4
        result.append(ClosureCarrier(name, shape, offset, nbytes))
        offset += nbytes
    return tuple(result)


def closure_payload_nbytes(grid: ClosureGrid) -> int:
    layout = closure_layout(grid)
    return layout[-1].offset_nbytes + layout[-1].nbytes


def closure_tile_plan(grid: ClosureGrid, *, tile: int = DEFAULT_TILE) -> Gm1TilePlan:
    grid = require_grid(grid)
    if isinstance(tile, bool) or not isinstance(tile, int) or tile < 1:
        raise ValueError("tile must be a positive integer")
    return build_gm1_open_tile_plan(
        nx=grid.nx,
        ny=grid.ny,
        tile_nx=tile,
        tile_ny=tile,
        halo=SOURCE_HALO,
    )


def capacity_record(
    grid: ClosureGrid,
    *,
    free_bytes: int,
    reserve_bytes: int,
    planned_payload_copies: int = 1,
) -> dict[str, int | str]:
    grid = require_grid(grid)
    for name, value in (("free_bytes", free_bytes), ("reserve_bytes", reserve_bytes)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if planned_payload_copies != 1:
        raise ValueError("closure publication permits exactly one payload copy")
    payload = closure_payload_nbytes(grid)
    if free_bytes < payload + reserve_bytes:
        raise OSError(
            f"closure publication needs {payload + reserve_bytes} bytes, only {free_bytes} available"
        )
    return {
        "status": "PASS_SINGLE_COPY_STREAMED_CAPACITY",
        "free_bytes": free_bytes,
        "reserve_bytes": reserve_bytes,
        "planned_payload_copies": 1,
        "payload_nbytes": payload,
        "post_publish_free_bytes": free_bytes - payload,
        "post_publish_above_reserve_bytes": free_bytes - payload - reserve_bytes,
    }


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _sha_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * STREAM_CHUNK_BYTES):
            digest.update(block)
    return digest.hexdigest()


def _sha_region(handle: Any, offset: int, nbytes: int, full: Any) -> str:
    handle.seek(offset)
    remaining = nbytes
    digest = hashlib.sha256()
    while remaining:
        block = handle.read(min(8 * STREAM_CHUNK_BYTES, remaining))
        if not block:
            raise ValueError("closure payload ended during carrier hashing")
        digest.update(block)
        full.update(block)
        remaining -= len(block)
    return digest.hexdigest()


def _exact_mapping(value: Mapping[str, Any], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{label} must be a string-keyed mapping")
    return {key: value[key] for key in sorted(value)}


class ClosureStreamWriter:
    """Write unique owned rectangles into one hidden carrier-major payload."""

    def __init__(
        self,
        destination: Path,
        *,
        grid: ClosureGrid,
        plan: Gm1TilePlan,
        input_pins: Mapping[str, Any],
        producer: Mapping[str, Any],
        free_bytes: int,
        reserve_bytes: int,
    ) -> None:
        import numpy as np

        self.grid = require_grid(grid)
        self.plan = plan
        if (plan.nx, plan.ny, plan.halo) != (grid.nx, grid.ny, SOURCE_HALO):
            raise ValueError("closure tile plan differs from the frozen grid/halo")
        self.destination = Path(destination).resolve()
        if self.destination.exists():
            raise FileExistsError(f"refusing to replace {self.destination}")
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        # Free space is decided on the smaller of the caller's declared figure
        # and this writer's own observation of the destination volume, and the
        # capacity refusal below is the only gate.
        #
        # This used to require the two to be equal, which is not a property a
        # live filesystem has: two independent samples taken microseconds apart
        # legitimately differ whenever anything else on the host is writing, so
        # a correct caller was refused at random -- and the refusal named no
        # wrong result, only a disagreement. The wrong result worth preventing
        # is sizing the 355 MB payload against room the destination volume does
        # not have, which leaves a truncated closure when the write stops
        # part-way; taking the minimum prevents exactly that, in both
        # directions. A caller that measured a roomier volume than the
        # destination loses to the observation; one that measured a tighter
        # volume, or sampled before something else freed space, is merely
        # conservative and is admitted.
        observed_free = int(shutil.disk_usage(self.destination.parent).free)
        if isinstance(free_bytes, bool) or not isinstance(free_bytes, int) or free_bytes < 0:
            raise ValueError("free_bytes must be a nonnegative integer")
        self.capacity = capacity_record(
            grid,
            free_bytes=min(free_bytes, observed_free),
            reserve_bytes=reserve_bytes,
        )
        self.capacity["declared_free_bytes"] = free_bytes
        self.capacity["observed_free_bytes"] = observed_free
        self.inputs = _exact_mapping(input_pins, "input_pins")
        self.producer = _exact_mapping(producer, "producer")
        self.layout = closure_layout(grid)
        self.staging = self.destination.parent / (
            f".{self.destination.name}.partial-{os.getpid()}-{uuid.uuid4().hex}"
        )
        self.staging.mkdir(parents=False, exist_ok=False)
        self.payload_path = self.staging / "closure.f32le"
        with self.payload_path.open("xb") as handle:
            handle.truncate(closure_payload_nbytes(grid))
            handle.flush()
            os.fsync(handle.fileno())
        self.maps: dict[str, Any] = {
            carrier.name: np.memmap(
                self.payload_path,
                dtype=np.dtype("<f4"),
                mode="r+",
                offset=carrier.offset_nbytes,
                shape=carrier.shape,
                order="C",
            )
            for carrier in self.layout
        }
        self.written = [False] * plan.ntiles
        self.stats = {
            name: {
                "finite_count": 0,
                "nonzero_count": 0,
                "positive_zero_count": 0,
                "minimum": math.inf,
                "maximum": -math.inf,
            }
            for name in CARRIER_ORDER
        }
        self.closed = False

    def write_tile(self, spec: Gm1TileSpec, arrays: Mapping[str, Any]) -> None:
        import numpy as np

        if self.closed:
            raise RuntimeError("closure writer is already closed")
        if spec.index < 0 or spec.index >= self.plan.ntiles or self.plan.specs[spec.index] != spec:
            raise ValueError("tile spec is not a member of this closure plan")
        if self.written[spec.index]:
            raise ValueError(f"closure tile {spec.index} was already written")
        if tuple(arrays) != CARRIER_ORDER:
            raise ValueError("closure tile carrier order changed")
        y0, y1, x0, x1 = spec.owned_rect("mass")
        height, width = y1 - y0, x1 - x0
        for carrier in self.layout:
            value = arrays[carrier.name]
            expected = (carrier.shape[0], height, width)
            if (
                not isinstance(value, np.ndarray)
                or value.dtype != np.dtype("float32")
                or value.shape != expected
                or not value.flags.c_contiguous
                or not bool(np.all(np.isfinite(value)))
            ):
                raise ValueError(
                    f"closure tile {carrier.name} must be finite C-f32 {expected}"
                )
            self.maps[carrier.name][:, y0:y1, x0:x1] = value
            bits = value.view(np.uint32)
            stats = self.stats[carrier.name]
            stats["finite_count"] += int(value.size)
            stats["nonzero_count"] += int(np.count_nonzero(value))
            stats["positive_zero_count"] += int(np.count_nonzero(bits == np.uint32(0)))
            stats["minimum"] = min(float(stats["minimum"]), float(np.min(value)))
            stats["maximum"] = max(float(stats["maximum"]), float(np.max(value)))
        self.written[spec.index] = True

    def finish(self) -> dict[str, str]:
        import numpy as np

        if self.closed:
            raise RuntimeError("closure writer is already closed")
        if not all(self.written):
            missing = [index for index, value in enumerate(self.written) if not value]
            raise ValueError(f"closure tile ownership is incomplete: {missing[:8]}")
        for value in self.maps.values():
            value.flush()
            mmap_handle = getattr(value, "_mmap", None)
            if mmap_handle is not None:
                mmap_handle.close()
        self.maps.clear()
        del value
        gc.collect()
        with self.payload_path.open("rb+") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        carriers: list[dict[str, Any]] = []
        full = hashlib.sha256()
        with self.payload_path.open("rb") as handle:
            for carrier in self.layout:
                sha256 = _sha_region(handle, carrier.offset_nbytes, carrier.nbytes, full)
                stats = self.stats[carrier.name]
                minimum = np.float32(stats["minimum"])
                maximum = np.float32(stats["maximum"])
                carriers.append(
                    carrier.record()
                    | {
                        "sha256": sha256,
                        "finite_count": int(stats["finite_count"]),
                        "nonzero_count": int(stats["nonzero_count"]),
                        "positive_zero_count": int(stats["positive_zero_count"]),
                        "minimum_f32_bits": f"{int(minimum.view(np.uint32)):08x}",
                        "maximum_f32_bits": f"{int(maximum.view(np.uint32)):08x}",
                    }
                )
        payload_sha256 = full.hexdigest()
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "artifact_schema": SCHEMA,
            "abi": ABI,
            "status": STATUS,
            "model": "GM1",
            "grid": self.grid.record() | {"vertical_grid_sha256": VERTICAL_GRID_SHA256},
            "capacity": self.capacity,
            "payload": {
                "file": "closure.f32le",
                "dtype": "little-endian-f32",
                "order": "C; carrier-major in manifest order",
                "carrier_count": len(carriers),
                "nbytes": closure_payload_nbytes(self.grid),
                "sha256": payload_sha256,
            },
            "carriers": carriers,
            "tile_plan": {
                "tile_nx": self.plan.tile_nx,
                "tile_ny": self.plan.tile_ny,
                "halo": self.plan.halo,
                "ntiles": self.plan.ntiles,
                "ownership_sha256": self.plan.ownership_sha256,
                "all_tiles_written_exactly_once": True,
            },
            "inputs": self.inputs,
            "producer": self.producer,
            "authority": {
                "tarball_sha256": AUTHORITY_TARBALL_SHA256,
                "operator_source_sha256": {
                    "static_metrics": STATIC_METRIC_KERNEL_SHA256,
                    "calcnm": CALCNM_KERNEL_SHA256,
                    "stretched_calcdef": STRETCHED_CALCDEF_KERNEL_SHA256,
                    "generic_turbsmag": GENERIC_TURBSMAG_KERNEL_SHA256,
                },
            },
            "historical_uniform_closure_disposition": (
                "immutable historical evidence only; not relabelled as stretched-z"
            ),
            "claim_limit": CLAIM_LIMIT,
        }
        manifest_raw = _canonical(manifest)
        manifest_sha256 = hashlib.sha256(manifest_raw).hexdigest()
        receipt = {
            "schema": RECEIPT_SCHEMA,
            "artifact_schema": SCHEMA,
            "abi": ABI,
            "status": STATUS,
            "manifest_sha256": manifest_sha256,
            "payload_sha256": payload_sha256,
            "payload_nbytes": closure_payload_nbytes(self.grid),
            "carrier_count": len(carriers),
            "grid_name": self.grid.name,
            "tile_ownership_sha256": self.plan.ownership_sha256,
            "producer_source_sha256": self.producer.get("source_sha256"),
        }
        receipt_raw = _canonical(receipt)
        receipt_sha256 = hashlib.sha256(receipt_raw).hexdigest()
        for name, raw in (("manifest.json", manifest_raw), ("receipt.json", receipt_raw)):
            path = self.staging / name
            with path.open("xb") as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
        names = sorted(path.name for path in self.staging.iterdir())
        if names != ["closure.f32le", "manifest.json", "receipt.json"]:
            raise RuntimeError("closure staging inventory drifted")
        os.rename(self.staging, self.destination)
        try:
            descriptor = os.open(self.destination.parent, os.O_RDONLY)
        except OSError:
            descriptor = None
        if descriptor is not None:
            try:
                try:
                    os.fsync(descriptor)
                except OSError:
                    # Windows does not expose a fsync-capable directory handle.
                    pass
            finally:
                os.close(descriptor)
        self.closed = True
        return {
            "manifest_sha256": manifest_sha256,
            "payload_sha256": payload_sha256,
            "receipt_sha256": receipt_sha256,
        }


def inspect_closure_artifact(
    directory: Path,
    *,
    grid: ClosureGrid,
    expected_manifest_sha256: str,
    expected_payload_sha256: str,
    expected_receipt_sha256: str,
) -> dict[str, str]:
    root = Path(directory).resolve()
    if not root.is_dir() or root.is_symlink():
        raise ValueError("closure artifact must be a regular directory")
    names = sorted(path.name for path in root.iterdir())
    if names != ["closure.f32le", "manifest.json", "receipt.json"]:
        raise ValueError("closure artifact inventory changed")
    manifest_path, payload_path, receipt_path = (
        root / "manifest.json",
        root / "closure.f32le",
        root / "receipt.json",
    )
    if _sha_path(manifest_path) != expected_manifest_sha256:
        raise ValueError("closure manifest SHA-256 changed")
    if _sha_path(receipt_path) != expected_receipt_sha256:
        raise ValueError("closure receipt SHA-256 changed")
    manifest = json.loads(manifest_path.read_bytes())
    receipt = json.loads(receipt_path.read_bytes())
    if (
        manifest.get("schema") != MANIFEST_SCHEMA
        or manifest.get("artifact_schema") != SCHEMA
        or manifest.get("abi") != ABI
        or manifest.get("status") != STATUS
        or manifest.get("grid")
        != (
            require_grid(grid).record()
            | {"vertical_grid_sha256": VERTICAL_GRID_SHA256}
        )
        or tuple(item.get("name") for item in manifest.get("carriers", [])) != CARRIER_ORDER
        or receipt.get("schema") != RECEIPT_SCHEMA
        or receipt.get("manifest_sha256") != expected_manifest_sha256
        or receipt.get("payload_sha256") != expected_payload_sha256
        or receipt.get("payload_nbytes") != closure_payload_nbytes(grid)
    ):
        raise ValueError("closure manifest/receipt identity changed")
    layout = closure_layout(grid)
    records = manifest["carriers"]
    full = hashlib.sha256()
    with payload_path.open("rb") as handle:
        for carrier, record in zip(layout, records, strict=True):
            if any(record.get(key) != value for key, value in carrier.record().items()):
                raise ValueError(f"closure carrier layout changed: {carrier.name}")
            if _sha_region(handle, carrier.offset_nbytes, carrier.nbytes, full) != record.get("sha256"):
                raise ValueError(f"closure carrier bytes changed: {carrier.name}")
        if handle.read(1):
            raise ValueError("closure payload has trailing bytes")
    if full.hexdigest() != expected_payload_sha256:
        raise ValueError("closure payload SHA-256 changed")
    return {
        "action": "inspected",
        "manifest_sha256": expected_manifest_sha256,
        "payload_sha256": expected_payload_sha256,
        "receipt_sha256": expected_receipt_sha256,
    }


__all__ = (
    "ABI",
    "CALCNM_KERNEL_SHA256",
    "CARRIER_ORDER",
    "CLAIM_LIMIT",
    "ClosureCarrier",
    "ClosureGrid",
    "ClosureStreamWriter",
    "DEFAULT_TILE",
    "GENERIC_TURBSMAG_KERNEL_SHA256",
    "GRID_5M",
    "GRID_50M",
    "GRID_PRESETS",
    "MANIFEST_SCHEMA",
    "RECEIPT_SCHEMA",
    "SCHEMA",
    "SOURCE_HALO",
    "STATIC_METRIC_KERNEL_SHA256",
    "STATUS",
    "STRETCHED_CALCDEF_KERNEL_SHA256",
    "capacity_record",
    "closure_layout",
    "closure_payload_nbytes",
    "closure_tile_plan",
    "inspect_closure_artifact",
)


# ---------------------------------------------------------------------------
# Consumption side: open a published v3 closure and upload it to one stream.
#
# The historical `gm1.real-50m-child-fresh-closure/v2` artifact is refused here
# by name and by fingerprint.  This project's own evidence
# (`evidence/validation/gm1-t300-child-stretched-closure-v3/receipt.json`)
# records that its `defv`/`defh` carriers are byte-identical to the sealed
# uniform-z negative control, so they are not the stretched-grid deformation at
# all, and that the fresh km/kh/TKE the child starts from were derived from
# them.  A child that consumes it begins with a subgrid closure diagnosed for a
# vertical grid it does not have - precisely the near-ground mixing that decides
# whether a tornado-scale vortex can be sustained.
# ---------------------------------------------------------------------------

REFUSED_UNIFORM_Z_MANIFEST_SHA256 = (
    "7aba693cd3a5c984c5a0d969bffa0793f80dfea138aa661da34a86f69982d9d9"
)
REFUSED_UNIFORM_Z_PAYLOAD_SHA256 = (
    "dc277233d7bead2009749e09f402fdff6c072323a57568fef270dbe6904dca6c"
)
REFUSED_UNIFORM_Z_RECEIPT_SHA256 = (
    "9972f6e981dc99e5635a844b70906328339535450bc57cdb01e61469b8080985"
)
REFUSED_UNIFORM_Z_DEFV_SHA256 = (
    "d7f044de8f9dea821f3f71616d64c5cdc770617ec92c189d8f46936d3dc05a78"
)
REFUSED_UNIFORM_Z_DEFH_SHA256 = (
    "0ba40b06156908886c5eb6560e9030c19bc36a37913c3f50f1b84735c948d24d"
)
UPLOAD_SCHEMA = "gm1.cuda-source-coordinate-child-stretched-closure/v3"
UPLOAD_STATUS = "ENQUEUED_CORRECTED_STRETCHED_CHILD_FRESH_CLOSURE"
CONSUMER_CLAIM_LIMIT = (
    "receipt-anchored mmap and same-stream CUDA upload of one published "
    "corrected stretched-z fresh-child closure; the mapping is immutable but "
    "uploaded carrier storage is writable for the admitted time integrator; no "
    "recomputation, RK stage, completed timestep, forecast, storm, tornado, or "
    "performance claim"
)


@dataclass(frozen=True, slots=True)
class LoadedChildStretchedClosure:
    directory: Path
    manifest_path: Path
    payload_path: Path
    receipt_path: Path
    manifest_sha256: str
    payload_sha256: str
    receipt_sha256: str
    grid: ClosureGrid
    arrays: Mapping[str, Any]
    manifest: Mapping[str, Any]
    receipt: Mapping[str, Any]

    def field(self, name: str) -> Any:
        try:
            return self.arrays[name]
        except KeyError as error:
            raise KeyError(f"unknown child closure carrier: {name}") from error


@dataclass(frozen=True, slots=True)
class CudaChildStretchedClosure:
    ready_event: Any
    source: LoadedChildStretchedClosure
    arrays: Mapping[str, Any]
    receipt: Mapping[str, Any]


def _read_canonical(path: Path, label: str) -> tuple[bytes, dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file")
    raw = path.read_bytes()
    value = json.loads(raw.decode("ascii"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    if _canonical(value) != raw:
        raise ValueError(
            f"{label} is not canonical sorted-key compact JSON; two byte "
            "spellings of one document would hash differently and no digest "
            "could bind it"
        )
    return raw, value


def refuse_uniform_z_closure(
    *,
    manifest_sha256: str,
    payload_sha256: str,
    receipt_sha256: str,
    carrier_sha256: Mapping[str, str],
) -> None:
    """Refuse the superseded uniform-z closure, by identity and by fingerprint."""

    if (
        manifest_sha256 == REFUSED_UNIFORM_Z_MANIFEST_SHA256
        or payload_sha256 == REFUSED_UNIFORM_Z_PAYLOAD_SHA256
        or receipt_sha256 == REFUSED_UNIFORM_Z_RECEIPT_SHA256
    ):
        raise ValueError(
            "the superseded uniform-z child closure is refused: its defv/defh "
            "carriers are byte-identical to the sealed uniform-z negative "
            "control, so the child's fresh km/kh and TKE were diagnosed for a "
            "vertical grid the child does not have; rebuild the closure from "
            "the corrected stretched-z oracle"
        )
    for name, refused in (
        ("defv", REFUSED_UNIFORM_Z_DEFV_SHA256),
        ("defh", REFUSED_UNIFORM_Z_DEFH_SHA256),
    ):
        if carrier_sha256.get(name) == refused:
            raise ValueError(
                f"closure carrier {name} is byte-identical to the sealed "
                "uniform-z negative control, so this closure did not use the "
                "stretched-grid deformation; the child's fresh km/kh and TKE "
                "would be diagnosed for a vertical grid it does not have"
            )


def load_gm1_child_stretched_closure(
    directory: str | os.PathLike[str],
    *,
    grid: ClosureGrid = GRID_50M,
) -> LoadedChildStretchedClosure:
    """Open any published v3 closure by the identity it records itself.

    Nothing is pinned to one historical run: the receipt must anchor the
    manifest that is present, the manifest must describe the payload that is
    present, and every carrier's recorded digest and statistics must be the
    bytes on disk.  A rerun of the corrected oracle publishes different digests
    and is admitted; the discredited uniform-z result is refused whatever digest
    it carries.
    """

    grid = require_grid(grid)
    requested = Path(directory)
    if requested.is_symlink():
        raise ValueError("child closure directory must not be a symlink")
    root = requested.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("child closure publication must be a real directory")
    inventory = sorted(entry.name for entry in os.scandir(root))
    if inventory != ["closure.f32le", "manifest.json", "receipt.json"]:
        raise ValueError(f"child closure directory inventory is {inventory}")
    manifest_path = root / "manifest.json"
    payload_path = root / "closure.f32le"
    receipt_path = root / "receipt.json"
    if payload_path.is_symlink() or not payload_path.is_file():
        raise ValueError("child closure payload must be a regular non-symlink file")

    manifest_raw, manifest = _read_canonical(manifest_path, "child closure manifest")
    receipt_raw, receipt = _read_canonical(receipt_path, "child closure receipt")
    manifest_sha256 = hashlib.sha256(manifest_raw).hexdigest()
    receipt_sha256 = hashlib.sha256(receipt_raw).hexdigest()
    if receipt.get("manifest_sha256") != manifest_sha256:
        raise ValueError(
            "child closure manifest bytes are not the manifest its own receipt "
            "anchors; the published carrier layout and the published receipt "
            "describe different artifacts"
        )
    if (
        manifest.get("schema") != MANIFEST_SCHEMA
        or manifest.get("artifact_schema") != SCHEMA
        or manifest.get("abi") != ABI
        or manifest.get("status") != STATUS
        or receipt.get("schema") != RECEIPT_SCHEMA
        or receipt.get("artifact_schema") != SCHEMA
        or receipt.get("abi") != ABI
        or receipt.get("status") != STATUS
    ):
        raise ValueError(
            "child closure schema, ABI or status is not the corrected "
            "stretched-z family this consumer reads; another family's arena "
            "would be uploaded with this one's carrier offsets"
        )
    expected_grid = require_grid(grid).record() | {
        "vertical_grid_sha256": VERTICAL_GRID_SHA256
    }
    if manifest.get("grid") != expected_grid:
        raise ValueError(
            "child closure covers a different grid than the child being built; "
            "every subgrid coefficient would be read at an offset belonging to "
            "a different column"
        )
    payload = manifest.get("payload")
    if not isinstance(payload, dict):
        raise ValueError("child closure manifest payload descriptor is missing")
    nbytes = closure_payload_nbytes(grid)
    payload_sha256 = _sha_path(payload_path)
    if (
        payload.get("file") != "closure.f32le"
        or payload.get("nbytes") != nbytes
        or payload.get("sha256") != payload_sha256
        or payload_path.stat().st_size != nbytes
        or receipt.get("payload_sha256") != payload_sha256
        or receipt.get("payload_nbytes") != nbytes
    ):
        raise ValueError(
            "child closure payload is not the arena its own manifest and "
            "receipt record; the child would upload bytes no published layout "
            "describes"
        )

    records = manifest.get("carriers")
    if not isinstance(records, list) or tuple(
        item.get("name") for item in records
    ) != CARRIER_ORDER:
        raise ValueError("child closure carrier order differs from the source order")
    carrier_sha256 = {item["name"]: item.get("sha256") for item in records}
    refuse_uniform_z_closure(
        manifest_sha256=manifest_sha256,
        payload_sha256=payload_sha256,
        receipt_sha256=receipt_sha256,
        carrier_sha256=carrier_sha256,
    )

    import numpy as np

    layout = closure_layout(grid)
    arrays: dict[str, Any] = {}
    full = hashlib.sha256()
    with payload_path.open("rb") as handle:
        for carrier, record in zip(layout, records, strict=True):
            if any(record.get(key) != value for key, value in carrier.record().items()):
                raise ValueError(f"child closure carrier layout changed: {carrier.name}")
            if (
                _sha_region(handle, carrier.offset_nbytes, carrier.nbytes, full)
                != record.get("sha256")
            ):
                raise ValueError(f"child closure carrier bytes changed: {carrier.name}")
        if handle.read(1):
            raise ValueError("child closure payload has trailing bytes")
    if full.hexdigest() != payload_sha256:
        raise ValueError("child closure payload aggregate SHA-256 changed")

    for carrier, record in zip(layout, records, strict=True):
        array = np.memmap(
            payload_path,
            dtype=np.dtype("<f4"),
            mode="r",
            offset=carrier.offset_nbytes,
            shape=carrier.shape,
            order="C",
        )
        bits = array.view(np.uint32)
        if (
            array.shape != carrier.shape
            or not array.flags.c_contiguous
            or array.flags.writeable
            or not bool(np.all(np.isfinite(array)))
            or int(np.count_nonzero(array)) != record.get("nonzero_count")
            or int(np.count_nonzero(bits == np.uint32(0)))
            != record.get("positive_zero_count")
            or f"{int(np.min(array).view(np.uint32)):08x}"
            != record.get("minimum_f32_bits")
            or f"{int(np.max(array).view(np.uint32)):08x}"
            != record.get("maximum_f32_bits")
        ):
            raise ValueError(
                f"child closure carrier {carrier.name} words or layout differ "
                "from the statistics its own manifest records"
            )
        arrays[carrier.name] = array
    if not np.array_equal(
        arrays["tkea"].view(np.uint32), arrays["tke3d"].view(np.uint32)
    ) or not bool(np.all(arrays["tketen"].view(np.uint32) == np.uint32(0))):
        raise ValueError(
            "child closure tke3d is not a raw-bit copy of tkea with tketen at "
            "positive zero; CM1 r21.1 diagnoses tkea from the fresh kmv and "
            "length scale and copies it into tke3d word for word "
            "(src/init3d.F:2723 and src/init3d.F:2750-2756, the "
            "interp_on_restart branch that rediagnoses subgrid TKE on a new "
            "grid - the same situation a fresh child is in), and allocates "
            "tketen at exactly zero (src/cm1.F:1739-1740); the first large "
            "step would otherwise advance a tendency that was never zeroed"
        )

    return LoadedChildStretchedClosure(
        directory=root,
        manifest_path=manifest_path,
        payload_path=payload_path,
        receipt_path=receipt_path,
        manifest_sha256=manifest_sha256,
        payload_sha256=payload_sha256,
        receipt_sha256=receipt_sha256,
        grid=grid,
        arrays=MappingProxyType(arrays),
        manifest=MappingProxyType(dict(manifest)),
        receipt=MappingProxyType(dict(receipt)),
    )


def upload_gm1_child_stretched_closure(
    closure: LoadedChildStretchedClosure, *, stream: Any | None = None
) -> CudaChildStretchedClosure:
    """Enqueue all thirteen admitted host carriers on one supplied stream."""

    if not isinstance(closure, LoadedChildStretchedClosure):
        raise TypeError("closure must be LoadedChildStretchedClosure")
    if tuple(closure.arrays) != CARRIER_ORDER:
        raise ValueError("child closure loaded carrier inventory changed")
    import cupy

    stream = cupy.cuda.get_current_stream() if stream is None else stream
    layout = closure_layout(closure.grid)
    device_arrays: dict[str, Any] = {}
    with stream:
        for carrier in layout:
            host = closure.arrays[carrier.name]
            value = cupy.asarray(host, dtype=cupy.float32, order="C")
            if (
                value.dtype != cupy.dtype(cupy.float32)
                or value.shape != carrier.shape
                or not value.flags.c_contiguous
            ):
                raise ValueError(
                    f"CUDA child closure carrier {carrier.name} layout changed"
                )
            device_arrays[carrier.name] = value
        ready = cupy.cuda.Event(disable_timing=True)
        ready.record(stream)
    receipt = MappingProxyType(
        {
            "schema": UPLOAD_SCHEMA,
            "status": UPLOAD_STATUS,
            "source": {
                "schema": SCHEMA,
                "abi": ABI,
                "manifest_sha256": closure.manifest_sha256,
                "payload_sha256": closure.payload_sha256,
                "receipt_sha256": closure.receipt_sha256,
                "payload_nbytes": closure_payload_nbytes(closure.grid),
                "grid_name": closure.grid.name,
            },
            "carrier_order": list(CARRIER_ORDER),
            "carrier_shapes": {
                carrier.name: list(carrier.shape) for carrier in layout
            },
            "stream_pointer": int(stream.ptr),
            "same_stream_no_host_sync": True,
            "mapping_immutable_device_storage_writable": True,
            "claim_limit": CONSUMER_CLAIM_LIMIT,
        }
    )
    return CudaChildStretchedClosure(
        ready_event=ready,
        source=closure,
        arrays=MappingProxyType(device_arrays),
        receipt=receipt,
    )
