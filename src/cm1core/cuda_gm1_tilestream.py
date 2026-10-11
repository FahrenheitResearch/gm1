"""Hybrid pinned-host tilestream seam for the real 50 m GM1 child.

This module admits one deliberately narrow local stage: the stretched-grid
fifth-order scalar-advection tendency.  The authoritative full-domain state
lives in exact-size CUDA-pinned host arrays.  Two persistent device buffers
move fixed-shape windows through copy-in, compute, and copy-out streams with
explicit CUDA-event edges.  Only the source operator's interior is scattered;
the three-cell radiative-open/LBC band remains owned by the existing boundary
stage.

CM1's psolver=3 stage is not tile-local.  Its full-domain scratch is inventoried
separately and any attempt to substitute a tiled pressure solve fails closed.
Likewise, in-place local stages are refused until the exact ring masks and
their save/patch event schedule exposed here are connected to a ring arena.

The geometry and event rules are an independent GM1 adaptation of immutable
ArWen tilestream engineering.  The exact upstream commit and file hashes are
pinned below; this package never imports from or writes the sibling checkout.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any


GM1_TILESTREAM_SCHEMA = "gm1.hybrid-tilestream/v1"
GM1_TILESTREAM_ABI = "gm1.hybrid-tilestream-scalar-advection5-f32/v1"
GM1_TILESTREAM_STORE_ABI = "gm1.hybrid-pinned-host-carrier-store-f32/v1"
GM1_TILESTREAM_PRESSURE_ABI = "gm1.domain-global-psolver3-workspace-f32/v1"
GM1_TILESTREAM_SOURCE_BINDING_ABI = "gm1.hybrid-tilestream-local-source-binding-f32/v1"

# DESIGN CITATION, NOT AN ADMISSION SURFACE.  GM1 does not import ArWen's
# tilestream package -- it never appears on an import path, and the receipt
# below says so with `immutable_external_checkout`.  These hashes record the
# exact bytes the GM1 host-store design was transcribed FROM, so the pins stay
# at the commit that was read and do NOT move with the physics estate: a
# citation that silently re-pointed at newer source would stop being evidence
# of what was transcribed.
#
# The estate pin moved to ArWen 2.7.4 on 2026-09-13 (CM1-ARWEN-SEAM.md).  Six
# of the seven cited files are byte-identical in 2.7.4; `tilestream/driver.py`
# is not.  That is recorded here rather than acted on, because re-reading the
# newer driver is a transcription decision with its own audit, not a side
# effect of a dependency bump.
ARWEN_TILESTREAM_COMMIT = "f47f177d4b2e0da95568d70230f5ce9785e632c7"
ARWEN_TILESTREAM_PATH_COMMIT = "dbf50d177b9d1dd0839e59413d11665906caca23"
#: Cited files whose bytes differ in the currently pinned estate.  Empty would
#: mean the citation and the pinned estate agree; this one does not, on
#: purpose.  See the note above before changing it.
ARWEN_TILESTREAM_DIVERGED_IN_PINNED_ESTATE = ("tilestream/driver.py",)
ARWEN_TILESTREAM_SOURCE_SHA256 = MappingProxyType(
    {
        "tilestream/hoststore.py": "d531b3cedebe4f358efe18b718711155b973d4e5ebaec827ff0c5dce7a3275cb",
        "tilestream/spec.py": "635c71da5ec98e118043a69e5ec33c5f0395587e4e27a93e258f5c9b29a57fdd",
        "tilestream/rings.py": "d190103c82efb8c05330404cb752fc27e71c18b86e2f127ca687ec5952a2b81f",
        "tilestream/driver.py": "d16c1e837a51e3cf9f8021d55511d180ebd0478bad17dc58ad6361c6bac3c088",
        "tilestream/gather.py": "36bcb74696a75bff60a392a1a19084156fc2a97868f31fede63b0039be0721df",
        "tilestream/overlap.py": "940307d189fff2773311acb45f37a0c4107423cee1df399ff139a482b975109c",
        "tilestream/shared_workspace.py": "60f78a6912c3013f113f15cb136ba0f75445a0176c9c98a8e22e4ef5cac6a86e",
    }
)

GM1_SCALAR_MODULE_SHA256 = (
    "b495571b51ca53cdf0acefdaea2b2f65041928dd2901dcc301d1f7e5eb610a6c"
)
GM1_SCALAR_KERNEL_SHA256 = (
    "d617958150e19db023d79f98878e2c0e3d20ee268f0ac3b6ee9865fe0c6fa900"
)
GM1_STRETCHED_MOIST_COMMIT = "5b3da165e97981dff7d1fd4af3bb8e22f647a67b"
GM1_STRETCHED_MOIST_MODULE_SHA256 = (
    "79588bba9ec40899472e36841192dcdf47c7daa13b140fbf4e4b8d96714ad922"
)
GM1_STRETCHED_MOIST_KERNEL_SHA256 = (
    "9e899ae56709ec406b836bba0bcac8941dcab3b69ef690d555a4f65ba2d3bbdf"
)
GM1_PSOLVER3_MODULE_SHA256 = (
    "6173f5defbda3e19c52033a3e19d5501838aeea6bc1617331f1e7ddd7eaaa94e"
)
GM1_PSOLVER3_CORE_SHA256 = (
    "99cd3ce11b67fd716bfc0a9cda7aecdc8050866bff32a742628b365e6cb71489"
)
GM1_PSOLVER3_BOUNDARY_SHA256 = (
    "a24700aed3a9040d12e102b8ec58685c47e2f049fdca9007f9ba0bb40d05209e"
)
GM1_RESUME_COMMIT = "509567c060e982013082752abf356ee85c5e7abb"
GM1_RESUME_MODULE_SHA256 = (
    "0f59ae76927b4886f6314d65b6078c0944719b68efb7820535ebb2772d075d7f"
)
GM1_RESUME_OPERATOR_MANIFEST_SHA256 = (
    "6d2523a36ba320568939a820ece9ac7b78a8a7e33db0a18eade66a7b31599d13"
)
GM1_POST_STEP_STREAM_RUNNER_COMMIT = "7a78d84ff8ac7be2e8780a1ea0f1388acac7385e"
GM1_POST_STEP_STREAM_RUNNER_SHA256 = (
    "f3afa9a67f94ecb4bf7dbc67025c7b952b1f587c0d969eabf997783063ad15e7"
)
GM1_POST_STEP_TRANSFER_CHUNK_WORDS = 2 * 1024 * 1024

CM1_OPEN_NGXY = 3
GM1_SOURCE_STENCIL_RADIUS = 3
GM1_TRANSFER_RADIUS = 2 * GM1_SOURCE_STENCIL_RADIUS
GM1_PRODUCTION_SHAPE = (75, 300, 300)
GM1_PRODUCTION_DX_DY = (50.0, 50.0)
GM1_DEFAULT_TILE = 96
CUDA_MEMORY_TYPE_HOST = 1
GIB = 1 << 30

VARIANT_STAGGER = MappingProxyType(
    {"mass": (0, 0), "u": (0, 1), "v": (1, 0), "w": (0, 0)}
)
VARIANTS = tuple(VARIANT_STAGGER)

STAGE_STORE_SHAPES = MappingProxyType(
    {
        "scalar": "padded-mass",
        "rru": "padded-u",
        "rrv": "padded-v",
        "rrw": "padded-w",
        "divx": "mass",
        "c1": "zface",
        "c2": "zface",
        "rr0": "mass",
        "gz": "surface",
        "tendency": "mass",
    }
)

TILE_LOCAL_IMMUTABLE_NAMES = (
    "c1",
    "c2",
    "rr0",
    "rho0",
    "rf0",
    "gz",
    "rgzu",
    "rgzv",
    "dzdx",
    "dzdy",
    "th0",
)

CHECKPOINT_NAMES = (
    "closure.tkea",
    "component.effc",
    "component.effi",
    "component.effs",
    "component.h_diabatic",
    "component.mp_graupelnc",
    "component.mp_graupelncv",
    "component.mp_rainnc",
    "component.mp_rainncv",
    "component.mp_snownc",
    "component.mp_snowncv",
    "component.mp_sr",
    "held.qcten",
    "held.qpten",
    "held.qtten",
    "held.qvten",
    "pp3d",
    "ppi",
    "ppx",
    "prs",
    "q3d",
    "qa",
    "radbce",
    "radbcn",
    "radbcs",
    "radbcw",
    "rf",
    "rho",
    "rr",
    "th3d",
    "tha",
    "u3d",
    "ua",
    "v3d",
    "va",
    "w3d",
    "wa",
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("ascii")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _variant_shape(ny: int, nx: int, variant: str) -> tuple[int, int]:
    try:
        ey, ex = VARIANT_STAGGER[variant]
    except KeyError:
        raise ValueError(f"unknown horizontal variant {variant!r}") from None
    return ny + ey, nx + ex


@dataclass(frozen=True, slots=True)
class Gm1TileSpec:
    """One fixed-shape nonperiodic compute window and its owned mass cells."""

    index: int
    ty: int
    tx: int
    nx: int
    ny: int
    tile_nx: int
    tile_ny: int
    halo: int
    i0: int
    i1: int
    j0: int
    j1: int
    ci0: int
    cj0: int
    cnx: int
    cny: int

    def gather_rect(
        self, variant: str, *, include_kernel_padding: bool = False
    ) -> tuple[int, int, int, int]:
        """Return physical-domain ``(y0,y1,x0,x1)`` read coordinates.

        The rectangle is clipped to physical cells because open-domain ghost
        cells are immutable external-boundary inputs and never participate in
        ring ownership.  ``include_kernel_padding`` adds the operator ABI's
        second three-cell surround; this is the actual host-transfer footprint.
        """

        ey, ex = VARIANT_STAGGER[variant]
        extra = CM1_OPEN_NGXY if include_kernel_padding else 0
        y0 = max(0, self.cj0 - extra)
        x0 = max(0, self.ci0 - extra)
        y1 = min(self.ny + ey, self.cj0 + self.cny + ey + extra)
        x1 = min(self.nx + ex, self.ci0 + self.cnx + ex + extra)
        return y0, y1, x0, x1

    def owned_rect(self, variant: str) -> tuple[int, int, int, int]:
        """Return the unique left/south-owned scatter rectangle."""

        ey, ex = VARIANT_STAGGER[variant]
        y1 = self.j1 + (ey if self.j1 == self.ny else 0)
        x1 = self.i1 + (ex if self.i1 == self.nx else 0)
        return self.j0, y1, self.i0, x1

    def scalar_scatter(self) -> tuple[tuple[slice, slice], tuple[slice, slice]] | None:
        """Full and tile slices for the source operator's admitted interior."""

        x0, x1 = max(self.i0, 3), min(self.i1, self.nx - 3)
        y0, y1 = max(self.j0, 3), min(self.j1, self.ny - 3)
        if x0 >= x1 or y0 >= y1:
            return None
        return (
            (slice(y0, y1), slice(x0, x1)),
            (
                slice(y0 - self.cj0, y1 - self.cj0),
                slice(x0 - self.ci0, x1 - self.ci0),
            ),
        )

    def record(self) -> dict[str, object]:
        return {
            "index": self.index,
            "grid_index_yx": [self.ty, self.tx],
            "owned_mass_yx": [self.j0, self.j1, self.i0, self.i1],
            "compute_mass_yx": [
                self.cj0,
                self.cj0 + self.cny,
                self.ci0,
                self.ci0 + self.cnx,
            ],
            "source_stencil_radius": self.halo,
            "host_transfer_radius": self.halo + CM1_OPEN_NGXY,
        }


@dataclass(frozen=True, slots=True)
class Gm1TilePlan:
    nx: int
    ny: int
    tile_nx: int
    tile_ny: int
    halo: int
    specs: tuple[Gm1TileSpec, ...]
    ownership_sha256: str

    @property
    def ntiles(self) -> int:
        return len(self.specs)


def build_gm1_open_tile_plan(
    *,
    nx: int,
    ny: int,
    tile_nx: int = GM1_DEFAULT_TILE,
    tile_ny: int = GM1_DEFAULT_TILE,
    halo: int = GM1_SOURCE_STENCIL_RADIUS,
) -> Gm1TilePlan:
    """Build fixed-shape, all-open tiles and prove exact scatter ownership."""

    nx, ny = _positive_int("nx", nx), _positive_int("ny", ny)
    tile_nx = _positive_int("tile_nx", tile_nx)
    tile_ny = _positive_int("tile_ny", tile_ny)
    halo = _positive_int("halo", halo)
    if halo != GM1_SOURCE_STENCIL_RADIUS:
        raise ValueError("GM1 scalar tilestream is sealed to the source radius 3")
    cnx, cny = tile_nx + 2 * halo, tile_ny + 2 * halo
    if cnx > nx or cny > ny:
        raise ValueError("nonperiodic fixed compute windows must fit inside the domain")
    ntx, nty = math.ceil(nx / tile_nx), math.ceil(ny / tile_ny)
    specs: list[Gm1TileSpec] = []
    for ty in range(nty):
        j0, j1 = ty * tile_ny, min((ty + 1) * tile_ny, ny)
        cj0 = min(max(j0 - halo, 0), ny - cny)
        for tx in range(ntx):
            i0, i1 = tx * tile_nx, min((tx + 1) * tile_nx, nx)
            ci0 = min(max(i0 - halo, 0), nx - cnx)
            specs.append(
                Gm1TileSpec(
                    len(specs),
                    ty,
                    tx,
                    nx,
                    ny,
                    tile_nx,
                    tile_ny,
                    halo,
                    i0,
                    i1,
                    j0,
                    j1,
                    ci0,
                    cj0,
                    cnx,
                    cny,
                )
            )

    np = importlib.import_module("numpy")
    ownership: dict[str, str] = {}
    for variant in VARIANTS:
        counts = np.zeros(_variant_shape(ny, nx, variant), dtype=np.uint16)
        for spec in specs:
            y0, y1, x0, x1 = spec.owned_rect(variant)
            counts[y0:y1, x0:x1] += 1
        if not bool(np.all(counts == 1)):
            raise RuntimeError(f"{variant} scatter ownership is not exactly one")
        ownership[variant] = _sha256(counts.astype("<u2", copy=False).tobytes())
    scalar_counts = np.zeros((ny, nx), dtype=np.uint16)
    for spec in specs:
        scatter = spec.scalar_scatter()
        if scatter is not None:
            full, _ = scatter
            scalar_counts[full] += 1
    expected = np.zeros_like(scalar_counts)
    expected[3 : ny - 3, 3 : nx - 3] = 1
    if not bool(np.array_equal(scalar_counts, expected)):
        raise RuntimeError(
            "scalar tile scatters do not exactly partition the CM1 interior"
        )
    record = {
        "nx": nx,
        "ny": ny,
        "tile_nx": tile_nx,
        "tile_ny": tile_ny,
        "halo": halo,
        "ntiles": len(specs),
        "variant_coverage_sha256": ownership,
        "scalar_interior_coverage_sha256": _sha256(
            scalar_counts.astype("<u2", copy=False).tobytes()
        ),
        "tiles": [spec.record() for spec in specs],
    }
    return Gm1TilePlan(
        nx, ny, tile_nx, tile_ny, halo, tuple(specs), _sha256(_canonical_json(record))
    )


def _rect_intersection(
    left: tuple[int, int, int, int], right: tuple[int, int, int, int]
) -> tuple[int, int, int, int] | None:
    y0, y1 = max(left[0], right[0]), min(left[1], right[1])
    x0, x1 = max(left[2], right[2]), min(left[3], right[3])
    return None if y0 >= y1 or x0 >= x1 else (y0, y1, x0, x1)


@dataclass(frozen=True, slots=True)
class Gm1RingPlan:
    """Exact transport-ring ownership and event dependencies.

    The plan is executable geometry, but this first scalar stage has immutable
    inputs and a disjoint tendency destination.  A request to use these masks
    for an in-place forecast stage is refused until a save/patch arena is bound.
    """

    tile_plan_sha256: str
    patch_waits: tuple[tuple[int, ...], ...]
    scatter_waits: tuple[tuple[int, ...], ...]
    gather_seam_waits: tuple[tuple[int, ...], ...]
    save_seam_waits: tuple[tuple[int, ...], ...]
    ring_cells_by_variant: Mapping[str, int]
    ring_mask_sha256: Mapping[str, str]
    schedule_sha256: str


def build_gm1_ring_plan(plan: Gm1TilePlan) -> Gm1RingPlan:
    """Derive masks and all four ArWen event-wait classes from rectangles."""

    if not isinstance(plan, Gm1TilePlan):
        raise TypeError("plan must be Gm1TilePlan")
    np = importlib.import_module("numpy")
    n = plan.ntiles
    patch: list[set[int]] = [set() for _ in range(n)]
    war: list[set[int]] = [set() for _ in range(n)]
    masks: dict[str, list[Any]] = {}
    origins: dict[str, list[tuple[int, int]]] = {}
    for variant in VARIANTS:
        masks[variant], origins[variant] = [], []
        for spec in plan.specs:
            y0, y1, x0, x1 = spec.owned_rect(variant)
            masks[variant].append(np.zeros((y1 - y0, x1 - x0), dtype=np.bool_))
            origins[variant].append((y0, x0))
    for variant in VARIANTS:
        reads = [
            spec.gather_rect(variant, include_kernel_padding=True)
            for spec in plan.specs
        ]
        writes = [spec.owned_rect(variant) for spec in plan.specs]
        for writer in range(n):
            for reader in range(n):
                hit = _rect_intersection(writes[writer], reads[reader])
                if hit is None or writer == reader:
                    continue
                if writer < reader:
                    patch[reader].add(writer)
                    y0, y1, x0, x1 = hit
                    origin_y, origin_x = origins[variant][writer]
                    masks[variant][writer][
                        y0 - origin_y : y1 - origin_y,
                        x0 - origin_x : x1 - origin_x,
                    ] = True
                else:
                    war[writer].add(reader)

    gather_seam: list[set[int]] = [{index} for index in range(n)]
    save_seam: list[set[int]] = [set() for _ in range(n)]
    for reader, deps in enumerate(patch):
        for writer in deps:
            gather_seam[reader].add(writer)
            save_seam[writer].add(reader)
    for writer, deps in enumerate(war):
        for reader in deps:
            gather_seam[reader].add(writer)

    variant_cells: dict[str, int] = {}
    variant_sha: dict[str, str] = {}
    for variant, entries in masks.items():
        variant_cells[variant] = sum(int(mask.sum()) for mask in entries)
        packed = b"".join(
            int(mask.shape[0]).to_bytes(4, "little")
            + int(mask.shape[1]).to_bytes(4, "little")
            + np.packbits(mask, bitorder="little").tobytes()
            for mask in entries
        )
        variant_sha[variant] = _sha256(packed)
        for reader, deps in enumerate(patch):
            read = plan.specs[reader].gather_rect(variant, include_kernel_padding=True)
            for writer in deps:
                hit = _rect_intersection(plan.specs[writer].owned_rect(variant), read)
                if hit is None:
                    raise RuntimeError(
                        "ring dependency lacks an intersecting rectangle"
                    )
                y0, y1, x0, x1 = hit
                origin_y, origin_x = origins[variant][writer]
                if not bool(
                    np.all(
                        entries[writer][
                            y0 - origin_y : y1 - origin_y,
                            x0 - origin_x : x1 - origin_x,
                        ]
                    )
                ):
                    raise RuntimeError("ring mask does not cover a later tile read")

    record = {
        "tile_plan_sha256": plan.ownership_sha256,
        "patch_waits": [sorted(value) for value in patch],
        "scatter_waits": [sorted(value) for value in war],
        "gather_seam_waits": [sorted(value) for value in gather_seam],
        "save_seam_waits": [sorted(value) for value in save_seam],
        "ring_cells_by_variant": variant_cells,
        "ring_mask_sha256": variant_sha,
    }
    return Gm1RingPlan(
        plan.ownership_sha256,
        tuple(tuple(sorted(value)) for value in patch),
        tuple(tuple(sorted(value)) for value in war),
        tuple(tuple(sorted(value)) for value in gather_seam),
        tuple(tuple(sorted(value)) for value in save_seam),
        MappingProxyType(variant_cells),
        MappingProxyType(variant_sha),
        _sha256(_canonical_json(record)),
    )


def _variant_ring_masks(
    plan: Gm1TilePlan, variant: str
) -> tuple[tuple[Any, ...], tuple[tuple[int, int], ...]]:
    """Reconstruct the exact per-writer ring masks sealed by ``Gm1RingPlan``."""

    if variant not in VARIANTS:
        raise ValueError(f"unknown horizontal variant {variant!r}")
    np = importlib.import_module("numpy")
    masks: list[Any] = []
    origins: list[tuple[int, int]] = []
    for spec in plan.specs:
        y0, y1, x0, x1 = spec.owned_rect(variant)
        masks.append(np.zeros((y1 - y0, x1 - x0), dtype=np.bool_))
        origins.append((y0, x0))
    reads = [
        spec.gather_rect(variant, include_kernel_padding=True)
        for spec in plan.specs
    ]
    writes = [spec.owned_rect(variant) for spec in plan.specs]
    for writer in range(plan.ntiles):
        for reader in range(writer + 1, plan.ntiles):
            hit = _rect_intersection(writes[writer], reads[reader])
            if hit is None:
                continue
            y0, y1, x0, x1 = hit
            origin_y, origin_x = origins[writer]
            masks[writer][
                y0 - origin_y : y1 - origin_y,
                x0 - origin_x : x1 - origin_x,
            ] = True
    return tuple(masks), tuple(origins)


@dataclass(slots=True)
class Gm1ScalarRingArena:
    """Exact saved-old-value arena for one in-place mass-grid scalar.

    Saving and patching occur on the pinned host side.  This class does not
    apply an RK update; it only proves that a later tile can reconstruct the
    immutable pre-stage halo after an earlier tile has overwritten its owned
    cells.
    """

    plan: Gm1TilePlan
    ring: Gm1RingPlan
    nz: int
    arrays: tuple[Any, ...]
    masks: tuple[Any, ...]
    packed_indices: tuple[Any, ...]
    origins: tuple[tuple[int, int], ...]
    saved: list[bool]
    pinned: bool

    @classmethod
    def allocate(
        cls, plan: Gm1TilePlan, ring: Gm1RingPlan, *, nz: int
    ) -> "Gm1ScalarRingArena":
        return cls._create(plan, ring, nz=nz, pinned=True)

    @classmethod
    def validation_arena(
        cls, plan: Gm1TilePlan, ring: Gm1RingPlan, *, nz: int
    ) -> "Gm1ScalarRingArena":
        return cls._create(plan, ring, nz=nz, pinned=False)

    @classmethod
    def _create(
        cls, plan: Gm1TilePlan, ring: Gm1RingPlan, *, nz: int, pinned: bool
    ) -> "Gm1ScalarRingArena":
        if not isinstance(plan, Gm1TilePlan) or not isinstance(ring, Gm1RingPlan):
            raise TypeError("scalar ring arena requires exact tile and ring plans")
        if ring.tile_plan_sha256 != plan.ownership_sha256:
            raise ValueError("ring arena plans have different tile identities")
        nz = _positive_int("nz", nz)
        np = importlib.import_module("numpy")
        masks, origins = _variant_ring_masks(plan, "mass")
        packed_indices: list[Any] = []
        arrays: list[Any] = []
        packed = bytearray()
        for mask in masks:
            index = np.full(mask.shape, -1, dtype=np.int64)
            index[mask] = np.arange(int(mask.sum()), dtype=np.int64)
            packed_indices.append(index)
            shape = (nz, int(mask.sum()))
            arrays.append(
                allocate_exact_pinned_array(shape)
                if pinned and shape[1]
                else np.empty(shape, dtype=np.dtype("<f4"))
            )
            packed.extend(int(mask.shape[0]).to_bytes(4, "little"))
            packed.extend(int(mask.shape[1]).to_bytes(4, "little"))
            packed.extend(np.packbits(mask, bitorder="little").tobytes())
        if _sha256(bytes(packed)) != ring.ring_mask_sha256["mass"]:
            raise RuntimeError("scalar ring mask bytes differ from the sealed plan")
        return cls(
            plan,
            ring,
            nz,
            tuple(arrays),
            masks,
            tuple(packed_indices),
            origins,
            [False] * plan.ntiles,
            pinned,
        )

    @property
    def nbytes(self) -> int:
        return sum(int(value.nbytes) for value in self.arrays)

    def save_tile(self, source: Any, tile_index: int) -> dict[str, object]:
        np = importlib.import_module("numpy")
        if (
            not isinstance(source, np.ndarray)
            or source.dtype != np.dtype("<f4")
            or source.shape != (self.nz, self.plan.ny, self.plan.nx)
            or not source.flags.c_contiguous
        ):
            raise ValueError("ring source must be host C-<f4 physical mass scalar")
        if (
            isinstance(tile_index, bool)
            or not isinstance(tile_index, int)
            or tile_index not in range(self.plan.ntiles)
        ):
            raise ValueError("ring tile_index is outside the plan")
        if self.saved[tile_index]:
            raise ValueError("ring values for this tile were already saved")
        y0, x0 = self.origins[tile_index]
        mask = self.masks[tile_index]
        values = self.arrays[tile_index]
        np.copyto(
            values,
            source[:, y0 : y0 + mask.shape[0], x0 : x0 + mask.shape[1]][
                :, mask
            ],
        )
        self.saved[tile_index] = True
        return {
            "schema": "gm1.hybrid-tilestream-scalar-ring-save/v1",
            "tile": tile_index,
            "saved_words": int(values.size),
            "ring_schedule_sha256": self.ring.schedule_sha256,
            "ring_mask_sha256": self.ring.ring_mask_sha256["mass"],
        }

    def patch_tile(self, target_scalar: Any, spec: Gm1TileSpec) -> dict[str, object]:
        np = importlib.import_module("numpy")
        expected = (
            self.nz + 2,
            spec.cny + 2 * CM1_OPEN_NGXY,
            spec.cnx + 2 * CM1_OPEN_NGXY,
        )
        if (
            not isinstance(target_scalar, np.ndarray)
            or target_scalar.dtype != np.dtype("<f4")
            or target_scalar.shape != expected
            or not target_scalar.flags.c_contiguous
            or not target_scalar.flags.writeable
        ):
            raise ValueError(f"ring patch target must be writable host C-<f4 {expected}")
        if spec.index not in range(self.plan.ntiles) or self.plan.specs[spec.index] != spec:
            raise ValueError("ring patch tile differs from its exact plan")
        raw_y0, raw_x0 = spec.cj0 - CM1_OPEN_NGXY, spec.ci0 - CM1_OPEN_NGXY
        gather = spec.gather_rect("mass", include_kernel_padding=True)
        target_global_y = np.clip(
            np.arange(raw_y0, raw_y0 + expected[1], dtype=np.int64),
            0,
            self.plan.ny - 1,
        )
        target_global_x = np.clip(
            np.arange(raw_x0, raw_x0 + expected[2], dtype=np.int64),
            0,
            self.plan.nx - 1,
        )
        patched = 0
        writers: list[int] = []
        for writer in self.ring.patch_waits[spec.index]:
            if not self.saved[writer]:
                raise ValueError("ring dependency was not saved before patch")
            owned = self.plan.specs[writer].owned_rect("mass")
            hit = _rect_intersection(owned, gather)
            if hit is None:
                raise RuntimeError("ring patch dependency has no geometric overlap")
            origin_y, origin_x = self.origins[writer]
            local_y = target_global_y[:, None] - origin_y
            local_x = target_global_x[None, :] - origin_x
            inside = (
                (local_y >= 0)
                & (local_y < self.masks[writer].shape[0])
                & (local_x >= 0)
                & (local_x < self.masks[writer].shape[1])
            )
            valid = np.zeros(inside.shape, dtype=np.bool_)
            iy, ix = np.nonzero(inside)
            valid[iy, ix] = self.masks[writer][local_y[iy, 0], local_x[0, ix]]
            ys, xs = np.nonzero(valid)
            if not len(ys):
                raise RuntimeError("ring patch overlap lacks saved words")
            packed = self.packed_indices[writer][
                target_global_y[ys] - origin_y,
                target_global_x[xs] - origin_x,
            ]
            target_scalar[
                1 : self.nz + 1,
                ys,
                xs,
            ] = self.arrays[writer][:, packed]
            patched += self.nz * len(ys)
            writers.append(writer)
        return {
            "schema": "gm1.hybrid-tilestream-scalar-ring-patch/v1",
            "tile": spec.index,
            "writers": writers,
            "patched_words": patched,
            "source_stage_values_restored": True,
            "canonical_state_mutated_by_patch": False,
        }


class Gm1PinnedBlock:
    """One exact-size raw ``cudaHostAlloc`` block with NumPy view support."""

    __slots__ = ("ptr", "nbytes", "__array_interface__")

    def __init__(self, nbytes: int):
        nbytes = _positive_int("nbytes", nbytes)
        runtime = importlib.import_module("cupy.cuda.runtime")
        self.ptr = int(runtime.hostAlloc(nbytes, 0))
        self.nbytes = nbytes
        self.__array_interface__ = {
            "data": (self.ptr, False),
            "shape": (nbytes,),
            "typestr": "|u1",
            "version": 3,
        }

    def free(self) -> None:
        ptr, self.ptr = getattr(self, "ptr", 0), 0
        if ptr:
            runtime = importlib.import_module("cupy.cuda.runtime")
            runtime.freeHost(ptr)

    def __del__(self) -> None:  # pragma: no cover - interpreter teardown
        try:
            self.free()
        except Exception:
            pass


def allocate_exact_pinned_array(shape: Sequence[int], dtype: str = "<f4") -> Any:
    """Allocate one zeroed exact-size pinned C array without CuPy pool rounding."""

    np = importlib.import_module("numpy")
    shape = tuple(_positive_int("shape extent", int(value)) for value in shape)
    dtype_value = np.dtype(dtype)
    count = math.prod(shape)
    block = Gm1PinnedBlock(count * dtype_value.itemsize)
    result = np.asarray(block).view(dtype_value).reshape(shape)
    result.fill(0)
    attrs = importlib.import_module("cupy.cuda.runtime").pointerGetAttributes(
        int(result.ctypes.data)
    )
    if int(attrs.type) != CUDA_MEMORY_TYPE_HOST:
        block.free()
        raise RuntimeError("cudaHostAlloc result is not CUDA host memory")
    return result


def _stage_shapes(*, nx: int, ny: int, nz: int) -> dict[str, tuple[int, ...]]:
    ng = CM1_OPEN_NGXY
    return {
        "scalar": (nz + 2, ny + 2 * ng, nx + 2 * ng),
        "rru": (nz + 2, ny + 2 * ng, nx + 1 + 2 * ng),
        "rrv": (nz + 2, ny + 1 + 2 * ng, nx + 2 * ng),
        "rrw": (nz + 3, ny + 2 * ng, nx + 2 * ng),
        "divx": (nz, ny, nx),
        "c1": (nz + 1, ny, nx),
        "c2": (nz + 1, ny, nx),
        "rr0": (nz, ny, nx),
        "gz": (ny, nx),
        "tendency": (nz, ny, nx),
    }


@dataclass(slots=True)
class Gm1PinnedHostCarrierStore:
    arrays: dict[str, Any]
    nx: int
    ny: int
    nz: int
    pinned: bool

    @classmethod
    def allocate_stage(
        cls, *, nx: int, ny: int, nz: int
    ) -> "Gm1PinnedHostCarrierStore":
        shapes = _stage_shapes(nx=nx, ny=ny, nz=nz)
        arrays = {
            name: allocate_exact_pinned_array(shape) for name, shape in shapes.items()
        }
        return cls(arrays, nx, ny, nz, True)

    @classmethod
    def validation_store(
        cls, arrays: Mapping[str, Any], *, nx: int, ny: int, nz: int
    ) -> "Gm1PinnedHostCarrierStore":
        """CPU-only pageable store; production launch always rejects it."""

        np = importlib.import_module("numpy")
        shapes = _stage_shapes(nx=nx, ny=ny, nz=nz)
        if set(arrays) != set(shapes):
            raise ValueError("stage store inventory changed")
        checked: dict[str, Any] = {}
        for name, shape in shapes.items():
            value = arrays[name]
            if (
                not isinstance(value, np.ndarray)
                or value.dtype != np.float32
                or value.shape != shape
                or not value.flags.c_contiguous
            ):
                raise ValueError(f"{name} must be host C-f32 {shape}")
            checked[name] = value
        return cls(checked, nx, ny, nz, False)

    @property
    def nbytes(self) -> int:
        return sum(int(value.nbytes) for value in self.arrays.values())

    def assert_pinned(self) -> dict[str, object]:
        runtime = importlib.import_module("cupy.cuda.runtime")
        types: dict[str, int] = {}
        for name, value in self.arrays.items():
            try:
                types[name] = int(
                    runtime.pointerGetAttributes(int(value.ctypes.data)).type
                )
            except Exception:
                types[name] = 0
        if (
            not self.pinned
            or not types
            or any(value != CUDA_MEMORY_TYPE_HOST for value in types.values())
        ):
            raise ValueError(
                "production GM1 tilestream requires every host carrier to be CUDA-pinned"
            )
        return {"all_pinned": True, "memory_types": types, "nbytes": self.nbytes}


@dataclass(slots=True)
class Gm1CanonicalHostCarrierStore:
    """The exact physical 37-carrier state, never a transient stage duplicate."""

    arrays: dict[str, Any]
    nx: int
    ny: int
    nz: int
    pinned: bool

    @classmethod
    def allocate(cls, *, nx: int, ny: int, nz: int) -> "Gm1CanonicalHostCarrierStore":
        arrays = {
            name: allocate_exact_pinned_array(
                checkpoint_carrier_shape(name, nx=nx, ny=ny, nz=nz)
            )
            for name in CHECKPOINT_NAMES
        }
        return cls(arrays, nx, ny, nz, True)

    @classmethod
    def validation_store(
        cls, arrays: Mapping[str, Any], *, nx: int, ny: int, nz: int
    ) -> "Gm1CanonicalHostCarrierStore":
        np = importlib.import_module("numpy")
        if tuple(arrays) != CHECKPOINT_NAMES:
            raise ValueError("canonical store must preserve the exact 37-carrier order")
        checked: dict[str, Any] = {}
        pointers: list[int] = []
        for name in CHECKPOINT_NAMES:
            value = arrays[name]
            shape = checkpoint_carrier_shape(name, nx=nx, ny=ny, nz=nz)
            if (
                not isinstance(value, np.ndarray)
                or value.dtype != np.dtype("<f4")
                or value.shape != shape
                or not value.flags.c_contiguous
            ):
                raise ValueError(f"canonical carrier {name} must be host C-<f4 {shape}")
            checked[name] = value
            pointers.append(int(value.ctypes.data))
        if len(pointers) != len(set(pointers)):
            raise ValueError("canonical carriers must not alias")
        return cls(checked, nx, ny, nz, False)

    @property
    def nbytes(self) -> int:
        return sum(int(value.nbytes) for value in self.arrays.values())

    def assert_pinned(self) -> dict[str, object]:
        runtime = importlib.import_module("cupy.cuda.runtime")
        types = {
            name: int(runtime.pointerGetAttributes(int(value.ctypes.data)).type)
            for name, value in self.arrays.items()
        }
        if not self.pinned or any(
            value != CUDA_MEMORY_TYPE_HOST for value in types.values()
        ):
            raise ValueError(
                "canonical production store requires all 37 carriers in CUDA host memory"
            )
        return {"all_pinned": True, "memory_types": types, "nbytes": self.nbytes}


def _immutable_source_shapes(
    *, nx: int, ny: int, nz: int
) -> Mapping[str, tuple[int, ...]]:
    return MappingProxyType(
        {
            "c1": (nz + 1, ny, nx),
            "c2": (nz + 1, ny, nx),
            "rr0": (nz, ny, nx),
            "rho0": (nz, ny, nx),
            "rf0": (nz + 1, ny, nx),
            "gz": (ny, nx),
            "rgzu": (ny, nx + 1),
            "rgzv": (ny + 1, nx),
            "dzdx": (ny, nx),
            "dzdy": (ny, nx),
            "th0": (nz, ny, nx),
        }
    )


@dataclass(frozen=True, slots=True)
class Gm1TileLocalSourceBinding:
    """Source-coordinate views needed to reconstruct one tile working set."""

    canonical: Gm1CanonicalHostCarrierStore
    immutable: Mapping[str, Any]
    scalar: str
    source_identity_sha256: str

    @classmethod
    def create(
        cls,
        canonical: Gm1CanonicalHostCarrierStore,
        immutable: Mapping[str, Any],
        *,
        scalar: str,
        source_identity_sha256: str,
    ) -> "Gm1TileLocalSourceBinding":
        if not isinstance(canonical, Gm1CanonicalHostCarrierStore):
            raise TypeError("canonical must be Gm1CanonicalHostCarrierStore")
        if scalar not in {"theta", "qv", "qc", "qr", "qi", "qs", "qg"}:
            raise ValueError("scalar must be theta or one exact WSM6 species")
        if (
            not isinstance(source_identity_sha256, str)
            or len(source_identity_sha256) != 64
            or any(value not in "0123456789abcdef" for value in source_identity_sha256)
        ):
            raise ValueError("source identity must be lowercase 64-hex sha256")
        np = importlib.import_module("numpy")
        shapes = _immutable_source_shapes(
            nx=canonical.nx, ny=canonical.ny, nz=canonical.nz
        )
        if tuple(immutable) != TILE_LOCAL_IMMUTABLE_NAMES:
            raise ValueError("immutable tile-source inventory or order changed")
        checked: dict[str, Any] = {}
        for name in TILE_LOCAL_IMMUTABLE_NAMES:
            value = immutable[name]
            if (
                not isinstance(value, np.ndarray)
                or value.dtype != np.dtype("<f4")
                or value.shape != shapes[name]
                or not value.flags.c_contiguous
            ):
                raise ValueError(
                    f"immutable source {name} must be host C-<f4 {shapes[name]}"
                )
            checked[name] = value
        return cls(
            canonical,
            MappingProxyType(checked),
            scalar,
            source_identity_sha256,
        )


def _tile_local_source_shapes(
    plan: Gm1TilePlan, *, nz: int
) -> dict[str, tuple[int, ...]]:
    shapes = _tile_stage_shapes(plan, nz=nz)
    cnx, cny = plan.specs[0].cnx, plan.specs[0].cny
    shapes.update(
        {
            "rho0": (nz + 2, cny + 2 * CM1_OPEN_NGXY, cnx + 2 * CM1_OPEN_NGXY),
            "rf0": (nz + 1, cny, cnx),
            "rgzu": (cny, cnx + 1),
            "rgzv": (cny + 1, cnx),
            "dzdx": (cny, cnx),
            "dzdy": (cny, cnx),
        }
    )
    return shapes


def tile_host_buffer_shapes(
    *, tile_nx: int, tile_ny: int, nz: int
) -> Mapping[str, tuple[int, ...]]:
    """Public exact pinned-host inventory for one duplicate-free tile."""

    tile_nx = _positive_int("tile_nx", tile_nx)
    tile_ny = _positive_int("tile_ny", tile_ny)
    nz = _positive_int("nz", nz)
    cnx = tile_nx + 2 * GM1_SOURCE_STENCIL_RADIUS
    cny = tile_ny + 2 * GM1_SOURCE_STENCIL_RADIUS
    shapes = _stage_shapes(nx=cnx, ny=cny, nz=nz)
    shapes.update(
        {
            "rho0": (nz + 2, cny + 2 * CM1_OPEN_NGXY, cnx + 2 * CM1_OPEN_NGXY),
            "rf0": (nz + 1, cny, cnx),
            "rgzu": (cny, cnx + 1),
            "rgzv": (cny + 1, cnx),
            "dzdx": (cny, cnx),
            "dzdy": (cny, cnx),
        }
    )
    return MappingProxyType(shapes)


def tile_device_buffer_shapes(
    *, tile_nx: int, tile_ny: int, nz: int
) -> Mapping[str, tuple[int, ...]]:
    """Public exact device inventory: host mirror plus fx/fy/fz scratch."""

    shapes = dict(tile_host_buffer_shapes(tile_nx=tile_nx, tile_ny=tile_ny, nz=nz))
    shapes.update(
        {
            "fx": shapes["rru"],
            "fy": shapes["rrv"],
            "fz": shapes["rrw"],
        }
    )
    return MappingProxyType(shapes)


def tile_pdef_host_buffer_shapes(
    *, tile_nx: int, tile_ny: int, nz: int
) -> Mapping[str, tuple[int, ...]]:
    """Pinned-host inventory for the final-RK moisture pdef tendency seam."""

    shapes = dict(tile_host_buffer_shapes(tile_nx=tile_nx, tile_ny=tile_ny, nz=nz))
    cnx = tile_nx + 2 * GM1_SOURCE_STENCIL_RADIUS
    cny = tile_ny + 2 * GM1_SOURCE_STENCIL_RADIUS
    shapes.update(
        {
            "moisture_a": (nz, cny, cnx),
            "rho0_3d": (nz, cny, cnx),
            "rgz": (cny, cnx),
        }
    )
    return MappingProxyType(shapes)


def tile_pdef_device_buffer_shapes(
    *, tile_nx: int, tile_ny: int, nz: int
) -> Mapping[str, tuple[int, ...]]:
    """Device inventory for getdivx plus final-RK pdef scratch."""

    shapes = dict(tile_device_buffer_shapes(tile_nx=tile_nx, tile_ny=tile_ny, nz=nz))
    cnx = tile_nx + 2 * GM1_SOURCE_STENCIL_RADIUS
    cny = tile_ny + 2 * GM1_SOURCE_STENCIL_RADIUS
    mass = (nz, cny, cnx)
    shapes.update(
        {
            "moisture_a": mass,
            "rho0_3d": mass,
            "rgz": (cny, cnx),
            "advx": mass,
            "advy": mass,
            "advz": mass,
            "candidate": mass,
        }
    )
    return MappingProxyType(shapes)


def tile_pdef_apply_host_buffer_shapes(
    *, tile_nx: int, tile_ny: int, nz: int
) -> Mapping[str, tuple[int, ...]]:
    """Pinned-host inventory through the source-ordered q3d tile apply."""

    shapes = dict(tile_pdef_host_buffer_shapes(tile_nx=tile_nx, tile_ny=tile_ny, nz=nz))
    cnx = tile_nx + 2 * GM1_SOURCE_STENCIL_RADIUS
    cny = tile_ny + 2 * GM1_SOURCE_STENCIL_RADIUS
    mass = (nz, cny, cnx)
    shapes.update({"moisture_held": mass, "moisture_current": mass})
    return MappingProxyType(shapes)


def tile_pdef_apply_device_buffer_shapes(
    *, tile_nx: int, tile_ny: int, nz: int
) -> Mapping[str, tuple[int, ...]]:
    """Device inventory through pdef and the final-RK moisture update."""

    shapes = dict(
        tile_pdef_device_buffer_shapes(tile_nx=tile_nx, tile_ny=tile_ny, nz=nz)
    )
    cnx = tile_nx + 2 * GM1_SOURCE_STENCIL_RADIUS
    cny = tile_ny + 2 * GM1_SOURCE_STENCIL_RADIUS
    mass = (nz, cny, cnx)
    shapes.update({"moisture_held": mass, "moisture_current": mass})
    return MappingProxyType(shapes)


def _edge_gather_xy(np: Any, value: Any, *, y0: int, x0: int, ny: int, nx: int) -> Any:
    """Gather a mass-grid rectangle with exact constant open-edge extension."""

    source_ny, source_nx = value.shape[-2:]
    yi = np.clip(np.arange(y0, y0 + ny, dtype=np.int64), 0, source_ny - 1)
    xi = np.clip(np.arange(x0, x0 + nx, dtype=np.int64), 0, source_nx - 1)
    return value[..., yi[:, None], xi]


def fill_gm1_tile_local_source(
    binding: Gm1TileLocalSourceBinding,
    spec: Gm1TileSpec,
    target: Mapping[str, Any],
) -> dict[str, object]:
    """Fill one tile from physical state/static sources without global scratch."""

    if not isinstance(binding, Gm1TileLocalSourceBinding):
        raise TypeError("binding must be Gm1TileLocalSourceBinding")
    if not isinstance(spec, Gm1TileSpec):
        raise TypeError("spec must be Gm1TileSpec")
    store = binding.canonical
    if (spec.nx, spec.ny) != (store.nx, store.ny):
        raise ValueError("tile plan and canonical store domains differ")
    np = importlib.import_module("numpy")
    plan = Gm1TilePlan(
        store.nx,
        store.ny,
        spec.tile_nx,
        spec.tile_ny,
        spec.halo,
        (spec,),
        "local-shape-only",
    )
    shapes = _tile_local_source_shapes(plan, nz=store.nz)
    if set(target) != set(shapes):
        raise ValueError("tile-local target inventory changed")
    for name, shape in shapes.items():
        value = target[name]
        if (
            not isinstance(value, np.ndarray)
            or value.dtype != np.dtype("<f4")
            or value.shape != shape
            or not value.flags.c_contiguous
            or not value.flags.writeable
        ):
            raise ValueError(f"tile-local target {name} must be writable C-<f4 {shape}")
        value.fill(np.float32(0.0))

    ng, x0, y0, cnx, cny = (
        CM1_OPEN_NGXY,
        spec.ci0,
        spec.cj0,
        spec.cnx,
        spec.cny,
    )
    if binding.scalar == "theta":
        base = _edge_gather_xy(
            np,
            binding.immutable["th0"],
            y0=y0 - ng,
            x0=x0 - ng,
            ny=cny + 2 * ng,
            nx=cnx + 2 * ng,
        )
        perturbation = _edge_gather_xy(
            np,
            store.arrays["th3d"],
            y0=y0 - ng,
            x0=x0 - ng,
            ny=cny + 2 * ng,
            nx=cnx + 2 * ng,
        )
        np.add(
            np.subtract(base, np.float32(300.0), dtype=np.float32),
            perturbation,
            out=target["scalar"][1 : store.nz + 1],
            dtype=np.float32,
        )
    else:
        species = ("qv", "qc", "qr", "qi", "qs", "qg").index(binding.scalar)
        np.copyto(
            target["scalar"][1 : store.nz + 1],
            _edge_gather_xy(
                np,
                store.arrays["q3d"][species],
                y0=y0 - ng,
                x0=x0 - ng,
                ny=cny + 2 * ng,
                nx=cnx + 2 * ng,
            ),
        )

    np.copyto(
        target["rru"][1 : store.nz + 1, ng : ng + cny, ng : ng + cnx + 1],
        store.arrays["u3d"][:, y0 : y0 + cny, x0 : x0 + cnx + 1],
    )
    np.copyto(
        target["rrv"][1 : store.nz + 1, ng : ng + cny + 1, ng : ng + cnx],
        store.arrays["v3d"][:, y0 : y0 + cny + 1, x0 : x0 + cnx],
    )
    np.copyto(
        target["rrw"][1 : store.nz + 2, ng : ng + cny, ng : ng + cnx],
        store.arrays["w3d"][:, y0 : y0 + cny, x0 : x0 + cnx],
    )
    np.copyto(
        target["rho0"][1 : store.nz + 1],
        _edge_gather_xy(
            np,
            binding.immutable["rho0"],
            y0=y0 - ng,
            x0=x0 - ng,
            ny=cny + 2 * ng,
            nx=cnx + 2 * ng,
        ),
    )
    direct = {
        "c1": (slice(None), slice(y0, y0 + cny), slice(x0, x0 + cnx)),
        "c2": (slice(None), slice(y0, y0 + cny), slice(x0, x0 + cnx)),
        "rr0": (slice(None), slice(y0, y0 + cny), slice(x0, x0 + cnx)),
        "rf0": (slice(None), slice(y0, y0 + cny), slice(x0, x0 + cnx)),
        "gz": (slice(y0, y0 + cny), slice(x0, x0 + cnx)),
        "dzdx": (slice(y0, y0 + cny), slice(x0, x0 + cnx)),
        "dzdy": (slice(y0, y0 + cny), slice(x0, x0 + cnx)),
        "rgzu": (slice(y0, y0 + cny), slice(x0, x0 + cnx + 1)),
        "rgzv": (slice(y0, y0 + cny + 1), slice(x0, x0 + cnx)),
    }
    for name, slices in direct.items():
        np.copyto(target[name], binding.immutable[name][slices])
    return {
        "schema": GM1_TILESTREAM_SOURCE_BINDING_ABI,
        "source_identity_sha256": binding.source_identity_sha256,
        "scalar": binding.scalar,
        "tile": spec.index,
        "compute_mass_yx": [y0, y0 + cny, x0, x0 + cnx],
        "full_domain_transient_stage_store_used": False,
        "open_scalar_edge_rule": "constant extension, matching selected open scalar halo kernels",
        "mass_flux_disposition": "physical u/v/w staged; exact getdivx kernels required before scalar advection",
    }


def fill_gm1_tile_local_pdef_source(
    binding: Gm1TileLocalSourceBinding,
    spec: Gm1TileSpec,
    target: Mapping[str, Any],
    *,
    species: int,
    rgz: Any,
) -> dict[str, object]:
    """Fill the exact extra inputs for one final-RK moisture pdef tile."""

    if isinstance(species, bool) or not isinstance(species, int) or species not in range(6):
        raise ValueError("species must index qv,qc,qr,qi,qs,qg")
    expected_scalar = ("qv", "qc", "qr", "qi", "qs", "qg")[species]
    if binding.scalar != expected_scalar:
        raise ValueError("pdef species and bound padded scalar differ")
    np = importlib.import_module("numpy")
    store = binding.canonical
    shapes = tile_pdef_host_buffer_shapes(
        tile_nx=spec.tile_nx, tile_ny=spec.tile_ny, nz=store.nz
    )
    if set(target) != set(shapes):
        raise ValueError("tile-local pdef host inventory changed")
    base_names = tuple(
        tile_host_buffer_shapes(
            tile_nx=spec.tile_nx, tile_ny=spec.tile_ny, nz=store.nz
        )
    )
    fill = fill_gm1_tile_local_source(
        binding, spec, {name: target[name] for name in base_names}
    )
    extra_shapes = {name: shapes[name] for name in ("moisture_a", "rho0_3d", "rgz")}
    for name, shape in extra_shapes.items():
        value = target[name]
        if (
            not isinstance(value, np.ndarray)
            or value.dtype != np.dtype("<f4")
            or value.shape != shape
            or not value.flags.c_contiguous
            or not value.flags.writeable
        ):
            raise ValueError(f"tile-local pdef {name} must be writable C-<f4 {shape}")
    if (
        not isinstance(rgz, np.ndarray)
        or rgz.dtype != np.dtype("<f4")
        or rgz.shape != (store.ny, store.nx)
        or not rgz.flags.c_contiguous
    ):
        raise ValueError("pdef rgz source must be host C-<f4 physical surface")
    y0, x0, cny, cnx = spec.cj0, spec.ci0, spec.cny, spec.cnx
    np.copyto(
        target["moisture_a"],
        store.arrays["qa"][species, :, y0 : y0 + cny, x0 : x0 + cnx],
    )
    np.copyto(
        target["rho0_3d"],
        binding.immutable["rho0"][:, y0 : y0 + cny, x0 : x0 + cnx],
    )
    np.copyto(target["rgz"], rgz[y0 : y0 + cny, x0 : x0 + cnx])
    return {
        **fill,
        "abi": "gm1.hybrid-tilestream-local-pdef-source-f32/v1",
        "species": species,
        "species_name": expected_scalar,
        "moisture_a_disposition": "exact canonical qa current-A physical tile",
        "scalar_padded_disposition": "read-only exact q3d ng=3 source halo",
        "q3d_inplace_apply_performed": False,
    }


def fill_gm1_tile_local_pdef_apply_source(
    binding: Gm1TileLocalSourceBinding,
    spec: Gm1TileSpec,
    target: Mapping[str, Any],
    *,
    species: int,
    rgz: Any,
    moisture_held: Any,
) -> dict[str, object]:
    """Fill one pdef/apply tile, including its exact solve1 held tendency."""

    np = importlib.import_module("numpy")
    store = binding.canonical
    shapes = tile_pdef_apply_host_buffer_shapes(
        tile_nx=spec.tile_nx, tile_ny=spec.tile_ny, nz=store.nz
    )
    if set(target) != set(shapes):
        raise ValueError("tile-local pdef/apply host inventory changed")
    pdef_names = tuple(
        tile_pdef_host_buffer_shapes(
            tile_nx=spec.tile_nx, tile_ny=spec.tile_ny, nz=store.nz
        )
    )
    fill = fill_gm1_tile_local_pdef_source(
        binding,
        spec,
        {name: target[name] for name in pdef_names},
        species=species,
        rgz=rgz,
    )
    expected = (store.nz, store.ny, store.nx)
    if (
        not isinstance(moisture_held, np.ndarray)
        or moisture_held.dtype != np.dtype("<f4")
        or moisture_held.shape != expected
        or not moisture_held.flags.c_contiguous
    ):
        raise ValueError(f"moisture_held must be host C-<f4 {expected}")
    if any(
        np.shares_memory(moisture_held, value)
        for value in (store.arrays["q3d"], store.arrays["qa"])
    ):
        raise ValueError("moisture_held must not alias q3d or qa")
    y0, x0, cny, cnx = spec.cj0, spec.ci0, spec.cny, spec.cnx
    np.copyto(
        target["moisture_held"],
        moisture_held[:, y0 : y0 + cny, x0 : x0 + cnx],
    )
    target["moisture_current"].fill(np.float32(0.0))
    return {
        **fill,
        "abi": "gm1.hybrid-tilestream-local-pdef-apply-source-f32/v1",
        "moisture_held_disposition": "exact live solve1 qten physical tile",
        "moisture_current_disposition": "pinned D2H destination; initialized +0",
        "source_order": "solve2.F:1430-1526",
    }


def scatter_gm1_tile_local_pdef_apply(
    canonical: Gm1CanonicalHostCarrierStore,
    spec: Gm1TileSpec,
    tile_current: Any,
    *,
    species: int,
) -> dict[str, object]:
    """Publish one unique q3d owned rectangle after its old-value ring save."""

    if not isinstance(canonical, Gm1CanonicalHostCarrierStore):
        raise TypeError("canonical must be Gm1CanonicalHostCarrierStore")
    if not isinstance(spec, Gm1TileSpec):
        raise TypeError("spec must be Gm1TileSpec")
    if (spec.nx, spec.ny) != (canonical.nx, canonical.ny):
        raise ValueError("pdef/apply tile and canonical domains differ")
    if isinstance(species, bool) or not isinstance(species, int) or species not in range(6):
        raise ValueError("species must index qv,qc,qr,qi,qs,qg")
    np = importlib.import_module("numpy")
    expected = (canonical.nz, spec.cny, spec.cnx)
    if (
        not isinstance(tile_current, np.ndarray)
        or tile_current.dtype != np.dtype("<f4")
        or tile_current.shape != expected
        or not tile_current.flags.c_contiguous
    ):
        raise ValueError(f"tile_current must be host C-<f4 {expected}")
    y0, y1, x0, x1 = spec.owned_rect("mass")
    ty0, tx0 = y0 - spec.cj0, x0 - spec.ci0
    ty1, tx1 = ty0 + (y1 - y0), tx0 + (x1 - x0)
    destination = canonical.arrays["q3d"][species, :, y0:y1, x0:x1]
    np.copyto(destination, tile_current[:, ty0:ty1, tx0:tx1])
    boundary = np.zeros((y1 - y0, x1 - x0), dtype=np.bool_)
    yy = np.arange(y0, y1)[:, None]
    xx = np.arange(x0, x1)[None, :]
    boundary |= yy < CM1_OPEN_NGXY
    boundary |= yy >= canonical.ny - CM1_OPEN_NGXY
    boundary |= xx < CM1_OPEN_NGXY
    boundary |= xx >= canonical.nx - CM1_OPEN_NGXY
    return {
        "schema": "gm1.hybrid-tilestream-q3d-owned-scatter/v1",
        "tile": spec.index,
        "species": species,
        "owned_yx": [y0, y1, x0, x1],
        "published_words": int(destination.size),
        "open_physical_boundary_band_words": int(boundary.sum()) * canonical.nz,
        "unique_full_physical_ownership": True,
        "solve3_bcs_or_open_history_performed": False,
        "pdefq_compatible_physical_q3d": True,
    }


def checkpoint_carrier_shape(
    name: str, *, nx: int, ny: int, nz: int
) -> tuple[int, ...]:
    """Shape from the committed 37-carrier post-step checkpoint ABI."""

    mass = (nz, ny, nx)
    if name in {"q3d", "qa"}:
        return (6, *mass)
    if name in {"u3d", "ua"}:
        return (nz, ny, nx + 1)
    if name in {"v3d", "va"}:
        return (nz, ny + 1, nx)
    if name in {"w3d", "wa", "rf", "closure.tkea"}:
        return (nz + 1, ny, nx)
    if name in {"radbcw", "radbce"}:
        return (nz + 2, ny + 2 * CM1_OPEN_NGXY)
    if name in {"radbcs", "radbcn"}:
        return (nz + 2, nx + 2 * CM1_OPEN_NGXY)
    if name.startswith("component.mp_"):
        return (ny, nx)
    if name not in CHECKPOINT_NAMES:
        raise KeyError(name)
    return mass


def checkpoint_store_bytes(*, nx: int, ny: int, nz: int) -> int:
    return sum(
        math.prod(checkpoint_carrier_shape(name, nx=nx, ny=ny, nz=nz)) * 4
        for name in CHECKPOINT_NAMES
    )


def render_output_staging_bytes(*, nx: int, ny: int, nz: int) -> int:
    """Exact 13-carrier first-step history/render staging inventory."""

    mass = nz * ny * nx
    u = nz * ny * (nx + 1)
    v = nz * (ny + 1) * nx
    zface = (nz + 1) * ny * nx
    surface = ny * nx
    # pp3d, th3d, prs, rho, plus six q3d species; u3d/v3d; and
    # w3d, rf, zf, total_geopotential_zf, closure.tkea; terrain_msl.
    return (10 * mass + u + v + 5 * zface + surface) * 4


def _checkpoint_variant(name: str) -> str | None:
    if name in {"u3d", "ua"}:
        return "u"
    if name in {"v3d", "va"}:
        return "v"
    if name in {"w3d", "wa", "rf", "closure.tkea"}:
        return "w"
    if name.startswith("radbc"):
        return None
    return "mass"


def checkpoint_ring_bytes(plan: Gm1TilePlan, ring: Gm1RingPlan, *, nz: int) -> int:
    """Exact arena bytes implied by the 37-carrier ABI and ring masks."""

    total_words = 0
    for name in CHECKPOINT_NAMES:
        variant = _checkpoint_variant(name)
        if variant is None:
            continue
        shape = checkpoint_carrier_shape(name, nx=plan.nx, ny=plan.ny, nz=nz)
        lead = math.prod(shape[:-2])
        total_words += lead * int(ring.ring_cells_by_variant[variant])
    return total_words * 4


def pressure_workspace_shapes(
    *, nx: int, ny: int, nz: int
) -> Mapping[str, tuple[int, ...]]:
    """Exact arrays allocated inside the selected full-domain psolver3 stage."""

    return MappingProxyType(
        {
            "pua": (nz, ny + 2, nx + 3),
            "pu": (nz, ny + 2, nx + 3),
            "pva": (nz, ny + 3, nx + 2),
            "pv": (nz, ny + 3, nx + 2),
            "pruf": (nx + 3,),
            "prvf": (ny + 3,),
            "prw": (nz, ny),
            "pre": (nz, ny),
            "prs": (nz, nx),
            "prn": (nz, nx),
            "us": ((nz + 1) * ny, 4),
            "vs": ((nz + 1) * nx, 4),
        }
    )


def pressure_workspace_inventory(*, nx: int, ny: int, nz: int) -> dict[str, object]:
    shapes = pressure_workspace_shapes(nx=nx, ny=ny, nz=nz)
    rows = [
        {
            "name": name,
            "shape": list(shape),
            "dtype": "<f4",
            "nbytes": math.prod(shape) * 4,
        }
        for name, shape in shapes.items()
    ]
    total = sum(int(row["nbytes"]) for row in rows)
    result = {
        "schema": GM1_TILESTREAM_PRESSURE_ABI,
        "selection": "CM1 psolver=3; terrain/open; stretch_z=1",
        "domain": {"nx": nx, "ny": ny, "nz": nz},
        "arrays": rows,
        "nbytes": total,
        "tile_local": False,
        "drain_required_before_pressure": True,
        "source": {
            "module_sha256": GM1_PSOLVER3_MODULE_SHA256,
            "core_kernel_sha256": GM1_PSOLVER3_CORE_SHA256,
            "boundary_kernel_sha256": GM1_PSOLVER3_BOUNDARY_SHA256,
        },
        "library_subsets": {
            "cufft": {
                "selected": False,
                "workspace_nbytes": 0,
                "reason": "selected psolver3 uses an explicit column solve, not an FFT",
            },
            "cusolver": {
                "selected": False,
                "workspace_nbytes": 0,
                "reason": "selected source-ordered tridiagonal solve uses sealed custom kernels",
            },
        },
        "claim_limit": "scratch allocation inventory only; no tiled/global pressure composition claim",
    }
    result["inventory_sha256"] = _sha256(_canonical_json(result))
    return result


def _tile_stage_shapes(plan: Gm1TilePlan, *, nz: int) -> dict[str, tuple[int, ...]]:
    return _stage_shapes(nx=plan.specs[0].cnx, ny=plan.specs[0].cny, nz=nz)


def _tile_stage_bytes(*, tile_nx: int, tile_ny: int, nz: int) -> int:
    shapes = _stage_shapes(
        nx=tile_nx + 2 * GM1_SOURCE_STENCIL_RADIUS,
        ny=tile_ny + 2 * GM1_SOURCE_STENCIL_RADIUS,
        nz=nz,
    )
    return sum(math.prod(shape) * 4 for shape in shapes.values())


def _tile_local_source_bytes(*, tile_nx: int, tile_ny: int, nz: int) -> int:
    """One duplicate-free pinned tile including local getdivx sources."""

    return sum(
        math.prod(shape) * 4
        for shape in tile_host_buffer_shapes(
            tile_nx=tile_nx, tile_ny=tile_ny, nz=nz
        ).values()
    )


def _tile_local_device_bytes(*, tile_nx: int, tile_ny: int, nz: int) -> int:
    """One device tile plus the exact fx/fy/fz scalar-advection scratch."""

    return sum(
        math.prod(shape) * 4
        for shape in tile_device_buffer_shapes(
            tile_nx=tile_nx, tile_ny=tile_ny, nz=nz
        ).values()
    )


def _max_square_tile(
    *,
    nx: int,
    ny: int,
    nz: int,
    pressure_bytes: int,
    device_budget_bytes: int,
) -> tuple[int, int]:
    """Largest exact square interior whose two buffers fit with pressure."""

    lo, hi, best = 1, min(nx, ny) - 2 * GM1_SOURCE_STENCIL_RADIUS, 0
    while lo <= hi:
        middle = (lo + hi) // 2
        peak = (
            2 * _tile_local_device_bytes(tile_nx=middle, tile_ny=middle, nz=nz)
            + pressure_bytes
        )
        if peak <= device_budget_bytes:
            best, lo = middle, middle + 1
        else:
            hi = middle - 1
    return best, (
        2 * _tile_local_device_bytes(tile_nx=best, tile_ny=best, nz=nz) if best else 0
    )


def rtx5090_1500_memory_plan(
    *,
    host_total_bytes: int,
    host_available_bytes: int,
    device_total_bytes: int,
    nx: int = 1500,
    ny: int = 1500,
    nz: int = 75,
    tile_nx: int = 192,
    tile_ny: int = 192,
) -> dict[str, object]:
    """Exact first-seam capacity plan against measured RTX 5090 host resources."""

    for name, value in (
        ("host_total_bytes", host_total_bytes),
        ("host_available_bytes", host_available_bytes),
        ("device_total_bytes", device_total_bytes),
    ):
        _positive_int(name, value)
    plan = build_gm1_open_tile_plan(nx=nx, ny=ny, tile_nx=tile_nx, tile_ny=tile_ny)
    ring = build_gm1_ring_plan(plan)
    checkpoint_bytes = checkpoint_store_bytes(nx=nx, ny=ny, nz=nz)
    ring_bytes = checkpoint_ring_bytes(plan, ring, nz=nz)
    host_one_tile_bytes = _tile_local_source_bytes(
        tile_nx=tile_nx, tile_ny=tile_ny, nz=nz
    )
    device_one_tile_bytes = _tile_local_device_bytes(
        tile_nx=tile_nx, tile_ny=tile_ny, nz=nz
    )
    host_tile_buffers = 2 * host_one_tile_bytes
    device_tile_buffers = 2 * device_one_tile_bytes
    pressure = pressure_workspace_inventory(nx=nx, ny=ny, nz=nz)
    pinned_total = checkpoint_bytes + ring_bytes + host_tile_buffers
    reserve = 8 * GIB
    arwen_total_limit = math.floor(host_total_bytes * 0.47)
    arwen_pinned_limit = math.floor(host_total_bytes * 0.50)
    available_limit = max(0, host_available_bytes - reserve)
    admitted_limit = min(arwen_total_limit, arwen_pinned_limit, available_limit)
    device_peak = device_tile_buffers + int(pressure["nbytes"])
    result = {
        "schema": "gm1.hybrid-tilestream-memory-plan/v1",
        "domain": {"nx": nx, "ny": ny, "nz": nz, "dx_m": 10.0, "dy_m": 10.0},
        "resources": {
            "host_total_bytes": host_total_bytes,
            "host_available_bytes": host_available_bytes,
            "device_total_bytes": device_total_bytes,
            "host_reserve_bytes": reserve,
            "arwen_max_total_fraction": 0.47,
            "arwen_pinned_ceiling_fraction": 0.50,
            "admitted_pinned_limit_bytes": admitted_limit,
        },
        "host_pinned": {
            "checkpoint_carrier_count": len(CHECKPOINT_NAMES),
            "checkpoint_store_bytes": checkpoint_bytes,
            "ring_arena_bytes": ring_bytes,
            "double_buffer_stage_bytes": host_tile_buffers,
            "total_bytes": pinned_total,
            "fits": pinned_total <= admitted_limit,
        },
        "device": {
            "double_buffer_stage_bytes": device_tile_buffers,
            "domain_global_pressure_workspace_bytes": pressure["nbytes"],
            "sequential_peak_bytes": device_peak,
            "fits": device_peak <= device_total_bytes,
            "pressure_requires_tile_drain": True,
        },
        "tile_plan": {
            "tile_nx": tile_nx,
            "tile_ny": tile_ny,
            "compute_nx": plan.specs[0].cnx,
            "compute_ny": plan.specs[0].cny,
            "ntiles": plan.ntiles,
            "ownership_sha256": plan.ownership_sha256,
            "ring_schedule_sha256": ring.schedule_sha256,
        },
        "pressure_workspace": pressure,
        "resume": {
            "commit": GM1_RESUME_COMMIT,
            "module_sha256": GM1_RESUME_MODULE_SHA256,
            "operator_manifest_sha256": GM1_RESUME_OPERATOR_MANIFEST_SHA256,
        },
        "refusals": [
            "terrain+adaptive-imove",
            "MPI decomposition",
            "tile-local substitution for domain-global psolver3",
            "in-place local stage without ring arena save/patch bindings",
            "lossy compression or precision conversion",
        ],
    }
    result["plan_sha256"] = _sha256(_canonical_json(result))
    return result


def rtx5090_sizing_matrix(
    *,
    host_total_bytes: int,
    host_available_bytes: int,
    device_total_bytes: int,
    device_available_bytes: int,
    memlock_soft_bytes: int,
    memlock_hard_bytes: int,
    tile_nx: int = 192,
    tile_ny: int = 192,
) -> dict[str, object]:
    """Exact theoretical/practical matrix requested for the 10 m campaign.

    ``nz=100`` and ``nz=150`` are capacity calculations only: no official
    stretched profiles exist for those counts.  Likewise, 15 km / 4.5 m is
    non-integral; the row sizes the smallest enclosing 3334-cell grid but
    explicitly refuses it as an exact 15 km CM1 domain.
    """

    for name, value in (
        ("host_total_bytes", host_total_bytes),
        ("host_available_bytes", host_available_bytes),
        ("device_total_bytes", device_total_bytes),
        ("device_available_bytes", device_available_bytes),
        ("memlock_soft_bytes", memlock_soft_bytes),
        ("memlock_hard_bytes", memlock_hard_bytes),
    ):
        _positive_int(name, value)
    if host_available_bytes > host_total_bytes:
        raise ValueError("host available bytes cannot exceed total bytes")
    if device_available_bytes > device_total_bytes:
        raise ValueError("device available bytes cannot exceed total bytes")
    if memlock_soft_bytes > memlock_hard_bytes:
        raise ValueError("effective soft memlock cannot exceed the hard limit")

    reserve = 8 * GIB
    recommended_host = min(
        math.floor(host_total_bytes * 0.47), max(0, host_available_bytes - reserve)
    )
    absolute_host = max(0, host_available_bytes - reserve)
    recommended_device = min(
        math.floor(device_total_bytes * 0.80), device_available_bytes
    )
    absolute_device = device_available_bytes
    transfer_chunk_bytes = GM1_POST_STEP_TRANSFER_CHUNK_WORDS * 4
    streamed_double_buffer_bytes = 2 * transfer_chunk_bytes
    requested = [
        (15_000.0, 10.0),
        (15_000.0, 7.5),
        (15_000.0, 6.0),
        (15_000.0, 5.0),
        (15_000.0, 4.5),
        (30_000.0, 10.0),
    ]
    rows: list[dict[str, object]] = []
    horizontal_cache: dict[tuple[int, int], tuple[Gm1TilePlan, Gm1RingPlan]] = {}
    for extent_m, dx_m in requested:
        exact_count = extent_m / dx_m
        integral = float(exact_count).is_integer()
        nx = int(exact_count) if integral else math.ceil(exact_count)
        ny = nx
        cache_key = (nx, ny)
        if cache_key not in horizontal_cache:
            tile_plan = build_gm1_open_tile_plan(
                nx=nx, ny=ny, tile_nx=tile_nx, tile_ny=tile_ny
            )
            horizontal_cache[cache_key] = (
                tile_plan,
                build_gm1_ring_plan(tile_plan),
            )
        tile_plan, ring = horizontal_cache[cache_key]
        for nz in (75, 100, 150):
            checkpoint_bytes = checkpoint_store_bytes(nx=nx, ny=ny, nz=nz)
            ring_bytes = checkpoint_ring_bytes(tile_plan, ring, nz=nz)
            legacy_one_tile = _tile_stage_bytes(tile_nx=tile_nx, tile_ny=tile_ny, nz=nz)
            host_one_tile = _tile_local_source_bytes(
                tile_nx=tile_nx, tile_ny=tile_ny, nz=nz
            )
            device_one_tile = _tile_local_device_bytes(
                tile_nx=tile_nx, tile_ny=tile_ny, nz=nz
            )
            host_buffers = 2 * host_one_tile
            pinned_steady = checkpoint_bytes + ring_bytes + host_buffers
            current_stage_store = sum(
                math.prod(shape) * 4
                for shape in _stage_shapes(nx=nx, ny=ny, nz=nz).values()
            )
            current_executable_steady = (
                checkpoint_bytes
                + ring_bytes
                + 2 * legacy_one_tile
                + current_stage_store
            )
            checkpoint_staging = checkpoint_bytes
            render_staging = render_output_staging_bytes(nx=nx, ny=ny, nz=nz)
            checkpoint_peak = pinned_steady + checkpoint_staging
            output_peak = pinned_steady + render_staging
            practical_peak = max(checkpoint_peak, output_peak)
            streamed_peak = pinned_steady + streamed_double_buffer_bytes
            current_executable_streamed_peak = (
                current_executable_steady + streamed_double_buffer_bytes
            )
            pressure = pressure_workspace_inventory(nx=nx, ny=ny, nz=nz)
            pressure_bytes = int(pressure["nbytes"])
            device_peak = 2 * device_one_tile + pressure_bytes
            max_recommended_tile, max_recommended_buffers = _max_square_tile(
                nx=nx,
                ny=ny,
                nz=nz,
                pressure_bytes=pressure_bytes,
                device_budget_bytes=recommended_device,
            )
            max_absolute_tile, max_absolute_buffers = _max_square_tile(
                nx=nx,
                ny=ny,
                nz=nz,
                pressure_bytes=pressure_bytes,
                device_budget_bytes=absolute_device,
            )
            cfl_multiplier = max(50.0 / dx_m, nz / 75.0)
            rows.append(
                {
                    "requested_extent_m": extent_m,
                    "dx_m": dx_m,
                    "nx": nx,
                    "ny": ny,
                    "sized_extent_m": nx * dx_m,
                    "exact_integer_extent": integral,
                    "horizontal_grid_status": (
                        "CAPACITY_ONLY"
                        if integral
                        else "REFUSED_NONINTEGER_CELL_COUNT_NEAREST_ENCLOSING_SIZE_ONLY"
                    ),
                    "nz": nz,
                    "vertical_grid_status": (
                        "SEALED_75_LEVEL_CAPACITY_GEOMETRY_ONLY"
                        if nz == 75
                        else "REFUSED_UNSEALED_VERTICAL_PROFILE_CAPACITY_ONLY"
                    ),
                    "host": {
                        "checkpoint_store_bytes": checkpoint_bytes,
                        "ring_arena_bytes": ring_bytes,
                        "double_pinned_tile_buffers_bytes": host_buffers,
                        "pinned_steady_bytes": pinned_steady,
                        "current_scalar_stage_store_bytes": current_stage_store,
                        "current_executable_steady_bytes": current_executable_steady,
                        "checkpoint_staging_bytes": checkpoint_staging,
                        "render_13_carrier_staging_bytes": render_staging,
                        "checkpoint_peak_bytes": checkpoint_peak,
                        "render_peak_bytes": output_peak,
                        "practical_peak_bytes": practical_peak,
                        "streamed_checkpoint_output_double_buffer_bytes": (
                            streamed_double_buffer_bytes
                        ),
                        "streamed_peak_bytes": streamed_peak,
                        "current_executable_streamed_peak_bytes": (
                            current_executable_streamed_peak
                        ),
                        "effective_memlock_soft_bytes": memlock_soft_bytes,
                        "effective_memlock_hard_bytes": memlock_hard_bytes,
                        "pinned_execution_fit": pinned_steady <= memlock_soft_bytes,
                        "pinned_execution_status": (
                            "ADMITTED_EFFECTIVE_MEMLOCK"
                            if pinned_steady <= memlock_soft_bytes
                            else "REFUSED_EFFECTIVE_MEMLOCK_BELOW_STEADY_PINNED_STORE"
                        ),
                        "steady_recommended_headroom_fit": pinned_steady
                        <= recommended_host,
                        "steady_absolute_memory_fit": pinned_steady <= absolute_host,
                        "current_executable_steady_recommended_headroom_fit": (
                            current_executable_steady <= recommended_host
                        ),
                        "current_executable_steady_absolute_memory_fit": (
                            current_executable_steady <= absolute_host
                        ),
                        "streamed_recommended_headroom_fit": streamed_peak
                        <= recommended_host,
                        "streamed_absolute_memory_fit": streamed_peak <= absolute_host,
                        "current_executable_streamed_recommended_headroom_fit": (
                            current_executable_streamed_peak <= recommended_host
                        ),
                        "current_executable_streamed_absolute_memory_fit": (
                            current_executable_streamed_peak <= absolute_host
                        ),
                        "recommended_headroom_fit": practical_peak <= recommended_host,
                        "absolute_memory_fit": practical_peak <= absolute_host,
                    },
                    "device": {
                        "one_pinned_host_tile_bytes": host_one_tile,
                        "double_pinned_host_tile_bytes": 2 * host_one_tile,
                        "one_device_tile_bytes": device_one_tile,
                        "double_device_tile_bytes": 2 * device_one_tile,
                        "domain_global_pressure_workspace_bytes": pressure_bytes,
                        "planned_sequential_peak_bytes": device_peak,
                        "recommended_headroom_fit": device_peak <= recommended_device,
                        "absolute_memory_fit": device_peak <= absolute_device,
                        "max_recommended_square_tile_cells": max_recommended_tile,
                        "max_recommended_double_buffer_bytes": max_recommended_buffers,
                        "max_absolute_square_tile_cells": max_absolute_tile,
                        "max_absolute_double_buffer_bytes": max_absolute_buffers,
                    },
                    "tile_plan": {
                        "tile_nx": tile_nx,
                        "tile_ny": tile_ny,
                        "ntiles": tile_plan.ntiles,
                        "ownership_sha256": tile_plan.ownership_sha256,
                        "ring_schedule_sha256": ring.schedule_sha256,
                    },
                    "predicted_cfl": {
                        "baseline_dx_m": 50.0,
                        "baseline_dt_s": 0.5,
                        "step_count_multiplier": cfl_multiplier,
                        "candidate_dt_s": 0.5 / cfl_multiplier,
                        "status": "PREDICTION_ONLY_REQUIRES_COUPLED_STABILITY_GATE",
                    },
                }
            )
    result = {
        "schema": "gm1.hybrid-tilestream-sizing-matrix/v1",
        "resources": {
            "host_total_bytes": host_total_bytes,
            "host_available_bytes": host_available_bytes,
            "host_reserve_bytes": reserve,
            "recommended_host_limit_bytes": recommended_host,
            "absolute_host_limit_bytes": absolute_host,
            "device_total_bytes": device_total_bytes,
            "device_available_bytes": device_available_bytes,
            "recommended_device_limit_bytes": recommended_device,
            "absolute_device_limit_bytes": absolute_device,
            "effective_memlock_soft_bytes": memlock_soft_bytes,
            "effective_memlock_hard_bytes": memlock_hard_bytes,
        },
        "rows": rows,
        "interpretation": {
            "recommended": "peak fits ArWen 47% host and 80% device headroom policies",
            "absolute": "peak fits measured available RAM minus 8 GiB and physical VRAM; no operating headroom claim",
            "checkpoint_and_render_staging": "reported separately and assumed sequential; simultaneous staging is not admitted",
            "streamed_staging": (
                "capacity-only alternative using two exact 8 MiB chunks from the "
                "accepted post-step runner; publication from the pinned carrier "
                "store is not admitted until raw-bit replay and durability gates pass"
            ),
            "current_executable_stage_store": (
                "the first executable scalar seam still owns one full-domain "
                "padded scalar/flux/metric/tendency store in addition to the 37 "
                "persistent carriers; architectural steady-state rows exclude "
                "that transient duplication, while current_executable_* rows "
                "include it. Tile-local reconstruction/source-producer binding "
                "must remove the duplication before an architectural ceiling is "
                "an executable ceiling"
            ),
            "memlock": (
                "every row remains execution-refused unless the effective launch "
                "soft limit covers its complete steady pinned store and every "
                "cudaHostAlloc pointer is independently reported as host memory; "
                "pageable fallback is forbidden"
            ),
        },
        "streaming_authority": {
            "runner_commit": GM1_POST_STEP_STREAM_RUNNER_COMMIT,
            "runner_source_sha256": GM1_POST_STEP_STREAM_RUNNER_SHA256,
            "transfer_chunk_words": GM1_POST_STEP_TRANSFER_CHUNK_WORDS,
            "transfer_chunk_bytes": transfer_chunk_bytes,
            "double_buffer_bytes": streamed_double_buffer_bytes,
        },
    }
    result["matrix_sha256"] = _sha256(_canonical_json(result))
    return result


@dataclass(frozen=True, slots=True)
class CudaGm1TileStreamLaunch:
    ready_events: tuple[Any, ...]
    compiled: Any
    receipt: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class CudaGm1TileMassFluxLaunch:
    ready_event: Any
    compiled: Any
    receipt: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class CudaGm1TileLocalScalarLaunch:
    """One source-ordered tile-local getdivx -> scalar-advection enqueue."""

    ready_event: Any
    mass_flux: CudaGm1TileMassFluxLaunch
    compiled_scalar: Any
    receipt: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class CudaGm1TileLocalPdefLaunch:
    """One source-ordered tile-local getdivx -> final-RK pdef enqueue."""

    ready_event: Any
    mass_flux: CudaGm1TileMassFluxLaunch
    compiled_pdef: Any
    receipt: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class CudaGm1TileLocalPdefApplyLaunch:
    """One pdef tile followed by the exact final-RK q3d composition."""

    ready_event: Any
    pdef: CudaGm1TileLocalPdefLaunch
    compiled_apply: Any
    receipt: Mapping[str, Any]


def launch_cuda_gm1_tile_local_mass_flux(
    arrays: Mapping[str, Any],
    vertical: Any,
    *,
    ni: int,
    nj: int,
    nk: int,
    dx: float,
    dy: float,
    source_identity_sha256: str,
    tile_index: int,
    stream: Any | None = None,
    compiled: Any | None = None,
) -> CudaGm1TileMassFluxLaunch:
    """Rebuild getdivx carriers inside one tile using the sealed live kernels."""

    ni, nj, nk = (
        _positive_int("ni", ni),
        _positive_int("nj", nj),
        _positive_int("nk", nk),
    )
    for name, value in (("dx", dx), ("dy", dy)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be numeric")
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be finite and > 0")
    if (
        not isinstance(source_identity_sha256, str)
        or len(source_identity_sha256) != 64
        or any(value not in "0123456789abcdef" for value in source_identity_sha256)
    ):
        raise ValueError("source identity must be lowercase 64-hex sha256")
    if (
        isinstance(tile_index, bool)
        or not isinstance(tile_index, int)
        or tile_index < 0
    ):
        raise ValueError("tile_index must be a nonnegative integer")
    cupy = importlib.import_module("cupy")
    ng = CM1_OPEN_NGXY
    shapes = {
        "rru": (nk + 2, nj + 2 * ng, ni + 1 + 2 * ng),
        "rrv": (nk + 2, nj + 1 + 2 * ng, ni + 2 * ng),
        "rrw": (nk + 3, nj + 2 * ng, ni + 2 * ng),
        "divx": (nk, nj, ni),
        "rho0": (nk + 2, nj + 2 * ng, ni + 2 * ng),
        "rf0": (nk + 1, nj, ni),
        "rgzu": (nj, ni + 1),
        "rgzv": (nj + 1, ni),
        "dzdx": (nj, ni),
        "dzdy": (nj, ni),
        "gz": (nj, ni),
    }
    if set(arrays) < set(shapes):
        raise ValueError("tile-local mass-flux inventory is incomplete")
    checked: dict[str, Any] = {}
    for name, shape in shapes.items():
        value = arrays[name]
        if (
            not isinstance(value, cupy.ndarray)
            or value.dtype != cupy.float32
            or value.shape != shape
            or not value.flags.c_contiguous
        ):
            raise ValueError(f"tile-local CUDA {name} must be C-f32 {shape}")
        checked[name] = value
    if len({int(value.data.ptr) for value in checked.values()}) != len(checked):
        raise ValueError("tile-local mass-flux arrays must not alias")
    metrics_module = importlib.import_module("cm1core.cuda_gm1_child_stretched_metrics")
    if not isinstance(vertical, metrics_module.Gm1ChildStretchedMetrics):
        raise TypeError("vertical must be Gm1ChildStretchedMetrics")
    moist_module = importlib.import_module("cm1core.cuda_gm1_child_stretched_moist")
    if moist_module.GM1_CHILD_STRETCHED_MOIST_SOURCE_SHA256 != (
        GM1_STRETCHED_MOIST_KERNEL_SHA256
    ):
        raise RuntimeError("stretched getdivx source pin changed")
    compiled = compiled or moist_module.compile_cuda_gm1_child_stretched_moist()
    if compiled.target_arch != vertical.target_arch:
        raise ValueError(
            "tile-local getdivx and vertical images target different arches"
        )
    stream = stream or cupy.cuda.get_current_stream()
    horizontal = nk * nj * (ni + 1) + nk * (nj + 1) * ni
    vertical_words = (nk + 1) * nj * ni
    mass = nk * nj * ni
    counts = (horizontal, vertical_words, mass)
    threads = (256, 1, 1)
    grids = tuple(((count + 255) // 256, 1, 1) for count in counts)
    f32, u32 = cupy.float32, cupy.uint32
    _, horizontal_kernel, vertical_kernel, divergence_kernel = compiled.kernels
    with stream:
        horizontal_kernel(
            grids[0],
            threads,
            (
                checked["rho0"],
                checked["rgzu"],
                checked["rgzv"],
                checked["rru"],
                checked["rrv"],
                u32(ni),
                u32(nj),
                u32(nk),
                u32(ng),
            ),
            stream=stream,
        )
        vertical_kernel(
            grids[1],
            threads,
            (
                checked["rf0"],
                checked["dzdx"],
                checked["dzdy"],
                checked["gz"],
                vertical.sigma,
                vertical.sigmaf,
                vertical.rds,
                checked["rru"],
                checked["rrv"],
                checked["rrw"],
                u32(ni),
                u32(nj),
                u32(nk),
                u32(ng),
                f32(vertical.ztop_f32),
            ),
            stream=stream,
        )
        divergence_kernel(
            grids[2],
            threads,
            (
                checked["rru"],
                checked["rrv"],
                checked["rrw"],
                vertical.rdsf,
                checked["divx"],
                u32(ni),
                u32(nj),
                u32(nk),
                u32(ng),
                f32(dx),
                f32(dy),
                f32(1.0e-20),
            ),
            stream=stream,
        )
        event = cupy.cuda.Event(disable_timing=True)
        event.record(stream)
    receipt = {
        "schema": GM1_TILESTREAM_SOURCE_BINDING_ABI,
        "operation": "tile-local-source-ordered-getdivx",
        "tile_index": tile_index,
        "compute_shape_zyx": [nk, nj, ni],
        "source_identity_sha256": source_identity_sha256,
        "source": {
            "commit": GM1_STRETCHED_MOIST_COMMIT,
            "module_sha256": GM1_STRETCHED_MOIST_MODULE_SHA256,
            "kernel_sha256": GM1_STRETCHED_MOIST_KERNEL_SHA256,
            "anchors": "misclibs.F:222-370 getdivx horizontal/vertical/divergence",
        },
        "source_order": [
            "horizontal rho0-weighted rru/rrv",
            "terrain-following rf0-weighted rrw",
            "stretched divergence divx",
        ],
        "getdiv_theta_side_output_produced": False,
        "full_domain_transient_stage_store_used": False,
        "ready_event_recorded": True,
    }
    receipt["receipt_sha256"] = _sha256(_canonical_json(receipt))
    return CudaGm1TileMassFluxLaunch(event, compiled, MappingProxyType(receipt))


def launch_cuda_gm1_tile_local_scalar_advection5(
    arrays: Mapping[str, Any],
    vertical: Any,
    *,
    ni: int,
    nj: int,
    nk: int,
    nrk: int,
    dx: float,
    dy: float,
    weps: float,
    source_identity_sha256: str,
    tile_index: int,
    stream: Any | None = None,
    mass_flux_compiled: Any | None = None,
    scalar_compiled: Any | None = None,
) -> CudaGm1TileLocalScalarLaunch:
    """Enqueue the first two admitted local operators on one tile.

    This is intentionally not an RK stage.  It rebuilds the destructive
    ``rru/rrv/rrw/divx`` carriers from the staged physical winds and then
    evaluates one stretched fifth-order scalar-advection tendency.  The
    boundary/LBC band remains external to this function and only an owning
    driver may scatter the tile interior.
    """

    cupy = importlib.import_module("cupy")
    stream = stream or cupy.cuda.get_current_stream()
    mass_flux = launch_cuda_gm1_tile_local_mass_flux(
        arrays,
        vertical,
        ni=ni,
        nj=nj,
        nk=nk,
        dx=dx,
        dy=dy,
        source_identity_sha256=source_identity_sha256,
        tile_index=tile_index,
        stream=stream,
        compiled=mass_flux_compiled,
    )
    scalar_module = importlib.import_module(
        "cm1core.cuda_gm1_child_stretched_scalar"
    )
    scalar_compiled = (
        scalar_compiled
        or scalar_module.compile_cuda_gm1_child_stretched_scalar_advection()
    )
    scalar = scalar_module.launch_cuda_gm1_child_stretched_scalar_advection5(
        arrays["scalar"],
        {name: arrays[name] for name in ("rru", "rrv", "rrw")},
        {name: arrays[name] for name in ("c1", "c2", "rr0", "gz")},
        arrays["divx"],
        arrays["tendency"],
        {
            axis: arrays[name]
            for axis, name in (("x", "fx"), ("y", "fy"), ("z", "fz"))
        },
        vertical,
        ni=ni,
        nj=nj,
        nk=nk,
        nrk=nrk,
        dx=dx,
        dy=dy,
        weps=weps,
        stream=stream,
        compiled=scalar_compiled,
        allow_compact_validation=True,
    )
    receipt: dict[str, Any] = {
        "schema": GM1_TILESTREAM_SCHEMA,
        "abi": GM1_TILESTREAM_ABI,
        "operation": "tile-local-source-getdivx-scalar-advection5-tendency",
        "tile_index": tile_index,
        "compute_shape_zyx": [nk, nj, ni],
        "source_identity_sha256": source_identity_sha256,
        "source_order": [
            "getdivx horizontal rho0-weighted rru/rrv",
            "getdivx terrain-following rf0-weighted rrw",
            "getdivx stretched divergence divx",
            "adv.F stretched fifth-order scalar fluxes and tendency",
        ],
        "mass_flux_receipt_sha256": mass_flux.receipt["receipt_sha256"],
        "scalar_source_sha256": GM1_SCALAR_KERNEL_SHA256,
        "same_stream_no_host_sync": True,
        "ready_event_recorded": True,
        "scatter_disposition": (
            "caller may scatter only Gm1TileSpec.scalar_scatter(); "
            "radiative-open/LBC band is externally owned"
        ),
        "claim_limit": (
            "one tile-local getdivx plus scalar-advection tendency; no RK "
            "update, boundary-history mutation, pressure solve, or forecast"
        ),
    }
    receipt["receipt_sha256"] = _sha256(_canonical_json(receipt))
    return CudaGm1TileLocalScalarLaunch(
        scalar.ready_event,
        mass_flux,
        scalar_compiled,
        MappingProxyType(receipt),
    )


def launch_cuda_gm1_tile_local_pdef_moisture(
    arrays: Mapping[str, Any],
    vertical: Any,
    *,
    ni: int,
    nj: int,
    nk: int,
    dttmp: float,
    dx: float,
    dy: float,
    weps: float,
    species: int,
    source_identity_sha256: str,
    tile_index: int,
    stream: Any | None = None,
    mass_flux_compiled: Any | None = None,
    pdef_compiled: Any | None = None,
) -> CudaGm1TileLocalPdefLaunch:
    """Enqueue getdivx and the final-RK moisture pdef tendency only.

    The routine deliberately stops before the RK current-state update and
    before any boundary-history mutation, pdefq repair, or WSM6 call.  Thus the
    canonical ``q3d`` carrier stays read-only until a separately admitted ring
    save/patch lifecycle is supplied by the owning driver.
    """

    cupy = importlib.import_module("cupy")
    stream = stream or cupy.cuda.get_current_stream()
    mass_flux = launch_cuda_gm1_tile_local_mass_flux(
        arrays,
        vertical,
        ni=ni,
        nj=nj,
        nk=nk,
        dx=dx,
        dy=dy,
        source_identity_sha256=source_identity_sha256,
        tile_index=tile_index,
        stream=stream,
        compiled=mass_flux_compiled,
    )
    scalar_module = importlib.import_module(
        "cm1core.cuda_gm1_child_stretched_scalar"
    )
    pdef_compiled = (
        pdef_compiled or scalar_module.compile_cuda_gm1_child_stretched_scalar_pdef()
    )
    pdef = scalar_module.launch_cuda_gm1_child_stretched_pdef_moisture(
        arrays["moisture_a"],
        arrays["scalar"],
        {name: arrays[name] for name in ("rru", "rrv", "rrw")},
        {
            name: arrays[name]
            for name in ("c1", "c2", "rho0_3d", "rr0", "gz", "rgz")
        },
        arrays["divx"],
        {
            axis: arrays[name]
            for axis, name in (("x", "fx"), ("y", "fy"), ("z", "fz"))
        },
        arrays["advx"],
        arrays["advy"],
        arrays["advz"],
        arrays["candidate"],
        arrays["tendency"],
        vertical,
        ni=ni,
        nj=nj,
        nk=nk,
        dttmp=dttmp,
        dx=dx,
        dy=dy,
        weps=weps,
        species=species,
        stream=stream,
        compiled=pdef_compiled,
        allow_compact_validation=True,
    )
    receipt: dict[str, Any] = {
        "schema": GM1_TILESTREAM_SCHEMA,
        "abi": "gm1.hybrid-tilestream-final-rk-pdef-tendency-f32/v1",
        "operation": "tile-local-source-getdivx-final-rk-moisture-pdef",
        "tile_index": tile_index,
        "compute_shape_zyx": [nk, nj, ni],
        "species": species,
        "source_identity_sha256": source_identity_sha256,
        "source_order": [
            "getdivx horizontal/vertical/divergence",
            "solve2.F:1430-1533 final-RK moisture pdef flux/candidate/correction/tendency",
        ],
        "mass_flux_receipt_sha256": mass_flux.receipt["receipt_sha256"],
        "pdef_source_sha256": scalar_module.GM1_CHILD_STRETCHED_SCALAR_PDEF_SOURCE_SHA256,
        "same_stream_no_host_sync": True,
        "ready_event_recorded": True,
        "q3d_inplace_apply_performed": False,
        "pdefq_or_wsm6_performed": False,
        "pressure_workspace_disposition": "domain-global stage must be drained separately",
        "claim_limit": (
            "final-RK moisture pdef tendency only; no q3d update, halo/history "
            "mutation, pdefq, WSM6, pressure solve, or full RK-stage claim"
        ),
    }
    receipt["receipt_sha256"] = _sha256(_canonical_json(receipt))
    return CudaGm1TileLocalPdefLaunch(
        pdef.ready_event,
        mass_flux,
        pdef_compiled,
        MappingProxyType(receipt),
    )


def launch_cuda_gm1_tile_local_pdef_apply_moisture(
    arrays: Mapping[str, Any],
    vertical: Any,
    *,
    ni: int,
    nj: int,
    nk: int,
    dttmp: float,
    dx: float,
    dy: float,
    weps: float,
    species: int,
    source_identity_sha256: str,
    tile_index: int,
    stream: Any | None = None,
    mass_flux_compiled: Any | None = None,
    pdef_compiled: Any | None = None,
) -> CudaGm1TileLocalPdefApplyLaunch:
    """Enqueue getdivx -> pdef -> ``qa+dttmp*(qten+sten)``.

    Only the physical q3d tile result is produced.  The caller must save old
    ring values before launch and may publish only ``owned_rect('mass')``.
    Pdefq/WSM6 follow this result, while solve3 scalar boundary fills remain a
    later source-ordered stage.
    """

    cupy = importlib.import_module("cupy")
    rk_module = importlib.import_module("cm1core.cuda_scalar_moist_rk")
    stream = stream or cupy.cuda.get_current_stream()
    ni, nj, nk = (
        _positive_int("ni", ni),
        _positive_int("nj", nj),
        _positive_int("nk", nk),
    )
    physical = (nk, nj, ni)
    moisture_held = rk_module._require_cuda_f32(
        cupy,
        arrays["moisture_held"],
        "moisture_held",
        physical,
        writable=False,
    )
    moisture_current = rk_module._require_cuda_f32(
        cupy,
        arrays["moisture_current"],
        "moisture_current",
        physical,
    )
    protected = (
        arrays["moisture_a"],
        arrays["scalar"],
        arrays["tendency"],
        moisture_held,
        moisture_current,
    )
    if len({int(value.data.ptr) for value in protected}) != len(protected):
        raise ValueError("pdef/apply A, held, scalar, tendency, and current must not alias")
    pdef = launch_cuda_gm1_tile_local_pdef_moisture(
        arrays,
        vertical,
        ni=ni,
        nj=nj,
        nk=nk,
        dttmp=dttmp,
        dx=dx,
        dy=dy,
        weps=weps,
        species=species,
        source_identity_sha256=source_identity_sha256,
        tile_index=tile_index,
        stream=stream,
        mass_flux_compiled=mass_flux_compiled,
        pdef_compiled=pdef_compiled,
    )
    words = ni * nj * nk
    threads_count = rk_module.CUDA_SCALAR_MOIST_RK_THREADS
    grid = ((words + threads_count - 1) // threads_count, 1, 1)
    threads = (threads_count, 1, 1)
    with stream:
        pdef.compiled_pdef.kernels[3](
            grid,
            threads,
            (
                arrays["moisture_a"],
                moisture_held,
                arrays["tendency"],
                moisture_current,
                cupy.uint64(words),
                cupy.float32(dttmp),
            ),
            stream=stream,
        )
        event = cupy.cuda.Event(disable_timing=True)
        event.record(stream)
    receipt: dict[str, Any] = {
        "schema": GM1_TILESTREAM_SCHEMA,
        "abi": "gm1.hybrid-tilestream-final-rk-pdef-apply-f32/v1",
        "operation": "tile-local-source-getdivx-final-rk-pdef-q3d-apply",
        "tile_index": tile_index,
        "species": species,
        "compute_shape_zyx": [nk, nj, ni],
        "source_identity_sha256": source_identity_sha256,
        "source_order": [
            "solve2.F:1430-1523 pdef tendency",
            "solve2.F:1524-1526 qa+dttmp*(qten+sten) and smeps zero",
            "caller ring save/patch then unique physical owned scatter",
            "cm1.F:3405-3427 pdefq/WSM6 is downstream",
            "solve3.F:242-260 scalar bcs is downstream",
        ],
        "pdef_receipt_sha256": pdef.receipt["receipt_sha256"],
        "apply_source_sha256": rk_module.CUDA_SCALAR_MOIST_RK_SOURCE_SHA256,
        "dttmp_f32": float(cupy.float32(dttmp)),
        "same_stream_no_host_sync": True,
        "ready_event_recorded": True,
        "q3d_physical_tile_apply_performed": True,
        "pdefq_or_wsm6_performed": False,
        "solve3_boundary_or_open_history_performed": False,
        "pressure_workspace_disposition": "domain-global stage remains separately drained",
        "claim_limit": (
            "one physical q3d final-RK tile result through pdef apply; no "
            "pdefq, WSM6, solve3 halo/history, pressure, or full RK-stage claim"
        ),
    }
    receipt["receipt_sha256"] = _sha256(_canonical_json(receipt))
    return CudaGm1TileLocalPdefApplyLaunch(
        event,
        pdef,
        pdef.compiled_pdef,
        MappingProxyType(receipt),
    )


@dataclass(slots=True)
class _TileBuffer:
    host: dict[str, Any]
    device: dict[str, Any]
    copy_in: Any
    compute: Any
    copy_out: Any
    pending_spec: Gm1TileSpec | None = None
    gather_event: Any | None = None
    step_event: Any | None = None
    scatter_event: Any | None = None


def _fill_tile_host(
    buffer: _TileBuffer, store: Gm1PinnedHostCarrierStore, spec: Gm1TileSpec
) -> None:
    np = importlib.import_module("numpy")
    source, target = store.arrays, buffer.host
    y0, x0, cny, cnx = spec.cj0, spec.ci0, spec.cny, spec.cnx
    ng = CM1_OPEN_NGXY
    np.copyto(
        target["scalar"],
        source["scalar"][:, y0 : y0 + cny + 2 * ng, x0 : x0 + cnx + 2 * ng],
    )
    np.copyto(
        target["rru"],
        source["rru"][:, y0 : y0 + cny + 2 * ng, x0 : x0 + cnx + 1 + 2 * ng],
    )
    np.copyto(
        target["rrv"],
        source["rrv"][:, y0 : y0 + cny + 1 + 2 * ng, x0 : x0 + cnx + 2 * ng],
    )
    np.copyto(
        target["rrw"], source["rrw"][:, y0 : y0 + cny + 2 * ng, x0 : x0 + cnx + 2 * ng]
    )
    for name in ("divx", "c1", "c2", "rr0", "gz", "tendency"):
        np.copyto(target[name], source[name][..., y0 : y0 + cny, x0 : x0 + cnx])


def _scatter_tile_host(buffer: _TileBuffer, store: Gm1PinnedHostCarrierStore) -> None:
    spec = buffer.pending_spec
    if spec is None:
        return
    scatter = spec.scalar_scatter()
    if scatter is not None:
        full, tile = scatter
        store.arrays["tendency"][(slice(None), *full)] = buffer.host["tendency"][
            (slice(None), *tile)
        ]
    buffer.pending_spec = None


def launch_cuda_gm1_tiled_scalar_advection5(
    store: Gm1PinnedHostCarrierStore,
    vertical: Any,
    *,
    nrk: int,
    dx: float,
    dy: float,
    weps: float,
    tile_nx: int = GM1_DEFAULT_TILE,
    tile_ny: int = GM1_DEFAULT_TILE,
    compiled: Any | None = None,
    inplace: bool = False,
    pressure_stage: bool = False,
    cuda_graphs: bool = False,
) -> CudaGm1TileStreamLaunch:
    """Run the first exact GM1 local stage through pinned double buffers."""

    if not isinstance(store, Gm1PinnedHostCarrierStore):
        raise TypeError("store must be Gm1PinnedHostCarrierStore")
    if inplace:
        raise ValueError(
            "in-place tiling requires a bound exact ring arena save/patch implementation"
        )
    if pressure_stage:
        raise ValueError(
            "CM1 psolver3 is domain-global and cannot be substituted by a tile-local stage"
        )
    if cuda_graphs:
        raise ValueError(
            "CUDA graph capture is not admitted until event-record nodes are replay-proven"
        )
    if (store.nz, store.ny, store.nx) != GM1_PRODUCTION_SHAPE:
        raise ValueError("production GM1 tilestream gate requires exactly 300x300x75")
    if (float(dx), float(dy)) != GM1_PRODUCTION_DX_DY:
        raise ValueError("production GM1 tilestream requires dx=dy=50 m")
    if isinstance(nrk, bool) or not isinstance(nrk, int) or nrk not in (1, 2, 3):
        raise ValueError("nrk must be 1, 2, or 3")
    pinning = store.assert_pinned()
    plan = build_gm1_open_tile_plan(
        nx=store.nx, ny=store.ny, tile_nx=tile_nx, tile_ny=tile_ny
    )
    ring = build_gm1_ring_plan(plan)

    cupy = importlib.import_module("cupy")
    scalar_module = importlib.import_module("cm1core.cuda_gm1_child_stretched_scalar")
    compiled = (
        compiled or scalar_module.compile_cuda_gm1_child_stretched_scalar_advection()
    )
    tile_shapes = _tile_stage_shapes(plan, nz=store.nz)
    buffers: list[_TileBuffer] = []
    for _ in range(2):
        host = {
            name: allocate_exact_pinned_array(shape)
            for name, shape in tile_shapes.items()
        }
        device = {
            name: cupy.empty(shape, dtype=cupy.float32)
            for name, shape in tile_shapes.items()
        }
        device.update(
            {
                "fx": cupy.empty(tile_shapes["rru"], dtype=cupy.float32),
                "fy": cupy.empty(tile_shapes["rrv"], dtype=cupy.float32),
                "fz": cupy.empty(tile_shapes["rrw"], dtype=cupy.float32),
            }
        )
        buffers.append(
            _TileBuffer(
                host,
                device,
                cupy.cuda.Stream(non_blocking=True),
                cupy.cuda.Stream(non_blocking=True),
                cupy.cuda.Stream(non_blocking=True),
            )
        )

    event_rows: list[dict[str, object]] = []
    for index, spec in enumerate(plan.specs):
        buffer_index = index % 2
        buffer = buffers[buffer_index]
        if buffer.scatter_event is not None:
            buffer.scatter_event.synchronize()
            _scatter_tile_host(buffer, store)
        _fill_tile_host(buffer, store, spec)
        with buffer.copy_in:
            for name in STAGE_STORE_SHAPES:
                buffer.device[name].set(buffer.host[name], stream=buffer.copy_in)
            gather_event = cupy.cuda.Event(disable_timing=True)
            gather_event.record(buffer.copy_in)
        buffer.compute.wait_event(gather_event)
        launch = scalar_module.launch_cuda_gm1_child_stretched_scalar_advection5(
            buffer.device["scalar"],
            {name: buffer.device[name] for name in ("rru", "rrv", "rrw")},
            {name: buffer.device[name] for name in ("c1", "c2", "rr0", "gz")},
            buffer.device["divx"],
            buffer.device["tendency"],
            {
                "x": buffer.device["fx"],
                "y": buffer.device["fy"],
                "z": buffer.device["fz"],
            },
            vertical,
            ni=spec.cnx,
            nj=spec.cny,
            nk=store.nz,
            nrk=nrk,
            dx=dx,
            dy=dy,
            weps=weps,
            stream=buffer.compute,
            compiled=compiled,
            allow_compact_validation=True,
        )
        buffer.copy_out.wait_event(launch.ready_event)
        with buffer.copy_out:
            buffer.device["tendency"].get(
                out=buffer.host["tendency"], stream=buffer.copy_out, blocking=False
            )
            scatter_event = cupy.cuda.Event(disable_timing=True)
            scatter_event.record(buffer.copy_out)
        buffer.pending_spec = spec
        buffer.gather_event = gather_event
        buffer.step_event = launch.ready_event
        buffer.scatter_event = scatter_event
        event_rows.append(
            {
                "tile": index,
                "buffer": buffer_index,
                "copy_in_stream": int(buffer.copy_in.ptr),
                "compute_stream": int(buffer.compute.ptr),
                "copy_out_stream": int(buffer.copy_out.ptr),
                "edges": ["buffer-reuse->gather", "gather->step", "step->scatter"],
            }
        )
    for buffer in buffers:
        if buffer.scatter_event is not None:
            buffer.scatter_event.synchronize()
            _scatter_tile_host(buffer, store)

    receipt: dict[str, Any] = {
        "schema": GM1_TILESTREAM_SCHEMA,
        "abi": GM1_TILESTREAM_ABI,
        "status": "PASS_ENQUEUED_AND_SCATTERED",
        "domain": {
            "nx": store.nx,
            "ny": store.ny,
            "nz": store.nz,
            "dx_m": dx,
            "dy_m": dy,
        },
        "selection": {
            "terrain": True,
            "imove": 0,
            "nproc": 1,
            "stretch_z": 1,
            "boundaries": [2, 2, 2, 2],
            "nrk": nrk,
        },
        "operator": {
            "name": "stretched-scalar-advection5-tendency",
            "module_sha256": GM1_SCALAR_MODULE_SHA256,
            "kernel_sha256": GM1_SCALAR_KERNEL_SHA256,
            "source_stencil_radius": GM1_SOURCE_STENCIL_RADIUS,
            "boundary_band_owner": "existing radiative-open/LBC stage; not scattered here",
        },
        "host_store": {"abi": GM1_TILESTREAM_STORE_ABI, **pinning},
        "tile_plan": {
            "tile_nx": tile_nx,
            "tile_ny": tile_ny,
            "compute_nx": plan.specs[0].cnx,
            "compute_ny": plan.specs[0].cny,
            "ntiles": plan.ntiles,
            "ownership_sha256": plan.ownership_sha256,
        },
        "ring": {
            "schedule_sha256": ring.schedule_sha256,
            "cells_by_variant": dict(ring.ring_cells_by_variant),
            "mask_sha256": dict(ring.ring_mask_sha256),
            "runtime_disposition": "not-needed-for-immutable-input/disjoint-output stage",
            "inplace_refused": True,
        },
        "pipeline": {
            "persistent_device_buffers": 2,
            "streams_per_buffer": 3,
            "multistream_overlap": True,
            "event_order": event_rows,
            "cuda_graphs": "REFUSED_PENDING_EVENT_NODE_REPLAY_PROOF",
        },
        "domain_global_pressure": pressure_workspace_inventory(
            nx=store.nx, ny=store.ny, nz=store.nz
        ),
        "arwen_authority": {
            "commit": ARWEN_TILESTREAM_COMMIT,
            "path_commit": ARWEN_TILESTREAM_PATH_COMMIT,
            "source_sha256": dict(ARWEN_TILESTREAM_SOURCE_SHA256),
            "immutable_external_checkout": True,
            "diverged_in_pinned_estate": list(ARWEN_TILESTREAM_DIVERGED_IN_PINNED_ESTATE),
        },
        "claim_limit": "one scalar-advection tendency stage; no RK update, pressure solve, full step, in-place ring execution, forecast, or performance claim",
    }
    receipt["receipt_sha256"] = _sha256(_canonical_json(receipt))
    return CudaGm1TileStreamLaunch(
        tuple(
            buffer.scatter_event
            for buffer in buffers
            if buffer.scatter_event is not None
        ),
        compiled,
        MappingProxyType(receipt),
    )


def require_tile_local_stage(stage: str) -> None:
    """Fail closed on stage substitution beyond the first admitted seam."""

    if stage != "stretched-scalar-advection5-tendency":
        raise ValueError(
            "only stretched-scalar-advection5-tendency is admitted; pressure, "
            "RK update, boundary history, reductions, and physics remain source-owned"
        )


__all__ = (
    "ARWEN_TILESTREAM_COMMIT",
    "ARWEN_TILESTREAM_DIVERGED_IN_PINNED_ESTATE",
    "ARWEN_TILESTREAM_PATH_COMMIT",
    "ARWEN_TILESTREAM_SOURCE_SHA256",
    "CHECKPOINT_NAMES",
    "CudaGm1TileLocalPdefApplyLaunch",
    "CudaGm1TileLocalScalarLaunch",
    "CudaGm1TileLocalPdefLaunch",
    "CudaGm1TileMassFluxLaunch",
    "CudaGm1TileStreamLaunch",
    "GM1_TILESTREAM_ABI",
    "GM1_TILESTREAM_PRESSURE_ABI",
    "GM1_TILESTREAM_SCHEMA",
    "GM1_TILESTREAM_SOURCE_BINDING_ABI",
    "GM1_TILESTREAM_STORE_ABI",
    "Gm1CanonicalHostCarrierStore",
    "Gm1PinnedHostCarrierStore",
    "Gm1ScalarRingArena",
    "Gm1RingPlan",
    "Gm1TileLocalSourceBinding",
    "Gm1TilePlan",
    "Gm1TileSpec",
    "allocate_exact_pinned_array",
    "build_gm1_open_tile_plan",
    "build_gm1_ring_plan",
    "checkpoint_carrier_shape",
    "checkpoint_ring_bytes",
    "checkpoint_store_bytes",
    "fill_gm1_tile_local_source",
    "fill_gm1_tile_local_pdef_apply_source",
    "fill_gm1_tile_local_pdef_source",
    "launch_cuda_gm1_tile_local_mass_flux",
    "launch_cuda_gm1_tile_local_pdef_apply_moisture",
    "launch_cuda_gm1_tile_local_pdef_moisture",
    "launch_cuda_gm1_tile_local_scalar_advection5",
    "launch_cuda_gm1_tiled_scalar_advection5",
    "rtx5090_1500_memory_plan",
    "rtx5090_sizing_matrix",
    "pressure_workspace_inventory",
    "pressure_workspace_shapes",
    "render_output_staging_bytes",
    "require_tile_local_stage",
    "scatter_gm1_tile_local_pdef_apply",
    "tile_device_buffer_shapes",
    "tile_host_buffer_shapes",
    "tile_pdef_apply_device_buffer_shapes",
    "tile_pdef_apply_host_buffer_shapes",
    "tile_pdef_device_buffer_shapes",
    "tile_pdef_host_buffer_shapes",
)
