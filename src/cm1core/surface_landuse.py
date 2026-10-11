"""Source-exact CM1 r21.1 ``getlanduse`` on the packaged LANDUSE.TBL.

``src/init_surface.F:239-243`` calls ``getlanduse`` for every run with a
surface layer or a semi-slip bottom, and ``src/init_surface.F:444-466`` then
**overwrites** the roughness length, moisture availability, emissivity,
thermal inertia, albedo and the land/water flag from the land-use category.
This matters more than it looks: it runs *after* the ``initsfc`` branches, so
``initsfc=3``'s rough-west/smooth-east roughness assignment
(``src/init_surface.F:97-107``) does not survive into the run unless the
land-use categories differ across the domain.  A port that skipped this call
would produce a west/east roughness contrast CM1 does not produce.

``src/init_surface.F:246-254`` then re-imposes ``cnst_znt`` for ``set_znt=1``,
after the table, which is why the prescribed-roughness branch is the one that
does reach the surface.

CM1 reads the table from the working directory beside the executable.  This
port reads the same numbers from a record whose bytes are pinned to that
file, so the model needs no working directory.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from collections.abc import Mapping, MutableMapping
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any

from cm1core.errors import refuse


CM1_LANDUSE_SCHEMA = "gpuwm-cm1.landuse/v1"
CM1_LANDUSE_RECORD = "CM1-LANDUSE-TBL.json"
# The bytes of CM1 r21.1's own run/LANDUSE.TBL the packaged record was built
# from; tools/build_cm1_landuse_table.py records the same digest.
CM1_LANDUSE_TBL_SHA256 = (
    "92fef4134ac35dca6177e5afc79a1d0517383e9061cf113629d4d41755b88806"
)
CM1_LANDUSE_SOURCE_ANCHORS = MappingProxyType(
    {
        "documentation": "README.namelist:1003-1015",
        "call_site": "src/init_surface.F:239-243",
        "table_read": "src/init_surface.F:397-408",
        "category_assignment": "src/init_surface.F:441-466",
        "water_category": "src/init_surface.F:380",
        "prescribed_roughness_after_table": "src/init_surface.F:246-254",
        "water_moisture_availability": "src/init_surface.F:277",
        "xland_reality_check": "src/init_surface.F:279-289",
        "skin_temperature_reality_check": "src/init_surface.F:301-309",
    }
)
CM1_LANDUSE_CLAIM_LIMIT = (
    "CM1 r21.1 init_surface.F:441-466 getlanduse category lookup and the "
    "init_surface.F:270-322 reality checks on supplied compact carriers, from "
    "the packaged run/LANDUSE.TBL record; no file read from a working "
    "directory, no MPI broadcast, no soil model, no ocean model and no "
    "coupled-step claim"
)
# src/init_surface.F:455-459.  CM1 divides the table's albedo and thermal
# inertia by 100 and its roughness by 100 (the table is in per-cent and
# centimetres); the rest are used as read.
_PERCENT_COLUMNS = MappingProxyType(
    {"albedo": "albd", "thc": "therin", "znt": "sfz0"}
)


@lru_cache(maxsize=1)
def landuse_table() -> Mapping[str, Any]:
    """The packaged table, checked against the authority file it came from."""

    path = Path(__file__).with_name("records") / CM1_LANDUSE_RECORD
    record = json.loads(path.read_text())
    if record.get("schema") != "gm1.cm1-landuse-table/v1":
        raise RuntimeError("packaged land-use record carries another schema")
    if record["authority"]["sha256"] != CM1_LANDUSE_TBL_SHA256:
        raise RuntimeError(
            "packaged land-use record was not built from CM1 r21.1's "
            "run/LANDUSE.TBL"
        )
    if record["max_categories"] != 33 or record["max_seasons"] != 2:
        raise RuntimeError("land-use record does not have CM1's 33x2 shape")
    return MappingProxyType(record)


def landuse_record_sha256() -> str:
    path = Path(__file__).with_name("records") / CM1_LANDUSE_RECORD
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cpu_getlanduse_inplace(
    arrays: MutableMapping[str, Any],
    *,
    ni: int,
    nj: int,
    season: int,
) -> dict[str, object]:
    """``src/init_surface.F:441-466``, in place, on supplied carriers.

    ``lu_index`` selects the row; category 0 means "no data" and is read as
    water (``src/init_surface.F:447-450``).  A category whose tabulated
    albedo is not positive is CM1's own stop (``src/init_surface.F:451-454``);
    this port refuses instead, naming the field that would otherwise be left
    at whatever the allocator had.
    """

    np = importlib.import_module("numpy")
    if season not in (1, 2):
        refuse(
            code="CM1-LANDUSE-SEASON",
            gate="GM1-PHYSICS-SUPPORT",
            option="season",
            value=season,
            reason=(
                "LANDUSE.TBL carries exactly two seasons (init_surface.F:378, "
                ":442); any other index reads past the table, so the "
                "roughness length, albedo and moisture availability of every "
                "column would come from unallocated memory"
            ),
            remedy="set season to 1 (summer) or 2 (winter)",
        )
    record = landuse_table()
    water = int(record["water_category"])
    rows = record["seasons"][season - 1]
    for name in ("lu_index", "xland", "znt", "mavail", "emiss", "thc", "albedo"):
        if name not in arrays:
            raise ValueError(f"missing land-use carrier: {name}")
        value = arrays[name]
        if tuple(value.shape) != (nj, ni):
            raise ValueError(f"{name} must have shape {(nj, ni)}")
    lookup = {}
    for index, row in enumerate(rows, start=1):
        if not row["albd"] > 0.0:
            continue
        lookup[index] = (
            np.float32(np.float32(row["albd"]) / np.float32(100.0)),
            np.float32(np.float32(row["therin"]) / np.float32(100.0)),
            np.float32(row["sfem"]),
            np.float32(np.float32(row["sfz0"]) / np.float32(100.0)),
            np.float32(row["slmo"]),
        )
    categories = np.asarray(arrays["lu_index"]).astype(np.int64, copy=False)
    used = sorted({int(value) for value in np.unique(categories)})
    for category in used:
        resolved = water if category == 0 else category
        if resolved not in lookup:
            refuse(
                code="CM1-LANDUSE-CATEGORY",
                gate="GM1-PHYSICS-SUPPORT",
                option="lu_index",
                value=category,
                reason=(
                    "init_surface.F:451-454 stops CM1 when a category has no "
                    "row in LANDUSE.TBL, because the roughness length, "
                    "moisture availability and albedo of every column with "
                    "that category would otherwise stay at whatever the "
                    "allocator left there, and the surface layer would build "
                    "its drag from it"
                ),
                remedy=(
                    "use a LANDUSE.TBL category in [1, 33] with a positive "
                    "tabulated albedo"
                ),
            )
    for category in used:
        resolved = water if category == 0 else category
        albedo, thc, emiss, znt, mavail = lookup[resolved]
        mask = categories == category
        arrays["albedo"][mask] = albedo
        arrays["thc"][mask] = thc
        arrays["emiss"][mask] = emiss
        arrays["znt"][mask] = znt
        arrays["mavail"][mask] = mavail
        arrays["xland"][mask] = np.float32(
            2.0 if resolved == water else 1.0
        )
    return {
        "applied": True,
        "season": season,
        "categories_used": used,
        "water_category": water,
        "record_sha256": landuse_record_sha256(),
        "authority_sha256": CM1_LANDUSE_TBL_SHA256,
        "source_anchors": dict(CM1_LANDUSE_SOURCE_ANCHORS),
    }


def cpu_surface_reality_checks(
    arrays: Mapping[str, Any],
    *,
    sfcmodel: int,
) -> dict[str, object]:
    """``src/init_surface.F:270-322``, as refusals rather than prints.

    CM1 forces ``mavail=1`` over water first (``:277``), then stops on an
    ``xland`` that is neither land nor water and on a skin temperature below
    100 K.  ``lu_index`` and ``tmn`` are checked only for the external
    surface models, exactly as the source does.
    """

    np = importlib.import_module("numpy")
    water = np.asarray(arrays["xland"]) > np.float32(1.5)
    arrays["mavail"][water] = np.float32(1.0)
    rounded = np.rint(np.asarray(arrays["xland"])).astype(np.int64)
    invalid = (rounded <= 0) | (rounded >= 3)
    if bool(np.any(invalid)):
        refuse(
            code="CM1-SURFACE-XLAND-RANGE",
            gate="GM1-PHYSICS-SUPPORT",
            option="xland",
            value=float(np.asarray(arrays["xland"])[invalid][0]),
            reason=(
                "init_surface.F:279-289 stops CM1 unless every column is land "
                "(1) or water (2); getcecd branches on xland>1.5 "
                "(sfcphys.F:141), so a third value silently takes the land "
                "branch and the run reports a water surface it never "
                "integrated"
            ),
            remedy="set xland0 to 1.0 for land or 2.0 for water",
        )
    cold = np.asarray(arrays["tsk"]) < np.float32(100.0)
    if bool(np.any(cold)):
        refuse(
            code="CM1-SURFACE-TSK-RANGE",
            gate="GM1-PHYSICS-SUPPORT",
            option="tsk0",
            value=float(np.asarray(arrays["tsk"])[cold][0]),
            reason=(
                "init_surface.F:301-309 stops CM1 below 100 K; sfcflux "
                "divides the skin temperature by the surface Exner function "
                "(sfcphys.F:341) and takes its saturation mixing ratio "
                "(cm1libs.F:38), and below 100 K that saturation pressure "
                "underflows to zero, so the moisture flux is the full "
                "lowest-level mixing ratio drawn into the ground every step"
            ),
            remedy="declare a physical skin temperature in kelvin",
        )
    external = sfcmodel in (2, 3, 4, 6, 7)
    if external:
        categories = np.asarray(arrays["lu_index"]).astype(np.int64, copy=False)
        if bool(np.any((categories <= 0) | (categories >= 34))):
            refuse(
                code="CM1-SURFACE-LU-INDEX-RANGE",
                gate="GM1-PHYSICS-SUPPORT",
                option="lu0",
                value=int(categories.min()),
                reason=(
                    "init_surface.F:290-300 stops CM1 outside [1,33] for the "
                    "external surface models, which index LANDUSE.TBL "
                    "directly rather than through getlanduse"
                ),
                remedy="use a LANDUSE.TBL category in [1, 33]",
            )
        if bool(np.any(np.asarray(arrays["tmn"]) < np.float32(100.0))):
            refuse(
                code="CM1-SURFACE-TMN-RANGE",
                gate="GM1-PHYSICS-SUPPORT",
                option="tmn0",
                value=float(np.asarray(arrays["tmn"]).min()),
                reason=(
                    "init_surface.F:310-318 stops CM1 below 100 K; the soil "
                    "slab interpolates between the skin and deep-layer "
                    "temperatures (init_surface.F:160-166), so a deep layer "
                    "below 100 K drives the whole soil column below it"
                ),
                remedy="declare a physical deep-soil temperature in kelvin",
            )
    return {
        "applied": True,
        "water_columns": int(np.count_nonzero(water)),
        "external_surface_model_checks": bool(external),
    }


__all__ = [
    "CM1_LANDUSE_CLAIM_LIMIT",
    "CM1_LANDUSE_SCHEMA",
    "CM1_LANDUSE_SOURCE_ANCHORS",
    "CM1_LANDUSE_TBL_SHA256",
    "cpu_getlanduse_inplace",
    "cpu_surface_reality_checks",
    "landuse_record_sha256",
    "landuse_table",
]
