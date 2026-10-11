"""Non-overridable identities for the declared fixed profiles.

The installed profile manifest is the single source of truth. Keeping the
identity in a package record (rather than duplicating it in Python constants)
makes the wheel itself auditable and prevents code/record drift.
"""

from __future__ import annotations

import copy
import json
from importlib import resources

from .config_types import LaneName


def _load_manifest() -> dict[str, object]:
    record = resources.files("cm1core.records").joinpath("CM1-PROFILE-MANIFEST.json")
    return json.loads(record.read_text(encoding="utf-8"))


_MANIFEST = _load_manifest()
_OFFICIAL = _MANIFEST["official_supercell"]
_REAL = _MANIFEST["real_atmosphere_100m"]
_CARTESIAN = _MANIFEST["cartesian_runtime"]
#: The lane's LIVE admission.  Deliberately not part of ``_CARTESIAN``: that
#: block is a sealed identity copied verbatim into every canonical document,
#: and widening the lane may not move a document whose recorded receipts carry
#: its digest as ``plan_sha256``.  See the record's own
#: ``why_this_is_not_inside_cartesian_runtime``.
_CARTESIAN_ADMISSION = _MANIFEST["cartesian_runtime_admission"]
_NAMELIST = _OFFICIAL["native_namelist"]

# The configurable Cartesian lane has no LaneName constant yet; the record
# carries its identity string so the manifest stays the single source of truth.
CARTESIAN_LANE_NAME = _CARTESIAN["lane_name"]

OFFICIAL_NAMELIST_SHA256 = _NAMELIST["sha256"]
OFFICIAL_NAMELIST_BYTES = _NAMELIST["bytes"]
OFFICIAL_NAMELIST_OPTION_COUNT = _NAMELIST["option_count"]


def cartesian_live_admission() -> dict[str, object]:
    """Return what the configurable Cartesian lane admits right now.

    ``profile_identity`` returns a sealed identity; this returns the live
    envelope.  They are different records on purpose, and
    ``tests/test_cartesian_live_admission.py`` holds this one to the code that
    enforces it so it cannot become the stale half.
    """

    return copy.deepcopy(_CARTESIAN_ADMISSION)


def profile_identity(lane: str) -> dict[str, object]:
    if lane == LaneName.PRODUCT:
        result = copy.deepcopy(_OFFICIAL)
        result["lane"] = LaneName.PRODUCT
        return result
    if lane == LaneName.REAL_PRODUCT:
        result = copy.deepcopy(_REAL)
        result["lane"] = LaneName.REAL_PRODUCT
        return result
    if lane == LaneName.NATIVE_CONTEXT:
        result = copy.deepcopy(_OFFICIAL)
        result.pop("product_differences")
        result["lane"] = LaneName.NATIVE_CONTEXT
        result["execution_policy"] = "reference/context outside package; never a production door"
        return result
    if lane == CARTESIAN_LANE_NAME:
        result = copy.deepcopy(_CARTESIAN)
        result["lane"] = CARTESIAN_LANE_NAME
        return result
    if lane == LaneName.DRY_DYCORE:
        result = copy.deepcopy(_MANIFEST["dry_bootstrap"])
        result["lane"] = LaneName.DRY_DYCORE
        return result
    raise ValueError(f"unknown profile lane: {lane!r}")


__all__ = [
    "CARTESIAN_LANE_NAME",
    "cartesian_live_admission",
    "OFFICIAL_NAMELIST_BYTES",
    "OFFICIAL_NAMELIST_OPTION_COUNT",
    "OFFICIAL_NAMELIST_SHA256",
    "profile_identity",
]
