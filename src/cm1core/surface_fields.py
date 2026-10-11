"""CM1 r21.1 ``init_surface`` end to end, in CM1's own order.

``src/init_surface.F`` is one subroutine with five consecutive parts, and the
order between them is load-bearing:

1. the ``initsfc`` branches fill ``tsk``/``tmn``/``xland``/``lu_index`` and,
   for ``initsfc=3``, ``znt`` (``:40-145``);
2. the soil slab and the horizontal surface length scale (``:152-203``);
3. ``dosfcflx`` (``:207-227``);
4. ``getlanduse`` **overwrites** ``znt``, ``mavail``, ``emiss``, ``thc``,
   ``albedo`` and ``xland`` from ``lu_index`` (``:239-243``);
5. ``set_znt=1`` re-imposes ``cnst_znt`` on top of the table (``:246-254``),
   and the reality checks run last (``:270-322``).

Step 4 is why this module exists rather than a caller stitching the two
bounded operators together in whichever order looked reasonable.  Because
``getlanduse`` runs after the ``initsfc`` branches, ``initsfc=3``'s
rough-west/smooth-east roughness does not survive into a CM1 run at all: the
table overwrites both halves with the roughness of ``lu0``.  A port that
placed ``getlanduse`` first, or left it out, would integrate a west/east
roughness contrast CM1 never integrates.  The receipt says so per run.

:mod:`cm1core.surface_initialization` owns part 1-3 and
:mod:`cm1core.surface_landuse` owns parts 4-5; this module owns only the
order and the whole-domain coordinate they are evaluated on.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from cm1core.errors import refuse
from cm1core.surface_initialization import (
    SurfaceInitializationSelection,
    cpu_surface_initialization_inplace,
    surface_initialization_shapes,
)
from cm1core.surface_landuse import (
    CM1_LANDUSE_TBL_SHA256,
    cpu_getlanduse_inplace,
    cpu_surface_reality_checks,
    landuse_record_sha256,
)


SURFACE_FIELDS_SCHEMA = "gm1.surface-fields/v1"
SURFACE_FIELDS_GATE = "surface-drag"
SURFACE_FIELDS_SOURCE_ANCHORS = MappingProxyType(
    {
        "init_surface": "src/init_surface.F:40-343",
        "initsfc_branches": "src/init_surface.F:67-145",
        "soil_slab": "src/init_surface.F:152-172",
        "surface_length_scale": "src/init_surface.F:174-203",
        "dosfcflx": "src/init_surface.F:207-227",
        "getlanduse": "src/init_surface.F:239-243",
        "prescribed_roughness": "src/init_surface.F:246-254",
        "reality_checks": "src/init_surface.F:270-322",
        "ocean_mixed_layer": "src/init_surface.F:327-336",
        "horizontal_coordinate": "src/param.F:6646-6664,7328-7334,7406-7408",
        "namelist": "README.namelist:971-1015",
    }
)
SURFACE_FIELDS_CLAIM_LIMIT = (
    "CM1 r21.1 init_surface.F:40-322 in source order on a uniform iorigin=2 "
    "Cartesian grid: the four initsfc branches, the soil slab, dsxy, "
    "dosfcflx, the packaged getlanduse table, the set_znt override and the "
    "reality checks; oceanmodel=2 omlinit, stretched horizontal grids, "
    "iorigin=1, terrain, MPI subdomains and testcase=13 are excluded"
)

# src/init_surface.F:152.  CM1's own default for the MM5/WRF slab
# (README.namelist:1044-1050 num_soil_layers); the slab is only initialized
# for sfcmodel=2,3,4,6,7, so for sfcmodel=1 the count only sizes an array
# nothing reads.
CM1_DEFAULT_SOIL_LAYERS = 5

# src/param.F:6656-6664.  Every CM1 r21.1 configuration deck that ships with
# the release uses iorigin=2, the domain-centred origin, and GM1's Cartesian
# lane carries no other; the initsfc=2/3/4 branches are the only consumers.
CM1_IORIGIN = 2


@dataclass(frozen=True, slots=True)
class SurfaceFieldSelection:
    """CM1's ``param12`` surface selectors, by CM1's own namelist names."""

    initsfc: int
    sfcmodel: int
    bbc: int
    isfcflx: int
    set_znt: int
    cnst_znt: float
    tsk0_k: float
    tmn0_k: float
    xland0: float
    lu0: int
    season: int
    oceanmodel: int
    num_soil_layers: int = CM1_DEFAULT_SOIL_LAYERS
    cm1setup: int = 1
    ipbl: int = 0
    axisymm: int = 0


@dataclass(frozen=True, slots=True)
class SurfaceFields:
    """Whole-domain surface state and the receipt that names its order."""

    arrays: Mapping[str, Any]
    dosfcflx: bool
    receipt: Mapping[str, object]


SURFACE_FIELD_NAMES = (
    "tsk",
    "tmn",
    "xland",
    "lu_index",
    "znt",
    "mavail",
    "emiss",
    "thc",
    "albedo",
    "dsxy",
    "slab_zs",
    "slab_dzs",
    "tslb",
)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()


def _coordinate(np: Any, count: int, spacing: float) -> Any:
    """``src/param.F:6659`` then ``:7332`` for ``iorigin=2``, in binary32.

    CM1 builds the face coordinate in double precision and averages two of
    them into the cell centre; the result is stored in a default REAL.  This
    reproduces that: double throughout, one rounding at the end.
    """

    faces = np.arange(count + 1, dtype=np.float64) * np.float64(spacing) - (
        np.float64(0.5) * np.float64(spacing) * np.float64(count)
    )
    return np.asarray(0.5 * (faces[:-1] + faces[1:]), dtype=np.float32)


def build_surface_fields(
    *,
    selection: SurfaceFieldSelection,
    nx: int,
    ny: int,
    dx_m: float,
    dy_m: float,
) -> SurfaceFields:
    """Run ``init_surface`` once for the whole domain, in CM1's order."""

    np = importlib.import_module("numpy")
    if not isinstance(selection, SurfaceFieldSelection):
        raise TypeError("selection must be a SurfaceFieldSelection")
    for name, value in (("nx", nx), ("ny", ny)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    # src/init_surface.F:327-336
    if selection.oceanmodel not in (0, 1):
        refuse(
            code="CM1-SFC-OCEANMODEL-UNSUPPORTED",
            gate=SURFACE_FIELDS_GATE,
            option="oceanmodel",
            value=selection.oceanmodel,
            reason=(
                "oceanmodel=2 is the Pollard et al. mixed layer "
                "(src/oml.F, init_surface.F:329-336, turb.F:1305): it makes "
                "the skin temperature a prognostic variable that cools under "
                "the storm.  No GM1 carrier evolves it, so selecting it would "
                "hold the sea surface at its initial temperature for the "
                "whole run while the receipt named an interactive ocean"
            ),
            remedy=(
                "use oceanmodel=1, the fixed sea-surface temperature CM1 "
                "requires with sfcmodel=1 (README.namelist:946)"
            ),
        )
    # README.namelist:946
    if selection.sfcmodel == 1 and selection.oceanmodel == 0:
        oceanmodel = 1
    else:
        oceanmodel = selection.oceanmodel

    minx = np.float32(-0.5 * float(dx_m) * nx)
    maxx = np.float32(0.5 * float(dx_m) * nx)
    centerx = np.float32(float(minx) + 0.5 * (float(maxx) - float(minx)))

    shapes = surface_initialization_shapes(nx, ny, selection.num_soil_layers)
    arrays: dict[str, Any] = {}
    for name, shape in shapes.items():
        dtype = np.int32 if name == "lu_index" else np.float32
        arrays[name] = np.zeros(shape, dtype=dtype)
    arrays["xh"][:] = _coordinate(np, nx, dx_m)
    arrays["ruh"][:] = np.float32(1.0)
    arrays["rvh"][:] = np.float32(1.0)
    # init_surface writes tsk/tmn/xland/lu_index before it reads them, but
    # znt for initsfc != 3 is left at whatever the caller supplied and is
    # then overwritten by getlanduse.  Start it at CM1's allocator zero so a
    # skipped table would be visible rather than plausible.
    initialization = cpu_surface_initialization_inplace(
        arrays,
        ni=nx,
        nj=ny,
        num_soil_layers=selection.num_soil_layers,
        selection=SurfaceInitializationSelection(
            initsfc=selection.initsfc,
            sfcmodel=selection.sfcmodel,
            bbc=selection.bbc,
            axisymm=selection.axisymm,
            cm1setup=selection.cm1setup,
            ipbl=selection.ipbl,
            isfcflx=selection.isfcflx,
            tsk0_k=selection.tsk0_k,
            tmn0_k=selection.tmn0_k,
            xland0=selection.xland0,
            lu0=selection.lu0,
            minx_m=float(minx),
            maxx_m=float(maxx),
            centerx_m=float(centerx),
            dx_m=float(dx_m),
            dy_m=float(dy_m),
        ),
    )
    for name in ("mavail", "emiss", "thc", "albedo"):
        arrays[name] = np.zeros((ny, nx), dtype=np.float32)

    initsfc_roughness = arrays["znt"].copy()
    landuse: dict[str, object] | None = None
    checks: dict[str, object] | None = None
    # src/init_surface.F:239-243
    if selection.sfcmodel >= 1 or selection.bbc == 3:
        landuse = cpu_getlanduse_inplace(
            {
                "lu_index": arrays["lu_index"],
                "xland": arrays["xland"],
                "znt": arrays["znt"],
                "mavail": arrays["mavail"],
                "emiss": arrays["emiss"],
                "thc": arrays["thc"],
                "albedo": arrays["albedo"],
            },
            ni=nx,
            nj=ny,
            season=selection.season,
        )
    # src/init_surface.F:246-254
    if selection.set_znt == 1:
        arrays["znt"][:, :] = max(np.float32(1.0e-12), np.float32(selection.cnst_znt))
    if selection.sfcmodel >= 1 or selection.bbc == 3:
        checks = cpu_surface_reality_checks(arrays, sfcmodel=selection.sfcmodel)

    table_overwrote_initsfc_roughness = bool(
        landuse is not None
        and selection.initsfc == 3
        and not np.array_equal(initsfc_roughness, arrays["znt"])
    )
    receipt: dict[str, object] = {
        "schema": SURFACE_FIELDS_SCHEMA,
        "claim_limit": SURFACE_FIELDS_CLAIM_LIMIT,
        "source_anchors": dict(SURFACE_FIELDS_SOURCE_ANCHORS),
        "landuse_authority_sha256": CM1_LANDUSE_TBL_SHA256,
        "landuse_record_sha256": landuse_record_sha256(),
        "selection": {
            "initsfc": selection.initsfc,
            "sfcmodel": selection.sfcmodel,
            "bbc": selection.bbc,
            "isfcflx": selection.isfcflx,
            "set_znt": selection.set_znt,
            "cnst_znt": float(selection.cnst_znt),
            "tsk0": float(selection.tsk0_k),
            "tmn0": float(selection.tmn0_k),
            "xland0": float(selection.xland0),
            "lu0": selection.lu0,
            "season": selection.season,
            "oceanmodel": oceanmodel,
            "num_soil_layers": selection.num_soil_layers,
        },
        "domain": {
            "nx": nx,
            "ny": ny,
            "dx_m": float(dx_m),
            "dy_m": float(dy_m),
            "iorigin": CM1_IORIGIN,
            "minx_m": float(minx),
            "maxx_m": float(maxx),
            "centerx_m": float(centerx),
        },
        "source_order": [
            "initsfc branches (init_surface.F:67-145)",
            "soil slab and dsxy (init_surface.F:152-203)",
            "dosfcflx (init_surface.F:207-227)",
            "getlanduse (init_surface.F:239-243)"
            if landuse is not None
            else "getlanduse not reached (sfcmodel=0 and bbc!=3)",
            "set_znt override (init_surface.F:246-254)"
            if selection.set_znt == 1
            else "set_znt override not reached",
            "reality checks (init_surface.F:270-322)"
            if checks is not None
            else "reality checks not reached",
        ],
        "initialization": dict(initialization.receipt),
        "landuse": landuse,
        "reality_checks": checks,
        # src/init_surface.F:97-107 sets znt from initsfc=3, and
        # src/init_surface.F:458 then overwrites it from lu_index.  When that
        # happened, the run has no west/east roughness contrast -- which is
        # what CM1 does, and which a reader of a receipt naming initsfc=3
        # would otherwise assume the opposite of.
        "initsfc3_roughness_overwritten_by_landuse": table_overwrote_initsfc_roughness,
        "dosfcflx": bool(initialization.dosfcflx),
        "field_sha256": {
            name: hashlib.sha256(arrays[name].tobytes(order="C")).hexdigest()
            for name in SURFACE_FIELD_NAMES
        },
    }
    receipt["operation_token_sha256"] = _canonical_sha256(receipt)
    return SurfaceFields(
        arrays=MappingProxyType({name: arrays[name] for name in SURFACE_FIELD_NAMES}),
        dosfcflx=bool(initialization.dosfcflx),
        receipt=MappingProxyType(receipt),
    )


__all__ = [
    "CM1_DEFAULT_SOIL_LAYERS",
    "CM1_IORIGIN",
    "SURFACE_FIELDS_CLAIM_LIMIT",
    "SURFACE_FIELDS_SCHEMA",
    "SURFACE_FIELD_NAMES",
    "SurfaceFieldSelection",
    "SurfaceFields",
    "build_surface_fields",
]
