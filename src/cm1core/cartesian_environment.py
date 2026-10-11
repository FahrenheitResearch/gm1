"""The ``writeout.F`` 2-D environment planes for one Cartesian history frame.

``writeout.F:2895-2932`` computes CAPE, CIN, LCL and LFC once per column of
the live state whenever any of the four ``output_*`` switches is on, and
``writeout.F:2937-3028`` then writes the selected planes beside the 3-D
fields.  This module is the GM1 equivalent: it allocates the column scratch
through the same registry the rest of the runtime uses, launches the packaged
kernel at the publication boundary, and hands the planes to the history
profile as ordinary carriers.

A document that declares no ``[diagnostics]`` table, or leaves all four
switches off, never reaches this module at all: the carrier table is then
exactly the twelve carriers it has always been and the published frame is
byte-for-byte what it was.  That is ``writeout.F:2895``'s own behaviour.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from .cuda_getcape import (
    GETCAPE_FIELD_NAMES,
    GetCapeFieldSelection,
    compile_cuda_getcape,
    getcape_field_shapes,
    getcape_scratch_bytes,
    launch_cuda_getcape,
    require_getcape_selection,
)


ENVIRONMENT_GROUP = "environment"
ENVIRONMENT_WORK_BUFFER = "environment.getcape-work"
ENVIRONMENT_PLANE_PREFIX = "environment.plane"
#: ``writeout.F:2907-2910`` reads these live carriers, in this order.
RUNTIME_FIELD_KEYS = {
    "prs": "prs",
    "th0": "th0",
    "th3d": "th3d",
    "pi0": "pi0",
    "pp3d": "pp3d",
}


def environment_field_names(selection: GetCapeFieldSelection) -> tuple[str, ...]:
    """The planes this selection publishes, in ``writeout.F``'s own order."""

    require_getcape_selection(selection)
    return selection.selected()


def environment_scratch_bytes(config: Any, selection: GetCapeFieldSelection) -> dict[str, int]:
    """Price the sweep before a device word is allocated."""

    if not selection.any_field:
        return {
            "column_scratch_bytes": 0,
            "field_plane_bytes": 0,
            "getcape_scratch_peak_bytes": 0,
        }
    g = config.grid
    return getcape_scratch_bytes(g.nx, g.ny, g.nz)


def _surface_weights(config: Any) -> tuple[float, float, float]:
    """``param.F:9038``'s ``cgs1..3`` for this grid's own levels.

    ``writeout.F:2912-2914`` extrapolates the getcape surface level with these
    three weights.  On ``param.F:7789-7811``'s uniform coordinate they are
    1.875, -1.25 and 0.375; on a stretched one they are not, and using the
    uniform constants there would place the surface parcel's pressure,
    temperature and moisture at the wrong height in every column.
    """

    return tuple(float(value) for value in config.grid.levels.cgs)


def compute_environment_planes(
    runtime: Any, selection: GetCapeFieldSelection
) -> Mapping[str, Any]:
    """Run the column sweep on the live state and return the four planes.

    The caller owns the release: every buffer this allocates is named with the
    :data:`ENVIRONMENT_WORK_BUFFER` / :data:`ENVIRONMENT_PLANE_PREFIX` prefixes
    so :func:`release_environment_planes` can free the whole set on any path.
    """

    require_getcape_selection(selection)
    if not selection.any_field:
        return MappingProxyType({})
    cupy, registry = runtime.cp, runtime.registry
    g = runtime.config.grid
    ni, nj, nk = g.nx, g.ny, g.nz
    shapes = getcape_field_shapes(ni, nj, nk)
    fields = runtime.dynamics.fields
    arrays: dict[str, Any] = {}
    for name, key in RUNTIME_FIELD_KEYS.items():
        try:
            value = fields[key]
        except KeyError:
            raise ValueError(
                f"writeout.F:2907-2910 reads {key} to build the getcape column; "
                "this runtime does not hold it, so no environment plane can be "
                "published.  pi0/th0/qv0 are released for the acoustic phase "
                "(cuda_cartesian_dynamics._drop_expanded_base) and restored "
                "before solve3 ends, so a sweep that finds them missing is not "
                "at the post-solve3 publication boundary"
            ) from None
        if value.dtype != cupy.dtype("float32") or tuple(value.shape) != shapes[name]:
            raise ValueError(
                f"the live {key} carrier is {value.dtype} {tuple(value.shape)}; "
                f"the getcape column sweep needs float32 {shapes[name]}"
            )
        arrays[name] = value
    # writeout.F:2909 reads qa(i,j,k,nqv); param.F:3644 sets nqv=1.
    arrays["qv"] = runtime.dynamics.fields["q3d"][0]
    if not arrays["qv"].flags.c_contiguous:
        # q3d is one C-order (6, nk, nj, ni) registry allocation, so its first
        # species is contiguous by construction.  Copying it here instead would
        # allocate a volume outside the registry budget every frame, which is
        # exactly the accounting this runtime refuses to lose.
        raise ValueError(
            "the qv species of q3d is not contiguous; the getcape column sweep "
            "reads it as one C-order (nk, nj, ni) block and would otherwise need "
            "an unbudgeted device copy of the whole volume at every frame"
        )
    compiled = getattr(runtime, "_getcape_kernel", None)
    if compiled is None:
        compiled = compile_cuda_getcape()
        runtime._getcape_kernel = compiled
    arrays["work"] = registry.allocate(ENVIRONMENT_WORK_BUFFER, shapes["work"], zero=True)
    for name in GETCAPE_FIELD_NAMES:
        arrays[name] = registry.allocate(
            f"{ENVIRONMENT_PLANE_PREFIX}.{name}", shapes[name], zero=True
        )
    launch_cuda_getcape(
        compiled,
        arrays,
        ni=ni,
        nj=nj,
        nk=nk,
        selection=selection,
        surface_weights=_surface_weights(runtime.config),
        stream=runtime.stream,
    )
    return MappingProxyType({name: arrays[name] for name in GETCAPE_FIELD_NAMES})


def release_environment_planes(runtime: Any) -> None:
    """Free every buffer :func:`compute_environment_planes` allocated."""

    registry = runtime.registry
    names = [
        name
        for name in (
            ENVIRONMENT_WORK_BUFFER,
            *(f"{ENVIRONMENT_PLANE_PREFIX}.{field}" for field in GETCAPE_FIELD_NAMES),
        )
        if name in registry.buffers
    ]
    if names:
        registry.release(names)


__all__ = [
    "ENVIRONMENT_GROUP",
    "ENVIRONMENT_PLANE_PREFIX",
    "ENVIRONMENT_WORK_BUFFER",
    "RUNTIME_FIELD_KEYS",
    "compute_environment_planes",
    "environment_field_names",
    "environment_scratch_bytes",
    "release_environment_planes",
]
