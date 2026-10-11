"""Exact vertical-metric binding for the stretched 50 m GM1 child.

The admitted child uses CM1 ``stretch_z=1``.  Its nominal ``dz`` remains the
100 m metric reference, but its physical top is 15 km and every vertical
derivative/interpolation must consume the source-generated ``sigma``,
``sigmaf``, ``rds``, and ``rdsf`` profiles.  This module is deliberately only
a validation/binding layer; it performs no coordinate reconstruction.
"""

from __future__ import annotations

import importlib
import math
import weakref
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from cm1core.cuda_gm1_child_terrain import CudaGm1ChildTerrainMetricsLaunch


GM1_CHILD_STRETCHED_METRIC_SCHEMA = "gm1.cuda-child-stretched-metrics/v1"
GM1_CHILD_STRETCHED_METRIC_ABI = "gm1.cuda-child-stretched-metrics-f32/v1"
GM1_CHILD_STRETCHED_GRID_SHA256 = (
    "f66aa7937e0f4942dfbc55001641f221b8a84352961058e81247ee4e3ca8db56"
)
GM1_CHILD_STRETCHED_PROOF_SHA256 = (
    "277f2b8831f316b2e99a81a903f5ac20ee5d781e2c2b058963539f648f9ec02e"
)
GM1_CHILD_STRETCHED_PROFILE_SHA256 = MappingProxyType(
    {
        "sigma": "e0c800cd95ac68f526efef46de91db5463784e91860f52ea90ad76c9e18037c3",
        "sigmaf": "ceb1624e4dc10eb31562c7e8003f939d6d157e1ecd73d1dee479de5459f2a90e",
        "rds": "b995cfc800e8b69c257d1f0e9e328c315d91652f7b0a86a748af49eadd4306fe",
        "rdsf": "008898d0ed25941c735f0f7a23746d3a7eea562d12d01ad7cc07b6d50ed6f416",
    }
)
GM1_CHILD_STRETCHED_ZTOP_F32 = 15000.0
GM1_CHILD_STRETCHED_DZ_REFERENCE_F32 = 100.0


@dataclass(frozen=True, slots=True, weakref_slot=True)
class Gm1ChildStretchedMetrics:
    sigma: Any
    sigmaf: Any
    rds: Any
    rdsf: Any
    ztop_f32: float
    dz_reference_f32: float
    target_arch: str
    identity: Mapping[str, Any]
    #: A flat column's (terrain_flag=.false.) param.F:8467 ``mh(1,1,k)`` for
    #: k=0..nk+1 and param.F:8476 ``mf(1,1,k)`` for k=0..nk+2.  ``None`` is
    #: the terrain-following child, whose operators divide by ``rds``/``rdsf``;
    #: a binding that carries both runs CM1's flat branches, which multiply
    #: by ``rdz`` and then by these words (adv.F:562-572, sound.F:119-143).
    mh: Any = None
    mf: Any = None

    @property
    def flat_column(self) -> bool:
        """Does this binding describe CM1's terrain_flag=.false. column?"""

        if (self.mh is None) != (self.mf is None):
            raise ValueError("a flat column binds both mh and mf, or neither")
        return self.mh is not None

    @property
    def rdz_f32(self) -> float:
        """``param.F:6327``: ``rdz = 1.0/dz`` of the metric reference, in binary32."""

        import numpy as np

        return float(np.float32(1.0) / np.float32(self.dz_reference_f32))

    @property
    def arrays(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "sigma": self.sigma,
                "sigmaf": self.sigmaf,
                "rds": self.rds,
                "rdsf": self.rdsf,
            }
        )


def stretched_vertical_identity_defect(
    identity: Any,
    *,
    nk: int,
    ztop_f32: float,
    dz_reference_f32: float,
) -> str | None:
    """Why these vertical profiles do not describe this column, or ``None``.

    Every stretched operator binds ``sigma``/``sigmaf``/``rds``/``rdsf`` and
    divides by the reference ``dz`` of ``param.F:8170``.  What has to be true
    is not that the run is one particular fixture, but that the profiles it
    binds were generated for *this* column: ``param.F:8193`` makes the model
    top ``sigmaf(nk+1)``, so profiles built for a different ``nk``, a
    different top, or a different metric reference would put every vertical
    derivative and every terrain-coordinate height factor on a column the run
    does not have, silently.

    Three identities reach here.  The sealed 15 km child carries its own
    SHA-256 words and is held to every one of them.  The flat uniform
    validation fixture carries its declared class.  Anything else is a
    coordinate this run generated, and is held to its own declared words.
    """

    import math

    if not isinstance(identity, Mapping) or not identity.get("schema"):
        return "vertical identity must be a schema-bearing mapping"
    if not math.isfinite(float(ztop_f32)) or float(ztop_f32) <= 0.0:
        return "vertical top must be finite and greater than zero"
    if not math.isfinite(float(dz_reference_f32)) or float(dz_reference_f32) <= 0.0:
        return (
            "vertical reference dz must be finite and greater than zero; "
            "param.F:8170 divides every mh word by it"
        )
    if "grid_sha256" in identity:
        if identity.get("grid_sha256") != GM1_CHILD_STRETCHED_GRID_SHA256:
            return "vertical grid SHA-256 differs from the sealed child"
        if identity.get("proof_sha256") != GM1_CHILD_STRETCHED_PROOF_SHA256:
            return "vertical proof SHA-256 differs from the sealed child"
        if float(ztop_f32) != GM1_CHILD_STRETCHED_ZTOP_F32:
            return "sealed child vertical top must be exactly 15000 m"
        if float(dz_reference_f32) != GM1_CHILD_STRETCHED_DZ_REFERENCE_F32:
            return "sealed child vertical reference dz must be exactly 100 m"
        return None
    declared = identity.get("vertical_coordinate")
    if not isinstance(declared, Mapping):
        return None
    if int(declared.get("nk", nk)) != nk:
        return (
            f"vertical coordinate carries {declared.get('nk')} levels, "
            f"not the bound nk={nk}"
        )
    if float(declared.get("ztop_m", ztop_f32)) != float(ztop_f32):
        return "vertical coordinate top differs from the bound top"
    if float(declared.get("dz_reference_m", dz_reference_f32)) != float(
        dz_reference_f32
    ):
        return "vertical coordinate dz reference differs from the bound one"
    return None


def _require_cuda_profile(
    cupy: Any, value: Any, name: str, shape: tuple[int, ...]
) -> Any:
    if (
        not isinstance(value, cupy.ndarray)
        or value.dtype != cupy.float32
        or value.shape != shape
        or not value.flags.c_contiguous
    ):
        raise ValueError(f"stretched child {name} must be C-f32 {shape}")
    if int(value.device.id) != int(cupy.cuda.Device().id):
        raise ValueError(f"stretched child {name} is resident on another device")
    return value


def bind_gm1_child_stretched_metrics(
    launch: CudaGm1ChildTerrainMetricsLaunch,
) -> Gm1ChildStretchedMetrics:
    """Bind only the sealed child profiles already resident on the device."""

    if not isinstance(launch, CudaGm1ChildTerrainMetricsLaunch):
        raise TypeError("launch must be CudaGm1ChildTerrainMetricsLaunch")
    receipt = launch.metrics.receipt
    vertical_receipt = receipt.get("vertical")
    if not isinstance(vertical_receipt, Mapping):
        raise ValueError("child static-metric receipt lacks its vertical identity")
    if vertical_receipt.get("grid_sha256") != GM1_CHILD_STRETCHED_GRID_SHA256:
        raise ValueError("child vertical grid differs from its admitted SHA-256")
    if vertical_receipt.get("proof_sha256") != GM1_CHILD_STRETCHED_PROOF_SHA256:
        raise ValueError("child vertical proof differs from its admitted SHA-256")
    geometry = receipt.get("geometry")
    if not isinstance(geometry, Mapping) or geometry.get("ztop_m") != 15000.0:
        raise ValueError("child static metrics do not attest the 15 km physical top")

    cupy = importlib.import_module("cupy")
    vertical = launch.metrics.state.vertical
    static = launch.metrics.state.arrays
    sigma = _require_cuda_profile(cupy, vertical.grid.sigma, "sigma", (77,))
    sigmaf = _require_cuda_profile(cupy, vertical.grid.sigmaf, "sigmaf", (78,))
    rds = _require_cuda_profile(cupy, static["rds"], "rds", (77,))
    rdsf = _require_cuda_profile(cupy, static["rdsf"], "rdsf", (78,))
    if len({int(value.data.ptr) for value in (sigma, sigmaf, rds, rdsf)}) != 4:
        raise ValueError("stretched child vertical profiles must not alias")
    if launch.metrics.compiled.target_arch != vertical.compiled.target_arch:
        raise ValueError("child terrain and vertical CUDA images target different architectures")
    identity = MappingProxyType(
        {
            "schema": GM1_CHILD_STRETCHED_METRIC_SCHEMA,
            "abi": GM1_CHILD_STRETCHED_METRIC_ABI,
            "grid_sha256": GM1_CHILD_STRETCHED_GRID_SHA256,
            "proof_sha256": GM1_CHILD_STRETCHED_PROOF_SHA256,
            "profile_sha256": dict(GM1_CHILD_STRETCHED_PROFILE_SHA256),
            "profile_indexing": "direct CM1 indices 0:nk+1 and 0:nk+2",
            "ztop_f32": GM1_CHILD_STRETCHED_ZTOP_F32,
            "dz_reference_f32": GM1_CHILD_STRETCHED_DZ_REFERENCE_F32,
        }
    )
    return Gm1ChildStretchedMetrics(
        sigma,
        sigmaf,
        rds,
        rdsf,
        GM1_CHILD_STRETCHED_ZTOP_F32,
        GM1_CHILD_STRETCHED_DZ_REFERENCE_F32,
        launch.metrics.compiled.target_arch,
        identity,
    )


# ---------------------------------------------------------------------------
# A coordinate this run generated -- param.F:7789-8209
# ---------------------------------------------------------------------------
#
# ``bind_gm1_child_stretched_metrics`` above binds one historical geometry by
# its three sealed hashes.  It is not the only stretched coordinate CM1 can
# build: ``param.F:7814-8139`` generates a level set from the ``param6`` words
# of whatever namelist a run declares, and ``cm1_vertical_grid`` reproduces
# four of those five branches.  A binding whose identity carries the
# ``vertical_coordinate`` record of such a generated grid is held to *that*
# grid's own words instead of to the sealed child's -- the same discipline
# ``cuda_gm1_child_psolver3_open._require_vertical`` already applies.
#
# The check that matters is ``param.F:8193``: ``maxz = sigmaf(nk+1)``.  The
# model top a consumer is told about and the top face of the profiles it reads
# must be the same height, because ``init_terrain.F:175-196`` divides the
# terrain-following coordinate by that top.  A mismatch puts every vertical
# derivative on a different column from the one the profiles describe, and
# nothing downstream would notice.

GENERATED_VERTICAL_IDENTITY_KEY = "vertical_coordinate"

#: Bindings whose ``sigmaf(nk+1)`` word has already been read back and matched
#: to their own top, keyed by ``(id(binding), nk)``.  Every stretched launch
#: asks the question again, and each answer was a blocking device-to-host read
#: (33 of them per large step).  A binding is frozen and every kernel takes its
#: profiles ``const``, so the word cannot change under the same object; an
#: entry dies with its binding, so a later object that reuses the id is read
#: again.
_TOP_FACE_VERIFIED: "weakref.WeakValueDictionary[tuple[int, int], Gm1ChildStretchedMetrics]" = (
    weakref.WeakValueDictionary()
)


def require_flat_column(
    vertical: Gm1ChildStretchedMetrics,
    *,
    nk: int,
    cupy: Any,
) -> tuple[Any, Any, float]:
    """A flat binding's ``mh``, ``mf`` and ``rdz``, each checked.

    ``mh`` must be the (nk+2,) and ``mf`` the (nk+3,) C-contiguous binary32
    device column, distinct from each other and from the terrain profiles,
    so a kernel reading ``mh[k]`` and ``mf[k]`` reads CM1's ``mh(1,1,k)`` and
    ``mf(1,1,k)`` at the same ``k``.
    """

    if not isinstance(vertical, Gm1ChildStretchedMetrics):
        raise TypeError("vertical must be Gm1ChildStretchedMetrics")
    if not vertical.flat_column:
        raise ValueError(
            "this binding is the terrain-following column; the flat operators "
            "multiply by rdz*mh and have no mh/mf to read"
        )
    mh = _require_cuda_profile(cupy, vertical.mh, "mh", (nk + 2,))
    mf = _require_cuda_profile(cupy, vertical.mf, "mf", (nk + 3,))
    pointers = {int(value.data.ptr) for value in (mh, mf)}
    others = {
        int(value.data.ptr)
        for value in (vertical.sigma, vertical.sigmaf, vertical.rds, vertical.rdsf)
        if value is not None and hasattr(value, "data")
    }
    if len(pointers) != 2 or pointers & others:
        raise ValueError("flat column mh/mf must not alias each other or the terrain profiles")
    dz = float(vertical.dz_reference_f32)
    if not math.isfinite(dz) or dz <= 0.0:
        raise ValueError("flat column reference dz must be finite and > 0 (param.F:6327)")
    return mh, mf, vertical.rdz_f32


def generated_vertical_coordinate(
    vertical: Gm1ChildStretchedMetrics,
) -> Mapping[str, Any] | None:
    """The coordinate record this run generated, or ``None``.

    ``None`` means the binding is either the sealed 50 m child (its identity
    carries ``grid_sha256``) or a validation fixture; both keep every check
    they have ever had.
    """

    if not isinstance(vertical, Gm1ChildStretchedMetrics):
        raise TypeError("vertical must be Gm1ChildStretchedMetrics")
    identity = vertical.identity
    if not isinstance(identity, Mapping) or "grid_sha256" in identity:
        return None
    declared = identity.get(GENERATED_VERTICAL_IDENTITY_KEY)
    return declared if isinstance(declared, Mapping) else None


def require_generated_vertical(
    vertical: Gm1ChildStretchedMetrics,
    *,
    nk: int,
    cupy: Any | None = None,
) -> Mapping[str, Any]:
    """Hold a generated coordinate to its own ``param.F:8193`` model top.

    Returns the coordinate record.  Raises if the binding is not a generated
    one, so a caller that reaches this having already selected the generated
    branch cannot silently accept a sealed or fixture binding instead.
    """

    declared = generated_vertical_coordinate(vertical)
    if declared is None:
        raise ValueError(
            "this binding does not carry a generated vertical coordinate record"
        )
    if isinstance(nk, bool) or not isinstance(nk, int) or nk < 4:
        raise ValueError("nk must be an integer >= 4")
    if int(declared.get("nk", nk)) != nk:
        raise ValueError(
            f"generated vertical coordinate has {declared.get('nk')} levels, "
            f"not the {nk} this launch integrates"
        )
    ztop = float(vertical.ztop_f32)
    dz_reference = float(vertical.dz_reference_f32)
    for name, value in (("ztop", ztop), ("dz reference", dz_reference)):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"generated vertical {name} must be finite and > 0")
    if float(declared.get("ztop_m", ztop)) != ztop:
        raise ValueError("generated vertical coordinate top differs from the bound top")
    if float(declared.get("dz_reference_m", dz_reference)) != dz_reference:
        raise ValueError(
            "generated vertical coordinate dz reference differs from the bound one; "
            "param.F:8170 divides every mh word by exactly this number"
        )
    cupy = importlib.import_module("cupy") if cupy is None else cupy
    for name, shape in (
        ("sigma", (nk + 2,)),
        ("sigmaf", (nk + 3,)),
        ("rds", (nk + 2,)),
        ("rdsf", (nk + 3,)),
    ):
        _require_cuda_profile(cupy, getattr(vertical, name), name, shape)
    key = (id(vertical), nk)
    if _TOP_FACE_VERIFIED.get(key) is not vertical:
        top_face = float(vertical.sigmaf[nk + 1].get())
        if top_face != ztop:
            raise ValueError(
                f"generated vertical top {ztop} m is not this coordinate's own "
                f"sigmaf(nk+1) of {top_face} m (param.F:8193)"
            )
        _TOP_FACE_VERIFIED[key] = vertical
    return declared


__all__ = (
    "GM1_CHILD_STRETCHED_DZ_REFERENCE_F32",
    "GM1_CHILD_STRETCHED_GRID_SHA256",
    "GM1_CHILD_STRETCHED_METRIC_ABI",
    "GM1_CHILD_STRETCHED_METRIC_SCHEMA",
    "GM1_CHILD_STRETCHED_PROFILE_SHA256",
    "GM1_CHILD_STRETCHED_PROOF_SHA256",
    "GM1_CHILD_STRETCHED_ZTOP_F32",
    "GENERATED_VERTICAL_IDENTITY_KEY",
    "Gm1ChildStretchedMetrics",
    "bind_gm1_child_stretched_metrics",
    "generated_vertical_coordinate",
    "require_flat_column",
    "require_generated_vertical",
)
