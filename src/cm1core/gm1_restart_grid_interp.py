"""CM1 r21.1 restart-grid interpolation contract for the fixed GM1 child.

CM1's ``interp_on_restart`` branch interpolates in the one-dimensional
terrain coordinate (``sigma`` for scalar/u/v carriers and ``sigmaf`` for
w carriers), not in three-dimensional physical height.  At a new scalar
level below the first old scalar level, ``prep_for_interp`` intentionally
sets both data brackets to the first old level.  This module preserves that
constant-extension rule and the source REAL-weight-to-double-sum statement
order used by ``do_interp``.  It is an independent CPU/oracle implementation;
the Rust artifact producer owns full-domain numerical remapping.
"""

from __future__ import annotations

import hashlib
import importlib
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from cm1core.stretch_z1_child_15km import build_stretch_z1_child_15km_grid


CM1_RESTART_GRID_INTERP_SCHEMA = "gm1.cm1-r21.1-restart-grid-interp/v1"
CM1_RESTART_READ_SHA256 = (
    "72a041737a4025ce445cd303cd169eb6ff4c001a1140f9fca9e9cdb918884e2a"
)
CM1_RESTART_GRID_INTERP_CLAIM_LIMIT = (
    "exact CM1 r21.1 restart-grid coordinate search, lower/upper constant "
    "extension, and compact CPU trilinear arithmetic (REAL weight expressions "
    "assigned to DOUBLE PRECISION weights) for the selected fixed "
    "50 m child; no full-domain atmospheric artifact, hydrostatic rebalance, "
    "open-boundary initialization, forecast, storm, or tornado claim"
)
SOURCE_ANCHORS = MappingProxyType(
    {
        "scalar_sigma_maps": "src/restart_read.F:539-576",
        "velocity_face_maps": "src/restart_read.F:577-609",
        "selected_state_reads": "src/restart_read.F:1140-1213",
        "tke_restart_interp_suppression": "src/restart_read.F:1206-1211",
        "held_tendency_zero_on_interp": "src/restart_read.F:2657-2664",
        "first_and_last_level_constant_extension": "src/restart_read.F:3302-3406",
        "double_weight_trilinear_statement": "src/restart_read.F:3414-3465",
        "a_to_current_copy": "src/init3d.F:2280-2333",
    }
)

CM1_DIRECT_RESTART_READ = (
    "ua",
    "va",
    "wa",
    "ppi",
    "ppx",
    "tha",
    "qa",
)
CM1_A_TO_CURRENT_COPY = MappingProxyType(
    {
        "u3d": "ua",
        "v3d": "va",
        "w3d": "wa",
        "pp3d": "ppi",
        "th3d": "tha",
        "q3d": "qa",
    }
)
CM1_FRESH_ZERO_AFTER_GRID_INTERP = (
    "held.qpten",
    "held.qtten",
    "held.qvten",
    "held.qcten",
)
CM1_REINITIALIZE_AFTER_GRID_INTERP = ("closure.tkea",)
ARWEN_COMPONENT_POLICY_SEAM = (
    "component.h_diabatic",
    "component.effc",
    "component.effi",
    "component.effs",
    "component.mp_graupelnc",
    "component.mp_graupelncv",
    "component.mp_rainnc",
    "component.mp_rainncv",
    "component.mp_snownc",
    "component.mp_snowncv",
    "component.mp_sr",
)


@dataclass(frozen=True, slots=True)
class Cm1RestartAxisMap:
    lower: Any
    upper: Any
    fraction: Any
    below_support: Any
    above_support: Any

    @property
    def below_support_count(self) -> int:
        np = importlib.import_module("numpy")
        return int(np.count_nonzero(self.below_support))

    @property
    def above_support_count(self) -> int:
        np = importlib.import_module("numpy")
        return int(np.count_nonzero(self.above_support))


@dataclass(frozen=True, slots=True)
class Gm1FixedChildRestartMaps:
    scalar_x: Cm1RestartAxisMap
    scalar_y: Cm1RestartAxisMap
    u_face_x: Cm1RestartAxisMap
    v_face_y: Cm1RestartAxisMap
    scalar_sigma: Cm1RestartAxisMap
    w_sigmaf: Cm1RestartAxisMap
    receipt: Mapping[str, Any]


def validate_cm1_restart_interp_authority(authority_root: str | Path) -> Path:
    root = Path(authority_root).resolve(strict=True)
    path = root / "src/restart_read.F"
    if not root.is_dir() or Path(authority_root).is_symlink():
        raise ValueError("restart interpolation authority must be a real directory")
    if path.is_symlink() or not path.is_file():
        raise ValueError("missing regular src/restart_read.F authority")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != CM1_RESTART_READ_SHA256:
        raise ValueError(f"src/restart_read.F differs from CM1 r21.1: {actual}")
    return root


def _f32_vector(value: object, label: str) -> Any:
    np = importlib.import_module("numpy")
    if (
        not isinstance(value, np.ndarray)
        or value.dtype != np.dtype("float32")
        or value.ndim != 1
        or not value.flags.c_contiguous
        or value.size < 2
        or not bool(np.all(np.isfinite(value)))
    ):
        raise ValueError(f"{label} must be finite C-order float32 rank-1 with >=2 words")
    return value


def build_cm1_restart_axis_map(
    old_coordinates: object,
    new_coordinates: object,
    *,
    allow_constant_extension: bool,
) -> Cm1RestartAxisMap:
    """Build the source ``while old(next) < new`` bracket map.

    Exact matches retain the lower bracket with fraction one, matching the
    strict-less-than search.  Outside support, the data bracket itself is
    duplicated exactly as ``prep_for_interp`` does; no slope is invented.
    """

    np = importlib.import_module("numpy")
    old = _f32_vector(old_coordinates, "old restart coordinate")
    new = _f32_vector(new_coordinates, "new restart coordinate")
    if not bool(np.all(np.diff(old) > np.float32(0.0))):
        raise ValueError("old restart coordinates must be strictly increasing")
    lower = np.empty(new.size, dtype=np.int32)
    upper = np.empty(new.size, dtype=np.int32)
    fraction = np.empty(new.size, dtype=np.float32)
    below = np.less(new, old[0])
    above = np.greater(new, old[-1])
    if not allow_constant_extension and bool(np.any(below | above)):
        raise ValueError("new restart coordinates exceed the admitted old support")
    for index, coordinate in enumerate(new):
        if coordinate <= old[0]:
            lower[index] = 0
            upper[index] = 0
            fraction[index] = np.float32(0.0)
        elif coordinate > old[-1]:
            lower[index] = old.size - 1
            upper[index] = old.size - 1
            fraction[index] = np.float32(0.0)
        else:
            high = int(np.searchsorted(old, coordinate, side="left"))
            low = high - 1
            lower[index] = low
            upper[index] = high
            numerator = np.float32(coordinate - old[low])
            denominator = np.float32(old[high] - old[low])
            fraction[index] = np.float32(numerator / denominator)
    return Cm1RestartAxisMap(lower, upper, fraction, below, above)


def _parent_scalar_sigma(np: Any) -> Any:
    return np.ascontiguousarray(
        np.float32(100.0) + np.arange(75, dtype=np.float32) * np.float32(200.0)
    )


def _parent_w_sigmaf(np: Any) -> Any:
    return np.ascontiguousarray(
        np.arange(76, dtype=np.float32) * np.float32(200.0)
    )


def build_gm1_fixed_child_restart_maps() -> Gm1FixedChildRestartMaps:
    """Build the exact admitted parent-to-child coordinate maps."""

    np = importlib.import_module("numpy")
    grid = build_stretch_z1_child_15km_grid()
    parent_mass = np.arange(300, dtype=np.float32)
    parent_face = np.arange(301, dtype=np.float32)
    child_mass = np.ascontiguousarray(
        np.float32(149.25)
        + np.arange(300, dtype=np.float32) * np.float32(0.5)
    )
    child_face = np.ascontiguousarray(
        np.float32(149.5)
        + np.arange(301, dtype=np.float32) * np.float32(0.5)
    )
    scalar_x = build_cm1_restart_axis_map(
        parent_mass, child_mass, allow_constant_extension=False
    )
    scalar_y = build_cm1_restart_axis_map(
        parent_mass, child_mass, allow_constant_extension=False
    )
    u_face_x = build_cm1_restart_axis_map(
        parent_face, child_face, allow_constant_extension=False
    )
    v_face_y = build_cm1_restart_axis_map(
        parent_face, child_face, allow_constant_extension=False
    )
    scalar_sigma = build_cm1_restart_axis_map(
        _parent_scalar_sigma(np),
        np.ascontiguousarray(grid.sigma[1:76]),
        allow_constant_extension=True,
    )
    w_sigmaf = build_cm1_restart_axis_map(
        _parent_w_sigmaf(np),
        np.ascontiguousarray(grid.sigmaf[1:77]),
        allow_constant_extension=True,
    )
    receipt = MappingProxyType(
        {
            "schema": CM1_RESTART_GRID_INTERP_SCHEMA,
            "status": "PASS_EXACT_CM1_RESTART_GRID_MAPS",
            "source_sha256": CM1_RESTART_READ_SHA256,
            "source_anchors": dict(SOURCE_ANCHORS),
            "horizontal": {
                "scalar_parent_array_coordinate": "149.25 + 0.5 * child_index",
                "face_parent_array_coordinate": "149.5 + 0.5 * child_face_index",
                "strict_less_than_exact_match_fraction": 1.0,
            },
            "vertical": {
                "scalar_coordinate": "sigma",
                "w_coordinate": "sigmaf",
                "scalar_below_parent_k1_levels": scalar_sigma.below_support_count,
                "w_below_parent_first_face_levels": w_sigmaf.below_support_count,
                "below_support_policy": "duplicate first old data plane exactly",
                "above_support_policy": "duplicate last old data plane exactly",
            },
            "carrier_disposition": {
                "direct_restart_read": list(CM1_DIRECT_RESTART_READ),
                "a_to_current_copy": dict(CM1_A_TO_CURRENT_COPY),
                "fresh_zero_after_grid_interp": list(
                    CM1_FRESH_ZERO_AFTER_GRID_INTERP
                ),
                "reinitialize_after_grid_interp": list(
                    CM1_REINITIALIZE_AFTER_GRID_INTERP
                ),
                "arwen_component_policy_seam": list(ARWEN_COMPONENT_POLICY_SEAM),
            },
            "claim_limit": CM1_RESTART_GRID_INTERP_CLAIM_LIMIT,
        }
    )
    return Gm1FixedChildRestartMaps(
        scalar_x,
        scalar_y,
        u_face_x,
        v_face_y,
        scalar_sigma,
        w_sigmaf,
        receipt,
    )


def cm1_do_interp_point(
    lower_plane: object,
    upper_plane: object,
    *,
    ix: int,
    iy: int,
    rx: object,
    ry: object,
    rz: object,
) -> Any:
    """Evaluate one ``do_interp`` point in written double statement order."""

    np = importlib.import_module("numpy")
    low = np.asarray(lower_plane)
    high = np.asarray(upper_plane)
    if (
        low.dtype != np.dtype("float32")
        or high.dtype != np.dtype("float32")
        or low.ndim != 2
        or high.shape != low.shape
        or ix < 0
        or iy < 0
        or ix + 1 >= low.shape[1]
        or iy + 1 >= low.shape[0]
    ):
        raise ValueError("restart interpolation planes/index do not satisfy one cell")
    # The source declares rx/ry/rz and literal 1.0 as default REAL, while the
    # destinations w1..w8 are DOUBLE PRECISION.  Consequently every RHS
    # subtraction/multiplication is rounded in binary32 before assignment
    # widens the completed weight to binary64.  Widening x/y/z first is a
    # scientifically material one-word deviation for some inputs.
    x = np.float32(rx)
    y = np.float32(ry)
    z = np.float32(rz)
    one = np.float32(1.0)
    one_x = np.float32(one - x)
    one_y = np.float32(one - y)
    one_z = np.float32(one - z)

    def source_real_product(left: Any, middle: Any, right: Any) -> Any:
        return np.float64(np.float32(np.float32(left * middle) * right))

    w1 = source_real_product(one_x, one_y, one_z)
    w2 = source_real_product(x, one_y, one_z)
    w3 = source_real_product(one_x, y, one_z)
    w4 = source_real_product(one_x, one_y, z)
    w5 = source_real_product(x, one_y, z)
    w6 = source_real_product(one_x, y, z)
    w7 = source_real_product(x, y, one_z)
    w8 = source_real_product(x, y, z)
    result = np.float64(low[iy, ix]) * w1
    result = result + np.float64(low[iy, ix + 1]) * w2
    result = result + np.float64(low[iy + 1, ix]) * w3
    result = result + np.float64(high[iy, ix]) * w4
    result = result + np.float64(high[iy, ix + 1]) * w5
    result = result + np.float64(high[iy + 1, ix]) * w6
    result = result + np.float64(low[iy + 1, ix + 1]) * w7
    result = result + np.float64(high[iy + 1, ix + 1]) * w8
    return np.float32(result)


__all__ = (
    "ARWEN_COMPONENT_POLICY_SEAM",
    "CM1_A_TO_CURRENT_COPY",
    "CM1_DIRECT_RESTART_READ",
    "CM1_FRESH_ZERO_AFTER_GRID_INTERP",
    "CM1_REINITIALIZE_AFTER_GRID_INTERP",
    "CM1_RESTART_GRID_INTERP_CLAIM_LIMIT",
    "CM1_RESTART_GRID_INTERP_SCHEMA",
    "CM1_RESTART_READ_SHA256",
    "Cm1RestartAxisMap",
    "Gm1FixedChildRestartMaps",
    "SOURCE_ANCHORS",
    "build_cm1_restart_axis_map",
    "build_gm1_fixed_child_restart_maps",
    "cm1_do_interp_point",
    "validate_cm1_restart_interp_authority",
)
