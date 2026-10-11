"""Immutable views of Rust-produced, source-ordered CM1 Rayleigh profiles.

Python allocates the output arrays and binds the C ABI.  Rust owns geometry,
coefficient arithmetic and source default-REAL rounding.  Production needs no
Fortran compiler, GPU context or Python numerical transcription.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from types import MappingProxyType
from typing import Mapping


RAYLEIGH_PROFILES_SCHEMA = "gm1.cm1-r21.1-uniform-rayleigh-profiles/v1"
RAYLEIGH_PARAMETER_NAMES = (
    "dz_m", "model_top_m", "damping_start_m", "damping_depth_m", "damping_rate_s_inv",
)
_SOURCE_AUTHORITY = {
    "param_f_sha256": "cac64a6cb4363c6b88367b5cb9391f1bcf2130c63ffedef6e5973c03b190c349",
    "constants_f_sha256": "71cb7f92163a247c6f76df9d9d563f9fe4e1470123d0480a8c16762c2e0c8f94",
    "uniform_grid": "src/param.F:7789-7811",
    "stretched_grid": "src/param.F:7814-8225",
    "model_top_maxz": "src/param.F:8193",
    "active_depth_guard": "src/param.F:8633-8650",
    "coefficients": "src/param.F:8652-8671",
    "pi": "src/constants.F:14",
}


@dataclass(frozen=True, slots=True)
class UniformRayleighProfiles:
    tauh: object
    taus: object
    tauf: object
    parameters: Mapping[str, float]
    receipt: Mapping[str, object]


def _configure(library: ctypes.CDLL) -> None:
    f32p = ctypes.POINTER(ctypes.c_float)
    library.gm1_rayleigh_profiles_fill.argtypes = (
        ctypes.c_size_t, ctypes.c_double, ctypes.c_double, ctypes.c_double,
        f32p, ctypes.c_size_t, f32p, ctypes.c_size_t,
        f32p, ctypes.c_size_t, f32p, ctypes.c_size_t,
    )
    library.gm1_rayleigh_profiles_fill.restype = ctypes.c_int32
    cf32p = ctypes.POINTER(ctypes.c_float)
    library.gm1_rayleigh_profiles_fill_levels.argtypes = (
        ctypes.c_size_t, cf32p, ctypes.c_size_t, cf32p, ctypes.c_size_t,
        ctypes.c_double, ctypes.c_double, ctypes.c_double,
        f32p, ctypes.c_size_t, f32p, ctypes.c_size_t,
        f32p, ctypes.c_size_t, f32p, ctypes.c_size_t,
    )
    library.gm1_rayleigh_profiles_fill_levels.restype = ctypes.c_int32
    library.gm1_rayleigh_profiles_last_error_len.argtypes = ()
    library.gm1_rayleigh_profiles_last_error_len.restype = ctypes.c_size_t
    library.gm1_rayleigh_profiles_last_error_copy.argtypes = (
        ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t,
    )
    library.gm1_rayleigh_profiles_last_error_copy.restype = ctypes.c_size_t


def _last_error(library: ctypes.CDLL) -> str:
    length = int(library.gm1_rayleigh_profiles_last_error_len())
    if length == 0:
        return "Rust Rayleigh profile producer refused without an error message"
    buffer = (ctypes.c_uint8 * length)()
    if int(library.gm1_rayleigh_profiles_last_error_copy(buffer, length)) != length:
        return "Rust Rayleigh profile error changed while being copied"
    return bytes(buffer).decode("utf-8", errors="strict")


def build_uniform_rayleigh_profiles(
    library_path: str | Path,
    *,
    nz: int,
    dz_m: float,
    damping_depth_m: float = 3000.0,
    damping_rate_s_inv: float = 0.0033333333333,
) -> UniformRayleighProfiles:
    """Build dimensionless tau profiles and return the effective f32 settings.

    Alpha is returned as ``parameters['damping_rate_s_inv']`` for the solve1
    consumer.  It is deliberately absent from the coefficient multiplication.
    The supplied values are rounded to CM1 default REAL inside Rust before
    source-ordered geometry or coefficient arithmetic is performed.
    """
    if isinstance(nz, bool) or not isinstance(nz, int) or not 1 <= nz <= 2**31 - 3:
        raise ValueError("Rayleigh nz must be a positive CM1 integer with room for ghost indices")
    for name, value in ("dz_m", dz_m), ("damping_depth_m", damping_depth_m), (
        "damping_rate_s_inv", damping_rate_s_inv,
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"Rayleigh {name} must be a real scalar")
    if sys.byteorder != "little":
        raise ValueError("Rayleigh C ABI requires a little-endian host for its raw f32 contract")
    path = Path(library_path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("Rayleigh library must be a regular non-symlink file")
    path = path.resolve(strict=True)
    library = ctypes.CDLL(str(path))
    _configure(library)
    import numpy as np

    tauh = np.empty(nz, dtype="<f4")
    taus = np.empty(nz, dtype="<f4")
    tauf = np.empty(nz + 1, dtype="<f4")
    parameter_words = np.empty(len(RAYLEIGH_PARAMETER_NAMES), dtype="<f4")
    f32p = ctypes.POINTER(ctypes.c_float)
    status = library.gm1_rayleigh_profiles_fill(
        nz, dz_m, damping_depth_m, damping_rate_s_inv,
        tauh.ctypes.data_as(f32p), tauh.size,
        taus.ctypes.data_as(f32p), taus.size,
        tauf.ctypes.data_as(f32p), tauf.size,
        parameter_words.ctypes.data_as(f32p), parameter_words.size,
    )
    if status != 0:
        raise ValueError(_last_error(library))
    for value in (tauh, taus, tauf, parameter_words):
        value.flags.writeable = False
    parameters = dict(zip(RAYLEIGH_PARAMETER_NAMES, map(float, parameter_words), strict=True))
    receipt = {
        "schema": RAYLEIGH_PROFILES_SCHEMA,
        "producer": "Rust gm1_rayleigh_profiles_fill",
        "library_path": str(path),
        "library_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_authority": dict(_SOURCE_AUTHORITY),
        "nz": nz,
        "requested_parameters": {
            "dz_m": dz_m, "damping_depth_m": damping_depth_m,
            "damping_rate_s_inv": damping_rate_s_inv,
        },
        "effective_f32_parameters": dict(parameters),
        "effective_f32_parameter_bits": {
            name: f"{int(bits):08x}" for name, bits in zip(
                RAYLEIGH_PARAMETER_NAMES, parameter_words.view(np.uint32), strict=True,
            )
        },
        "profile_sha256": {
            name: hashlib.sha256(value.tobytes(order="C")).hexdigest()
            for name, value in (("tauh", tauh), ("taus", taus), ("tauf", tauf))
        },
        "dtype": "<f4",
        "alpha_applied_to_coefficients": False,
        "cosine_implementation": "compiled Rust target f32.cos; host libm",
        "python_scientific_arithmetic": False,
        "claim_limit": (
            "source-ordered uniform-grid Rayleigh profiles only; native raw-bit "
            "and cross-platform cosine identity require separate measurements"
        ),
    }
    # Refuse unrepresentable receipt inputs instead of silently serializing NaN.
    json.dumps(receipt, allow_nan=False, sort_keys=True)
    return UniformRayleighProfiles(
        tauh, taus, tauf, MappingProxyType(parameters), MappingProxyType(receipt),
    )


def build_rayleigh_profiles(
    library_path: str | Path,
    *,
    levels: object,
    damping_depth_m: float = 3000.0,
    damping_rate_s_inv: float = 0.0033333333333,
) -> UniformRayleighProfiles:
    """Build the tau profiles of one declared CM1 vertical coordinate.

    ``levels`` is a :class:`cm1core.cm1_vertical_grid.VerticalLevels`.  A
    ``stretch_z=0`` coordinate takes the established ``nz``/``dz_m`` entry
    point so its raw words, receipt and library call are exactly what they
    were; anything else hands ``zh(1..nk)`` and ``zf(1..nk+1)`` to Rust,
    because ``param.F:8652-8671`` reads those two arrays and no scalar ``dz``,
    and ``param.F:8193``'s ``maxz`` is ``zf(nk+1)`` rather than ``nz*dz``.
    """

    from .cm1_vertical_grid import VerticalLevels

    if not isinstance(levels, VerticalLevels):
        raise TypeError("levels must be cm1_vertical_grid.VerticalLevels")
    if levels.stretch_z == 0:
        return build_uniform_rayleigh_profiles(
            library_path, nz=levels.nk, dz_m=levels.dz_reference_m,
            damping_depth_m=damping_depth_m, damping_rate_s_inv=damping_rate_s_inv,
        )
    for name, value in ("damping_depth_m", damping_depth_m), (
        "damping_rate_s_inv", damping_rate_s_inv,
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"Rayleigh {name} must be a real scalar")
    if sys.byteorder != "little":
        raise ValueError("Rayleigh C ABI requires a little-endian host for its raw f32 contract")
    path = Path(library_path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("Rayleigh library must be a regular non-symlink file")
    path = path.resolve(strict=True)
    library = ctypes.CDLL(str(path))
    _configure(library)
    import numpy as np

    nz = levels.nk
    zh = np.ascontiguousarray(np.asarray(levels.zh, dtype="<f4")[1 : nz + 1])
    zf = np.ascontiguousarray(np.asarray(levels.zf, dtype="<f4")[1 : nz + 2])
    tauh = np.empty(nz, dtype="<f4")
    taus = np.empty(nz, dtype="<f4")
    tauf = np.empty(nz + 1, dtype="<f4")
    parameter_words = np.empty(len(RAYLEIGH_PARAMETER_NAMES), dtype="<f4")
    f32p = ctypes.POINTER(ctypes.c_float)
    status = library.gm1_rayleigh_profiles_fill_levels(
        nz,
        zh.ctypes.data_as(f32p), zh.size,
        zf.ctypes.data_as(f32p), zf.size,
        levels.dz_reference_m, damping_depth_m, damping_rate_s_inv,
        tauh.ctypes.data_as(f32p), tauh.size,
        taus.ctypes.data_as(f32p), taus.size,
        tauf.ctypes.data_as(f32p), tauf.size,
        parameter_words.ctypes.data_as(f32p), parameter_words.size,
    )
    if status != 0:
        raise ValueError(_last_error(library))
    for value in (tauh, taus, tauf, parameter_words):
        value.flags.writeable = False
    parameters = dict(zip(RAYLEIGH_PARAMETER_NAMES, map(float, parameter_words), strict=True))
    receipt = {
        "schema": RAYLEIGH_PROFILES_SCHEMA,
        "producer": "Rust gm1_rayleigh_profiles_fill_levels",
        "library_path": str(path),
        "library_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_authority": dict(_SOURCE_AUTHORITY),
        "nz": nz,
        "vertical_coordinate": levels.record(),
        "requested_parameters": {
            "dz_reference_m": levels.dz_reference_m,
            "damping_depth_m": damping_depth_m,
            "damping_rate_s_inv": damping_rate_s_inv,
        },
        "effective_f32_parameters": dict(parameters),
        "effective_f32_parameter_bits": {
            name: f"{int(bits):08x}" for name, bits in zip(
                RAYLEIGH_PARAMETER_NAMES, parameter_words.view(np.uint32), strict=True,
            )
        },
        "profile_sha256": {
            name: hashlib.sha256(value.tobytes(order="C")).hexdigest()
            for name, value in (("tauh", tauh), ("taus", taus), ("tauf", tauf))
        },
        "dtype": "<f4",
        "alpha_applied_to_coefficients": False,
        "cosine_implementation": "compiled Rust target f32.cos; host libm",
        "python_scientific_arithmetic": False,
        "claim_limit": (
            "source-ordered Rayleigh profiles over one declared CM1 vertical "
            "coordinate only; native raw-bit and cross-platform cosine "
            "identity require separate measurements"
        ),
    }
    json.dumps(receipt, allow_nan=False, sort_keys=True)
    return UniformRayleighProfiles(
        tauh, taus, tauf, MappingProxyType(parameters), MappingProxyType(receipt),
    )


__all__ = [
    "RAYLEIGH_PARAMETER_NAMES", "RAYLEIGH_PROFILES_SCHEMA", "UniformRayleighProfiles",
    "build_rayleigh_profiles", "build_uniform_rayleigh_profiles",
]
