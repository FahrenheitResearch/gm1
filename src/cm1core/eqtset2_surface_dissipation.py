"""CM1 r21.1 ``bbc=3`` surface dissipative heating at the lowest model level.

With a free-slip or no-slip lower boundary CM1 heats every mass level with
the z-face average of the SGS dissipation rate, which is what
:class:`cm1core.eqtset2_energy_terms.Eqtset2IdissConsumer` owns.  With the
semi-slip lower boundary CM1 does something different at exactly one level:
``src/solve1.F:932-935`` sets the interior loop's first index ``k1`` to 2, so
the average never reaches ``k=1``, and ``src/solve1.F:950-962`` (moist,
``eqtset=2``) and ``src/solve1.F:976-988`` (dry/traditional) then heat
``k=1`` with the surface-layer dissipation rate

    epsd = ust**3 * log((dz1 + znt)/znt) / (karman * dz1)

where ``dz1 = zf(i,j,2)-zf(i,j,1)``.  That is the frictional heat the
surface stress actually does on the flow, and without it a semi-slip run
removes kinetic energy at the ground and never puts the equivalent internal
energy back.

This module owns that one level.  The interior levels stay with the already
native-verified consumer, driven over the ``k=2..nk`` slice, so ``bbc=3``
reuses the sealed operator for every level it did not change and nothing is
transcribed twice.
"""

from __future__ import annotations

import hashlib
import importlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, NoReturn

from .errors import refuse
from .eqtset2_energy_terms import (
    CM1_CP,
    CM1_CPDCV,
    CM1_CPI,
    CM1_CPL,
    CM1_CV,
    CM1_CVV,
    EQTSET2_SPECIES,
)


SURFACE_DISSIPATION_SCHEMA = "gpuwm-cm1.eqtset2-surface-dissipation/v1"
SURFACE_DISSIPATION_GATE = "surface-drag"
SURFACE_DISSIPATION_CLAIM_LIMIT = (
    "CM1 r21.1 bbc=3 lowest-model-level dissipative heating for moist "
    "eqtset=2 (src/solve1.F:950-962) and the dry/traditional equations "
    "(src/solve1.F:976-988), on a supplied surface dissipation rate; the "
    "interior levels, the surface stress that produces the rate, coupled "
    "steps, trajectory, forecast, storm and tornado claims are excluded"
)

SOURCE_ANCHORS = MappingProxyType(
    {
        "interior_loop_first_index": "src/solve1.F:932-935",
        "moist_surface_level": "src/solve1.F:950-962",
        "dry_surface_level": "src/solve1.F:976-988",
        "moist_heat_capacity": "src/solve1.F:744-834",
        "surface_dissipation_rate": "src/solve1.F:956-957",
        "karman": "src/constants.F:23",
    }
)
AUTHORITY_TARBALL_SHA256 = (
    "dc49fe84531056d1ae6249b37a5e3ee453fd96861c3b6bafd63828d92e64edf7"
)
AUTHORITY_SOURCE_SHA256 = MappingProxyType(
    {
        "src/constants.F": (
            "71cb7f92163a247c6f76df9d9d563f9fe4e1470123d0480a8c16762c2e0c8f94"
        ),
        "src/solve1.F": (
            "515346f18268aea39e7a919536ee6187386e5db6d184e6e4c597509c701e2076"
        ),
    }
)

# The order the heating reads its operands, exactly as solve1.F writes them.
CARRIER_NAMES: tuple[str, ...] = (
    "thten1",
    "epsd_surface",
    "pi0",
    "ppi",
) + EQTSET2_SPECIES


@dataclass(frozen=True, slots=True)
class SurfaceDissipationCarriers:
    """The lowest model level's heating operands.

    Every 3-D carrier is the full ``(nz, ny, nx)`` array the caller already
    owns; this operator reads and writes index 0 only, so nothing is copied
    and no second atmosphere exists.  ``epsd_surface`` is the ``(ny, nx)``
    surface dissipation rate :mod:`cm1core.cuda_surface_drag` produces from
    ``ust``, ``znt`` and ``zf``.
    """

    thten1: Any
    epsd_surface: Any
    pi0: Any
    ppi: Any
    qv: Any
    qc: Any
    qr: Any
    qi: Any
    qs: Any
    qg: Any


def _refuse(*, code: str, reason: str, remedy: str, option: str, value: object) -> NoReturn:
    refuse(
        code=code,
        gate=SURFACE_DISSIPATION_GATE,
        reason=reason,
        remedy=remedy,
        option=option,
        value=value,
    )


def _shape(value: object) -> tuple[int, ...] | None:
    raw = getattr(value, "shape", None)
    if raw is None:
        return None
    try:
        return tuple(int(item) for item in raw)
    except (TypeError, ValueError, OverflowError):
        return None


def _is_f32(value: object) -> bool:
    dtype = getattr(value, "dtype", None)
    return bool(dtype is not None and getattr(dtype, "name", None) == "float32")


def _validate(
    carriers: SurfaceDissipationCarriers, *, nx: int, ny: int, nz: int
) -> None:
    if not isinstance(carriers, SurfaceDissipationCarriers):
        _refuse(
            code="CM1-SFC-DISS-CARRIERS",
            reason="the surface heating needs one complete named carrier set",
            remedy="construct SurfaceDissipationCarriers with all ten fields",
            option="carriers",
            value=type(carriers).__name__,
        )
    for name, value, minimum in (("nx", nx, 1), ("ny", ny, 1), ("nz", nz, 2)):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            _refuse(
                code="CM1-SFC-DISS-DIMENSION",
                reason=f"{name} must be an integer >= {minimum}",
                remedy="bind the exact physical-domain dimensions",
                option=name,
                value=value,
            )
    volume = (nz, ny, nx)
    for name in CARRIER_NAMES:
        value = getattr(carriers, name)
        if name == "epsd_surface":
            admitted = ((ny, nx),)
        elif name == "pi0":
            admitted = ((nz,), volume)
        else:
            admitted = (volume,)
        if _shape(value) not in admitted:
            _refuse(
                code="CM1-SFC-DISS-SHAPE",
                reason=f"{name} has shape {_shape(value)}, not one of {admitted}",
                remedy="bind exact C-contiguous z-y-x or y-x binary32 carriers",
                option=f"carriers.{name}.shape",
                value=_shape(value),
            )
        if not _is_f32(value):
            _refuse(
                code="CM1-SFC-DISS-LAYOUT",
                reason=f"{name} is not IEEE-754 binary32",
                remedy="bind raw binary32 storage without conversion",
                option=f"carriers.{name}.dtype",
                value=str(getattr(value, "dtype", None)),
            )
    if any(
        carriers.thten1 is getattr(carriers, name) for name in CARRIER_NAMES[1:]
    ):
        _refuse(
            code="CM1-SFC-DISS-ALIAS",
            reason="the mutable tendency aliases one of its own inputs",
            remedy="bind distinct tendency and input arrays",
            option="carriers.thten1.identity",
            value="aliased",
        )


def _require_equation_set(equation_set: object) -> bool:
    """Select CM1's moist or dry heat capacity, or refuse.

    ``src/solve1.F:947`` branches on ``imoist.eq.1.and.eqtset.eq.2``: the
    moist branch divides by ``cpdcv*cvm*(pi0+ppi)`` and the traditional one
    by ``cp*(pi0+ppi)``.  Those are different amounts of heat for the same
    dissipation rate, so the branch is selected by value, never inferred.
    """

    if isinstance(equation_set, bool) or not isinstance(equation_set, int):
        _refuse(
            code="CM1-SFC-DISS-EQTSET",
            reason="the equation set selects a different heat capacity",
            remedy="pass equation_set=1 or 2",
            option="equation_set",
            value=equation_set,
        )
    if equation_set not in (1, 2):
        _refuse(
            code="CM1-SFC-DISS-EQTSET",
            reason=(
                "CM1's dissipative heating has exactly two heat-capacity "
                "branches (src/solve1.F:947-988); another equation set would "
                "silently take the moist one and deposit the wrong amount of "
                "internal energy at the ground"
            ),
            remedy="pass equation_set=2 for the moist energy-conserving set, or 1",
            option="equation_set",
            value=equation_set,
        )
    return equation_set == 2


def _surface_level(value: Any, *, ny: int, nx: int) -> Any:
    """Return the ``k=1`` slab of a carrier, as a view, never a copy."""

    shape = _shape(value)
    if shape == (ny, nx):
        return value
    if len(shape) == 1:
        return value[0]
    return value[0]


def _apply(xp: Any, carriers: SurfaceDissipationCarriers, *, nx: int, ny: int, nz: int, moist: bool) -> Any:
    """Transcribe ``src/solve1.F:950-962`` for the single ``k=1`` level.

    Every statement is one binary32 array operation in the source's own
    order, which is the same order the interior consumer uses at
    ``src/solve1.F:937-949`` for the levels it keeps.
    """

    f32 = xp.float32
    level = {
        name: _surface_level(getattr(carriers, name), ny=ny, nx=nx)
        for name in CARRIER_NAMES
    }
    pi0 = level["pi0"]
    if _shape(pi0) == ():
        pi0 = xp.full((ny, nx), pi0, dtype=f32)
    dheat = xp.empty((ny, nx), dtype=f32)
    work = xp.empty((ny, nx), dtype=f32)
    if moist:
        # solve1.F:750-834, exact species and accumulation order.
        ql = xp.empty((ny, nx), dtype=f32)
        qice = xp.empty((ny, nx), dtype=f32)
        cvm = xp.empty((ny, nx), dtype=f32)
        xp.copyto(ql, level["qc"])
        xp.add(ql, level["qr"], out=ql)
        xp.copyto(qice, level["qi"])
        xp.add(qice, level["qs"], out=qice)
        xp.add(qice, level["qg"], out=qice)
        xp.multiply(level["qv"], f32(CM1_CVV), out=cvm)
        xp.add(cvm, f32(CM1_CV), out=cvm)
        xp.multiply(ql, f32(CM1_CPL), out=work)
        xp.add(cvm, work, out=cvm)
        xp.multiply(qice, f32(CM1_CPI), out=work)
        xp.add(cvm, work, out=cvm)
        # solve1.F:958.  dheat = epsd/( cpdcv*cvm*(pi0+ppi) )
        xp.add(pi0, level["ppi"], out=work)
        xp.multiply(cvm, f32(CM1_CPDCV), out=cvm)
        xp.multiply(cvm, work, out=cvm)
        xp.divide(level["epsd_surface"], cvm, out=dheat)
    else:
        # solve1.F:984.  The traditional cloud-modeling and dry equations
        # divide by cp alone: dheat = epsd/( cp*(pi0+ppi) ).
        xp.add(pi0, level["ppi"], out=work)
        xp.multiply(work, f32(CM1_CP), out=work)
        xp.divide(level["epsd_surface"], work, out=dheat)
    xp.add(level["thten1"], dheat, out=level["thten1"])
    return dheat


def cpu_surface_dissipative_heating_inplace(
    carriers: SurfaceDissipationCarriers,
    *,
    nx: int,
    ny: int,
    nz: int,
    equation_set: int = 2,
) -> dict[str, object]:
    """Host reference for the same statements the device path executes."""

    np = importlib.import_module("numpy")
    _validate(carriers, nx=nx, ny=ny, nz=nz)
    moist = _require_equation_set(equation_set)
    _apply(np, carriers, nx=nx, ny=ny, nz=nz, moist=moist)
    return {
        "schema": SURFACE_DISSIPATION_SCHEMA,
        "claim_limit": SURFACE_DISSIPATION_CLAIM_LIMIT,
        "authority": dict(AUTHORITY_SOURCE_SHA256),
        "authority_tarball_sha256": AUTHORITY_TARBALL_SHA256,
        "anchors": dict(SOURCE_ANCHORS),
        "bottom_boundary": 3,
        "equation_set": int(equation_set),
        "levels_heated": 1,
        "surface_words": ny * nx,
        "host_reference": True,
    }


def apply_surface_dissipative_heating(
    carriers: SurfaceDissipationCarriers,
    *,
    nx: int,
    ny: int,
    nz: int,
    equation_set: int = 2,
    stream: Any | None = None,
) -> dict[str, object]:
    """Heat the lowest model level with the surface dissipation rate.

    Device-resident: every operand stays where the caller put it and only
    the ``k=1`` slab of ``thten1`` is written.
    """

    cupy = importlib.import_module("cupy")
    _validate(carriers, nx=nx, ny=ny, nz=nz)
    moist = _require_equation_set(equation_set)
    ndarray = cupy.ndarray
    for name in CARRIER_NAMES:
        value = getattr(carriers, name)
        if not isinstance(value, ndarray):
            _refuse(
                code="CM1-SFC-DISS-NONCUDA",
                reason=f"{name} is not a CuPy device array",
                remedy="bind the live CUDA carrier",
                option=f"carriers.{name}",
                value=type(value).__name__,
            )
    stream = cupy.cuda.get_current_stream() if stream is None else stream
    with stream:
        _apply(cupy, carriers, nx=nx, ny=ny, nz=nz, moist=moist)
        event = cupy.cuda.Event(disable_timing=True)
        event.record(stream)
    return {
        "schema": SURFACE_DISSIPATION_SCHEMA,
        "claim_limit": SURFACE_DISSIPATION_CLAIM_LIMIT,
        "authority": dict(AUTHORITY_SOURCE_SHA256),
        "authority_tarball_sha256": AUTHORITY_TARBALL_SHA256,
        "anchors": dict(SOURCE_ANCHORS),
        "bottom_boundary": 3,
        "equation_set": int(equation_set),
        "levels_heated": 1,
        "surface_words": ny * nx,
        "ready_event": event,
        "host_copy": False,
    }


def module_sha256() -> str:
    from pathlib import Path

    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


__all__ = [
    "AUTHORITY_SOURCE_SHA256",
    "AUTHORITY_TARBALL_SHA256",
    "CARRIER_NAMES",
    "SOURCE_ANCHORS",
    "SURFACE_DISSIPATION_CLAIM_LIMIT",
    "SURFACE_DISSIPATION_SCHEMA",
    "SurfaceDissipationCarriers",
    "apply_surface_dissipative_heating",
    "cpu_surface_dissipative_heating_inplace",
    "module_sha256",
]
