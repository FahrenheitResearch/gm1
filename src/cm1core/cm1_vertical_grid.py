"""CM1 r21.1 vertical coordinate: the one place a level set is built.

Every GM1 consumer of a vertical grid -- the closure metrics, the Rayleigh
profiles, the acoustic solver's sigma/sigmaf/rds/rdsf, the terrain-open
Fornberg lower/upper boundary weights, and the Rust initializer's mass
heights -- reads its levels from here rather than reconstructing them from a
scalar ``dz``.  A uniform grid is the ``stretch_z=0`` member of this family,
not a separate code path: ``param.F:7789-7811`` builds it with exactly the
arithmetic ``param.F:8162-8209`` then applies to every stretched branch.

Authority (CM1 r21.1, ``src/param.F``):

* ``:2608``   ``stretch_z<1`` overwrites ``ztop = dz*float(nk)``.
* ``:2609``   ``stretch_z=2`` overwrites ``dz = ztop/float(nk)``.
* ``:7789-7811``  uniform ``zf``/``zh``/``sigmaf``/``sigma`` and unit metrics.
* ``:7814``       the ``stretch_z>=1`` selector.
* ``:7819-7953``  ``stretch_z=1``, the Wilhelmson-Chen analytic function:
  ``:7823-7825`` the segment counts, ``:7828-7877`` the four ``stopcm1``
  feasibility checks, ``:7881-7882`` ``c1``/``c2``, ``:7925-7934`` ``zf``,
  ``:7941-7953`` ``sigmaf`` and its two ghosts.
* ``:7959-8003``  ``stretch_z=2``, the COMMAS geometric ``ZGRID`` generator
  (``:9877-9943``) over ``ZHEIGHT`` (``:9951-9996``).  Read, and refused by
  value: see :data:`STRETCH_Z2_REFUSAL`.
* ``:8005-8064``  ``stretch_z=3``, user scalar (half) levels; ``:8025`` is the
  alternating face recursion and ``:8037`` takes the generated top.
* ``:8072-8132``  ``stretch_z=4``, user w (full) levels; ``:8087`` reads them
  and ``:8105`` takes the generated top.
* ``:8162-8182``  the shared ghost/``zh``/``mh``/``mf`` tail.
* ``:8189-8209``  ``sigma``, ``maxz``, ``rmh``, ``rmf``.
* ``:8212-8224``  the positive-spacing dummy check.
* ``:8985-9060``  Fornberg (1988) 2nd-order extrapolation coefficients
  ``cgs1..3`` (``:9038-9040``) and ``cgt1..3`` (``:9046-9048``).
* ``init_terrain.F:292-298``  ``rds`` and ``rdsf``.

This module performs no device work and allocates no atmospheric field.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import math
from typing import Any, Sequence


VERTICAL_GRID_SCHEMA = "gm1.cm1-r21.1-vertical-grid/v1"

#: ``stretch_z`` values this port builds.  ``param.F:7814-8139`` defines one
#: more, ``stretch_z=2``; :data:`STRETCH_Z2_REFUSAL` says why it is not here.
ADMITTED_STRETCH_Z = (0, 1, 3, 4)

SOURCE_ANCHORS = {
    "ztop_overwrite": "src/param.F:2608",
    "dz_overwrite_stretch_z2": "src/param.F:2609",
    "uniform_levels": "src/param.F:7789-7811",
    "selector": "src/param.F:7814-7819",
    "wilhelmson_chen": "src/param.F:7819-7953",
    "geometric_zgrid": "src/param.F:7959-8003;9877-9943;9951-9996",
    "user_scalar_levels": "src/param.F:8005-8064",
    "user_w_levels": "src/param.F:8072-8132",
    "shared_tail": "src/param.F:8162-8182",
    "inverse_metrics": "src/param.F:8189-8209",
    "positive_spacing_check": "src/param.F:8212-8224",
    "fornberg_weights": "src/param.F:8985-9060",
    "rds_rdsf": "src/init_terrain.F:292-298",
}

#: Why ``stretch_z=2`` is refused rather than generated.
#:
#: ``param.F:2609`` forces ``dz = ztop/nk`` for this branch and
#: ``param.F:9897`` makes ``dz*(ke-1)`` the bisection target, but ``ZGRID``
#: never checks its own residual: ``param.F:9901-9917`` halves ``zx`` at most
#: fifty times, accepts whatever ``stretch`` it is holding, and
#: ``param.F:9933-9938`` then fills ``gze`` from that ratio.  CM1 keeps the two
#: resulting numbers apart -- the namelist ``ztop`` still drives the terrain
#: coordinate at ``init_terrain.F:175-196``, while ``maxz = sigmaf(nk+1)``
#: (``param.F:8193``) drives the Rayleigh damper at ``param.F:8633-8650`` --
#: and neither is derived from the other.  GM1's sealed geometry carries
#: exactly one model top, and :class:`VerticalLevels` publishes it in the run
#: receipt, so admitting this branch would bind one of those two consumers to
#: the wrong height by an amount no CM1 check bounds.  ``ZGRID`` is also the
#: one branch with no distributed deck: ``run/config_files`` selects
#: ``stretch_z`` 0, 1 and 3 only, so there is no oracle to fix the choice
#: against.
STRETCH_Z2_REFUSAL = (
    "CM1 r21.1 param.F:9897-9937 accepts whatever geometric ratio its "
    "fifty-step bisection is holding and never bounds the residual, so the "
    "generated top face sigmaf(nk+1) and the namelist ztop are two different "
    "heights; init_terrain.F:175-196 uses the second and param.F:8633-8650 uses "
    "the first, while a GM1 grid carries exactly one model top, so one of the "
    "terrain coordinate and the Rayleigh layer would sit at the wrong height"
)

#: The f32 words the compiled GM1 kernels carry inline for the lower and upper
#: boundary quadratic extrapolation.  They are the exact Fornberg weights of
#: *any* grid whose lowest (respectively highest) three scalar levels are
#: uniformly spaced, which includes every uniform grid and every stretch_z=1
#: grid with nk1>=3 and nk3>=3.
#:
#: They are *analytic*, and the spacing cancels out of them.  ``param.F:8990-
#: 8993`` divides every node by ``sigmaf(2)-sigmaf(1)``, so on a column whose
#: lowest three layers all have depth ``h`` the Fornberg nodes are
#: ``alpha = (0.5, 1.5, 2.5)`` and the extrapolation point is ``x0 = 0``
#: whatever ``h`` is; the Lagrange weights at ``x0`` are then
#: ``(1.875, -1.25, 0.375)`` exactly, for every ``h``.  ``param.F:9004-9007``
#: normalizes the upper stencil the same way against ``sigmaf(nk+1)-
#: sigmaf(nk)``, with the same result.
PINNED_FORNBERG_F32 = (1.875, -1.25, 0.375)


class VerticalGridError(ValueError):
    """A level set CM1 itself would stop on, or that GM1 cannot carry."""


# ---------------------------------------------------------------------------
# Is the pinned quadratic this grid's own?  -- param.F:8985-9060
# ---------------------------------------------------------------------------
#
# The question the compiled kernels ask of a coordinate is geometric: are the
# three scalar levels their inline parabola is fitted through equally spaced?
# It is NOT "does a fresh f32 recomputation of the weights return the same
# three words".  Those are different questions, and the second one is not a
# usable test:
#
#   ``param.F:8985-9060`` reads its nodes out of ``sigma``/``sigmaf``, which
#   are f32.  A face at 13 332 m carries an f32 ulp of about 1 mm, so on a
#   perfectly uniform 40 x 333.3 m column the differenced top layer depths come
#   out as 333.30078125, 333.30078125 and 333.298828125 m -- three spellings of
#   one number.  Feeding those into the recursion returns
#   ``(1.8750014, -1.2500057, 0.3750043)``: the analytic answer plus seven
#   digits of representation noise.  Testing that triple for *bitwise* equality
#   against ``PINNED_FORNBERG_F32`` therefore refused a plain uniform grid for
#   the sole crime of having a ``dz`` that is not exactly representable in
#   binary32, while admitting the arithmetically identical 500.0 m column.  A
#   refusal that fires on valid input is a defect, so the test was replaced by
#   the property it was always trying to stand in for.
#
# The property is decided on the *construction* -- the heights the generator
# was asked to build, before they are rounded into the f32 coordinate -- so no
# rounding of the output can turn a uniform grid into a refused one.
def _f32_ulp(np: Any, value: float) -> float:
    """The distance from ``value`` to the next f32 word away from zero."""

    magnitude = np.float32(abs(float(value)))
    if magnitude == np.float32(0.0):
        return float(np.finfo(np.float32).tiny)
    return float(np.float32(np.nextafter(magnitude, np.float32(np.inf))) - magnitude)


def _triplet_equally_spaced(np: Any, faces: Sequence[float]) -> bool:
    """Whether four consecutive construction faces bound one layer depth.

    ``faces`` are ordered as the extrapolation walks them, so ``faces[i+1] -
    faces[i]`` are the three layer depths whose equality makes
    :data:`PINNED_FORNBERG_F32` this coordinate's own weights.

    The coordinate the kernels read is f32, so each of these heights reaches a
    kernel with at most half an f32 ulp of its own magnitude of representation
    error.  A layer depth differenced from two of them carries at most one ulp,
    and two depths that are equal in the construction can therefore be observed
    to differ by at most two.  Two ulp of the largest height involved is the
    smallest band that cannot separate them -- it is forced by the storage
    format, not chosen to make a case pass.  Inside the band the levels are one
    spacing written twice; outside it they are genuinely different spacings and
    the inline parabola is fitted through the wrong points.
    """

    depths = [float(faces[index + 1]) - float(faces[index]) for index in range(3)]
    slack = 2.0 * _f32_ulp(np, max(abs(float(face)) for face in faces))
    return (max(depths) - min(depths)) <= slack


def _construction_faces_m(
    *,
    stretch_z: int,
    nk: int,
    dz_m: float,
    dz_bot_m: float,
    dz_top_m: float,
    derived: Any,
    levels_m: Sequence[float],
) -> list[float]:
    """``zf(1..nk+1)`` as the generator was *asked* for it, in f64.

    Every branch below is its ``param.F`` formula evaluated in double
    precision from the same words the f32 generator uses, so the only
    difference from :attr:`VerticalLevels.zf` is that these heights have not
    been rounded into the coordinate yet.  The top segment in particular is
    written as one multiply rather than ``param.F:7935-7937``'s repeated
    ``zf(k-1)+dz_top``, so a run of equal layers stays a run of equal layers
    instead of accumulating one rounding step per level.
    """

    if stretch_z == 0:
        # param.F:7789-7791: zf(k) = (k-1)*dz, every layer exactly dz.
        return [float(index) * float(dz_m) for index in range(nk + 1)]
    if stretch_z == 1:
        # param.F:7925-7934, the three segments of the Wilhelmson-Chen column.
        nk1, nk2 = derived.nk1, derived.nk2
        nominal = float(derived.nominal_dz_f32)
        c1, c2 = float(derived.c1_f32), float(derived.c2_f32)
        bot, top = float(dz_bot_m), float(dz_top_m)
        base = float(nk1) * bot
        faces: list[float] = [float(index) * bot for index in range(nk1 + 1)]
        for step in range(1, nk2 + 1):
            distance = step * nominal
            faces.append(base + (c1 + c2 * distance) * distance)
        crest = faces[nk1 + nk2]
        faces.extend(crest + float(step) * top for step in range(1, derived.nk3 + 1))
        return faces
    if stretch_z == 3:
        # param.F:8019-8026: sigmaf(1)=0 then sigmaf(k+1)=2*sigma(k)-sigmaf(k).
        faces = [0.0]
        for value in levels_m:
            faces.append(2.0 * float(value) - faces[-1])
        return faces
    # param.F:8085-8089: stretch_z=4 declares the w levels themselves.
    return [float(value) for value in levels_m]


def _pinned_quadratic_ends(np: Any, faces: Sequence[float], nk: int) -> tuple[bool, bool]:
    """``(lowest, highest)``: whether each end's three layers share one depth.

    ``param.F:8990-8993`` fits its lower parabola through ``sigma(1..3)`` and
    ``:9004-9007`` its upper one through ``sigma(nk..nk-2)``.  ``sigma(k)`` is
    the midpoint of ``sigmaf(k)`` and ``sigmaf(k+1)`` (``param.F:8189-8191``),
    so those three scalar levels are equally spaced exactly when the three
    layers they sit in have one depth, and the normalizing divisor
    ``sigmaf(2)-sigmaf(1)`` (resp. ``sigmaf(nk+1)-sigmaf(nk)``) is that depth.
    """

    return (
        _triplet_equally_spaced(np, faces[0:4]),
        _triplet_equally_spaced(np, faces[nk : nk - 4 : -1]),
    )


def _f32(np: Any, value: Any) -> Any:
    return np.float32(value)


def _add(np: Any, a: Any, b: Any) -> Any:
    return np.float32(np.float32(a) + np.float32(b))


def _sub(np: Any, a: Any, b: Any) -> Any:
    return np.float32(np.float32(a) - np.float32(b))


def _mul(np: Any, a: Any, b: Any) -> Any:
    return np.float32(np.float32(a) * np.float32(b))


def _div(np: Any, a: Any, b: Any) -> Any:
    return np.float32(np.float32(a) / np.float32(b))


# ---------------------------------------------------------------------------
# Fornberg (1988) extrapolation weights -- param.F:8985-9060
# ---------------------------------------------------------------------------

def _fornberg_delta(alpha: Sequence[float], x0: float) -> list[list[list[float]]]:
    """``delta[n][m+1][nu]`` mirrors ``delta(n,m,nu)`` with source ``m=-1..4``.

    Transcribed from ``param.F:9016-9035`` with ``bigm=4`` (param.F:107) and
    ``bign=3`` (param.F:108) in double precision, the source's own type.
    """

    delta = [[[0.0 for _ in range(4)] for _ in range(6)] for _ in range(4)]
    delta[0][1][0] = 1.0
    b1 = 1.0
    for n in range(1, 4):
        b2 = 1.0
        for nu in range(n):
            b3 = alpha[n] - alpha[nu]
            if not math.isfinite(b3) or b3 == 0.0:
                raise VerticalGridError(
                    "Fornberg extrapolation nodes are duplicated; two of the "
                    "lowest or highest scalar levels share a height"
                )
            b2 = b2 * b3
            delta[n - 1][n + 1][nu] = 0.0
            for m in range(0, min(n, 4) + 1):
                delta[n][m + 1][nu] = (
                    (alpha[n] - x0) * delta[n - 1][m + 1][nu]
                    - m * delta[n - 1][m][nu]
                ) / b3
        for m in range(0, min(n, 4) + 1):
            delta[n][m + 1][n] = (b1 / b2) * (
                m * delta[n - 1][m][n - 1]
                - (alpha[n - 1] - x0) * delta[n - 1][m + 1][n - 1]
            )
        b1 = b2
    return delta


def fornberg_extrapolation_weights(
    sigma: Any, sigmaf: Any, *, nk: int, top: bool
) -> tuple[float, float, float]:
    """``cgs1..3`` (``top=False``) or ``cgt1..3`` (``top=True``).

    ``sigma`` and ``sigmaf`` carry CM1 indices, so ``sigma[1]`` is the first
    scalar level and ``sigmaf[1]`` the ground.  ``param.F:8990-9013`` forms
    ``x0`` and ``alpha`` by default-REAL divisions before promoting them to
    DOUBLE PRECISION; the same rounding is reproduced here.
    """

    np = importlib.import_module("numpy")
    if nk < 4:
        raise VerticalGridError(
            "Fornberg extrapolation reads four scalar levels; a grid with "
            "fewer cannot fill the lower and upper boundary stencils"
        )
    if top:
        denominator = _sub(np, sigmaf[nk + 1], sigmaf[nk])
        x0_num = sigmaf[nk + 1]
        nodes = [sigma[nk], sigma[nk - 1], sigma[nk - 2], sigma[nk - 3]]
    else:
        denominator = _sub(np, sigmaf[2], sigmaf[1])
        x0_num = sigmaf[1]
        nodes = [sigma[1], sigma[2], sigma[3], sigma[4]]
    if not bool(np.isfinite(denominator)) or float(denominator) <= 0.0:
        raise VerticalGridError(
            "Fornberg extrapolation spacing is not positive and finite"
        )
    x0 = float(_div(np, x0_num, denominator))
    alpha = [float(_div(np, node, denominator)) for node in nodes]
    delta = _fornberg_delta(alpha, x0)
    return tuple(float(np.float32(delta[2][1][nu])) for nu in range(3))  # type: ignore[return-value]


def fornberg_derivative_weights(
    sigma: Any, sigmaf: Any, *, nk: int, top: bool
) -> tuple[float, float, float]:
    """``dgs1..3`` (``top=False``) or ``dgt1..3`` (``top=True``).

    ``param.F:9042-9044`` and ``:9050-9052`` read ``delta(2,1,nu)`` from the
    same Fornberg table that supplies the value weights ``cgs``/``cgt``: the
    ``m=1`` row is the FIRST DERIVATIVE at the boundary face, expressed per
    unit of the same normalising spacing.  ``misclibs.F:2137-2148`` multiplies
    the result by ``rdz*mh``, so on a uniform column the pair is the exact
    three-point one-sided derivative (-2, 3, -1) at the ground and
    (2, -3, 1) at the top.
    """

    np = importlib.import_module("numpy")
    if nk < 4:
        raise VerticalGridError(
            "Fornberg extrapolation reads four scalar levels; a grid with "
            "fewer cannot fill the lower and upper boundary stencils"
        )
    if top:
        denominator = _sub(np, sigmaf[nk + 1], sigmaf[nk])
        x0_num = sigmaf[nk + 1]
        nodes = [sigma[nk], sigma[nk - 1], sigma[nk - 2], sigma[nk - 3]]
    else:
        denominator = _sub(np, sigmaf[2], sigmaf[1])
        x0_num = sigmaf[1]
        nodes = [sigma[1], sigma[2], sigma[3], sigma[4]]
    if not bool(np.isfinite(denominator)) or float(denominator) <= 0.0:
        raise VerticalGridError(
            "Fornberg extrapolation spacing is not positive and finite"
        )
    x0 = float(_div(np, x0_num, denominator))
    alpha = [float(_div(np, node, denominator)) for node in nodes]
    delta = _fornberg_delta(alpha, x0)
    return tuple(float(np.float32(delta[2][2][nu])) for nu in range(3))  # type: ignore[return-value]


def fornberg_first_derivative_weights(
    sigma: Any, sigmaf: Any, *, nk: int, top: bool
) -> tuple[float, float, float]:
    """``dgs1..3`` (``top=False``) or ``dgt1..3`` (``top=True``).

    ``param.F:9042-9044`` and ``param.F:9050-9052`` read ``delta(2,1,nu)`` --
    the first-derivative weights of the same quadratic whose value weights
    :func:`fornberg_extrapolation_weights` returns.  ``calcvort`` uses them to
    close the vertical potential-temperature gradient at the ground and the
    model top (``misclibs.F:2158-2163``); on equally spaced ends they are
    ``(-2, 3, -1)`` in units of the reference ``dz``.
    """

    np = importlib.import_module("numpy")
    if nk < 4:
        raise VerticalGridError(
            "Fornberg extrapolation reads four scalar levels; a grid with "
            "fewer cannot fill the lower and upper boundary stencils"
        )
    if top:
        denominator = _sub(np, sigmaf[nk + 1], sigmaf[nk])
        x0_num = sigmaf[nk + 1]
        nodes = [sigma[nk], sigma[nk - 1], sigma[nk - 2], sigma[nk - 3]]
    else:
        denominator = _sub(np, sigmaf[2], sigmaf[1])
        x0_num = sigmaf[1]
        nodes = [sigma[1], sigma[2], sigma[3], sigma[4]]
    if not bool(np.isfinite(denominator)) or float(denominator) <= 0.0:
        raise VerticalGridError(
            "Fornberg extrapolation spacing is not positive and finite"
        )
    x0 = float(_div(np, x0_num, denominator))
    alpha = [float(_div(np, node, denominator)) for node in nodes]
    delta = _fornberg_delta(alpha, x0)
    return tuple(float(np.float32(delta[2][2][nu])) for nu in range(3))  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# The shared tail -- param.F:8162-8209
# ---------------------------------------------------------------------------

def _finish_levels(np: Any, zf_physical: Sequence[Any], nk: int, dz: Any) -> dict[str, Any]:
    """Complete ``zf``/``zh``/``sigmaf``/``sigma``/``mh``/``mf`` from faces.

    ``zf_physical`` supplies the ``nk+1`` physical w levels, CM1 ``k=1..nk+1``.
    """

    dz = np.float32(dz)
    zf = np.zeros(nk + 3, dtype=np.float32)
    for k in range(1, nk + 2):
        zf[k] = np.float32(zf_physical[k - 1])
    # param.F:8165-8166
    zf[0] = np.float32(-zf[2])
    zf[nk + 2] = _add(np, zf[nk + 1], _sub(np, zf[nk + 1], zf[nk]))
    # param.F:8212-8224 stops CM1 on a non-positive layer depth.  It is run
    # before the metrics rather than after, because param.F:8170's dz/(zf(k+1)
    # -zf(k)) would otherwise divide by zero on the way to the same refusal.
    for k in range(1, nk + 1):
        if not float(zf[k + 1]) > float(zf[k]):
            raise VerticalGridError(
                f"level {k} has non-positive physical spacing "
                f"zf({k + 1})-zf({k}) = {float(zf[k + 1]) - float(zf[k])}"
            )
    zh = np.empty(nk + 2, dtype=np.float32)
    mh = np.empty(nk + 2, dtype=np.float32)
    # param.F:8168-8171
    for k in range(nk + 2):
        zh[k] = _mul(np, np.float32(0.5), _add(np, zf[k + 1], zf[k]))
        mh[k] = _div(np, dz, _sub(np, zf[k + 1], zf[k]))
    # param.F:8172-8173
    zh[0] = np.float32(-zh[1])
    zh[nk + 1] = _add(np, zh[nk], _mul(np, np.float32(2.0), _sub(np, zf[nk + 1], zh[nk])))
    mf = np.empty(nk + 3, dtype=np.float32)
    # param.F:8175-8179
    for k in range(1, nk + 2):
        mf[k] = _div(np, dz, _sub(np, zh[k], zh[k - 1]))
    mf[0], mf[nk + 2] = mf[1], mf[nk + 1]
    # With terrain_flag=false the nominal coordinate is the physical one, and
    # param.F carries the same ghost rule for sigmaf that it applies to zf
    # (:7947-7948 for stretch_z=1).  The stretch_z=3 and 4 branches write
    # sigmaf(1..nk+1) only and leave the two ghosts unset, which param.F:8189
    # then reads into sigma(0) and sigma(nk+1); GM1's device profiles are
    # (nk+3,) words and rdsf would divide by a stale zero, so the ghost rule
    # param.F:7952-7953 applies to stretch_z=1 is used for every branch.  The
    # ghosts enter no rds/rdsf word this port consumes: init_terrain.F:292-298
    # reads sigma(1..nk) and sigmaf(1..nk+1) only.
    sigmaf = zf.copy()
    sigma = np.empty(nk + 2, dtype=np.float32)
    # param.F:8189-8191
    for k in range(nk + 2):
        sigma[k] = _mul(np, np.float32(0.5), _add(np, sigmaf[k], sigmaf[k + 1]))
    rmh = np.empty_like(mh)
    rmf = np.empty_like(mf)
    # param.F:8195-8209
    for k in range(nk + 2):
        rmh[k] = _div(np, np.float32(1.0), mh[k])
    for k in range(nk + 3):
        rmf[k] = _div(np, np.float32(1.0), mf[k])
    return {
        "zf": zf, "zh": zh, "sigmaf": sigmaf, "sigma": sigma,
        "mh": mh, "mf": mf, "rmh": rmh, "rmf": rmf,
    }


def _rds_rdsf(np: Any, sigma: Any, sigmaf: Any, nk: int) -> tuple[Any, Any]:
    """``init_terrain.F:292-298``, in the GM1 device shapes."""

    rds = np.zeros(nk + 2, dtype=np.float32)
    rdsf = np.zeros(nk + 3, dtype=np.float32)
    for k in range(2, nk + 1):
        rds[k] = _div(np, np.float32(1.0), _sub(np, sigma[k], sigma[k - 1]))
    for k in range(1, nk + 1):
        rdsf[k] = _div(np, np.float32(1.0), _sub(np, sigmaf[k + 1], sigmaf[k]))
    return rds, rdsf


# ---------------------------------------------------------------------------
# The level family
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True, eq=False)
class VerticalLevels:
    """One complete CM1 vertical coordinate, CM1-indexed and f32-exact.

    Equality and hashing use the sealed identity words rather than the arrays,
    so a geometry that carries one stays hashable and two independently built
    copies of the same coordinate compare equal.
    """

    stretch_z: int
    nk: int
    dz_reference_m: float
    ztop_m: float
    str_bot_m: float
    str_top_m: float
    dz_bot_m: float
    dz_top_m: float
    levels_m: tuple[float, ...]
    zf: Any
    zh: Any
    sigmaf: Any
    sigma: Any
    mh: Any
    mf: Any
    rmh: Any
    rmf: Any
    rds: Any
    rdsf: Any
    cgs: tuple[float, float, float]
    cgt: tuple[float, float, float]
    #: Whether the lowest (resp. highest) three layers of the *construction*
    #: share one depth, which is exactly when :data:`PINNED_FORNBERG_F32` is
    #: this coordinate's own quadratic.  Decided by
    #: :func:`_pinned_quadratic_ends` before the heights are rounded into the
    #: f32 coordinate, so it is a fact about the grid and not about ``dz``'s
    #: binary32 spelling.  Derived, so it stays out of :meth:`_identity`.
    lowest_layers_equally_spaced: bool
    highest_layers_equally_spaced: bool
    sha256: str

    def _identity(self) -> tuple[Any, ...]:
        return (
            self.stretch_z, self.nk, self.dz_reference_m, self.ztop_m,
            self.str_bot_m, self.str_top_m, self.dz_bot_m, self.dz_top_m,
            self.levels_m, self.sha256,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, VerticalLevels):
            return NotImplemented
        return self._identity() == other._identity()

    def __hash__(self) -> int:
        return hash(self._identity())

    @property
    def uniform(self) -> bool:
        return self.stretch_z == 0

    @property
    def lowest_mass_level_m(self) -> float:
        """First scalar level above ground, CM1 ``zh(1)``."""

        return float(self.zh[1])

    @property
    def highest_mass_level_m(self) -> float:
        return float(self.zh[self.nk])

    @property
    def mass_heights_m(self) -> tuple[float, ...]:
        """``zh(1..nk)``: the physical scalar levels."""

        return tuple(float(self.zh[k]) for k in range(1, self.nk + 1))

    @property
    def face_heights_m(self) -> tuple[float, ...]:
        """``zf(1..nk+1)``: the physical w levels."""

        return tuple(float(self.zf[k]) for k in range(1, self.nk + 2))

    @property
    def spacings_m(self) -> tuple[float, ...]:
        return tuple(
            float(self.zf[k + 1]) - float(self.zf[k]) for k in range(1, self.nk + 1)
        )

    def levels_below_m(self, height_m: float) -> int:
        return sum(1 for value in self.mass_heights_m if value < height_m)

    def stretch_parameters(self) -> Any:
        """The exact ``StretchZ1Parameters`` for the device generator."""

        from .cuda_stretch_z1 import StretchZ1Parameters

        if self.stretch_z == 0:
            return StretchZ1Parameters(
                self.nk, self.dz_reference_m, self.ztop_m, 0.0,
                self.ztop_m, self.dz_reference_m, self.dz_reference_m,
            )
        if self.stretch_z == 1:
            return StretchZ1Parameters(
                self.nk, self.dz_reference_m, self.ztop_m, self.str_bot_m,
                self.str_top_m, self.dz_bot_m, self.dz_top_m,
            )
        raise VerticalGridError(
            f"stretch_z={self.stretch_z} has no analytic StretchZ1Parameters; "
            "param.F:8005-8132 carries its levels as explicit words, so the "
            "param.F:7819-7953 generator cannot reproduce them"
        )

    def fornberg_matches_pinned_kernels(self) -> bool:
        """Whether the inline kernel words are this grid's own weights.

        The lowest and highest three layers each having one depth is the
        analytic condition for :data:`PINNED_FORNBERG_F32`; see the module
        comment above :func:`_f32_ulp` for why this is asked of the
        construction rather than of :attr:`cgs`/:attr:`cgt`, which are CM1's
        own f32 recomputation and carry CM1's own f32 rounding with them.
        """

        return self.lowest_layers_equally_spaced and self.highest_layers_equally_spaced

    def record(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "schema": VERTICAL_GRID_SCHEMA,
            "stretch_z": self.stretch_z,
            "nk": self.nk,
            "dz_reference_m": self.dz_reference_m,
            "ztop_m": self.ztop_m,
            "lowest_mass_level_m": self.lowest_mass_level_m,
            "lowest_spacing_m": self.spacings_m[0],
            "highest_spacing_m": self.spacings_m[-1],
            "levels_below_1000m": self.levels_below_m(1000.0),
            "cgs": list(self.cgs),
            "cgt": list(self.cgt),
            "sha256": self.sha256,
            "authority": dict(SOURCE_ANCHORS),
        }
        if self.stretch_z in (1, 2):
            body.update(
                str_bot_m=self.str_bot_m, str_top_m=self.str_top_m,
                dz_bot_m=self.dz_bot_m, dz_top_m=self.dz_top_m,
            )
        if self.stretch_z in (3, 4):
            body["levels_m"] = list(self.levels_m)
        return body


def _seal(np: Any, arrays: Sequence[Any]) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        digest.update(np.ascontiguousarray(value, dtype=np.float32).astype("<f4", copy=False).tobytes())
    return digest.hexdigest()


def _positive(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise VerticalGridError(f"{name} must be numeric")
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise VerticalGridError(f"{name} must be finite and greater than zero")
    return value


def build_vertical_levels(
    *,
    stretch_z: int,
    nk: int,
    dz_m: float,
    ztop_m: float,
    str_bot_m: float = 0.0,
    str_top_m: float = 0.0,
    dz_bot_m: float = 0.0,
    dz_top_m: float = 0.0,
    levels_m: Sequence[float] = (),
) -> VerticalLevels:
    """Build one CM1 vertical coordinate.

    ``stretch_z=0`` reproduces ``param.F:7789-7811`` exactly.  It is realized
    through the ``stretch_z=1`` generator's degenerate parameters
    (``str_bot=0``, ``str_top=ztop``, ``dz_bot=dz_top=dz``), for which
    ``param.F:7881-7882`` yields ``c1=1`` and ``c2=0`` and ``:7925-7934``
    collapses to ``zf(k)=(k-1)*dz``; ``tests/test_cm1_vertical_grid.py`` holds
    the two to raw-word equality.
    """

    np = importlib.import_module("numpy")
    if isinstance(stretch_z, bool) or not isinstance(stretch_z, int):
        raise VerticalGridError("stretch_z must be an integer")
    if stretch_z == 2:
        raise VerticalGridError(STRETCH_Z2_REFUSAL)
    if stretch_z not in ADMITTED_STRETCH_Z:
        raise VerticalGridError(
            f"stretch_z={stretch_z} is not one of CM1 r21.1's branches "
            f"{ADMITTED_STRETCH_Z} (param.F:7814-8139)"
        )
    if isinstance(nk, bool) or not isinstance(nk, int) or nk < 4:
        raise VerticalGridError("nk must be an integer >= 4")
    dz_m = _positive("dz_m", dz_m)
    ztop_m = _positive("ztop_m", ztop_m)
    levels_m = tuple(float(value) for value in levels_m)

    if stretch_z in (0, 1):
        from .cuda_stretch_z1 import StretchZ1Parameters, build_stretch_z1_grid

        if stretch_z == 0:
            # param.F:2608 overwrites ztop for every unstretched grid.
            expected_top = float(np.float32(np.float32(dz_m) * np.float32(nk)))
            if float(np.float32(ztop_m)) != expected_top:
                raise VerticalGridError(
                    f"stretch_z=0 fixes the model top at dz*nk = {expected_top} m "
                    f"(param.F:2608); {ztop_m} m was declared"
                )
            if levels_m:
                raise VerticalGridError("stretch_z=0 carries no explicit levels")
            parameters = StretchZ1Parameters(nk, dz_m, expected_top, 0.0, expected_top, dz_m, dz_m)
            str_bot_m, str_top_m = 0.0, expected_top
            dz_bot_m = dz_top_m = dz_m
            ztop_m = expected_top
        else:
            if levels_m:
                raise VerticalGridError("stretch_z=1 carries no explicit levels")
            str_bot_m = float(str_bot_m)
            str_top_m = _positive("str_top_m", str_top_m)
            dz_bot_m = _positive("dz_bot_m", dz_bot_m)
            dz_top_m = _positive("dz_top_m", dz_top_m)
            parameters = StretchZ1Parameters(
                nk, dz_m, ztop_m, str_bot_m, str_top_m, dz_bot_m, dz_top_m
            )
        try:
            grid = build_stretch_z1_grid(parameters)
        except ValueError as error:
            raise VerticalGridError(str(error)) from None
        derived = grid.derived
        pieces = {
            "zf": grid.zf, "zh": grid.zh, "sigmaf": grid.sigmaf, "sigma": grid.sigma,
            "mh": grid.mh, "mf": grid.mf, "rmh": grid.rmh, "rmf": grid.rmf,
        }
    else:
        expected = nk if stretch_z == 3 else nk + 1
        if len(levels_m) != expected:
            raise VerticalGridError(
                f"stretch_z={stretch_z} requires exactly {expected} explicit "
                f"{'scalar' if stretch_z == 3 else 'w'} levels; {len(levels_m)} were given"
            )
        for index, value in enumerate(levels_m):
            if not math.isfinite(value):
                raise VerticalGridError(f"explicit level {index} is not finite")
        if stretch_z == 3:
            # param.F:8019-8026: sigmaf(1)=0 then sigmaf(k+1)=2*sigma(k)-sigmaf(k).
            faces = [np.float32(0.0)]
            for value in levels_m:
                faces.append(
                    _sub(np, _mul(np, np.float32(2.0), np.float32(value)), faces[-1])
                )
        else:
            # param.F:8085-8089: the file holds the w levels themselves.
            if float(np.float32(levels_m[0])) != 0.0:
                raise VerticalGridError(
                    "stretch_z=4 w levels must begin at the ground, 0 m"
                )
            faces = [np.float32(value) for value in levels_m]
        # param.F:8037 and :8107 overwrite ztop with the generated top face.
        generated_top = float(faces[nk])
        if float(np.float32(ztop_m)) != generated_top:
            raise VerticalGridError(
                f"stretch_z={stretch_z} fixes the model top at its own top level "
                f"{generated_top} m (param.F:{'8037' if stretch_z == 3 else '8105'}); "
                f"{ztop_m} m was declared"
            )
        ztop_m = generated_top
        str_bot_m = str_top_m = dz_bot_m = dz_top_m = 0.0
        derived = None
        pieces = _finish_levels(np, faces, nk, dz_m)

    for value in pieces.values():
        value.flags.writeable = False
    rds, rdsf = _rds_rdsf(np, pieces["sigma"], pieces["sigmaf"], nk)
    rds.flags.writeable = False
    rdsf.flags.writeable = False
    cgs = fornberg_extrapolation_weights(pieces["sigma"], pieces["sigmaf"], nk=nk, top=False)
    cgt = fornberg_extrapolation_weights(pieces["sigma"], pieces["sigmaf"], nk=nk, top=True)
    # cgs/cgt above stay CM1's own f32 recomputation, word for word, because
    # the CPU reference in cuda_terrain_open_boundary consumes them and must
    # keep matching native CM1.  Whether the *kernels'* inline quadratic is
    # this grid's own is a separate, geometric question, and is asked of the
    # construction heights rather than of those rounded words.
    lowest_uniform, highest_uniform = _pinned_quadratic_ends(
        np,
        _construction_faces_m(
            stretch_z=stretch_z, nk=nk, dz_m=dz_m, dz_bot_m=dz_bot_m,
            dz_top_m=dz_top_m, derived=derived, levels_m=levels_m,
        ),
        nk,
    )
    order = ("zf", "zh", "sigmaf", "sigma", "mh", "mf", "rmh", "rmf")
    sha256 = _seal(np, [pieces[name] for name in order] + [rds, rdsf])
    return VerticalLevels(
        stretch_z=stretch_z, nk=nk, dz_reference_m=float(np.float32(dz_m)),
        ztop_m=float(np.float32(ztop_m)), str_bot_m=float(np.float32(str_bot_m)),
        str_top_m=float(np.float32(str_top_m)), dz_bot_m=float(np.float32(dz_bot_m)),
        dz_top_m=float(np.float32(dz_top_m)), levels_m=levels_m,
        rds=rds, rdsf=rdsf, cgs=cgs, cgt=cgt,
        lowest_layers_equally_spaced=lowest_uniform,
        highest_layers_equally_spaced=highest_uniform,
        sha256=sha256,
        **pieces,
    )


def uniform_vertical_levels(nk: int, dz_m: float) -> VerticalLevels:
    """The ``stretch_z=0`` member: ``param.F:7789-7811`` with ``ztop=dz*nk``."""

    np = importlib.import_module("numpy")
    top = float(np.float32(np.float32(dz_m) * np.float32(nk)))
    return build_vertical_levels(stretch_z=0, nk=nk, dz_m=dz_m, ztop_m=top)


__all__ = (
    "ADMITTED_STRETCH_Z",
    "STRETCH_Z2_REFUSAL",
    "PINNED_FORNBERG_F32",
    "SOURCE_ANCHORS",
    "VERTICAL_GRID_SCHEMA",
    "VerticalGridError",
    "VerticalLevels",
    "build_vertical_levels",
    "fornberg_extrapolation_weights",
    "uniform_vertical_levels",
)
