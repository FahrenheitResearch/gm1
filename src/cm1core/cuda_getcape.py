"""Device-resident CM1 ``getcape`` 2-D environment fields.

``src/writeout.F:2895-2932`` runs ``getcape.F`` once per column of the live
model state and fills four 2-D planes; ``writeout.F:2937-3028`` then writes
whichever of them the ``output_cape``/``output_cin``/``output_lcl``/
``output_lfc`` switches select.  This module is that block: one CUDA kernel
whose thread index is a column, a host oracle that evaluates the same
arithmetic in source order, and the compile/launch pair the runtime uses.

Three things are worth stating plainly about the transcription.

* The column handed to ``getcape`` is CM1's, not a new one.
  ``writeout.F:2907-2910`` builds it from the *live* carriers -- ``prs``,
  ``(th0+tha)*(pi0+ppi)`` and ``qa(:,:,:,nqv)`` -- and ``writeout.F:2912-2914``
  extrapolates a surface level below the lowest mass level with the
  ``cgs1/cgs2/cgs3`` bottom weights ``param.F:9038`` forms.  At this runtime's
  publication boundary ``tha``/``ppi``/``qa`` and ``th3d``/``pp3d``/``q3d``
  hold the same words, which is exactly the post-solve3 quiescence
  ``cartesian_history.require_cartesian_publication_boundary`` already proves
  before a frame is taken.
* ``writeout.F:2921`` fixes ``source=3``, the mixed-layer parcel.  The other
  two ``getcape.F`` parcels are reachable here because ``getcape.F`` itself
  implements them and the t=0 report uses all three; the default is CM1's.
* The parcel ascent needs ``expf``, ``logf`` and ``powf``.  Those are not the
  same functions on the device as in glibc, so the host oracle is checked
  against the packaged kernel *text* compiled by a C++ compiler -- where both
  sides call glibc and agree raw word for raw word -- and the device
  comparison is a bounded one.  Nothing here feeds back into the trajectory:
  these are diagnostic planes, published beside the state and never into it.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import hashlib
import importlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .errors import refuse


CUDA_GETCAPE_SCHEMA = "gm1.cuda-getcape-fields/v1"
CUDA_GETCAPE_ABI = "gm1.cuda-getcape-fields-kernels/v1"
CUDA_GETCAPE_THREADS = 128
CUDA_GETCAPE_KERNEL_NAME = "cm1_getcape_fields_v1"
CUDA_GETCAPE_CLAIM_LIMIT = (
    "CM1 r21.1 getcape.F parcel integration run per column over one completed "
    "post-solve3 Cartesian state, published as the four writeout.F 2-D planes; "
    "the device transcendental library is not glibc, so the device planes agree "
    "with the host oracle to a bounded difference and not raw word for raw word; "
    "not a severe-weather product, a forecast, or a claim about any observed storm"
)

SOURCE_ANCHORS = {
    "parcel_integration": "src/getcape.F:14-520",
    "saturation_liquid": "src/getcape.F:526-538",
    "saturation_ice": "src/getcape.F:544-556",
    "equivalent_potential_temperature": "src/getcape.F:562-577",
    "dewpoint": "src/getcape.F:583-594",
    "field_column_assembly": "src/writeout.F:2907-2914",
    "field_parcel_selector": "src/writeout.F:2921",
    "field_switches": "src/writeout.F:2895,2937-3030",
    "surface_extrapolation_weights": "src/param.F:8988-9052",
    "t0_report": "src/base.F:3143-3189",
}

#: ``writeout.F:2937-3028`` in the order CM1 writes them.
GETCAPE_FIELD_NAMES = ("cape", "cin", "lcl", "lfc")
#: ``writeout.F:2938-2941`` etc: CM1's own name, description and unit per field.
GETCAPE_FIELD_META = {
    "cape": ("convective available potential energy", "J/kg"),
    "cin": ("convective inhibition", "J/kg"),
    "lcl": ("lifted condensation level", "m"),
    "lfc": ("level of free convection", "m"),
}
#: The CM1 namelist switch that selects each plane (``input.F:122``).
GETCAPE_FIELD_SWITCHES = {
    "cape": "output_cape",
    "cin": "output_cin",
    "lcl": "output_lcl",
    "lfc": "output_lfc",
}

INPUT_NAMES = ("prs", "th0", "th3d", "pi0", "pp3d", "qv")

#: ``getcape.F:60``.
DEFAULT_PRESSURE_INCREMENT_PA = 200.0
#: ``getcape.F:70``.
DEFAULT_MIXED_LAYER_DEPTH_M = 500.0
#: ``getcape.F:73``.
DEFAULT_ADIABAT = 1
#: ``writeout.F:2921``.
DEFAULT_PARCEL_SOURCE = 3
#: ``getcape.F:123``.
CONVERGE = 0.0002
#: ``getcape.F:383``.
MAX_ADIABAT_ITERATIONS = 100
#: ``param.F:9038`` on ``param.F:7789-7811``'s uniform coordinate.
UNIFORM_SURFACE_WEIGHTS = (1.875, -1.25, 0.375)

PARCEL_SOURCE_NAMES = {1: "surface", 2: "most-unstable", 3: "mixed-layer"}
ADIABAT_NAMES = {
    1: "pseudoadiabatic-liquid",
    2: "reversible-liquid",
    3: "pseudoadiabatic-ice",
    4: "reversible-ice",
}


def getcape_field_shapes(ni: int, nj: int, nk: int) -> dict[str, tuple[int, ...]]:
    """Input and output extents for one ``ni x nj`` column sweep."""

    for name, value, minimum in (("ni", ni, 1), ("nj", nj, 1), ("nk", nk, 3)):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    volume = (nk, nj, ni)
    plane = (nj, ni)
    shapes = {name: volume for name in INPUT_NAMES}
    shapes.update({name: plane for name in GETCAPE_FIELD_NAMES})
    # getcape.F:89 holds p, pi, thv and z for every level of the column.  On
    # the device those four live in one strided scratch volume of nk+1 levels,
    # which is the surface level writeout.F:2912 prepends plus the model's own.
    shapes["work"] = (4, nk + 1, nj, ni)
    return shapes


def getcape_scratch_bytes(ni: int, nj: int, nk: int) -> dict[str, int]:
    """Price the column scratch before a device word is allocated."""

    shapes = getcape_field_shapes(ni, nj, nk)
    work = 4
    for extent in shapes["work"]:
        work *= extent
    return {
        "column_scratch_bytes": work,
        "field_plane_bytes": 4 * nj * ni,
        "getcape_scratch_peak_bytes": work + 4 * len(GETCAPE_FIELD_NAMES) * nj * ni,
    }


@dataclass(frozen=True, slots=True)
class GetCapeFieldSelection:
    """Which planes this run publishes and how the parcel is lifted."""

    output_cape: bool = False
    output_cin: bool = False
    output_lcl: bool = False
    output_lfc: bool = False
    parcel_source: int = DEFAULT_PARCEL_SOURCE
    adiabat: int = DEFAULT_ADIABAT
    mixed_layer_depth_m: float = DEFAULT_MIXED_LAYER_DEPTH_M
    pressure_increment_pa: float = DEFAULT_PRESSURE_INCREMENT_PA

    @property
    def any_field(self) -> bool:
        """``writeout.F:2895``: the column sweep runs only if a plane wants it."""

        return bool(
            self.output_cape or self.output_cin or self.output_lcl or self.output_lfc
        )

    def selected(self) -> tuple[str, ...]:
        return tuple(
            name
            for name in GETCAPE_FIELD_NAMES
            if getattr(self, GETCAPE_FIELD_SWITCHES[name])
        )

    def record(self) -> dict[str, Any]:
        return {
            "schema": CUDA_GETCAPE_SCHEMA,
            "output_cape": self.output_cape,
            "output_cin": self.output_cin,
            "output_lcl": self.output_lcl,
            "output_lfc": self.output_lfc,
            "published_fields": list(self.selected()),
            "parcel_source": self.parcel_source,
            "parcel_source_name": PARCEL_SOURCE_NAMES[self.parcel_source],
            "adiabat": self.adiabat,
            "adiabat_name": ADIABAT_NAMES[self.adiabat],
            "mixed_layer_depth_m": self.mixed_layer_depth_m,
            "pressure_increment_pa": self.pressure_increment_pa,
            "authority": dict(SOURCE_ANCHORS),
            "claim_limit": CUDA_GETCAPE_CLAIM_LIMIT,
        }


def require_getcape_selection(
    selection: GetCapeFieldSelection, *, option_prefix: str = "diagnostics"
) -> None:
    """Refuse every parcel, adiabat, depth and increment not transcribed."""

    if not isinstance(selection, GetCapeFieldSelection):
        raise TypeError("selection must be GetCapeFieldSelection")
    for name in GETCAPE_FIELD_SWITCHES.values():
        value = getattr(selection, name)
        if not isinstance(value, bool):
            refuse(
                code="CM1-GETCAPE-SWITCH",
                gate="support-matrix",
                reason=(
                    f"writeout.F:2937-3028 tests {name}.eq.1 to decide whether the "
                    "plane is written at all; a non-boolean would be truthy for any "
                    "nonzero value and publish a field the document did not ask for"
                ),
                remedy=f"declare {option_prefix}.{name} as true or false",
                option=f"{option_prefix}.{name}",
                value=value,
            )
    if selection.parcel_source not in PARCEL_SOURCE_NAMES:
        refuse(
            code="CM1-GETCAPE-PARCEL",
            gate="support-matrix",
            reason=(
                f"getcape.F:242-250 stops CM1 on source={selection.parcel_source}; an "
                "unrecognized parcel would leave kmax at whatever the previous column "
                "left it and lift a parcel from an arbitrary level, so every published "
                "CAPE would describe a parcel the document never named"
            ),
            remedy=(
                "declare parcel_source = 1 (surface), 2 (most-unstable) or 3 "
                "(mixed-layer, which is what writeout.F:2921 uses)"
            ),
            option=f"{option_prefix}.parcel_source",
            value=selection.parcel_source,
        )
    if selection.adiabat not in ADIABAT_NAMES:
        refuse(
            code="CM1-GETCAPE-ADIABAT",
            gate="support-matrix",
            reason=(
                f"getcape.F:419-423 stops CM1 on adiabat={selection.adiabat}; an "
                "unrecognized formulation would leave the condensate neither removed "
                "nor carried, so the parcel would follow neither a pseudoadiabat nor "
                "a reversible adiabat and the published CAPE would belong to no "
                "thermodynamic path"
            ),
            remedy=(
                "declare adiabat = 1 (pseudoadiabatic liquid, which is getcape.F:73's "
                "own value), 2, 3 or 4"
            ),
            option=f"{option_prefix}.adiabat",
            value=selection.adiabat,
        )
    for name, value, authority, consequence in (
        (
            "mixed_layer_depth_m",
            selection.mixed_layer_depth_m,
            "getcape.F:233-234",
            "divides the depth-weighted mixed-layer sums by this depth, so a "
            "non-positive or non-finite value makes the source parcel theta and qv "
            "infinite or NaN and every published plane is NaN",
        ),
        (
            "pressure_increment_pa",
            selection.pressure_increment_pa,
            "getcape.F:328-333",
            "divides each layer's pressure drop by this increment to choose the "
            "sub-step count, so a non-positive or non-finite value gives the ascent "
            "zero or a non-finite number of sub-steps and the parcel never leaves "
            "its source level",
        ),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be numeric")
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            refuse(
                code="CM1-GETCAPE-CONTROL",
                gate="support-matrix",
                reason=f"{authority} {consequence}",
                remedy=f"declare {option_prefix}.{name} as a finite positive number",
                option=f"{option_prefix}.{name}",
                value=value,
            )


# ---------------------------------------------------------------------------
# Host oracle, in getcape.F's own statement order.
# ---------------------------------------------------------------------------

_LIBM: Any = None


def _libm() -> Any:
    """glibc's binary32 transcendentals, bound once.

    The oracle and the packaged kernel text compiled by a C++ compiler must
    call the same ``expf``/``logf``/``powf``; NumPy's own float32 loops are a
    third implementation and would make a raw-word comparison meaningless.
    """

    global _LIBM
    if _LIBM is None:
        name = ctypes.util.find_library("m")
        library = ctypes.CDLL(name) if name else ctypes.CDLL(None)
        for symbol, arity in (("expf", 1), ("logf", 1), ("powf", 2)):
            handle = getattr(library, symbol)
            handle.restype = ctypes.c_float
            handle.argtypes = (ctypes.c_float,) * arity
        _LIBM = library
    return _LIBM


def _expf(value: Any) -> Any:
    np = importlib.import_module("numpy")
    return np.float32(_libm().expf(ctypes.c_float(float(value))))


def _logf(value: Any) -> Any:
    np = importlib.import_module("numpy")
    return np.float32(_libm().logf(ctypes.c_float(float(value))))


def _powf(base: Any, exponent: Any) -> Any:
    np = importlib.import_module("numpy")
    return np.float32(
        _libm().powf(ctypes.c_float(float(base)), ctypes.c_float(float(exponent)))
    )


# getcape.F:100-121.  The constants getcape.F declares for itself.
CM1_G = 9.81
CM1_P00 = 100000.0
CM1_CP = 1005.7
CM1_RD = 287.04
CM1_RV = 461.5
CM1_XLV = 2501000.0
CM1_XLS = 2836017.0
CM1_T0 = 273.15
CM1_CPV = 1875.0
CM1_CPL = 4190.0
CM1_CPI = 2118.636


def _constants(np: Any) -> dict[str, Any]:
    f = np.float32
    g, cp, rd, rv = f(CM1_G), f(CM1_CP), f(CM1_RD), f(CM1_RV)
    cpv, cpl, cpi, t0 = f(CM1_CPV), f(CM1_CPL), f(CM1_CPI), f(CM1_T0)
    return {
        "g": g,
        "p00": f(CM1_P00),
        "cp": cp,
        "rd": rd,
        "rv": rv,
        "cpv": cpv,
        "cpl": cpl,
        "cpi": cpi,
        "lv1": f(f(CM1_XLV) + f(cpl - cpv) * t0),
        "lv2": f(cpl - cpv),
        "ls1": f(f(CM1_XLS) + f(cpi - cpv) * t0),
        "ls2": f(cpi - cpv),
        "rp00": f(f(1.0) / f(CM1_P00)),
        "eps": f(rd / rv),
        "reps": f(rv / rd),
        "rddcp": f(rd / cp),
        "cpdg": f(cp / g),
        "converge": f(CONVERGE),
    }


def getqvl(np: Any, p: Any, t: Any) -> Any:
    """getcape.F:526-538."""

    f = np.float32
    eps = f(f(CM1_RD) / f(CM1_RV))
    es = f(f(611.2) * _expf(f(f(17.67) * f(t - f(273.15)) / f(t - f(29.65)))))
    es = f(min(es, f(p * f(0.5))))
    return f(f(eps * es) / f(p - es))


def getqvi(np: Any, p: Any, t: Any) -> Any:
    """getcape.F:544-556."""

    f = np.float32
    eps = f(f(CM1_RD) / f(CM1_RV))
    es = f(f(611.2) * _expf(f(f(21.8745584) * f(t - f(273.15)) / f(t - f(7.66)))))
    es = f(min(es, f(p * f(0.5))))
    return f(f(eps * es) / f(p - es))


def getthx(np: Any, p: Any, t: Any, td: Any, q: Any) -> Any:
    """getcape.F:562-577."""

    f = np.float32
    if f(td - t) >= f(-0.1):
        tlcl = f(t)
    else:
        inner = f(f(f(1.0) / f(td - f(56.0))) + f(f(0.00125) * _logf(f(t / td))))
        tlcl = f(f(56.0) + f(f(1.0) / inner))
    first = _powf(f(f(100000.0) / p), f(f(0.2854) * f(f(1.0) - f(f(0.28) * q))))
    second = _expf(
        f(f(f(f(3376.0) / tlcl) - f(2.54)) * q * f(f(1.0) + f(f(0.81) * q)))
    )
    return f(f(t * first) * second)


def gettd(np: Any, p: Any, q: Any) -> Any:
    """getcape.F:583-594."""

    f = np.float32
    eps = f(f(CM1_RD) / f(CM1_RV))
    el = _logf(f(f(f(f(q / eps) * p) / f(100.0)) / f(f(1.0) + f(q / eps))))
    return f(f(273.15) + f(f(f(243.5) * el - f(440.8)) / f(f(19.48) - el)))


def cpu_getcape(
    p_mb: Any,
    t_c: Any,
    q_kg_per_kg: Any,
    *,
    source: int = DEFAULT_PARCEL_SOURCE,
    adiabat: int = DEFAULT_ADIABAT,
    mixed_layer_depth_m: float = DEFAULT_MIXED_LAYER_DEPTH_M,
    pressure_increment_pa: float = DEFAULT_PRESSURE_INCREMENT_PA,
) -> dict[str, float]:
    """``getcape.F:14-520`` for one column, in binary32 and in source order."""

    np = importlib.import_module("numpy")
    f = np.float32
    c = _constants(np)
    nk = len(p_mb)
    if nk < 3 or len(t_c) != nk or len(q_kg_per_kg) != nk:
        raise ValueError("getcape needs three equal-length profiles of >= 3 levels")
    if source not in PARCEL_SOURCE_NAMES:
        raise ValueError(f"getcape.F:242-250 stops CM1 on source={source}")
    if adiabat not in ADIABAT_NAMES:
        raise ValueError(f"getcape.F:419-423 stops CM1 on adiabat={adiabat}")
    pinc = f(pressure_increment_pa)
    ml_depth = f(mixed_layer_depth_m)

    # getcape.F:131-143.
    p = [f(0.0)] * nk
    t = [f(0.0)] * nk
    q = [f(0.0)] * nk
    td = [f(0.0)] * nk
    pi = [f(0.0)] * nk
    th = [f(0.0)] * nk
    thv = [f(0.0)] * nk
    for k in range(nk):
        p[k] = f(f(100.0) * f(p_mb[k]))
        t[k] = f(f(273.15) + f(t_c[k]))
        q[k] = f(max(f(q_kg_per_kg[k]), f(1.0e-10)))
        ee = _logf(
            f(f(f(f(q[k] / c["eps"]) * p[k]) / f(100.0)) / f(f(1.0) + f(q[k] / c["eps"])))
        )
        td[k] = f(f(273.15) + f(f(f(243.5) * ee - f(440.8)) / f(f(19.48) - ee)))
        pi[k] = _powf(f(p[k] * c["rp00"]), c["rddcp"])
        th[k] = f(t[k] / pi[k])
        thv[k] = f(f(th[k] * f(f(1.0) + f(c["reps"] * q[k]))) / f(f(1.0) + q[k]))

    # getcape.F:147-151.
    z = [f(0.0)] * nk
    for k in range(1, nk):
        dz = f(
            f(-c["cpdg"]) * f(0.5) * f(thv[k] + thv[k - 1]) * f(pi[k] - pi[k - 1])
        )
        z[k] = f(z[k - 1] + dz)

    # getcape.F:155-251.
    kmax = 0
    avgth = 0.0
    avgqv = 0.0
    if source == 1:
        kmax = 0
    elif source == 2:
        if p[0] < f(50000.0):
            kmax = 0
        else:
            maxthe = f(0.0)
            for k in range(nk):
                if p[k] >= f(50000.0):
                    the = getthx(np, p[k], t[k], td[k], q[k])
                    if the > maxthe:
                        maxthe = the
                        kmax = k
    else:
        if f(z[1] - z[0]) > ml_depth:
            avgth, avgqv, kmax = float(th[0]), float(q[0]), 0
        elif z[nk - 1] < ml_depth:
            avgth, avgqv, kmax = float(th[nk - 1]), float(q[nk - 1]), nk - 1
        else:
            k = 1
            while k < nk and z[k] <= ml_depth:
                avgth += float(f(f(0.5) * f(z[k] - z[k - 1]) * f(th[k] + th[k - 1])))
                avgqv += float(f(f(0.5) * f(z[k] - z[k - 1]) * f(q[k] + q[k - 1])))
                k += 1
            if k >= nk:
                raise ValueError("getcape.F:212-224 ran past the top of the column")
            th2s = f(
                th[k - 1]
                + f(f(th[k] - th[k - 1]) * f(ml_depth - z[k - 1]) / f(z[k] - z[k - 1]))
            )
            qv2s = f(
                q[k - 1]
                + f(f(q[k] - q[k - 1]) * f(ml_depth - z[k - 1]) / f(z[k] - z[k - 1]))
            )
            avgth += float(f(f(0.5) * f(ml_depth - z[k - 1]) * f(th2s + th[k - 1])))
            avgqv += float(f(f(0.5) * f(ml_depth - z[k - 1]) * f(qv2s + q[k - 1])))
            avgth /= float(ml_depth)
            avgqv /= float(ml_depth)
            kmax = 0

    # getcape.F:253-312.
    narea = f(0.0)
    k = kmax
    pi2 = pi[kmax]
    p2 = p[kmax]
    if source in (1, 2):
        th2 = th[kmax]
        t2 = t[kmax]
        qv2 = q[kmax]
        b2 = f(0.0)
    else:
        th2 = f(avgth)
        qv2 = f(avgqv)
        thv_source = f(f(th2 * f(f(1.0) + f(c["reps"] * qv2))) / f(f(1.0) + qv2))
        t2 = f(th2 * pi2)
        b2 = f(f(c["g"] * f(thv_source - thv[kmax])) / thv[kmax])
    psource, tsource, qvsource = float(p2), float(t2), float(qv2)

    ql2 = f(0.0)
    qi2 = f(0.0)
    qt = qv2
    cape = f(0.0)
    cin = f(0.0)
    doit = True
    cloud = False
    ice = adiabat in (3, 4)
    pseudo = adiabat in (1, 3)
    zlcl = f(-1.0)
    zlfc = f(-1.0)
    zel = f(-1.0)
    converged = True

    # getcape.F:321-513.
    while doit and k + 1 < nk:
        k += 1
        b1 = b2
        dp = f(p[k - 1] - p[k])
        if dp < pinc:
            nloop = 1
        else:
            nloop = 1 + int(dp / pinc)
            dp = f(dp / f(nloop))
        for n in range(1, nloop + 1):
            p1, t1, th1, qv1, ql1, qi1 = p2, t2, th2, qv2, ql2, qi2
            p2 = f(p2 - dp)
            pi2 = _powf(f(p2 * c["rp00"]), c["rddcp"])
            thlast = th1
            iteration = 0
            while True:
                iteration += 1
                t2 = f(thlast * pi2)
                if ice:
                    fliq = f(
                        max(
                            min(f(f(t2 - f(233.15)) / f(f(273.15) - f(233.15))), f(1.0)),
                            f(0.0),
                        )
                    )
                    fice = f(f(1.0) - fliq)
                else:
                    fliq, fice = f(1.0), f(0.0)
                qv2 = f(
                    min(
                        qt,
                        f(
                            f(fliq * getqvl(np, p2, t2))
                            + f(fice * getqvi(np, p2, t2))
                        ),
                    )
                )
                qi2 = f(max(f(fice * f(qt - qv2)), f(0.0)))
                ql2 = f(max(f(f(qt - qv2) - qi2), f(0.0)))
                tbar = f(f(0.5) * f(t1 + t2))
                qvbar = f(f(0.5) * f(qv1 + qv2))
                qlbar = f(f(0.5) * f(ql1 + ql2))
                qibar = f(f(0.5) * f(qi1 + qi2))
                lhv = f(c["lv1"] - f(c["lv2"] * tbar))
                lhs = f(c["ls1"] - f(c["ls2"] * tbar))
                rm = f(c["rd"] + f(c["rv"] * qvbar))
                cpm = f(
                    f(f(c["cp"] + f(c["cpv"] * qvbar)) + f(c["cpl"] * qlbar))
                    + f(c["cpi"] * qibar)
                )
                exponent = f(
                    f(
                        f(f(lhv * f(ql2 - ql1)) / f(cpm * tbar))
                        + f(f(lhs * f(qi2 - qi1)) / f(cpm * tbar))
                    )
                    + f(f(f(rm / cpm) - f(c["rd"] / c["cp"])) * _logf(f(p2 / p1)))
                )
                th2 = f(th1 * _expf(exponent))
                if iteration > MAX_ADIABAT_ITERATIONS:
                    # getcape.F:383-396.  CAPE, CIN and the three source words
                    # are zeroed and the routine returns; zlcl, zlfc and zel
                    # keep whatever the ascent had reached.
                    return {
                        "cape": 0.0,
                        "cin": 0.0,
                        "lcl": float(zlcl),
                        "lfc": float(zlfc),
                        "el": float(zel),
                        "psource": 0.0,
                        "tsource": 0.0,
                        "qvsource": 0.0,
                        "converged": False,
                    }
                if abs(f(th2 - thlast)) > c["converge"]:
                    thlast = f(thlast + f(f(0.3) * f(th2 - thlast)))
                else:
                    break
            if ql2 >= f(1.0e-10):
                cloud = True
            if cloud and zlcl < f(0.0):
                zlcl = f(
                    z[k - 1] + f(f(f(z[k] - z[k - 1]) * f(n)) / f(nloop))
                )
            if pseudo:
                qt = qv2
                ql2 = f(0.0)
                qi2 = f(0.0)

        thv2 = f(
            f(th2 * f(f(1.0) + f(c["reps"] * qv2)))
            / f(f(f(f(1.0) + qv2) + ql2) + qi2)
        )
        b2 = f(f(c["g"] * f(thv2 - thv[k])) / thv[k])
        dz = f(
            f(-c["cpdg"]) * f(0.5) * f(thv[k] + thv[k - 1]) * f(pi[k] - pi[k - 1])
        )
        if zlcl > f(0.0) and zlfc < f(0.0) and b2 > f(0.0):
            if b1 > f(0.0):
                zlfc = zlcl
            else:
                zlfc = f(
                    z[k - 1] + f(f(f(z[k] - z[k - 1]) * f(f(0.0) - b1)) / f(b2 - b1))
                )
        if zlfc > f(0.0) and zel < f(0.0) and b2 < f(0.0):
            zel = f(
                z[k - 1] + f(f(f(z[k] - z[k - 1]) * f(f(0.0) - b1)) / f(b2 - b1))
            )
        if b2 >= f(0.0) and b1 < f(0.0):
            frac = f(b2 / f(b2 - b1))
            parea = f(f(0.5) * b2 * dz * frac)
            narea = f(narea - f(f(0.5) * b1 * dz * f(f(1.0) - frac)))
            cin = f(cin + narea)
            narea = f(0.0)
        elif b2 < f(0.0) and b1 > f(0.0):
            frac = f(b1 / f(b1 - b2))
            parea = f(f(0.5) * b1 * dz * frac)
            narea = f(f(-0.5) * b2 * dz * f(f(1.0) - frac))
        elif b2 < f(0.0):
            parea = f(0.0)
            narea = f(narea - f(f(0.5) * dz * f(b1 + b2)))
        else:
            parea = f(f(0.5) * dz * f(b1 + b2))
            narea = f(0.0)
        cape = f(cape + f(max(f(0.0), parea)))
        if p[k] <= f(10000.0) and b2 < f(0.0):
            doit = False

    return {
        "cape": float(cape),
        "cin": float(cin),
        "lcl": float(zlcl),
        "lfc": float(zlfc),
        "el": float(zel),
        "psource": psource,
        "tsource": tsource,
        "qvsource": qvsource,
        "converged": converged,
    }


def _require_host_inputs(
    arrays: Mapping[str, Any], *, ni: int, nj: int, nk: int
) -> dict[str, Any]:
    np = importlib.import_module("numpy")
    shapes = getcape_field_shapes(ni, nj, nk)
    missing = [name for name in INPUT_NAMES if name not in arrays]
    if missing:
        raise ValueError(f"missing getcape arrays: {', '.join(missing)}")
    result: dict[str, Any] = {}
    for name in INPUT_NAMES:
        value = arrays[name]
        if (
            not isinstance(value, np.ndarray)
            or value.dtype != np.dtype("float32")
            or value.shape != shapes[name]
            or not value.flags.c_contiguous
        ):
            raise ValueError(
                f"getcape input {name} must have C-order float32 shape {shapes[name]}"
            )
        if not bool(np.isfinite(value).all()):
            raise ValueError(f"getcape input {name} contains non-finite values")
        result[name] = value
    if len({int(value.ctypes.data) for value in result.values()}) != len(result):
        raise ValueError("getcape input arrays must not alias")
    return result


def cpu_getcape_columns(
    arrays: Mapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
    selection: GetCapeFieldSelection,
    surface_weights: tuple[float, float, float] = UNIFORM_SURFACE_WEIGHTS,
) -> dict[str, Any]:
    """``writeout.F:2895-2932`` on the host: four planes from one live state."""

    np = importlib.import_module("numpy")
    require_getcape_selection(selection)
    a = _require_host_inputs(arrays, ni=ni, nj=nj, nk=nk)
    f = np.float32
    weights = tuple(f(value) for value in surface_weights)
    if len(weights) != 3:
        raise ValueError("param.F:9038 forms exactly three bottom weights")
    planes = {
        name: np.zeros((nj, ni), dtype=np.float32) for name in GETCAPE_FIELD_NAMES
    }
    for j in range(nj):
        for i in range(ni):
            # writeout.F:2907-2910.
            p_mb = [f(0.0)] * (nk + 1)
            t_c = [f(0.0)] * (nk + 1)
            q = [f(0.0)] * (nk + 1)
            for k in range(nk):
                p_mb[k + 1] = f(f(0.01) * a["prs"][k, j, i])
                theta = f(a["th0"][k, j, i] + a["th3d"][k, j, i])
                exner = f(a["pi0"][k, j, i] + a["pp3d"][k, j, i])
                t_c[k + 1] = f(f(theta * exner) - f(273.15))
                q[k + 1] = f(a["qv"][k, j, i])
            # writeout.F:2912-2914.
            for column in (p_mb, t_c, q):
                column[0] = f(
                    f(f(weights[0] * column[1]) + f(weights[1] * column[2]))
                    + f(weights[2] * column[3])
                )
            result = cpu_getcape(
                p_mb,
                t_c,
                q,
                source=selection.parcel_source,
                adiabat=selection.adiabat,
                mixed_layer_depth_m=selection.mixed_layer_depth_m,
                pressure_increment_pa=selection.pressure_increment_pa,
            )
            for name in GETCAPE_FIELD_NAMES:
                planes[name][j, i] = np.float32(result[name])
    return planes


CUDA_GETCAPE_SOURCE = r'''// gm1.cuda-getcape-fields-kernels/v1
// CM1 r21.1 src/getcape.F:14-594 per column, assembled as src/writeout.F:2907-2914.
// Every arithmetic operation is an explicit round-to-nearest intrinsic so no
// multiply-add is fused; expf/logf/powf are the device library's, which is not
// glibc, and that is the one difference this text carries against the host
// oracle it was checked with.
__device__ __forceinline__ float A(float a,float b){return __fadd_rn(a,b);}
__device__ __forceinline__ float S(float a,float b){return __fsub_rn(a,b);}
__device__ __forceinline__ float M(float a,float b){return __fmul_rn(a,b);}
__device__ __forceinline__ float D(float a,float b){return __fdiv_rn(a,b);}

// getcape.F:100-121.
#define CM1_G        9.81f
#define CM1_P00      100000.0f
#define CM1_CP       1005.7f
#define CM1_RD       287.04f
#define CM1_RV       461.5f
#define CM1_CPV      1875.0f
#define CM1_CPL      4190.0f
#define CM1_CPI      2118.636f
#define CM1_CONVERGE 0.0002f
#define CM1_MAXITER  100u

// getcape.F:526-538.  Bolton (1980) over liquid, with the 171023 clamp.
__device__ __forceinline__ float cm1_getqvl(float p,float t){
  const float eps=D(CM1_RD,CM1_RV);
  float es=M(611.2f,expf(D(M(17.67f,S(t,273.15f)),S(t,29.65f))));
  es=fminf(es,M(p,0.5f));
  return D(M(eps,es),S(p,es));
}
// getcape.F:544-556.
__device__ __forceinline__ float cm1_getqvi(float p,float t){
  const float eps=D(CM1_RD,CM1_RV);
  float es=M(611.2f,expf(D(M(21.8745584f,S(t,273.15f)),S(t,7.66f))));
  es=fminf(es,M(p,0.5f));
  return D(M(eps,es),S(p,es));
}
// getcape.F:562-577.
__device__ __forceinline__ float cm1_getthx(float p,float t,float td,float q){
  float tlcl;
  if(S(td,t)>=-0.1f){tlcl=t;}
  else{tlcl=A(56.0f,D(1.0f,A(D(1.0f,S(td,56.0f)),M(0.00125f,logf(D(t,td))))));}
  float first=powf(D(100000.0f,p),M(0.2854f,S(1.0f,M(0.28f,q))));
  float second=expf(M(M(S(D(3376.0f,tlcl),2.54f),q),A(1.0f,M(0.81f,q))));
  return M(M(t,first),second);
}

// writeout.F:2907-2910 for level>=1 and writeout.F:2912-2914 for level 0.
#define CGS(f0,f1,f2) A(A(M(w0,(f0)),M(w1,(f1))),M(w2,(f2)))

__device__ __forceinline__ float cm1_col_p_mb(const float* prs,unsigned k,
    unsigned long long q,unsigned long long npts,float w0,float w1,float w2){
  if(k>0u)return M(0.01f,prs[(unsigned long long)(k-1u)*npts+q]);
  return CGS(M(0.01f,prs[q]),M(0.01f,prs[npts+q]),M(0.01f,prs[2ull*npts+q]));
}
__device__ __forceinline__ float cm1_level_t_c(const float* th0,const float* th3d,
    const float* pi0,const float* pp3d,unsigned long long o){
  return S(M(A(th0[o],th3d[o]),A(pi0[o],pp3d[o])),273.15f);
}
__device__ __forceinline__ float cm1_col_t_c(const float* th0,const float* th3d,
    const float* pi0,const float* pp3d,unsigned k,unsigned long long q,
    unsigned long long npts,float w0,float w1,float w2){
  if(k>0u)return cm1_level_t_c(th0,th3d,pi0,pp3d,(unsigned long long)(k-1u)*npts+q);
  return CGS(cm1_level_t_c(th0,th3d,pi0,pp3d,q),
             cm1_level_t_c(th0,th3d,pi0,pp3d,npts+q),
             cm1_level_t_c(th0,th3d,pi0,pp3d,2ull*npts+q));
}
__device__ __forceinline__ float cm1_col_qv(const float* qv,unsigned k,
    unsigned long long q,unsigned long long npts,float w0,float w1,float w2){
  if(k>0u)return qv[(unsigned long long)(k-1u)*npts+q];
  return CGS(qv[q],qv[npts+q],qv[2ull*npts+q]);
}

extern "C" __global__ void cm1_getcape_fields_v1(
    const float* prs,const float* th0,const float* th3d,const float* pi0,
    const float* pp3d,const float* qv,float* work,
    float* out_cape,float* out_cin,float* out_lcl,float* out_lfc,
    unsigned ni,unsigned nj,unsigned nk,unsigned source,unsigned adiabat,
    float ml_depth,float pinc,float w0,float w1,float w2){
  const unsigned long long npts=(unsigned long long)ni*nj;
  const unsigned long long first=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
  const unsigned long long stride=(unsigned long long)blockDim.x*gridDim.x;
  const unsigned levels=nk+1u;
  const float rp00=D(1.0f,CM1_P00);
  const float eps=D(CM1_RD,CM1_RV);
  const float reps=D(CM1_RV,CM1_RD);
  const float rddcp=D(CM1_RD,CM1_CP);
  const float cpdg=D(CM1_CP,CM1_G);
  const float lv1=A(2501000.0f,M(S(CM1_CPL,CM1_CPV),273.15f));
  const float lv2=S(CM1_CPL,CM1_CPV);
  const float ls1=A(2836017.0f,M(S(CM1_CPI,CM1_CPV),273.15f));
  const float ls2=S(CM1_CPI,CM1_CPV);
  const bool ice=(adiabat==3u||adiabat==4u);
  const bool pseudo=(adiabat==1u||adiabat==3u);

  for(unsigned long long c=first;c<npts;c+=stride){
    float* wp =work+(unsigned long long)0u*levels*npts;
    float* wpi=work+(unsigned long long)1u*levels*npts;
    float* wtv=work+(unsigned long long)2u*levels*npts;
    float* wz =work+(unsigned long long)3u*levels*npts;

    // getcape.F:131-143.
    for(unsigned k=0u;k<levels;++k){
      const unsigned long long o=(unsigned long long)k*npts+c;
      const float p=M(100.0f,cm1_col_p_mb(prs,k,c,npts,w0,w1,w2));
      const float t=A(273.15f,cm1_col_t_c(th0,th3d,pi0,pp3d,k,c,npts,w0,w1,w2));
      const float q=fmaxf(cm1_col_qv(qv,k,c,npts,w0,w1,w2),1.0e-10f);
      const float pi=powf(M(p,rp00),rddcp);
      const float th=D(t,pi);
      wp[o]=p; wpi[o]=pi;
      wtv[o]=D(M(th,A(1.0f,M(reps,q))),A(1.0f,q));
    }
    // getcape.F:147-151.
    wz[c]=0.0f;
    for(unsigned k=1u;k<levels;++k){
      const unsigned long long o=(unsigned long long)k*npts+c, o1=o-npts;
      wz[o]=A(wz[o1],M(M(M(-cpdg,0.5f),A(wtv[o],wtv[o1])),S(wpi[o],wpi[o1])));
    }

    // getcape.F:155-251.
    unsigned kmax=0u; double avgth=0.0, avgqv=0.0;
    if(source==2u){
      if(wp[c]>=50000.0f){
        float maxthe=0.0f;
        for(unsigned k=0u;k<levels;++k){
          const unsigned long long o=(unsigned long long)k*npts+c;
          if(wp[o]>=50000.0f){
            const float t=A(273.15f,cm1_col_t_c(th0,th3d,pi0,pp3d,k,c,npts,w0,w1,w2));
            const float q=fmaxf(cm1_col_qv(qv,k,c,npts,w0,w1,w2),1.0e-10f);
            const float ee=logf(D(D(M(D(q,eps),wp[o]),100.0f),A(1.0f,D(q,eps))));
            const float td=A(273.15f,D(S(M(243.5f,ee),440.8f),S(19.48f,ee)));
            const float the=cm1_getthx(wp[o],t,td,q);
            if(the>maxthe){maxthe=the;kmax=k;}
          }
        }
      }
    }else if(source==3u){
      const float z1=wz[npts+c];
      const float ztop=wz[(unsigned long long)(levels-1u)*npts+c];
      if(S(z1,wz[c])>ml_depth){
        const float t0c=A(273.15f,cm1_col_t_c(th0,th3d,pi0,pp3d,0u,c,npts,w0,w1,w2));
        avgth=(double)D(t0c,wpi[c]);
        avgqv=(double)fmaxf(cm1_col_qv(qv,0u,c,npts,w0,w1,w2),1.0e-10f);
        kmax=0u;
      }else if(ztop<ml_depth){
        const unsigned kt=levels-1u; const unsigned long long ot=(unsigned long long)kt*npts+c;
        const float tt=A(273.15f,cm1_col_t_c(th0,th3d,pi0,pp3d,kt,c,npts,w0,w1,w2));
        avgth=(double)D(tt,wpi[ot]);
        avgqv=(double)fmaxf(cm1_col_qv(qv,kt,c,npts,w0,w1,w2),1.0e-10f);
        kmax=kt;
      }else{
        unsigned k=1u;
        float thprev=D(A(273.15f,cm1_col_t_c(th0,th3d,pi0,pp3d,0u,c,npts,w0,w1,w2)),wpi[c]);
        float qprev=fmaxf(cm1_col_qv(qv,0u,c,npts,w0,w1,w2),1.0e-10f);
        float thk=thprev,qk=qprev;
        while(k<levels&&wz[(unsigned long long)k*npts+c]<=ml_depth){
          const unsigned long long o=(unsigned long long)k*npts+c;
          thk=D(A(273.15f,cm1_col_t_c(th0,th3d,pi0,pp3d,k,c,npts,w0,w1,w2)),wpi[o]);
          qk=fmaxf(cm1_col_qv(qv,k,c,npts,w0,w1,w2),1.0e-10f);
          avgth+=(double)M(M(0.5f,S(wz[o],wz[o-npts])),A(thk,thprev));
          avgqv+=(double)M(M(0.5f,S(wz[o],wz[o-npts])),A(qk,qprev));
          thprev=thk;qprev=qk;++k;
        }
        // getcape.F:194 has already established z(nk) >= ml_depth and z is
        // strictly increasing, so the scan above always stops inside the
        // column.  The clamp cannot fire on an admitted column and exists so
        // that a malformed one reads a level that exists rather than memory
        // that does not.
        if(k>=levels)k=levels-1u;
        const unsigned long long o=(unsigned long long)k*npts+c;
        thk=D(A(273.15f,cm1_col_t_c(th0,th3d,pi0,pp3d,k,c,npts,w0,w1,w2)),wpi[o]);
        qk=fmaxf(cm1_col_qv(qv,k,c,npts,w0,w1,w2),1.0e-10f);
        const float dzl=S(wz[o],wz[o-npts]);
        const float th2i=A(thprev,D(M(S(thk,thprev),S(ml_depth,wz[o-npts])),dzl));
        const float qv2i=A(qprev,D(M(S(qk,qprev),S(ml_depth,wz[o-npts])),dzl));
        avgth+=(double)M(M(0.5f,S(ml_depth,wz[o-npts])),A(th2i,thprev));
        avgqv+=(double)M(M(0.5f,S(ml_depth,wz[o-npts])),A(qv2i,qprev));
        avgth/=(double)ml_depth; avgqv/=(double)ml_depth;
        kmax=0u;
      }
    }

    // getcape.F:253-312.
    const unsigned long long om=(unsigned long long)kmax*npts+c;
    float narea=0.0f,th2,t2,qv2,b2,p2=wp[om],pi2=wpi[om];
    if(source==1u||source==2u){
      const float tk=A(273.15f,cm1_col_t_c(th0,th3d,pi0,pp3d,kmax,c,npts,w0,w1,w2));
      th2=D(tk,pi2); t2=tk;
      qv2=fmaxf(cm1_col_qv(qv,kmax,c,npts,w0,w1,w2),1.0e-10f);
      b2=0.0f;
    }else{
      th2=(float)avgth; qv2=(float)avgqv;
      const float thvs=D(M(th2,A(1.0f,M(reps,qv2))),A(1.0f,qv2));
      t2=M(th2,pi2);
      b2=D(M(CM1_G,S(thvs,wtv[om])),wtv[om]);
    }
    float ql2=0.0f,qi2=0.0f,qt=qv2,cape=0.0f,cin=0.0f;
    float zlcl=-1.0f,zlfc=-1.0f,zel=-1.0f;
    bool doit=true,cloud=false,converged=true;
    unsigned k=kmax;

    // getcape.F:321-513.
    while(doit&&k+1u<levels){
      k+=1u;
      const unsigned long long o=(unsigned long long)k*npts+c, o1=o-npts;
      const float b1=b2;
      float dp=S(wp[o1],wp[o]);
      int nloop=1;
      if(dp<pinc){nloop=1;}else{nloop=1+(int)(dp/pinc);dp=D(dp,(float)nloop);}
      for(int n=1;n<=nloop;++n){
        const float p1=p2,t1=t2,th1=th2,qv1=qv2,ql1=ql2,qi1=qi2;
        p2=S(p2,dp); pi2=powf(M(p2,rp00),rddcp);
        float thlast=th1; unsigned it=0u;
        for(;;){
          it+=1u; t2=M(thlast,pi2);
          float fliq=1.0f,fice=0.0f;
          if(ice){fliq=fmaxf(fminf(D(S(t2,233.15f),S(273.15f,233.15f)),1.0f),0.0f);fice=S(1.0f,fliq);}
          qv2=fminf(qt,A(M(fliq,cm1_getqvl(p2,t2)),M(fice,cm1_getqvi(p2,t2))));
          qi2=fmaxf(M(fice,S(qt,qv2)),0.0f);
          ql2=fmaxf(S(S(qt,qv2),qi2),0.0f);
          const float tbar=M(0.5f,A(t1,t2)),qvbar=M(0.5f,A(qv1,qv2));
          const float qlbar=M(0.5f,A(ql1,ql2)),qibar=M(0.5f,A(qi1,qi2));
          const float lhv=S(lv1,M(lv2,tbar)),lhs=S(ls1,M(ls2,tbar));
          const float rm=A(CM1_RD,M(CM1_RV,qvbar));
          const float cpm=A(A(A(CM1_CP,M(CM1_CPV,qvbar)),M(CM1_CPL,qlbar)),M(CM1_CPI,qibar));
          const float term=A(A(D(M(lhv,S(ql2,ql1)),M(cpm,tbar)),D(M(lhs,S(qi2,qi1)),M(cpm,tbar))),
                             M(S(D(rm,cpm),D(CM1_RD,CM1_CP)),logf(D(p2,p1))));
          th2=M(th1,expf(term));
          // getcape.F:383-396: abandon the fixed point and report zeros.
          if(it>CM1_MAXITER){converged=false;break;}
          if(fabsf(S(th2,thlast))>CM1_CONVERGE){thlast=A(thlast,M(0.3f,S(th2,thlast)));}
          else break;
        }
        if(!converged)break;
        if(ql2>=1.0e-10f)cloud=true;
        if(cloud&&zlcl<0.0f)zlcl=A(wz[o1],D(M(S(wz[o],wz[o1]),(float)n),(float)nloop));
        if(pseudo){qt=qv2;ql2=0.0f;qi2=0.0f;}
      }
      if(!converged){doit=false;break;}
      const float thv2=D(M(th2,A(1.0f,M(reps,qv2))),A(A(A(1.0f,qv2),ql2),qi2));
      b2=D(M(CM1_G,S(thv2,wtv[o])),wtv[o]);
      const float dz=M(M(M(-cpdg,0.5f),A(wtv[o],wtv[o1])),S(wpi[o],wpi[o1]));
      if(zlcl>0.0f&&zlfc<0.0f&&b2>0.0f){
        if(b1>0.0f)zlfc=zlcl;
        else zlfc=A(wz[o1],D(M(S(wz[o],wz[o1]),S(0.0f,b1)),S(b2,b1)));
      }
      if(zlfc>0.0f&&zel<0.0f&&b2<0.0f)
        zel=A(wz[o1],D(M(S(wz[o],wz[o1]),S(0.0f,b1)),S(b2,b1)));
      float parea;
      if(b2>=0.0f&&b1<0.0f){
        const float frac=D(b2,S(b2,b1));
        parea=M(M(M(0.5f,b2),dz),frac);
        narea=S(narea,M(M(M(0.5f,b1),dz),S(1.0f,frac)));
        cin=A(cin,narea); narea=0.0f;
      }else if(b2<0.0f&&b1>0.0f){
        const float frac=D(b1,S(b1,b2));
        parea=M(M(M(0.5f,b1),dz),frac);
        narea=M(M(M(-0.5f,b2),dz),S(1.0f,frac));
      }else if(b2<0.0f){
        parea=0.0f; narea=S(narea,M(M(0.5f,dz),A(b1,b2)));
      }else{
        parea=M(M(0.5f,dz),A(b1,b2)); narea=0.0f;
      }
      cape=A(cape,fmaxf(0.0f,parea));
      if(wp[o]<=10000.0f&&b2<0.0f)doit=false;
    }
    // getcape.F:383-396: cape and cin are zeroed and the routine returns;
    // zlcl and zlfc keep whatever the ascent had reached.
    if(!converged){cape=0.0f;cin=0.0f;}
    out_cape[c]=cape; out_cin[c]=cin; out_lcl[c]=zlcl; out_lfc[c]=zlfc;
  }
}
'''
CUDA_GETCAPE_SOURCE_BYTES = CUDA_GETCAPE_SOURCE.encode("utf-8")
CUDA_GETCAPE_SOURCE_SHA256 = (
    "bbf54f1f4cf7ff140ba1e43e179effad9a5545b322e3e0befc9dbd1058990b96"
)


@dataclass(frozen=True, slots=True)
class CompiledGetCapeKernel:
    module_owner: Any
    kernel: Any
    ptx: bytes
    cubin: bytes
    target_arch: str
    ptx_options: tuple[str, ...]
    cubin_options: tuple[str, ...]
    runtime_identity: tuple[tuple[str, str], ...]


def compile_cuda_getcape() -> CompiledGetCapeKernel:
    """Compile the packaged kernel text for the active device, pin-checked."""

    digest = hashlib.sha256(CUDA_GETCAPE_SOURCE_BYTES).hexdigest()
    if digest != CUDA_GETCAPE_SOURCE_SHA256:
        raise RuntimeError("packaged getcape CUDA source differs from pin")
    cupy = importlib.import_module("cupy")
    compiler = importlib.import_module("cupy.cuda.compiler")
    # One architecture rule for the package (nvrtc_module.nvrtc_target_arch):
    # compiler._get_arch(), the active device's capability clamped to what this
    # NVRTC accepts.  This function used to build the target from the device's
    # own compute_capability and refuse when the two differed; the clamp is the
    # target every other GM1 image and every CuPy compile already uses.
    arch = importlib.import_module("cm1core.nvrtc_module").nvrtc_target_arch(cupy)
    nvrtc = tuple(compiler._get_nvrtc_version())
    common = (
        "--std=c++14",
        "--fmad=false",
        "--ftz=false",
        "--prec-div=true",
        "--prec-sqrt=true",
    )
    if int(nvrtc[0]) >= 12:
        common += ("--device-as-default-execution-space",)
    ptx_options = common + (f"--gpu-architecture=compute_{arch}",)
    cubin_options = common + (f"--gpu-architecture=sm_{arch}",)
    ptx, _ = compiler._NVRTCProgram(
        CUDA_GETCAPE_SOURCE, "cm1_getcape_fields_v1_ptx.cu", method="ptx"
    ).compile(ptx_options)
    cubin, _ = compiler._NVRTCProgram(
        CUDA_GETCAPE_SOURCE, "cm1_getcape_fields_v1_cubin.cu", method="cubin"
    ).compile(cubin_options)
    cubin = cubin.encode("latin-1") if isinstance(cubin, str) else bytes(cubin)
    # The image that runs is the cubin compiled above (``--ftz=false``), not a
    # RawModule, which CuPy compiles with ``-ftz=true`` appended.
    module = importlib.import_module("cupy.cuda.function").Module()
    module.load(cubin)
    kernel = module.get_function(CUDA_GETCAPE_KERNEL_NAME)
    return CompiledGetCapeKernel(
        module_owner=module,
        kernel=kernel,
        ptx=ptx if isinstance(ptx, bytes) else bytes(ptx, "utf-8"),
        cubin=cubin,
        target_arch=arch,
        ptx_options=ptx_options,
        cubin_options=cubin_options,
        runtime_identity=(
            ("cupy", str(cupy.__version__)),
            ("nvrtc", ".".join(str(part) for part in nvrtc)),
            ("arch", arch),
        ),
    )


def launch_cuda_getcape(
    compiled: CompiledGetCapeKernel,
    arrays: Mapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
    selection: GetCapeFieldSelection,
    surface_weights: tuple[float, float, float] = UNIFORM_SURFACE_WEIGHTS,
    stream: Any = None,
) -> None:
    """Run one column sweep in place over already-resident device arrays."""

    require_getcape_selection(selection)
    cupy = importlib.import_module("cupy")
    shapes = getcape_field_shapes(ni, nj, nk)
    needed = (*INPUT_NAMES, "work", *GETCAPE_FIELD_NAMES)
    missing = [name for name in needed if name not in arrays]
    if missing:
        raise ValueError(f"missing getcape device arrays: {', '.join(missing)}")
    for name in needed:
        value = arrays[name]
        if value.dtype != cupy.dtype("float32") or tuple(value.shape) != shapes[name]:
            raise ValueError(
                f"getcape device array {name} must be float32 {shapes[name]}, "
                f"not {value.dtype} {tuple(value.shape)}"
            )
        if not value.flags.c_contiguous:
            raise ValueError(f"getcape device array {name} must be C-contiguous")
    columns = ni * nj
    blocks = (columns + CUDA_GETCAPE_THREADS - 1) // CUDA_GETCAPE_THREADS
    weights = tuple(cupy.float32(value) for value in surface_weights)
    if len(weights) != 3:
        raise ValueError("param.F:9038 forms exactly three bottom weights")
    arguments = (
        *(arrays[name] for name in INPUT_NAMES),
        arrays["work"],
        *(arrays[name] for name in GETCAPE_FIELD_NAMES),
        cupy.uint32(ni),
        cupy.uint32(nj),
        cupy.uint32(nk),
        cupy.uint32(selection.parcel_source),
        cupy.uint32(selection.adiabat),
        cupy.float32(selection.mixed_layer_depth_m),
        cupy.float32(selection.pressure_increment_pa),
        *weights,
    )
    launcher = compiled.kernel
    if stream is None:
        launcher((blocks,), (CUDA_GETCAPE_THREADS,), arguments)
    else:
        with stream:
            launcher((blocks,), (CUDA_GETCAPE_THREADS,), arguments)


def cuda_getcape_record(
    compiled: CompiledGetCapeKernel | None,
    selection: GetCapeFieldSelection,
    *,
    ni: int,
    nj: int,
    nk: int,
) -> dict[str, Any]:
    """What the receipt says about this run's environment-field sweep."""

    record: dict[str, Any] = {
        "abi": CUDA_GETCAPE_ABI,
        "kernel": CUDA_GETCAPE_KERNEL_NAME,
        "block_threads": CUDA_GETCAPE_THREADS,
        "source_sha256": CUDA_GETCAPE_SOURCE_SHA256,
        "selection": selection.record(),
        **getcape_scratch_bytes(ni, nj, nk),
    }
    if compiled is not None:
        record["target_arch"] = compiled.target_arch
        record["runtime_identity"] = dict(compiled.runtime_identity)
        record["cubin_options"] = list(compiled.cubin_options)
    return record


__all__ = [
    "CUDA_GETCAPE_ABI",
    "CUDA_GETCAPE_CLAIM_LIMIT",
    "CUDA_GETCAPE_KERNEL_NAME",
    "CUDA_GETCAPE_SCHEMA",
    "CUDA_GETCAPE_SOURCE",
    "CUDA_GETCAPE_SOURCE_BYTES",
    "CUDA_GETCAPE_SOURCE_SHA256",
    "CUDA_GETCAPE_THREADS",
    "DEFAULT_ADIABAT",
    "DEFAULT_MIXED_LAYER_DEPTH_M",
    "DEFAULT_PARCEL_SOURCE",
    "DEFAULT_PRESSURE_INCREMENT_PA",
    "GETCAPE_FIELD_META",
    "GETCAPE_FIELD_NAMES",
    "GETCAPE_FIELD_SWITCHES",
    "GetCapeFieldSelection",
    "SOURCE_ANCHORS",
    "UNIFORM_SURFACE_WEIGHTS",
    "compile_cuda_getcape",
    "cpu_getcape",
    "cpu_getcape_columns",
    "cuda_getcape_record",
    "getcape_field_shapes",
    "getcape_scratch_bytes",
    "launch_cuda_getcape",
    "require_getcape_selection",
]
