"""CM1 r21.1's three storm-starting forcings: iinit = 9, 10 and 12.

Three of ``init3d.F``'s arms build no initial perturbation at all.  They set
control words, and the storm is started by a forcing CM1 applies during the
first minutes of the integration:

* ``iinit = 12``, updraft nudging (Naylor and Gilmore 2012, MWR 140,
  3699-3705).  ``init3d.F:1370-1396`` sets the ellipsoid (``xc/yc/zc`` and
  ``xr/yr/zr``), the rate ``alpha_wnudge``, the target ``wmax_wnudge`` and the
  window ``t1_wnudge``/``t2_wnudge``.  ``misclibs.F:539-585`` (``get_wnudge``)
  builds ``alpha*gamm*max(wmax*cos^2(pi*beta/2) - w, 0)`` inside the ellipsoid,
  and ``sound.F:564-590`` adds it to ``wten`` inside every acoustic small step,
  after the horizontal momentum update and before the vertically implicit
  solve, on ``k = 2..nk`` and the physical columns, from the saved RK-stage
  tendency ``wtens`` (``if(n.eq.1) wtens = wten``).  It acts while
  ``mtime + dt <= t2_wnudge``; ``gamm`` ramps linearly from 1 at ``t1_wnudge``
  to 0 at ``t2_wnudge``.
* ``iinit = 9``, forced low-level convergence (Loftus et al. 2008, MWR 136,
  2408-2421).  ``init3d.F:1303-1333`` sets ``Dmax``, ``zdeep``, ``lamx``,
  ``lamy``, ``xcent``, ``ycent`` and ``convtime`` and derives ``Aconv``;
  ``misclibs.F:456-531`` (``convinitu``/``convinitv``) overwrite ``u3d``/``v3d``
  with ``u0 + umo`` wherever ``|umo| > 0.01 m/s`` below ``zdeep``, and
  ``sound.F:412-440`` calls them in every acoustic small step, right after the
  horizontal momentum update, while ``rtime = sngl(mtime+dt) <= convtime``
  (``solve2.F:291``; ``cm1.F:4363-4365`` then clears ``convinit``).
* ``iinit = 10``, momentum forcing (Morrison et al. 2015, JAS 72, 312-339).
  ``init3d.F:1334-1347`` sets the forcing box and its ``t1``/``t2`` ramp;
  ``solve1.F:1245-1265`` adds ``gamm*alpha*cos(pi/2*(xf-xc)/xr)*
  cosh(2.5*(zf-zs)/zr)**(-2)`` to the held u tendency ``uten1`` once per large
  step while ``mtime < t2_uforce``.

CM1 reads all of these words from its source; this port reads them from the
``[initialization]`` table under CM1's own names, with CM1's values as the
defaults.  Where CM1 writes ``centerx``/``centery`` (``param.F:7408``) an
undeclared centre is the domain centre in the ``iorigin`` frame.

What GM1 does differently, stated rather than hidden:

* a restart (``irst = 1``) turns the forcing off, because ``param.F:891-903``
  sets ``iinit = 0`` on a restart and ``init3d.F:175-176`` then leaves
  ``convinit = wnudge = 0``; a run resumed inside the forcing window therefore
  differs from one that was never stopped, in CM1 and here alike;
* ``cosh`` (iinit = 10) is glibc 2.41+'s correctly rounded ``coshf`` on
  every host; a CM1 built against glibc 2.40 or older calls another
  ``coshf`` that is an ulp off for about 0.8 % of arguments, so its u
  forcing differs from this one (and from the reference CM1 build) in
  a few words;
* a run with CUDA graphs on is refused while a forcing is selected
  (``cartesian_door.cuda_graph_configuration_refusal``), because the ramp is a
  function of the model time and a replayed graph would freeze it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

#: The arms this module owns.
FORCING_IINIT: frozenset[int] = frozenset({9, 10, 12})

INITIATION_FORCING_SCHEMA = "gm1.cm1-initiation-forcing/v1"

#: CM1's own names and values for each arm's words (init3d.F).  ``None`` is
#: CM1's ``centerx``/``centery`` or ``minx+0.5*(maxx-minx)``: the domain centre.
WNUDGE_DEFAULTS: Mapping[str, Any] = MappingProxyType(
    {
        # init3d.F:1372-1385
        "xc_wnudge": None,
        "xr_wnudge": 10000.0,
        "yc_wnudge": None,
        "yr_wnudge": 10000.0,
        "zc_wnudge": 1500.0,
        "zr_wnudge": 1500.0,
        "alpha_wnudge": 0.5,
        "wmax_wnudge": 10.0,
        "t1_wnudge": 900.0,
        "t2_wnudge": 1200.0,
    }
)
CONVERGENCE_DEFAULTS: Mapping[str, Any] = MappingProxyType(
    {
        # init3d.F:1309-1316
        "dmax": -1.0e-3,
        "zdeep": 2000.0,
        "lamx": 10000.0,
        "lamy": 10000.0,
        "xcent": 0.0,
        "ycent": 0.0,
        "convtime": 900.0,
    }
)
UFORCE_DEFAULTS: Mapping[str, Any] = MappingProxyType(
    {
        # init3d.F:1339-1344
        "xc_uforce": None,
        "xr_uforce": 10000.0,
        "zr_uforce": 10000.0,
        "alpha_uforce": 0.1,
        "t1_uforce": 3300.0,
        "t2_uforce": 3600.0,
    }
)
FORCING_DEFAULTS: Mapping[int, Mapping[str, Any]] = MappingProxyType(
    {12: WNUDGE_DEFAULTS, 9: CONVERGENCE_DEFAULTS, 10: UFORCE_DEFAULTS}
)
FORCING_KEYS: frozenset[str] = frozenset(
    set(WNUDGE_DEFAULTS) | set(CONVERGENCE_DEFAULTS) | set(UFORCE_DEFAULTS)
)

AUTHORITY: Mapping[int, str] = MappingProxyType(
    {
        12: (
            "CM1 r21.1 init3d.F:1370-1396 (control words), misclibs.F:539-585 "
            "get_wnudge, sound.F:564-590 (added to wten every acoustic small "
            "step while mtime+dt <= t2_wnudge); Naylor and Gilmore (2012)"
        ),
        9: (
            "CM1 r21.1 init3d.F:1303-1333 (control words, Aconv), "
            "misclibs.F:456-531 convinitu/convinitv, sound.F:412-440 (every "
            "acoustic small step while sngl(mtime+dt) <= convtime), "
            "cm1.F:4363-4365; Loftus et al. (2008)"
        ),
        10: (
            "CM1 r21.1 init3d.F:1334-1347 (control words), solve1.F:1245-1265 "
            "(added to the held u tendency once per large step while "
            "mtime < t2_uforce); Morrison et al. (2015)"
        ),
    }
)

#: constants.F's pi, as CM1 carries it in a default REAL.
_PI_F32 = 3.14159265358979323


def _f32(value: float) -> float:
    import numpy as np

    return float(np.float32(value))


# ------------------------------------------------ CM1's transcendentals
#
# gfortran's default-REAL cos, exp and cosh call the C library's cosf, expf
# and coshf (glibc on the Linux hosts CM1 and GM1 run on).  NumPy's float32
# functions and CUDA's cosf/expf are other implementations that differ from
# glibc's by an ulp or more (tools/oracle/initiation_forcing_oracle.py measured
# up to 512 ulp in a forced u tendency), so every transcendental these
# forcings need is evaluated here, once, on the host, and the device is handed
# the resulting words: its kernels do only correctly rounded +, -, *, / and
# max, as in the updraft-nudging forcing.
# cosf and expf are glibc's (the same words in glibc 2.39 and 2.43 over every
# argument in [-16, 16]); coshf is not the host's but the correctly rounded
# value glibc 2.41+ returns, which the reference CM1 build calls
# (_coshf_correctly_rounded).

_LIBM: Any = None


def _libm() -> Any:
    global _LIBM
    if _LIBM is None:
        import ctypes
        import ctypes.util

        name = ctypes.util.find_library("m")
        library = ctypes.CDLL(name) if name else ctypes.CDLL(None)
        for fn in ("cosf", "expf", "coshf"):
            getattr(library, fn).restype = ctypes.c_float
            getattr(library, fn).argtypes = (ctypes.c_float,)
        _LIBM = library
    return _LIBM


def _coshf_correctly_rounded(value: float) -> float:
    """binary32 ``cosh``, correctly rounded: glibc 2.41+'s ``coshf``.

    glibc 2.41 replaced ``coshf`` with CORE-MATH's correctly rounded one, so a
    CM1 built against it (the reference build, glibc 2.43) and one
    built against glibc 2.40 or older (Ubuntu 24.04's 2.39) differ by an ulp for
    about 0.8 % of arguments; on CM1's stock 1 km grid with iinit = 10's own
    words that moved 76 of the 380 forced ``aiu`` words (4 of the 20 forced
    levels, up to 4 ulp).  The binary64 ``cosh`` rounded once to binary32 is
    that correctly rounded value: over every non-NaN binary32 argument
    (4,278,190,082 words) it hashes identically to that build's ``coshf``.
    Using it makes the
    forcing the same on every host instead of following the host's C library.
    """

    import numpy as np

    try:
        wide = math.cosh(value)
    except OverflowError:
        return math.inf
    with np.errstate(over="ignore"):
        return float(np.float32(wide))


def libm_f32(name: str, values: Any) -> Any:
    """binary32 ``name`` (cosf, expf or coshf) of every word of ``values``.

    ``cosf`` and ``expf`` are the host glibc's own (identical over every
    argument these forcings reach in glibc 2.39 and 2.43); ``coshf`` is the
    correctly rounded value glibc 2.41+ returns (:func:`_coshf_correctly_rounded`).
    """

    import ctypes

    import numpy as np

    array = np.asarray(values, dtype=np.float32)
    distinct, inverse = np.unique(array, return_inverse=True)
    if name == "coshf":
        out = np.fromiter((_coshf_correctly_rounded(float(v)) for v in distinct), dtype=np.float32, count=distinct.size)
    else:
        fn = getattr(_libm(), name)
        out = np.fromiter((fn(ctypes.c_float(float(v))) for v in distinct), dtype=np.float32, count=distinct.size)
    return out[inverse.reshape(-1)].reshape(array.shape)


@dataclass(frozen=True, slots=True)
class InitiationForcing:
    """One selected forcing arm with every word resolved by value.

    Positions are metres in the run's ``iorigin`` frame (``param.F:7212-7213``),
    the frame CM1's ``xh``/``xf``/``yh``/``yf`` carry.  ``words`` holds the
    arm's CM1 words, resolved (a ``None`` centre replaced by the domain centre).
    """

    iinit: int
    #: ``(name, value)`` pairs, sorted by name; a tuple so the owning
    #: ``RuntimeConfig`` stays hashable and picklable.
    words: tuple[tuple[str, float], ...]
    #: Which words the document declared (the rest are CM1's own values).
    declared: frozenset[str] = frozenset()
    #: ``param.F:7212-7213``: 1 corner, 2 centre.
    iorigin: int = 1

    def __post_init__(self) -> None:
        if self.iinit not in FORCING_IINIT:
            raise ValueError(f"iinit={self.iinit} is not a forcing arm")
        words = dict(self.words)
        expected = set(FORCING_DEFAULTS[self.iinit])
        if set(words) != expected:
            raise ValueError(f"iinit={self.iinit} forcing needs exactly {sorted(expected)}")
        for name, value in words.items():
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        object.__setattr__(self, "words", tuple(sorted((k, float(v)) for k, v in words.items())))
        object.__setattr__(self, "declared", frozenset(self.declared))

    def __getitem__(self, name: str) -> float:
        for key, value in self.words:
            if key == name:
                return value
        raise KeyError(name)

    # -- time windows, in CM1's own precision ------------------------------

    def active_substeps(self, mtime_s: float, dt_s: float) -> bool:
        """Does ``sound.F`` apply this arm in the step that starts at mtime?"""

        dt32 = _f32(dt_s)
        if self.iinit == 12:
            # sound.F:569: IF( (mtime+dt).le.t2_wnudge ); mtime is double.
            return float(mtime_s) + dt32 <= _f32(self["t2_wnudge"])
        if self.iinit == 9:
            # sound.F:416/432: rtime.le.convtime, rtime = sngl(mtime+dt).
            return _f32(float(mtime_s) + dt32) <= _f32(self["convtime"])
        return False

    def active_held(self, mtime_s: float) -> bool:
        """``solve1.F:1245``: ``iinit.eq.10 .and. mtime.lt.t2_uforce``."""

        return self.iinit == 10 and float(mtime_s) < _f32(self["t2_uforce"])

    def gamm(self, mtime_s: float) -> float:
        """The linear ramp of ``misclibs.F:557-561`` / ``solve1.F:1247-1251``.

        CM1 evaluates it in double (``mtime`` is double) and stores it in a
        default REAL.
        """

        t1, t2 = (
            (self["t1_wnudge"], self["t2_wnudge"]) if self.iinit == 12
            else (self["t1_uforce"], self["t2_uforce"])
        )
        t1, t2 = _f32(t1), _f32(t2)
        gamm = 1.0
        if float(mtime_s) >= t1:
            # (mtime - t1) is double (mtime is double); (t2 - t1) is a
            # default-REAL difference, rounded before the division promotes it.
            gamm = 1.0 + (0.0 - 1.0) * (float(mtime_s) - t1) / _f32(t2 - t1)
        return _f32(gamm)

    def aconv(self) -> float:
        """``init3d.F:1318-1331``, the 3-D branch, in default REAL."""

        import numpy as np

        f = np.float32
        lamx, lamy = f(self["lamx"]), f(self["lamy"])
        return float((f(-0.5) * f(self["dmax"])) / ((f(1.0) / (lamx * lamx)) + (f(1.0) / (lamy * lamy))))

    def record(self) -> dict[str, object]:
        return {
            "schema": INITIATION_FORCING_SCHEMA,
            "iinit": self.iinit,
            "words": {k: v for k, v in self.words},
            "declared": sorted(self.declared),
            "iorigin": self.iorigin,
            "authority": AUTHORITY[self.iinit],
            "restart": (
                "off after a restart: param.F:891-903 sets iinit=0 when irst=1 and "
                "init3d.F:175-176 leaves convinit=wnudge=0"
            ),
        }


def resolve_initiation_forcing(
    initialization: Any, *, nx: int, ny: int, dx_m: float, dy_m: float
) -> InitiationForcing | None:
    """The forcing a document's ``[initialization]`` selects, or None."""

    if initialization is None or int(initialization.iinit) not in FORCING_IINIT:
        return None
    iinit = int(initialization.iinit)
    iorigin = int(getattr(initialization, "iorigin", 1))
    # param.F:6649/6659 and :7406-7408: minx = xfdp(1), maxx = xfdp(nx+1),
    # centerx is minx plus half of maxx - minx, all in default REAL.
    mult = 0.5 if iorigin == 2 else 0.0
    minx = _f32(-mult * dx_m * nx)
    maxx = _f32(dx_m * nx - mult * dx_m * nx)
    miny = _f32(-mult * dy_m * ny)
    maxy = _f32(dy_m * ny - mult * dy_m * ny)
    centerx = _f32(minx + _f32(0.5 * _f32(maxx - minx)))
    centery = _f32(miny + _f32(0.5 * _f32(maxy - miny)))
    words: dict[str, float] = {}
    declared = set()
    for name, default in FORCING_DEFAULTS[iinit].items():
        value = getattr(initialization, name)
        if name in getattr(initialization, "declared", frozenset()):
            declared.add(name)
        if value is None:
            value = centery if name.startswith("yc") else centerx
        words[name] = float(value)
    return InitiationForcing(iinit=iinit, words=tuple(words.items()), declared=frozenset(declared), iorigin=iorigin)


# ---------------------------------------------------------------- coordinates


def horizontal_coordinates(
    *, nx: int, ny: int, dx_m: float, dy_m: float, iorigin: int
) -> dict[str, Any]:
    """CM1's ``xh``/``xf``/``yh``/``yf`` for a uniform grid, as float32 arrays.

    ``param.F:6649,6659``: ``xfdp(i) = dble(dx)*(i-1) [- 0.5d0*dble(dx)*nx]``;
    ``param.F:6683-6688``: ``xf = xfdp``, ``xh = 0.5d0*(xfdp(i)+xfdp(i+1))``,
    both stored in default REAL.  Index 0 here is CM1's i = 1.
    """

    import numpy as np

    def axis(n: int, d: float) -> tuple[Any, Any]:
        dd = float(np.float32(d))
        faces = dd * np.arange(n + 1, dtype=np.float64)
        if iorigin == 2:
            faces = faces - 0.5 * dd * n
        return (0.5 * (faces[:-1] + faces[1:])).astype(np.float32), faces.astype(np.float32)

    xh, xf = axis(nx, dx_m)
    yh, yf = axis(ny, dy_m)
    return {"xh": xh, "xf": xf, "yh": yh, "yf": yf}


# ------------------------------------------------------ host reference forms


def wnudge_geometry(forcing: InitiationForcing, *, xh: Any, yh: Any, zf: Any) -> dict[str, Any]:
    """``misclibs.F:571-576``'s time-independent words: ``beta`` and ``wmag``.

    ``beta`` and ``wmag = wmax*(cos(0.5*pi*beta)**2)`` depend only on the
    grid, so they are built once, over the smallest box of CM1 faces
    ``k = 2..nk`` (0-based faces 1..nk-1 of ``zf``) and columns that holds
    every ``beta < 1`` point; inside the box a point with ``beta >= 1`` carries
    NaN (CM1 writes 0.0 there, ``misclibs.F:580``).  ``cos`` is glibc's
    ``cosf``; gfortran expands ``**2`` to one multiplication.
    """

    import numpy as np

    f = np.float32
    rxr = f(1.0) / f(max(forcing["xr_wnudge"], 1.0e-12))
    ryr = f(1.0) / f(max(forcing["yr_wnudge"], 1.0e-12))
    rzr = f(1.0) / f(max(forcing["zr_wnudge"], 1.0e-12))
    a = (np.asarray(xh, dtype=f) - f(forcing["xc_wnudge"])) * rxr
    b = (np.asarray(yh, dtype=f) - f(forcing["yc_wnudge"])) * ryr
    zf = np.asarray(zf, dtype=f)
    c = (zf - f(forcing["zc_wnudge"])) * rzr
    aa, bb, cc = (a * a).astype(f), (b * b).astype(f), (c * c).astype(f)
    nk = zf.size - 1

    def span(sq: Any, lo: int, hi: int) -> tuple[int, int]:
        idx = np.nonzero(sq[lo:hi] < f(1.0))[0]
        return (0, 0) if idx.size == 0 else (lo + int(idx[0]), lo + int(idx[-1]) + 1)

    i0, i1 = span(aa, 0, aa.size)
    j0, j1 = span(bb, 0, bb.size)
    k0, k1 = span(cc, 1, nk)
    if i1 <= i0 or j1 <= j0 or k1 <= k0:
        return {"box": (0, 0, 0, 0, 0, 0), "wmag": np.zeros((0, 0, 0), dtype=f), "points": 0}
    beta = np.sqrt(((aa[None, None, i0:i1] + bb[None, j0:j1, None]).astype(f) + cc[k0:k1, None, None]).astype(f))
    beta = beta.astype(f)
    inside = beta < f(1.0)
    wmag = np.full(beta.shape, np.nan, dtype=f)
    halfpi = f(f(0.5) * f(_PI_F32))
    q = libm_f32("cosf", (halfpi * beta[inside]).astype(f))
    wmag[inside] = (f(forcing["wmax_wnudge"]) * (q * q).astype(f)).astype(f)
    return {"box": (k0, k1, j0, j1, i0, i1), "wmag": wmag, "points": int(np.count_nonzero(inside))}


def wnudge_tendency_host(
    forcing: InitiationForcing, *, mtime_s: float, xh: Any, yh: Any, zf: Any, w: Any, geometry: Any = None
) -> Any:
    """``misclibs.F:539-585`` on the host: the nudging tendency for w faces.

    ``w`` is ``(nk+1, ny, nx)`` over faces k = 1..nk+1; faces 1 and nk+1 get
    zero because CM1 loops ``k = 2, nk``, and so does every point outside the
    ellipsoid (``misclibs.F:580``).
    """

    import numpy as np

    f = np.float32
    geo = geometry if geometry is not None else wnudge_geometry(forcing, xh=xh, yh=yh, zf=zf)
    tem = f(forcing["alpha_wnudge"]) * f(forcing.gamm(mtime_s))
    out = np.zeros(np.shape(w), dtype=f)
    if geo["points"]:
        k0, k1, j0, j1, i0, i1 = geo["box"]
        wmag = geo["wmag"]
        wb = np.asarray(w, dtype=f)[k0:k1, j0:j1, i0:i1]
        inside = ~np.isnan(wmag)
        d = np.zeros(wmag.shape, dtype=f)
        d[inside] = (tem * np.maximum((wmag[inside] - wb[inside]).astype(f), f(0.0))).astype(f)
        out[k0:k1, j0:j1, i0:i1] = d
    return out


def uforce_table_host(forcing: InitiationForcing, *, xf: Any, zf: Any) -> Any:
    """``solve1.F:1258-1260``'s ``aiu`` on the (k, i) u-point lattice.

    ``zf`` is the face height ``zf(i,j,k)`` for k = 1..nk (CM1 reads the face
    below each u level, above ground: ``zs`` is zero on a flat grid), ``xf``
    the u-point x for i = 1..ni+1.  Outside the box the entry is zero.
    """

    import numpy as np

    f = np.float32
    xc, xr, zr = f(forcing["xc_uforce"]), f(forcing["xr_uforce"]), f(forcing["zr_uforce"])
    alpha = f(forcing["alpha_uforce"])
    x = xf.astype(f)[None, :]
    z = zf.astype(f)[:, None]
    inside = (np.abs(x - xc) < xr) & (np.abs(z) < zr)
    # cos(0.5*pi*(xf-xc)/xr) is ((0.5*pi)*(xf-xc))/xr; glibc cosf and coshf;
    # gfortran's x**(-2) is libgcc's __powisf2: 1/(x*x).
    cosx = libm_f32("cosf", ((f(f(0.5) * f(_PI_F32)) * (x - xc)).astype(f) / xr).astype(f))
    ch = libm_f32("coshf", ((f(2.5) * z).astype(f) / zr).astype(f))
    aiu = (alpha * cosx).astype(f) * (f(1.0) / (ch * ch).astype(f)).astype(f)
    return np.where(inside, aiu, f(0.0)).astype(f)


def convergence_tables(forcing: InitiationForcing, *, along: Any, across: Any, zh: Any, axis: str) -> dict[str, Any]:
    """``misclibs.F:474-485`` (axis "u") or ``:515-526`` (axis "v"), per index.

    Every factor of ``umo`` -- ``term1``, the product of the two exponentials
    of ``term2`` and ``term3``, and ``term4``, multiplied in that grouping -- depends on
    one index only: ``term1`` and the along-axis exponential on the face
    coordinate, the across-axis exponential on the other scalar coordinate,
    ``term4`` on the level (``0.5*(zh(i-1)+zh(i))`` is ``zh`` on a flat column).
    ``exp`` is glibc's ``expf``.
    """

    import numpy as np

    f = np.float32
    aconv = f(forcing.aconv())
    zdeep = f(forcing["zdeep"])
    if axis == "u":
        lam_a, lam_c, c_a, c_c = f(forcing["lamx"]), f(forcing["lamy"]), f(forcing["xcent"]), f(forcing["ycent"])
    else:
        lam_a, lam_c, c_a, c_c = f(forcing["lamy"]), f(forcing["lamx"]), f(forcing["ycent"]), f(forcing["xcent"])
    d_a = (np.asarray(along, dtype=f) - c_a).astype(f)
    term1 = (-(((f(2.0) * aconv) * d_a).astype(f) / (lam_a * lam_a))).astype(f)
    r_a = (d_a / lam_a).astype(f)
    r_c = ((np.asarray(across, dtype=f) - c_c) / lam_c).astype(f)
    return {
        "term1": term1,
        "e_along": libm_f32("expf", (-(r_a * r_a)).astype(f)),
        "e_across": libm_f32("expf", (-(r_c * r_c)).astype(f)),
        "term4": ((zdeep - np.asarray(zh, dtype=f)) / zdeep).astype(f),
    }


def _convergence_host(tables: dict[str, Any], base0: Any, vel: Any, *, along_last: bool) -> Any:
    import numpy as np

    f = np.float32
    t1, ea, ec, t4 = tables["term1"], tables["e_along"], tables["e_across"], tables["term4"]
    if along_last:  # u: (k, j=across, i=along)
        prod = (ec[None, :, None] * ea[None, None, :]).astype(f)
        mo = ((t1[None, None, :] * prod).astype(f) * t4[:, None, None]).astype(f)
    else:           # v: (k, j=along, i=across)
        prod = (ec[None, None, :] * ea[None, :, None]).astype(f)
        mo = ((t1[None, :, None] * prod).astype(f) * t4[:, None, None]).astype(f)
    replace = (t4[:, None, None] > 0) & (np.abs(mo) > f(0.01))
    new = (np.asarray(base0, dtype=f)[:, None, None] + mo).astype(f)
    return np.where(replace, new, np.asarray(vel, dtype=f)).astype(f)


def convergence_u_host(forcing: InitiationForcing, *, xf: Any, yh: Any, zh: Any, u0: Any, u: Any) -> Any:
    """``misclibs.F:456-490`` on the host: the new ``u3d`` (nk, ny, nx+1)."""

    tables = convergence_tables(forcing, along=xf, across=yh, zh=zh, axis="u")
    return _convergence_host(tables, u0, u, along_last=True)


def convergence_v_host(forcing: InitiationForcing, *, xh: Any, yf: Any, zh: Any, v0: Any, v: Any) -> Any:
    """``misclibs.F:497-531`` on the host: the new ``v3d`` (nk, ny+1, nx)."""

    tables = convergence_tables(forcing, along=yf, across=xh, zh=zh, axis="v")
    return _convergence_host(tables, v0, v, along_last=False)


# ------------------------------------------------------- where the push reaches

#: The run-event and receipt code of a forcing that is centred outside the domain or reaches no point of it.
FORCING_REACH_WARNING_CODE = "GM1-FORCING-CENTRE-OUTSIDE"

_ARM_WORDS: Mapping[int, tuple[str, str | None]] = MappingProxyType(
    {12: ("xc_wnudge", "yc_wnudge"), 9: ("xcent", "ycent"), 10: ("xc_uforce", None)}
)
_ARM_NAMES: Mapping[int, str] = MappingProxyType(
    {12: "updraft nudging (iinit = 12)", 9: "forced convergence (iinit = 9)", 10: "momentum forcing (iinit = 10)"}
)


def _km(value_m: float) -> str:
    return f"{value_m / 1000.0:.1f} km"


def forcing_reach(
    forcing: InitiationForcing, *, nx: int, ny: int, dx_m: float, dy_m: float, zh: Any, zf: Any
) -> dict[str, Any]:
    """Whether the forcing's centre lies in the domain, and how many grid points it touches.

    CM1 applies each arm wherever its own test holds and nowhere else, whatever
    the centre: ``misclibs.F:575`` (``beta < 1``, w faces ``k = 2..nk``),
    ``misclibs.F:486-487``/``:527-528`` (``|umo| > 0.01`` below ``zdeep``, u and
    v points) and ``solve1.F:1256`` (the ``xr``/``zr`` box, u faces).  A centre
    outside ``[minx, maxx] x [miny, maxy]`` (``param.F:7406-7408``, the
    ``iorigin`` frame) is not an error in CM1 and is not one here: the run is
    CM1's run.  But when the push touches no point, nothing starts a storm,
    and a user who typed a centre in the wrong frame (metres from the corner
    for ``iorigin = 2``, or km for m) is owed that sentence before the GPU
    hours, not after.  This is host arithmetic on CM1's own words (the same
    geometry the device stage uploads), so it opens no device.
    """

    import numpy as np

    f = np.float32
    coords = horizontal_coordinates(nx=nx, ny=ny, dx_m=dx_m, dy_m=dy_m, iorigin=forcing.iorigin)
    xf, yf = coords["xf"], coords["yf"]
    minx, maxx, miny, maxy = float(xf[0]), float(xf[-1]), float(yf[0]), float(yf[-1])
    xname, yname = _ARM_WORDS[forcing.iinit]
    xc = float(forcing[xname])
    yc = None if yname is None else float(forcing[yname])
    inside = minx <= xc <= maxx and (yc is None or miny <= yc <= maxy)
    zh = np.asarray(zh, dtype=f)
    zf = np.asarray(zf, dtype=f)
    if forcing.iinit == 12:
        points = int(wnudge_geometry(forcing, xh=coords["xh"], yh=coords["yh"], zf=zf)["points"])
        what = "w points"
    elif forcing.iinit == 10:
        aiu = uforce_table_host(forcing, xf=xf, zf=zf[: zh.size])
        points = int(np.count_nonzero(aiu)) * int(ny)
        what = "u points"
    else:
        points = 0
        for axis, along, across, along_last in (("u", xf, coords["yh"], True), ("v", yf, coords["xh"], False)):
            tables = convergence_tables(forcing, along=along, across=across, zh=zh, axis=axis)
            t1, ea, ec, t4 = tables["term1"], tables["e_along"], tables["e_across"], tables["term4"]
            for k in np.nonzero(t4 > 0)[0]:
                if along_last:
                    mo = ((t1[None, :] * (ec[:, None] * ea[None, :]).astype(f)).astype(f) * t4[k]).astype(f)
                else:
                    mo = ((t1[:, None] * (ec[None, :] * ea[:, None]).astype(f)).astype(f) * t4[k]).astype(f)
                points += int(np.count_nonzero(np.abs(mo) > f(0.01)))
        what = "u and v points"
    centre = f"x = {_km(xc)}" + ("" if yc is None else f", y = {_km(yc)}")
    extent = f"x {_km(minx)} to {_km(maxx)}" + ("" if yc is None else f", y {_km(miny)} to {_km(maxy)}")
    words = f"initialization.{xname}" + ("" if yname is None else f" and initialization.{yname}")
    frame = "from the domain centre" if forcing.iorigin == 2 else "from the south-west corner"
    warning = None
    if not inside and points == 0:
        warning = (
            f"The push is centred outside the domain, so no storm will be started: CM1's {_ARM_NAMES[forcing.iinit]} "
            f"is centred at {centre} and the domain spans {extent} ({frame}, iorigin = {forcing.iorigin}), and it "
            f"reaches no grid point. CM1 runs this the same way. To start a storm, move {words} inside the domain."
        )
    elif not inside:
        warning = (
            f"The push is centred outside the domain, so only its edge reaches in ({points} {what}) and a storm "
            f"may not start: CM1's {_ARM_NAMES[forcing.iinit]} is centred at {centre} and the domain spans {extent} "
            f"({frame}, iorigin = {forcing.iorigin}). CM1 runs this the same way. To start a storm, move {words} "
            "inside the domain."
        )
    elif points == 0:
        warning = (
            f"The push reaches no grid point, so no storm will be started: CM1's {_ARM_NAMES[forcing.iinit]} is "
            f"centred at {centre}, inside the domain, but its size and height put no {what.split(' ')[0]} point "
            "inside it. CM1 runs this the same way."
        )
    return {
        "authority": (
            "CM1 r21.1 misclibs.F:575 (iinit = 12), misclibs.F:486-487 and 527-528 (iinit = 9), "
            "solve1.F:1256 (iinit = 10); param.F:7406-7408 domain extent"
        ),
        "iinit": forcing.iinit,
        "centre_m": [xc, yc],
        "domain_m": {"x": [minx, maxx], "y": [miny, maxy]},
        "centre_inside_domain": inside,
        "forced_points": points,
        "points_are": what,
        "code": None if warning is None else FORCING_REACH_WARNING_CODE,
        "warning": warning,
    }


# ------------------------------------------------------------ device kernels

_KERNELS = r"""
extern "C" __global__ void gm1_wnudge(
    float* __restrict__ wten, const float* __restrict__ wtens,
    const float* __restrict__ w, const float* __restrict__ wmag,
    unsigned ni, unsigned nj, unsigned nk, unsigned ng,
    unsigned sw_k, unsigned sw_j,
    unsigned k0, unsigned j0, unsigned i0, unsigned bk, unsigned bj, unsigned bi,
    float tem)
{
    // misclibs.F:571-583 over k = 2..nk, j = 1..nj, i = 1..ni, then
    // sound.F:584-588 wten = wtens + dum4.  wmag is misclibs.F:576's
    // wmax*cos(0.5*pi*beta)**2, built once on the host with glibc cosf over
    // the ellipsoid's box (NaN where beta >= 1), so this kernel does only
    // correctly rounded arithmetic.  Padded w carriers (halo 3, 1): 0-based
    // physical (k, j, i) -> (k+1, j+ng, i+ng).  wtens is compact (nk+1, nj, ni).
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned j = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned k = blockIdx.z + 1u;           // 0-based face 1 .. nk-1 == CM1 k = 2..nk
    if (i >= ni || j >= nj || k >= nk) return;
    size_t p = (size_t)(k + 1u) * sw_k + (size_t)(j + ng) * sw_j + (size_t)(i + ng);
    float d = 0.0f;
    if (k >= k0 && k < k0 + bk && j >= j0 && j < j0 + bj && i >= i0 && i < i0 + bi) {
        float wm = wmag[((size_t)(k - k0) * bj + (j - j0)) * bi + (i - i0)];
        if (!isnan(wm)) d = __fmul_rn(tem, fmaxf(__fsub_rn(wm, w[p]), 0.0f));
    }
    size_t q = ((size_t)k * nj + j) * ni + i;
    wten[p] = __fadd_rn(wtens[q], d);
}

extern "C" __global__ void gm1_wnudge_save(
    float* __restrict__ wtens, const float* __restrict__ wten,
    unsigned ni, unsigned nj, unsigned nk, unsigned ng,
    unsigned sw_k, unsigned sw_j)
{
    // sound.F:571-578: if(n.eq.1) wtens(i,j,k) = wten(i,j,k), k = 2..nk.
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned j = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned k = blockIdx.z + 1u;
    if (i >= ni || j >= nj || k >= nk) return;
    size_t p = (size_t)(k + 1u) * sw_k + (size_t)(j + ng) * sw_j + (size_t)(i + ng);
    wtens[((size_t)k * nj + j) * ni + i] = wten[p];
}

extern "C" __global__ void gm1_convinit(
    float* __restrict__ vel, const float* __restrict__ base0,
    const float* __restrict__ term1, const float* __restrict__ e_along,
    const float* __restrict__ e_across, const float* __restrict__ term4,
    unsigned n_along, unsigned n_across, unsigned nk, unsigned ng,
    unsigned s_k, unsigned s_across, unsigned s_along)
{
    // misclibs.F:474-487 (u: along = xf over i = 1..ni+1, across = yh over
    // j = 1..nj) and :515-528 (v: along = yf over j = 1..nj+1, across = xh):
    // umo = term1*(exp(term2)*exp(term3))*term4, each factor a per-index
    // table built on the host (glibc expf; convergence_tables).
    unsigned a = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned c = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned k = blockIdx.z;
    if (a >= n_along || c >= n_across || k >= nk) return;
    float t4 = term4[k];
    if (!(t4 > 0.0f)) return;
    float umo = __fmul_rn(__fmul_rn(term1[a], __fmul_rn(e_along[a], e_across[c])), t4);
    if (fabsf(umo) > 0.01f) {
        size_t p = (size_t)(k + 1u) * s_k + (size_t)(c + ng) * s_across + (size_t)(a + ng) * s_along;
        vel[p] = __fadd_rn(base0[k], umo);
    }
}

extern "C" __global__ void gm1_uforce_held(
    float* __restrict__ uten, const float* __restrict__ aiu,
    unsigned nip1, unsigned nj, unsigned nk, float gamm)
{
    // solve1.F:1255-1263: uten1 = uten1 + gamm*aiu on i = 1..ni+1, j, k.
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned j = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned k = blockIdx.z;
    if (i >= nip1 || j >= nj || k >= nk) return;
    float t = aiu[(size_t)k * nip1 + i];
    if (t != 0.0f) {
        size_t p = ((size_t)k * nj + j) * nip1 + i;
        uten[p] = __fadd_rn(uten[p], __fmul_rn(gamm, t));
    }
}
"""

_MODULE_CACHE: dict[int, Any] = {}


#: The kernels' names, extern "C" so NVRTC keeps them as written.
_KERNEL_NAMES = ("gm1_wnudge", "gm1_wnudge_save", "gm1_convinit", "gm1_uforce_held")


def _module(cupy: Any) -> Any:
    """The forcing kernels through the package's gradual-underflow compile path.

    CuPy's own ``RawModule`` appends ``-ftz=true``, which would flush CM1's
    binary32 sub-normals (``CM1-ARITHMETIC-CONTRACT.md``); ``--fmad=false``
    keeps every product and sum separately rounded, as gfortran's are.
    """

    key = id(cupy)
    if key not in _MODULE_CACHE:
        from .nvrtc_module import compile_nvrtc_module

        _MODULE_CACHE[key] = compile_nvrtc_module(
            cupy,
            code=_KERNELS,
            filename="gm1_initiation_forcing.cu",
            options=("--std=c++14", "--fmad=false", "--ftz=false", "--prec-div=true", "--prec-sqrt=true"),
            name_expressions=_KERNEL_NAMES,
        )
    return _MODULE_CACHE[key]


class _Substep:
    """The acoustic-substep hook: nest's ``begin/after_momentum/after_pressure``.

    It composes with a live nest's own hook, which it calls first at each point.
    """

    def __init__(self, stage: "InitiationForcingStage", inner: Any) -> None:
        self.stage, self.inner = stage, inner

    def begin(self, **kw: Any) -> None:
        if self.inner is not None:
            self.inner.begin(**kw)

    def after_momentum(self, *, nrk: int, n: int, dts: float) -> None:
        if self.inner is not None:
            self.inner.after_momentum(nrk=nrk, n=n, dts=dts)
        self.stage.apply_substep(n=n)

    def after_pressure(self, **kw: Any) -> None:
        if self.inner is not None:
            self.inner.after_pressure(**kw)

    def record(self) -> Any:
        return None if self.inner is None else self.inner.record()


@dataclass
class InitiationForcingStage:
    """Binds one :class:`InitiationForcing` to a ``CartesianDynamics``."""

    forcing: InitiationForcing
    dynamics: Any
    disabled_by_restart: bool = False
    counts: dict[str, int] = field(default_factory=lambda: {"held_steps": 0, "substeps": 0, "stage_saves": 0})
    last_active_time_s: float | None = None
    _wtens: Any = None
    _mtime_s: float = 0.0

    def __post_init__(self) -> None:
        d = self.dynamics
        g = d.geometry
        if getattr(g, "stretch_x", 0) or getattr(g, "stretch_y", 0):
            raise ValueError("the initiation forcing binds CM1's uniform horizontal grid only")
        cp = d.cupy
        coords = horizontal_coordinates(nx=g.nx, ny=g.ny, dx_m=g.dx_m, dy_m=g.dy_m, iorigin=self.forcing.iorigin)
        import numpy as np

        zh = np.asarray(g.mass_heights_m, dtype=np.float32)
        zf = np.asarray(g.face_heights_m, dtype=np.float32)
        self._host = {**coords, "zh": zh, "zf": zf}
        self._wgeo: dict[str, Any] | None = None
        tables: dict[str, Any] = {}
        if self.forcing.iinit == 12:
            self._wgeo = wnudge_geometry(self.forcing, xh=coords["xh"], yh=coords["yh"], zf=zf)
            tables["wmag"] = self._wgeo["wmag"] if self._wgeo["points"] else np.zeros(1, dtype=np.float32)
        elif self.forcing.iinit == 9:
            for axis, along, across in (("u", coords["xf"], coords["yh"]), ("v", coords["yf"], coords["xh"])):
                for name, value in convergence_tables(self.forcing, along=along, across=across, zh=zh, axis=axis).items():
                    tables[f"{axis}.{name}"] = value
        else:
            # zf(i,j,k) - zs(i,j) for k = 1..nk on a flat column is zf(k).
            tables["aiu"] = uforce_table_host(self.forcing, xf=coords["xf"], zf=zf[: g.nz])
        with d.stream:
            self._dev = {name: cp.asarray(np.ascontiguousarray(value)) for name, value in tables.items()}
        self._kernels = _module(cp)

    # -- clock --------------------------------------------------------------

    def begin_step(self, mtime_s: float) -> None:
        self._mtime_s = float(mtime_s)

    @property
    def dt_s(self) -> float:
        return float(self.dynamics.config.dt_s)

    # -- solve1: iinit = 10 ----------------------------------------------------

    def held(self) -> None:
        if self.disabled_by_restart or not self.forcing.active_held(self._mtime_s):
            return
        d, g, cp = self.dynamics, self.dynamics.geometry, self.dynamics.cupy
        uten = d.held["uten"]
        if tuple(uten.shape) != (g.nz, g.ny, g.nx + 1) or not uten.flags.c_contiguous:
            raise RuntimeError("held u tendency is not the compact (nz, ny, nx+1) carrier")
        gamm = self.forcing.gamm(self._mtime_s)
        self._kernels.get_function("gm1_uforce_held")(
            ((g.nx + 1 + 31) // 32, (g.ny + 7) // 8, g.nz), (32, 8, 1),
            (uten, self._dev["aiu"], cp.uint32(g.nx + 1), cp.uint32(g.ny), cp.uint32(g.nz), cp.float32(gamm)),
            stream=d.stream,
        )
        self.counts["held_steps"] += 1
        self.last_active_time_s = self._mtime_s

    # -- sound.F: iinit = 9 and 12 ----------------------------------------------

    def substep_hook(self, inner: Any) -> Any:
        if self.disabled_by_restart or not self.forcing.active_substeps(self._mtime_s, self.dt_s):
            if self._wtens is not None:
                self.dynamics.registry.release(["initiation.wtens"])
                self._wtens = None
            return inner
        if self.forcing.iinit == 12 and self._wtens is None:
            g = self.dynamics.geometry
            self._wtens = self.dynamics.registry.allocate("initiation.wtens", (g.nz + 1, g.ny, g.nx), zero=True)
        self.last_active_time_s = self._mtime_s
        return _Substep(self, inner)

    def apply_substep(self, *, n: int) -> None:
        d, g, cp = self.dynamics, self.dynamics.geometry, self.dynamics.cupy
        a = d.arrays
        ng = 3
        u32, f32 = cp.uint32, cp.float32
        if self.forcing.iinit == 12:
            wten, w = a["wten"], a["w3d"]
            if tuple(wten.shape) != (g.nz + 3, g.ny + 2 * ng, g.nx + 2 * ng):
                raise RuntimeError("padded wten is not the (nk+3, nj+6, ni+6) carrier")
            sw_k, sw_j = (g.ny + 2 * ng) * (g.nx + 2 * ng), g.nx + 2 * ng
            grid = ((g.nx + 31) // 32, (g.ny + 7) // 8, max(g.nz - 1, 1))
            dims = (u32(g.nx), u32(g.ny), u32(g.nz), u32(ng), u32(sw_k), u32(sw_j))
            if n == 1:
                self._kernels.get_function("gm1_wnudge_save")(grid, (32, 8, 1), (self._wtens, wten, *dims), stream=d.stream)
                self.counts["stage_saves"] += 1
            fz = self.forcing
            # misclibs.F:565: tem = alpha_wnudge*gamm, in default REAL.
            tem = _f32(_f32(fz["alpha_wnudge"]) * fz.gamm(self._mtime_s))
            k0, k1, j0, j1, i0, i1 = self._wgeo["box"]
            self._kernels.get_function("gm1_wnudge")(
                grid, (32, 8, 1),
                (wten, self._wtens, w, self._dev["wmag"], *dims,
                 u32(k0), u32(j0), u32(i0), u32(k1 - k0), u32(j1 - j0), u32(i1 - i0), f32(tem)),
                stream=d.stream,
            )
        else:
            u, v = a["u3d"], a["v3d"]
            if tuple(u.shape) != (g.nz + 2, g.ny + 2 * ng, g.nx + 1 + 2 * ng):
                raise RuntimeError("padded u3d is not the (nk+2, nj+6, ni+7) carrier")
            if tuple(v.shape) != (g.nz + 2, g.ny + 1 + 2 * ng, g.nx + 2 * ng):
                raise RuntimeError("padded v3d is not the (nk+2, nj+7, ni+6) carrier")
            kern = self._kernels.get_function("gm1_convinit")
            u0, v0 = d.base_profiles["u0"], d.base_profiles["v0"]
            t = self._dev
            # u: along x (xf, i = 1..ni+1), across y (yh).
            kern(
                ((g.nx + 1 + 31) // 32, (g.ny + 7) // 8, g.nz), (32, 8, 1),
                (u, u0, t["u.term1"], t["u.e_along"], t["u.e_across"], t["u.term4"],
                 u32(g.nx + 1), u32(g.ny), u32(g.nz), u32(ng),
                 u32((g.ny + 2 * ng) * (g.nx + 1 + 2 * ng)), u32(g.nx + 1 + 2 * ng), u32(1)),
                stream=d.stream,
            )
            # v: along y (yf, j = 1..nj+1), across x (xh).
            kern(
                ((g.ny + 1 + 31) // 32, (g.nx + 7) // 8, g.nz), (32, 8, 1),
                (v, v0, t["v.term1"], t["v.e_along"], t["v.e_across"], t["v.term4"],
                 u32(g.ny + 1), u32(g.nx), u32(g.nz), u32(ng),
                 u32((g.ny + 1 + 2 * ng) * (g.nx + 2 * ng)), u32(1), u32(g.nx + 2 * ng)),
                stream=d.stream,
            )
        self.counts["substeps"] += 1
        # The step's record is published from finalize_held_tendencies, before
        # its acoustic steps; refreshed here, a run whose last step still
        # forces reports that step too.
        receipts = getattr(d, "last_receipts", None)
        if isinstance(receipts, dict) and "initiation_forcing" in receipts:
            receipts["initiation_forcing"] = self.record()

    def record(self) -> dict[str, object]:
        return {
            **self.forcing.record(),
            "applied": dict(self.counts),
            "last_active_step_start_s": self.last_active_time_s,
            "disabled_by_restart": self.disabled_by_restart,
            # How many w faces lie inside the nudging ellipsoid (misclibs.F:575,
            # beta < 1): zero means CM1's own run would nudge nothing either.
            **({"nudged_w_faces": int(self._wgeo["points"])} if self._wgeo is not None else {}),
        }
