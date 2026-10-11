"""CM1's ``cm1out_stats`` statistics, computed from history frames.

CM1 r21.1 writes one record of domain statistics every ``statfrq`` seconds
(``src/statpack.F``, names from ``setup_stat_vars``, :666-1458) through
``writestat_nc`` (``src/writeout_nc.F:1631-1781``).  GM1 has no in-run
statistics yet, so this module computes the same statistics from each
published history frame, under CM1's names, descriptions and units; the
file's times are the frame times, not ``statfrq``.

Every statistic is a transcription of its CM1 routine, in CM1's own
precision: REAL (binary32) where CM1 declares REAL, DOUBLE PRECISION where
CM1 accumulates in double.  :func:`compute` takes a :class:`StatInputs`,
which :func:`inputs_from_frame` fills from a GM1 frame and
:func:`inputs_from_cm1_history` from one time of a CM1 history file, so the
same code can be checked against CM1's own statistics of CM1's own state.

Statistics GM1 cannot form from a frame (the eddy viscosities, the
microphysics budgets, the artificial and boundary sources, and the fields of
schemes GM1 does not run) are listed with a reason in ``not_produced``.

Where the frame carries no base state (``output_basestate = 0``, CM1's own
default), the total nondimensional pressure is recovered from the frame's
pressure as ``(prs/p00)**(rd/cp)``, the inverse of CM1's
``prs = p00*(pi0+ppi)**(cp/rd)``, and the base state as that minus ``ppi``.
The statistics that use it (``ppmax``/``ppmin``, relative humidity,
theta-e, the energies) then agree with CM1's to the rounding of one REAL
operation, not word for word; the file says which statistics that is.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .settings import Cm1Settings, Geometry, Species

STATS_FILENAME = "cm1out_stats.nc"


class _C:
    """CM1 r21.1 constants.F, as binary32 (set_constants, testcase 0)."""

    def __init__(self) -> None:
        import numpy as np

        f = np.float32
        self.g = f(9.81)
        self.rd = f(287.04)
        self.cp = f(1005.7)
        self.cpv = f(1870.0)
        self.xlv = f(2501000.0)
        self.xls = f(2834000.0)
        self.rv = f(461.5)
        self.cv = f(self.cp - self.rd)
        self.cvv = f(self.cpv - self.rv)
        self.rovcp = f(self.rd / self.cp)
        self.cpdrd = f(self.cp / self.rd)
        self.eps = f(self.rd / self.rv)
        self.reps = f(self.rv / self.rd)
        self.cpl = f(4190.0)
        self.cpi = f(2106.0)
        self.to = f(273.15)
        self.lv1 = f(self.xlv + f(f(self.cpl - self.cpv) * self.to))
        self.ls1 = f(self.xls + f(f(self.cpi - self.cpv) * self.to))
        self.p00 = f(1.0e5)
        self.clwsat = f(1.0e-6)
        self.l0 = f(2.555e6)


@dataclass
class StatInputs:
    """One model state, CM1-shaped, in CM1's own arrays."""

    time: float
    u: Any          # ua (nz, ny, nx+1)
    v: Any          # va (nz, ny+1, nx)
    w: Any          # wa (nz+1, ny, nx)
    tha: Any        # (nz, ny, nx)
    ppi: Any        # (nz, ny, nx)
    prs: Any        # (nz, ny, nx)
    rho: Any = None     # (nz, ny, nx)
    q: Sequence[Any] = ()   # qa(:,:,:,n), n = 1..numq
    tke: Any = None     # (nz+1, ny, nx)
    prate: Any = None   # (ny, nx)
    #: The base state, as CM1 holds it (a column per level) when known.
    th0: Any = None
    pi0: Any = None
    prs0: Any = None
    qv0: Any = None
    #: Total theta (th0+tha, REAL), used when th0 is not known.
    th: Any = None
    umove: float = 0.0
    vmove: float = 0.0
    #: Names of the statistics whose inputs were recovered rather than read.
    derived: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class StatSpec:
    name: str
    long_name: str
    units: str


# setup_stat_vars (statpack.F:666-1458): name, description, unit, in order.
# desc_stat is CHARACTER(len=40), so a file carries at most 40 characters.
def _spec(name: str, desc: str, unit: str) -> StatSpec:
    return StatSpec(name, desc[:40].rstrip(), unit)


def _names(settings: Cm1Settings, geometry: Geometry, species: Species) -> list[tuple[StatSpec, str]]:
    """Every statistic CM1 defines for these settings, with the group that forms it."""

    s = settings
    on = lambda key: s.int(key, 0) == 1  # noqa: E731
    imoist = s.int("imoist", 0) == 1
    rows: list[tuple[StatSpec, str]] = [(_spec("mtime", "model time (seconds since beginning of simulation)", "s"), "mtime")]
    if s.int("adapt_dt", 0) == 1:
        rows.append((_spec("dt", "average timestep dt", "s"), "dt"))
    if on("stat_w"):
        rows += [(_spec("wmax", "max vertical velocity", "m/s"), "w"),
                 (_spec("wmin", "min vertical velocity", "m/s"), "w"),
                 (_spec("zwmax", "level of max vertical velocity", "m AGL"), "w"),
                 (_spec("zwmin", "level of min vertical velocity", "m AGL"), "w")]
    if on("stat_wlevs") and geometry.maxz >= 10000.0:
        for level, tag, text in ((500, "500", "500 m"), (1000, "1000", "1000 m"), (2500, "2500", "2500 m"),
                                 (5000, "5000", "5000 m"), (10000, "10k", "10 km")):
            rows += [(_spec(f"wmax{tag}", f"max vertical velocity at {text} AGL", "m/s"), "wlevs"),
                     (_spec(f"wmin{tag}", f"min vertical velocity at {text} AGL", "m/s"), "wlevs")]
    if on("stat_u"):
        rows += [(_spec("umax", "max u velocity (grid-rel)", "m/s"), "u"),
                 (_spec("umin", "min u velocity (grid-rel)", "m/s"), "u"),
                 (_spec("sumax", "max u velocity at lowst mod lev (grid-rel)", "m/s"), "u"),
                 (_spec("sumin", "min u velocity at lowst mod lev (grid-rel)", "m/s"), "u")]
    if on("stat_v"):
        rows += [(_spec("vmax", "max v velocity (grid-rel)", "m/s"), "v"),
                 (_spec("vmin", "min v velocity (grid-rel)", "m/s"), "v"),
                 (_spec("svmax", "max v velocity at lowst mod lev (grid-rel)", "m/s"), "v"),
                 (_spec("svmin", "min v velocity at lowst mod lev (grid-rel)", "m/s"), "v")]
    if on("stat_rmw"):
        rows += [(_spec("rmw", "radius of maximum horizontal wind speed", "m"), "rmw"),
                 (_spec("zmw", "height of maximum horizontal wind speed", "m"), "rmw")]
    if on("stat_pipert"):
        rows += [(_spec("ppimax", "max nondimensional pressure perturbation", "nondimensional"), "pipert"),
                 (_spec("ppimin", "min nondimensional pressure perturbation", "nondimensional"), "pipert")]
    if on("stat_prspert"):
        rows += [(_spec("ppmax", "max pressure perturbation", "Pa"), "prspert"),
                 (_spec("ppmin", "min pressure perturbation", "Pa"), "prspert")]
    if on("stat_thpert"):
        rows += [(_spec("thpmax", "max potential temperature perturbation", "K"), "thpert"),
                 (_spec("thpmin", "min potential temperature perturbation", "K"), "thpert"),
                 (_spec("sthpmax", "max pot. temp. pert. at lowest model level", "K"), "thpert"),
                 (_spec("sthpmin", "min pot. temp. pert. at lowest model level", "K"), "thpert")]
    if on("stat_q"):
        for name, unit in zip(species.names, species.units):
            rows += [(_spec(f"max{name}", f"max {name}", unit), "q"),
                     (_spec(f"min{name}", f"min {name}", unit), "q")]
        rows += [(_spec("pratemax", "maximum surface precipitation rate", "kg/m2/s"), "prate"),
                 (_spec("pratemin", "minimum surface precipitation rate", "kg/m2/s"), "prate")]
    if on("stat_tke") and s.int("sgsmodel", 0) == 1 and s.int("cm1setup", 1) in (1, 4):  # param.F:752 idoles
        rows += [(_spec("tkemax", "max subgrid tke", "m^2/s^2"), "tke"),
                 (_spec("tkemin", "min subgrid tke", "m^2/s^2"), "tke")]
    sgs_or_pbl2 = s.int("sgsmodel", 0) >= 1 or s.int("ipbl", 0) == 2
    if on("stat_km"):
        rows += [(_spec("kmhmax", "max horiz eddy viscosity for momentum", "m^2/s"), "km"),
                 (_spec("kmhmin", "min horiz eddy viscosity for momentum", "m^2/s"), "km")]
        if sgs_or_pbl2:
            rows += [(_spec("kmvmax", "max vert eddy viscosity for momentum", "m^2/s"), "km"),
                     (_spec("kmvmin", "min vert eddy viscosity for momentum", "m^2/s"), "km")]
    if on("stat_kh"):
        rows += [(_spec("khhmax", "max horiz eddy diffusivity for scalars", "m^2/s"), "kh"),
                 (_spec("khhmin", "min horiz eddy diffusivity for scalars", "m^2/s"), "kh")]
        if sgs_or_pbl2:
            rows += [(_spec("khvmax", "max vert eddy diffusivity for scalars", "m^2/s"), "kh"),
                     (_spec("khvmin", "min vert eddy diffusivity for scalars", "m^2/s"), "kh")]
    if on("stat_div"):
        rows += [(_spec("divmax", "max 3d divergence", "1/s"), "div"),
                 (_spec("divmin", "min 3d divergence", "1/s"), "div")]
    if on("stat_rh"):
        rows += [(_spec("rhmax", "max relative humidity wrt liquid", "nondimensional"), "rh"),
                 (_spec("rhmin", "min relative humidity wrt liquid", "nondimensional"), "rh")]
    if on("stat_rhi"):
        # setup_stat_vars describes rhimax as a minimum; CM1's words are kept.
        rows += [(_spec("rhimax", "min relative humidity wrt ice", "nondimensional"), "rhi"),
                 (_spec("rhimin", "min relative humidity wrt ice", "nondimensional"), "rhi")]
    if s.int("iptra", 0) == 1:
        for n in range(1, s.int("npt", 1) + 1):
            rows += [(_spec(f"maxpt{n}", f"max pt{n}", "kg/kg"), "pt"),
                     (_spec(f"minpt{n}", f"min pt{n}", "kg/kg"), "pt")]
    if on("stat_the"):
        rows += [(_spec("themax", "max theta-e below 10 km", "K"), "the"),
                 (_spec("themin", "min theta-e below 10 km", "K"), "the"),
                 (_spec("sthemax", "max theta-e at lowest model level", "K"), "the"),
                 (_spec("sthemin", "min theta-e at lowest model level", "K"), "the")]
    if on("stat_cloud"):
        rows += [(_spec("qctop", "max cloud top height", "m"), "cloud"),
                 (_spec("qcbot", "min cloud base height", "m"), "cloud")]
    if on("stat_sfcprs"):
        rows += [(_spec("sprsmax", "max pressure at lowest model level", "Pa"), "sfcprs"),
                 (_spec("sprsmin", "min pressure at lowest model level", "Pa"), "sfcprs"),
                 (_spec("psfcmax", "max pressure at surface", "Pa"), "sfcprs"),
                 (_spec("psfcmin", "min pressure at surface", "Pa"), "sfcprs")]
    if on("stat_wsp"):
        rows += [(_spec("wspmax", "max grid-rel horiz wind speed", "m/s"), "wsp"),
                 (_spec("wspmin", "min grid-rel horiz wind speed", "m/s"), "wsp"),
                 (_spec("zwspmax", "level of max grid-rel horiz wind speed", "m AGL"), "wsp"),
                 (_spec("zwspmin", "level of min grid-rel horiz wind speed", "m AGL"), "wsp"),
                 (_spec("swspmax", "max grid-rel horiz wind speed at l.m.l.", "m/s"), "wsp"),
                 (_spec("swspmin", "min grid-rel horiz wind speed at l.m.l.", "m/s"), "wsp")]
        if s.int("bbc", 1) == 3:
            text = "horiz wind speed at 10m AGL" if s.int("imove", 0) != 1 else "ground-rel. horiz wind speed at 10m AGL"
            rows += [(_spec("wsp10max", f"max {text}", "m/s"), "wsp10"),
                     (_spec("wsp10min", f"min {text}", "m/s"), "wsp10")]
    if s.int("bbc", 1) == 3 and s.int("sfcmodel", 0) >= 1:
        rows += [(_spec("hpblmax", "max diagnosed pbl depth", "m"), "hpbl"),
                 (_spec("hpblmin", "min diagnosed pbl depth", "m"), "hpbl")]
    if on("stat_cfl"):
        text = "max Courant number (average)" if s.int("adapt_dt", 0) == 1 else "max Courant number"
        rows.append((_spec("cflmax", text, "nondimensional"), "cfl"))
        if sgs_or_pbl2 or s.int("horizturb", 0) == 1:
            rows += [(_spec("kshmax", "max horiz K stability factor", "nondimensional"), "ks"),
                     (_spec("ksvmax", "max vert K stability factor", "nondimensional"), "ks")]
    if on("stat_vort"):
        rows += [(_spec("vortsfc", "max vert vorticity at lowest model level", "1/s"), "vort")]
        rows += [(_spec(f"vort{n}km", f"max vert vorticity at {n} km AGL", "1/s"), "vort") for n in range(1, 6)]
    if on("stat_tmass"):
        rows.append((_spec("tmass", "total mass of dry air", "kg"), "tmass"))
    if on("stat_tmois"):
        rows.append((_spec("tmois", "total mass of moisture", "kg"), "tmois"))
    if on("stat_qmass"):
        for n, name in enumerate(species.names, 1):
            if _counts_as_water(n, species):
                rows.append((_spec(f"mass{name}", f"total mass of {name}", "kg"), "qmass"))
    if on("stat_tenerg"):
        rows += [(_spec("ek", "total kinetic energy", "kg m^2/s^2"), "tenerg"),
                 (_spec("ei", "total internal energy", "kg m^2/s^2"), "tenerg"),
                 (_spec("ep", "total potential energy", "kg m^2/s^2"), "tenerg"),
                 (_spec("le", "total latent energy", "kg m^2/s^2"), "tenerg"),
                 (_spec("et", "total energy", "kg m^2/s^2"), "tenerg")]
    if on("stat_mo"):
        rows += [(_spec("tmu", "total E-W momentum", "kg m/s"), "mo"),
                 (_spec("tmv", "total N-S momentum", "kg m/s"), "mo"),
                 (_spec("tmw", "total vertical momentum", "kg m/s"), "mo")]
    if on("stat_tmf"):
        rows += [(_spec("tmfu", "total upward dry-air mass flux", "kg m/s"), "tmf"),
                 (_spec("tmfd", "total downward dry-air mass flux", "kg m/s"), "tmf")]
    if on("stat_pcn") and imoist:
        for budget in ("tcond", "tevac", "tauto", "taccr", "tevar", "train", "erain", "qsfc", "esfc", "erad"):
            rows.append((_spec(budget, budget, "unk"), "pcn"))
    if on("stat_qsrc") and imoist:
        rows += [(_spec(f"as{name}", f"artificial source of {name}", "kg"), "qsrc") for name in species.names]
        rows += [(_spec(f"bs{name}", f"bndry source/sink of {name}", "kg"), "qsrc") for name in species.names]
    if not imoist:
        rows = [row for row in rows if row[1] not in ("q", "prate", "rh", "rhi", "the", "cloud", "tmois", "qmass")]
    return rows


def _counts_as_water(n: int, species: Species) -> bool:
    r = species.ranges
    return (n == species.nqv
            or (r.get("nql1", -2) > 0 and r["nql1"] <= n <= r["nql2"])
            or (species.iice == 1 and r.get("nqs1", -2) > 0 and r["nqs1"] <= n <= r["nqs2"]))


#: Why a statistic is not in a GM1 file.  Plain words.
_NOT_PRODUCED = {
    "dt": "GM1 runs a fixed time step here.",
    "rmw": "The radius of maximum wind is an axisymmetric (hurricane) statistic GM1 does not form.",
    "km": "GM1 does not save the eddy viscosity.",
    "kh": "GM1 does not save the eddy diffusivity.",
    "ks": "GM1 does not save the eddy viscosity and diffusivity these stability factors use.",
    "div": "The divergence uses the base-state density; the frames do not carry the base state (set output_basestate = 1).",
    "pt": "GM1 does not run passive tracers.",
    "wsp10": "GM1 has no surface-layer 10 m wind.",
    "hpbl": "GM1 runs no PBL scheme.",
    "pcn": "These are CM1's in-run microphysics budget totals; they cannot be formed from a snapshot.",
    "qsrc": "These are CM1's in-run source and sink totals; they cannot be formed from a snapshot.",
    "prate": "The frames carry no surface precipitation rate (output_rain = 0).",
    "tke": "The frames carry no subgrid TKE.",
    "missing": "The frame lacks a field this statistic needs.",
}


@dataclass
class StatsResult:
    specs: list[StatSpec]
    values: dict[str, float]
    not_produced: dict[str, str]
    derived: list[str]


def compute(inputs: StatInputs, settings: Cm1Settings, geometry: Geometry, species: Species) -> StatsResult:
    """Every statistic CM1 defines for these settings that the inputs allow."""

    import numpy as np

    c = _C()
    f32 = np.float32
    rows = _names(settings, geometry, species)
    values: dict[str, float] = {}
    missing: dict[str, str] = {}
    groups = {group for _spec_, group in rows}
    # A group whose inputs the frame does not carry is listed, not computed.
    have = {
        "w": inputs.w is not None, "u": inputs.u is not None, "v": inputs.v is not None,
        "tha": inputs.tha is not None, "ppi": inputs.ppi is not None, "prs": inputs.prs is not None,
        "rho": inputs.rho is not None, "q": len(inputs.q) == len(species.names) and len(inputs.q) > 0,
        "theta": inputs.th is not None or (inputs.th0 is not None and inputs.tha is not None),
    }
    needs = {
        "w": ("w",), "wlevs": ("w",), "u": ("u",), "v": ("v",), "pipert": ("ppi",),
        "prspert": ("prs", "ppi"), "thpert": ("tha",), "q": ("q",), "div": ("u", "v", "w"),
        "rh": ("prs", "ppi", "q", "theta"), "rhi": ("prs", "ppi", "q", "theta"),
        "the": ("prs", "ppi", "q", "theta"), "cloud": ("q",), "sfcprs": ("prs",), "wsp": ("u", "v"),
        "cfl": ("u", "v", "w"), "vort": ("u", "v"), "tmass": ("rho",), "tmois": ("rho", "q"),
        "qmass": ("rho", "q"), "tenerg": ("rho", "q", "u", "v", "w", "prs", "ppi", "theta"),
        "mo": ("rho", "q", "u", "v", "w"), "tmf": ("rho", "w"),
    }
    for group, required in needs.items():
        if group in groups and not all(have[r] for r in required):
            groups.discard(group)
            for spec, g in rows:
                if g == group:
                    missing[spec.name] = _NOT_PRODUCED["missing"]
    zf, zh, mh, rmh = geometry.zf, geometry.zh, geometry.mh, geometry.rmh
    nz = geometry.nz
    u, v, w = inputs.u, inputs.v, inputs.w

    def put(name: str, value: Any) -> None:
        values[name] = float(np.float32(value))

    def level_max_min(field: Any) -> tuple[Any, Any, int, int]:
        # maxmin.F:5-119: per-level extremes, then the lowest level holding
        # the overall one (a strict comparison keeps the first).
        flat = field.reshape(field.shape[0], -1)
        tmax = flat.max(axis=1)
        tmin = flat.min(axis=1)
        kmax = int(np.argmax(tmax))
        kmin = int(np.argmin(tmin))
        return tmax[kmax], tmin[kmin], kmax, kmin

    def maxmin(field: Any, hi: str, lo: str) -> tuple[int, int]:
        fmax, fmin, kmax, kmin = level_max_min(field)
        put(hi, fmax)
        put(lo, fmin)
        return kmax, kmin

    def maxmin2d(field: Any, hi: str, lo: str) -> None:
        put(hi, field.max())
        put(lo, field.min())

    put("mtime", inputs.time)

    # ---- base state and total pressure/theta, as CM1 holds them
    need_thermo = groups & {"prspert", "rh", "rhi", "the", "tenerg"}
    pi0 = th0 = pitot = thtot = None
    if need_thermo:
        # Every group here needs prs and ppi (the ``needs`` table above).
        if inputs.pi0 is not None:
            pi0 = np.asarray(inputs.pi0, dtype=f32).reshape(-1, 1, 1)
            pitot = (pi0 + inputs.ppi).astype(f32, copy=False)
        else:
            # The inverse of prs = p00*(pi0+ppi)**cpdrd (see the module note).
            pitot = np.power(inputs.prs.astype(np.float64) / 1.0e5, float(c.rovcp)).astype(f32, copy=False)
            pi0 = (pitot.astype(np.float64) - inputs.ppi).astype(f32, copy=False)
        if inputs.th0 is not None:
            th0 = np.asarray(inputs.th0, dtype=f32).reshape(-1, 1, 1)
            thtot = (th0 + inputs.tha).astype(f32, copy=False)
        elif inputs.th is not None:
            thtot = np.asarray(inputs.th, dtype=f32)
    thermo_derived = inputs.pi0 is None or inputs.th0 is None

    # ---- statpack.F:130-145, maxmin on wa (nk+1 faces)
    if "w" in groups:
        kmax, kmin = maxmin(w, "wmax", "wmin")
        put("zwmax", zf[kmax])
        put("zwmin", zf[kmin])

    # ---- statpack.F:147-210, w interpolated to five heights
    if "wlevs" in groups:
        for zlev, tag in ((500.0, "500"), (1000.0, "1000"), (2500.0, "2500"), (5000.0, "5000"), (10000.0, "10k")):
            zlev32 = f32(zlev)
            if zlev32 < zf[nz]:
                nkm = nz  # zero-based face index of CM1's nk+1
                while zf[nkm] > zlev32 and nkm > 0:
                    nkm -= 1
                r2 = f32(f32(zlev32 - zf[nkm]) / f32(zf[nkm + 1] - zf[nkm]))
                r1 = f32(f32(1.0) - r2)
                plane = (r1 * w[nkm] + r2 * w[nkm + 1]).astype(f32, copy=False)
            else:
                plane = np.zeros(w.shape[1:], dtype=f32)
            maxmin2d(plane, f"wmax{tag}", f"wmin{tag}")

    if "u" in groups:
        maxmin(u, "umax", "umin")
        maxmin2d(u[0], "sumax", "sumin")
    if "v" in groups:
        maxmin(v, "vmax", "vmin")
        maxmin2d(v[0], "svmax", "svmin")
    if "pipert" in groups:
        maxmin(inputs.ppi, "ppimax", "ppimin")
    if "prspert" in groups:
        # statpack.F:223-236: prs - p00*(pi0**cpdrd), REAL.
        pp = (inputs.prs - (c.p00 * _powf(np, np.broadcast_to(pi0, inputs.prs.shape), c.cpdrd)).astype(f32, copy=False)).astype(f32, copy=False)
        maxmin(pp, "ppmax", "ppmin")
        if inputs.pi0 is None:
            inputs.derived += ["ppmax", "ppmin"]
    if "thpert" in groups:
        maxmin(inputs.tha, "thpmax", "thpmin")
        maxmin2d(inputs.tha[0], "sthpmax", "sthpmin")
    if "q" in groups:
        for name, q in zip(species.names, inputs.q):
            maxmin(q, f"max{name}", f"min{name}")
    if "prate" in groups:
        if inputs.prate is not None:
            maxmin2d(inputs.prate, "pratemax", "pratemin")
        else:
            missing["pratemax"] = missing["pratemin"] = _NOT_PRODUCED["prate"]
    if "tke" in groups:
        if inputs.tke is not None:
            maxmin(inputs.tke, "tkemax", "tkemin")
        else:
            missing["tkemax"] = missing["tkemin"] = _NOT_PRODUCED["tke"]

    # ---- statpack.F:285-340, the 3-d divergence (needs rho0)
    if "div" in groups:
        if inputs.pi0 is not None and inputs.th0 is not None and inputs.prs0 is not None and inputs.qv0 is not None:
            prs0 = np.asarray(inputs.prs0, dtype=f32).reshape(-1)
            qv0 = np.asarray(inputs.qv0, dtype=f32).reshape(-1)
            p0 = np.asarray(inputs.pi0, dtype=f32).reshape(-1)
            t0 = np.asarray(inputs.th0, dtype=f32).reshape(-1)
            # base.F:2105
            rho0 = (prs0 / (c.rd * t0 * p0 * (f32(1.0) + qv0 * c.reps))).astype(f32, copy=False)
            rdx = f32(f32(1.0) / f32(geometry.dx))
            rdy = f32(f32(1.0) / f32(geometry.dy))
            rdz = f32(f32(1.0) / f32(geometry.dz))
            r = rho0.reshape(-1, 1, 1)
            two_r = (r + r).astype(f32, copy=False)
            half = f32(0.5)
            dudx = (half * (two_r * u[:, :, 1:] - two_r * u[:, :, :-1]) * rdx).astype(f32, copy=False)
            dvdy = (half * (two_r * v[:, 1:, :] - two_r * v[:, :-1, :]) * rdy).astype(f32, copy=False)
            r_up = np.concatenate([rho0[1:], rho0[-1:]]).reshape(-1, 1, 1)
            r_dn = np.concatenate([rho0[:1], rho0[:-1]]).reshape(-1, 1, 1)
            dwdz = (half * ((r + r_up) * w[1:] - (r + r_dn) * w[:-1]) * rdz * mh.reshape(-1, 1, 1)).astype(f32, copy=False)
            maxmin((dudx + dvdy + dwdz).astype(f32, copy=False), "divmax", "divmin")
        else:
            missing["divmax"] = missing["divmin"] = _NOT_PRODUCED["div"]

    rh = None
    qv = inputs.q[species.nqv - 1] if inputs.q else None
    if groups & {"rh", "the"} or "rhi" in groups:
        if thtot is None or qv is None:
            for name in ("rhmax", "rhmin", "rhimax", "rhimin", "themax", "themin", "sthemax", "sthemin"):
                missing[name] = _NOT_PRODUCED["missing"]
        else:
            tk = (thtot * pitot).astype(f32, copy=False)
            if groups & {"rh", "the"}:
                qvs = _rslf(np, c, inputs.prs, tk)
                rh = (qv * (f32(1.0) + qvs * c.reps) / (qvs * (f32(1.0) + qv * c.reps))).astype(f32, copy=False)
                if "rh" in groups:
                    maxmin(rh, "rhmax", "rhmin")
                    if thermo_derived:
                        inputs.derived += ["rhmax", "rhmin"]
            if "rhi" in groups and species.iice == 1:
                qvs = _rsif(np, c, inputs.prs, tk)
                rhi = (qv * (f32(1.0) + qvs * c.reps) / (qvs * (f32(1.0) + qv * c.reps))).astype(f32, copy=False)
                maxmin(rhi, "rhimax", "rhimin")
                if thermo_derived:
                    inputs.derived += ["rhimax", "rhimin"]
            if "the" in groups:
                the = _calcthe(np, c, zh, tk, qv, inputs.prs, rh)
                maxmin(the, "themax", "themin")
                maxmin2d(the[0], "sthemax", "sthemin")
                if thermo_derived:
                    inputs.derived += ["themax", "themin", "sthemax", "sthemin"]

    if "cloud" in groups:
        # statpack.F:420-445 + misclibs.F:2344-2416
        qci = np.zeros(inputs.q[0].shape, dtype=f32)
        for flag, q in zip(species.cloudvar, inputs.q):
            if flag:
                qci = (qci + q).astype(f32, copy=False)
        cloudy = (qci >= c.clwsat).any(axis=(1, 2))
        maxz = f32(geometry.maxz)
        top = f32(max([f32(0.0)] + [zh[k] for k in range(nz) if cloudy[k]]))
        bot = f32(min([maxz] + [zh[k] for k in range(nz) if cloudy[k]]))
        if bot == maxz:
            bot = f32(0.0)
        put("qctop", top)
        put("qcbot", bot)

    if "sfcprs" in groups:
        maxmin2d(inputs.prs[0], "sprsmax", "sprsmin")
        cgs1, cgs2, cgs3 = (f32(x) for x in fornberg_ends(geometry)["cgs"])
        psfc = ((cgs1 * inputs.prs[0] + cgs2 * inputs.prs[1]) + cgs3 * inputs.prs[2]).astype(f32, copy=False)
        maxmin2d(psfc, "psfcmax", "psfcmin")

    if "wsp" in groups:
        half = f32(0.5)
        us = (f32(inputs.umove) + half * (u[:, :, :-1] + u[:, :, 1:])).astype(f32, copy=False)
        vs = (f32(inputs.vmove) + half * (v[:, :-1, :] + v[:, 1:, :])).astype(f32, copy=False)
        wsp = np.sqrt((us * us + vs * vs).astype(f32, copy=False)).astype(f32, copy=False)
        kmax, kmin = maxmin(wsp, "wspmax", "wspmin")
        put("zwspmax", zh[kmax])
        put("zwspmin", zh[kmin])
        maxmin2d(wsp[0], "swspmax", "swspmin")
    for group in ("wsp10", "hpbl", "ks", "km", "kh", "pt", "pcn", "qsrc", "rmw", "dt"):
        if group in groups:
            for spec, g in rows:
                if g == group:
                    missing[spec.name] = _NOT_PRODUCED[group]

    if "cfl" in groups:
        # misclibs.F:2584-2718 (nx > 1 and ny > 1), REAL.
        dt = f32(geometry.dt)
        half = f32(0.5)
        dtdx = f32(f32(half * dt) * f32(f32(1.0) / f32(geometry.dx)))
        dtdy = f32(f32(half * dt) * f32(f32(1.0) / f32(geometry.dy)))
        dtdz = f32(f32(half * dt) * f32(f32(1.0) / f32(geometry.dz)))
        a = ((u[:, :, :-1] + u[:, :, 1:]) * dtdx).astype(f32, copy=False)
        b = ((v[:, :-1, :] + v[:, 1:, :]) * dtdy).astype(f32, copy=False)
        cz = ((w[:-1] + w[1:]) * dtdz * mh.reshape(-1, 1, 1)).astype(f32, copy=False)
        cfl = np.sqrt(((a * a + b * b) + cz * cz).astype(f32, copy=False)).astype(f32, copy=False)
        put("cflmax", cfl.max())

    if "vort" in groups:
        _vertvort(np, c, settings, geometry, u, v, put)

    rho = inputs.rho
    sums_needed = groups & {"tmass", "tmois", "qmass", "tenerg", "mo", "tmf"}
    if sums_needed and rho is None:
        for spec, g in rows:
            if g in sums_needed:
                missing[spec.name] = _NOT_PRODUCED["missing"]
        sums_needed = set()
    if sums_needed:
        volume = float(f32(f32(f32(geometry.dx) * f32(geometry.dy)) * f32(geometry.dz)))
        rmh3 = rmh.reshape(-1, 1, 1)
        ql, qi = _getqli(np, species, inputs.q, inputs.q[0].shape)
        if "tmass" in groups:
            put("tmass", _dsum(np, (rho * rmh3).astype(f32, copy=False)) * volume)
        if "tmois" in groups:
            water = ((qv + ql) + qi).astype(f32, copy=False)
            put("tmois", _dsum(np, ((rho * water).astype(f32, copy=False) * rmh3).astype(f32, copy=False)) * volume)
        if "qmass" in groups:
            for n, (name, q) in enumerate(zip(species.names, inputs.q), 1):
                if _counts_as_water(n, species):
                    put(f"mass{name}", _dsum(np, ((rho * q).astype(f32, copy=False) * rmh3).astype(f32, copy=False)) * volume)
        if groups & {"tenerg", "mo"}:
            tem = rmh3.astype(np.float64)
            qtot = ((qv + ql) + qi).astype(f32, copy=False).astype(np.float64)
            qtot += 1.0
            base = rho.astype(np.float64)
            base *= tem
            rt = base.copy() if "tenerg" in groups else None
            base *= qtot
            del qtot
        if "tenerg" in groups:
            if thtot is None:
                for name in ("ek", "ei", "ep", "le", "et"):
                    missing[name] = _NOT_PRODUCED["missing"]
            else:
                half = f32(0.5)
                vel = ((half * (u[:, :, :-1] * u[:, :, :-1] + u[:, :, 1:] * u[:, :, 1:])
                        + half * (v[:, :-1, :] * v[:, :-1, :] + v[:, 1:, :] * v[:, 1:, :]))
                       + half * (w[:-1] * w[:-1] + w[1:] * w[1:])).astype(f32, copy=False)
                ek = 0.5 * _ddot(np, base, vel.astype(np.float64))
                tmp = (thtot * pitot).astype(f32, copy=False).astype(np.float64)
                ei = _ddot(np, rt * tmp, (c.cv + c.cvv * qv).astype(f32, copy=False).astype(np.float64))
                ep = float(c.g) * _ddot(np, base, np.broadcast_to(zh.astype(np.float64).reshape(-1, 1, 1), base.shape))
                liquid = tmp * float(c.cpl)
                liquid -= float(c.lv1)
                liquid *= rt
                ice = tmp * float(c.cpi)
                ice -= float(c.ls1)
                ice *= rt
                le = _ddot(np, liquid, ql.astype(np.float64)) + _ddot(np, ice, qi.astype(np.float64))
                del liquid, ice, tmp
                ek, ei, ep, le = (x * volume for x in (ek, ei, ep, le))
                put("ek", ek)
                put("ei", ei)
                put("ep", ep)
                put("le", le)
                put("et", ek + ei + ep + le)
                if thermo_derived:
                    inputs.derived += ["ei", "le", "et"]
        if "mo" in groups:
            half = f32(0.5)
            us = (f32(inputs.umove) + half * (u[:, :, :-1] + u[:, :, 1:])).astype(f32, copy=False)
            vs = (f32(inputs.vmove) + half * (v[:, :-1, :] + v[:, 1:, :])).astype(f32, copy=False)
            ws = (half * (w[:-1] + w[1:])).astype(f32, copy=False)
            put("tmu", _ddot(np, base, us.astype(np.float64)) * volume)
            put("tmv", _ddot(np, base, vs.astype(np.float64)) * volume)
            put("tmw", _ddot(np, base, ws.astype(np.float64)) * volume)
        if "tmf" in groups:
            mf = ((rho * f32(0.5)) * (w[:-1] + w[1:])).astype(f32, copy=False).astype(np.float64)
            area = float(f32(geometry.dx)) * float(f32(geometry.dy))
            put("tmfu", _dsum(np, np.maximum(mf, 0.0)) * area)
            put("tmfd", _dsum(np, np.minimum(mf, 0.0)) * area)

    specs = [spec for spec, _group in rows if spec.name in values]
    not_produced = {spec.name: missing.get(spec.name, _NOT_PRODUCED["missing"])
                    for spec, _group in rows if spec.name not in values}
    return StatsResult(specs, values, not_produced, sorted(set(inputs.derived)))


def _expf(np: Any, x: Any) -> Any:
    """REAL ``exp``: glibc's expf is correctly rounded, so round the binary64 result once."""

    return np.exp(np.asarray(x, dtype=np.float64)).astype(np.float32, copy=False)


def _powf(np: Any, x: Any, y: Any) -> Any:
    """REAL ``**`` with a REAL exponent (gfortran calls powf, correctly rounded)."""

    return np.power(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)).astype(np.float32, copy=False)


def _ddot(np: Any, a: Any, b: Any) -> float:
    """sum(a*b) in DOUBLE PRECISION without forming the product array."""

    a = np.ascontiguousarray(a, dtype=np.float64).reshape(-1)
    b = np.ascontiguousarray(b, dtype=np.float64).reshape(-1)
    return float(np.dot(a, b))


def _dsum(np: Any, array: Any) -> float:
    """CM1 accumulates these sums in DOUBLE PRECISION, level by level."""

    return float(np.asarray(array, dtype=np.float64).reshape(array.shape[0], -1).sum(axis=1).sum())


def _getqli(np: Any, species: Species, q: Sequence[Any], shape: tuple[int, ...]) -> tuple[Any, Any]:
    # misclibs.F:923-967, REAL sums in species order.
    f32 = np.float32
    ql = np.zeros(shape, dtype=f32)
    qi = np.zeros(shape, dtype=f32)
    r = species.ranges
    if r.get("nql1", -2) > 0:
        for n in range(r["nql1"], r["nql2"] + 1):
            ql = (ql + q[n - 1]).astype(f32, copy=False)
    if species.iice == 1 and r.get("nqs1", -2) > 0:
        for n in range(r["nqs1"], r["nqs2"] + 1):
            qi = (qi + q[n - 1]).astype(f32, copy=False)
    return ql, qi


def _rslf(np: Any, c: _C, p: Any, t: Any) -> Any:
    # cm1libs.F:16-62, the Bolton (1980) branch every GM1 scheme takes.
    f32 = np.float32
    esl = (f32(611.2) * _expf(np, ((f32(17.67) * (t - f32(273.15))) / (t - f32(29.65))).astype(f32, copy=False))).astype(f32, copy=False)
    esl = np.minimum(esl, (p * f32(0.5)).astype(f32, copy=False))
    return ((c.eps * esl) / (p - esl)).astype(f32, copy=False)


def _rsif(np: Any, c: _C, p: Any, t: Any) -> Any:
    # cm1libs.F:69-115, Tao et al. (1989).
    f32 = np.float32
    esi = (f32(611.2) * _expf(np, ((f32(21.8745584) * (t - f32(273.15))) / (t - f32(7.66))).astype(f32, copy=False))).astype(f32, copy=False)
    esi = np.minimum(esi, (p * f32(0.5)).astype(f32, copy=False))
    return ((c.eps * esi) / (p - esi)).astype(f32, copy=False)


def _calcthe(np: Any, c: _C, zh: Any, tx: Any, qv: Any, prs: Any, rh: Any) -> Any:
    # misclibs.F:2300-2336 (Bryan 2008), REAL; above 10 km the level below is copied.
    f32 = np.float32
    cpm = c.cp
    a = _powf(np, ((c.p00 * (f32(1.0) + qv * c.reps)) / prs).astype(f32, copy=False), f32(c.rd / cpm))
    b = _powf(np, rh, (((-qv) * c.rv) / cpm).astype(f32, copy=False))
    e = _expf(np, ((c.l0 * qv) / (cpm * tx)).astype(f32, copy=False))
    the = (((tx * a) * b) * e).astype(f32, copy=False)
    for k in range(1, the.shape[0]):
        if zh[k] > f32(10000.0):
            the[k] = the[k - 1]
    return the


def _vertvort(np: Any, c: _C, settings: Cm1Settings, geometry: Geometry, u: Any, v: Any, put: Any) -> None:
    """misclibs.F:2424-2576, Cartesian grid without terrain, REAL."""

    f32 = np.float32
    nz = geometry.nz
    zh = geometry.zh
    rdx = f32(f32(1.0) / f32(geometry.dx))
    rdy = f32(f32(1.0) / f32(geometry.dy))
    # param.F:6678-6681 on one process: an edge that is not periodic is excluded.
    ibw = int(settings.int("wbc", 1) != 1)
    ibe = int(settings.int("ebc", 1) != 1)
    ibs = int(settings.int("sbc", 1) != 1)
    ibn = int(settings.int("nbc", 1) != 1)
    levels = [0]
    for height in (1000.0, 2000.0, 3000.0, 4000.0, 5000.0):
        above = [k for k in range(nz) if zh[k] >= f32(height)]
        levels.append(above[0] if above else nz)
    names = ("vortsfc", "vort1km", "vort2km", "vort3km", "vort4km", "vort5km")
    for name, k in zip(names, levels):
        if k >= nz:
            put(name, 0.0)
            continue
        va = v[k]  # (ny+1, nx): CM1 va(i=1..ni, j=1..nj+1)
        ua = u[k]  # (ny, nx+1): CM1 ua(i=1..ni+1, j=1..nj)
        # Periodic ghosts: va(0,j) = va(ni,j), va(ni+1,j) = va(1,j); ua(i,0) = ua(i,nj), ua(i,nj+1) = ua(i,1).
        va_x = np.concatenate([va[:, -1:], va, va[:, :1]], axis=1)   # columns CM1 i = 0..ni+1
        ua_y = np.concatenate([ua[-1:, :], ua, ua[:1, :]], axis=0)   # rows CM1 j = 0..nj+1
        nx, ny = geometry.nx, geometry.ny
        i0, i1 = 1 + ibw, nx + 1 - ibe
        j0, j1 = 1 + ibs, ny + 1 - ibn
        if i1 < i0 or j1 < j0:
            put(name, f32(-9999999.0))
            continue
        jj = slice(j0 - 1, j1)          # va rows (CM1 j -> row j-1)
        ii = slice(i0, i1 + 1)          # va_x columns (CM1 i -> column i)
        iim = slice(i0 - 1, i1)
        term1 = ((va_x[jj, ii] - va_x[jj, iim]) * rdx).astype(f32, copy=False)
        uj = slice(j0, j1 + 1)          # ua_y rows (CM1 j -> row j)
        ujm = slice(j0 - 1, j1)
        ui = slice(i0 - 1, i1)          # ua columns (CM1 i -> column i-1)
        term2 = ((ua_y[uj, ui] - ua_y[ujm, ui]) * rdy).astype(f32, copy=False)
        vort = (term1 - term2).astype(f32, copy=False)
        put(name, max(f32(-9999999.0), vort.max()))


def fornberg_ends(geometry: Geometry) -> Mapping[str, tuple[float, float, float]]:
    """cgs/dgs/cgt/dgt (param.F:8985-9060) for this column."""

    from ..cm1_vertical_grid import fornberg_extrapolation_weights, fornberg_first_derivative_weights

    nk = geometry.nz
    sigma = [float("nan")] + [float(x) for x in geometry.zh]
    sigmaf = [float("nan")] + [float(x) for x in geometry.zf]
    return {
        "cgs": fornberg_extrapolation_weights(sigma, sigmaf, nk=nk, top=False),
        "dgs": fornberg_first_derivative_weights(sigma, sigmaf, nk=nk, top=False),
        "cgt": fornberg_extrapolation_weights(sigma, sigmaf, nk=nk, top=True),
        "dgt": fornberg_first_derivative_weights(sigma, sigmaf, nk=nk, top=True),
    }


# --------------------------------------------------------------------------
# Inputs

def inputs_from_frame(frame: Any, species: Species) -> StatInputs:
    """The statistics' inputs from one GM1 frame (views of its arena)."""

    import numpy as np

    from .fields import reconstructed_prs

    def get(name: str) -> Any:
        return frame.array(name) if frame.has(name) else None

    q: list[Any] = []
    if frame.has("q3d"):
        q3d = frame.array("q3d")
        q.extend(q3d[i] for i in range(q3d.shape[0]))
    if frame.has("nc3d"):
        nc3d = frame.array("nc3d")
        q.extend(nc3d[i] for i in range(nc3d.shape[0]))
    if q and len(q) != len(species.names):
        raise ValueError(f"{frame.directory}: q3d/nc3d carry {len(q)} species, the record names {len(species.names)}")
    moving = frame.moving_frame
    column = lambda name: None if not frame.has(name) else np.asarray(frame.array(name), dtype=np.float32).reshape(-1)  # noqa: E731
    return StatInputs(
        time=frame.model_time_s,
        u=get("u3d"), v=get("v3d"), w=get("w3d"),
        tha=get("th3d"), ppi=get("pp3d"), prs=get("prs") if frame.has("prs") else reconstructed_prs(frame),
        rho=get("rho"),
        q=q, tke=get("closure.tkea"), prate=get("prate"),
        th0=column("th0"), pi0=column("pi0") if frame.has("pi0") else column("base.pi0"),
        prs0=column("prs0"), qv0=column("qv0"), th=get("th"),
        umove=float(moving.get("umove", 0.0) or 0.0), vmove=float(moving.get("vmove", 0.0) or 0.0),
    )


def inputs_from_cm1_history(dataset: Any, index: int, species: Species) -> StatInputs:
    """The same inputs from one time of a CM1 netCDF history file.

    Needs ``output_thpert``, ``output_pipert``, ``output_rho`` and, for the
    exact thermodynamic statistics, ``output_basestate``.
    """

    import numpy as np

    def get(name: str) -> Any:
        if name not in dataset.variables:
            return None
        return np.asarray(dataset.variables[name][index], dtype=np.float32)

    def column(name: str) -> Any:
        array = get(name)
        return None if array is None else array[:, 0, 0].copy()

    q = [get(name) for name in species.names]
    return StatInputs(
        time=float(dataset.variables["time"][index]),
        u=get("u"), v=get("v"), w=get("w"), tha=get("thpert"), ppi=get("pipert"), prs=get("prs"),
        rho=get("rho"), q=q, tke=get("tke"), prate=get("prate"),
        th0=column("th0"), pi0=column("pi0"), prs0=column("prs0"), qv0=column("qv0"), th=get("th"),
        umove=float(dataset.variables["umove"][index]) if "umove" in dataset.variables else 0.0,
        vmove=float(dataset.variables["vmove"][index]) if "vmove" in dataset.variables else 0.0,
    )


# --------------------------------------------------------------------------
# The file

def write_stats(path: Any, rows: Sequence[tuple[float, Mapping[str, float]]], specs: Sequence[StatSpec],
                *, global_attributes: Mapping[str, Any], deflate: int = 0) -> None:
    """``cm1out_stats.nc`` as writestat_nc (writeout_nc.F:1631-1781) lays it out."""

    import netCDF4
    import numpy as np

    ds = netCDF4.Dataset(path, "w", format="NETCDF4_CLASSIC")
    try:
        ds.set_fill_off()
        for dim in ("xh", "yh", "zh"):
            ds.createDimension(dim, 1)
        ds.createDimension("time", None)
        for dim, long_name, units, axis in (
            ("xh", "west-east location", "degree_east", "X"),
            ("yh", "south-north location", "degree_north", "Y"),
            ("zh", "height", "m", "Z"),
        ):
            var = ds.createVariable(dim, "f4", (dim,))
            var.long_name, var.units, var.axis = long_name, units, axis
        var = ds.createVariable("time", "f4", ("time",))
        var.long_name, var.units, var.axis = "time", "seconds", "T"
        variables = {}
        for spec in specs:
            var = ds.createVariable(spec.name, "f4", ("time",), chunksizes=(1,),
                                    zlib=deflate > 0, complevel=deflate or 1, shuffle=deflate > 0)
            var.long_name, var.units = spec.long_name, spec.units
            variables[spec.name] = var
        for key, value in global_attributes.items():
            ds.setncattr(key, value)
        ds["xh"][:] = np.float32(0.0)
        ds["yh"][:] = np.float32(0.0)
        ds["zh"][:] = np.float32(0.0)
        missing = np.float32(-999999.9)
        ds["time"][:] = np.asarray([time for time, _values in rows], dtype=np.float32)
        for spec in specs:
            column = np.asarray([values.get(spec.name, missing) for _time, values in rows], dtype=np.float32)
            variables[spec.name][:] = column
    finally:
        ds.close()


__all__ = [
    "STATS_FILENAME",
    "StatInputs",
    "StatSpec",
    "StatsResult",
    "compute",
    "fornberg_ends",
    "inputs_from_cm1_history",
    "inputs_from_frame",
    "write_stats",
]
