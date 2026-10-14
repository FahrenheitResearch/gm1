"""How big the model's cloud and precipitation particles are, by each microphysics scheme's own rules.

The infrared picture of the clouds (:mod:`.infrared`) needs each species' effective radius: the same mass of water in
small droplets hides the warm air below far better than in large drops, so the sizes set how deep into a cloud the
infrared sees.  Each scheme sizes its particles its own way, so each is sized by its own rules, from CM1 r21.1's
source; where CM1 computes the radii it hands its radiation code, those are used:

* Morrison (``morrison-mp10``): ``morrison.F:5001-5178``.  The slopes of cloud ice, rain, snow and graupel or hail
  come from the frame's own number concentrations (``nci, ncr, ncs, ncg``) with the scheme's slope limits, and the
  radii are ``EFFI, EFFR, EFFS, EFFG = 3 / (2 lambda)``; cloud droplets have the fixed droplet number ``ndcnst``
  (per cm3, the run's namelist; CM1's default is 250) and the droplet shape the scheme sets from that number
  (``PGAM``), ``EFFC``.
* WSM6 (``wsm6-mp6``): ``mp_wsm6_effectRad.F:50-184`` with the background and largest radii CM1 passes it
  (``mp_driver.F:1586-1607``) for cloud water (300 droplets per cm3), cloud ice (WSM6's crystal number from the
  ice mass) and snow (the intercept that grows as it gets colder); rain and graupel or hail, which that routine
  does not size, get ``3 / (2 lambda)`` of WSM6's exponential distributions with its fixed intercepts
  (``mp_wsm6.F:16, 42-43, 89-96, 138-139, 155``).
* Thompson (``thompson-mp8``, the scheme ArWen runs): ``calc_effectRad`` (``thompson.F:5229-5333``) for cloud
  water (the scheme's fixed 100 droplets per cm3), cloud ice (from its number) and snow (the scheme's fits of the
  snow's size moments); rain from its number and graupel from the scheme's intercept rule
  (``thompson.F:1980-1987``), each ``3 / (2 lambda)``.
* any other scheme: the fixed radii of the clouds-from-above look (:data:`.satellite.R_EFF`), and the words say so.

The visible extinction of a species is the clouds-from-above look's ``beta = 3 rho q / (2 rho_w r_e)`` with the radius
from here instead of a fixed one.  Nothing here is smoothed or tuned: a radius is computed cell by cell.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np

from .frame import FrameUnreadable

PI = math.pi
RHO_W = 1000.0
#: CM1's own fixed radii (m) of the clouds-from-above look, for a scheme this module has no rules for.
FIXED = {"qc": 10.0e-6, "qi": 30.0e-6, "qs": 100.0e-6, "qg": 500.0e-6, "qr": 500.0e-6}
SPECIES = ("qc", "qi", "qs", "qg", "qr")
WORD = {"qc": "cloud water", "qi": "cloud ice", "qs": "snow", "qg": "graupel or hail", "qr": "rain"}

# ---------------------------------------------------------------- Morrison (morrison.F)
MORRISON_QSMALL = 1.0e-14                                    # morrison.F:460
MORRISON_RHOW, MORRISON_RHOI, MORRISON_RHOSN = 997.0, 500.0, 100.0   # :437-439
MORRISON_RHOG = {"graupel": 400.0, "hail": 900.0}            # :442, :444 (ihail)
MORRISON_DCS = 125.0e-6                                      # :449
MORRISON_LIMITS = {                                          # :486-498 (1/m)
    "qi": (1.0 / (2.0 * MORRISON_DCS + 100.0e-6), 1.0 / 1.0e-6),
    "qr": (1.0 / 2800.0e-6, 1.0 / 20.0e-6),
    "qs": (1.0 / 2000.0e-6, 1.0 / 10.0e-6),
    "qg": (1.0 / 2000.0e-6, 1.0 / 20.0e-6),
}
#: CM1 r21.1's ndcnst default (run/namelist.input:108), droplets per cm3.
NDCNST_DEFAULT = 250.0

# ---------------------------------------------------------------- WSM6 (mp_wsm6.F, mp_wsm6_effectRad.F)
WSM6_DENR, WSM6_DENS = 1000.0, 100.0                         # mp_wsm6.F denr, :38
WSM6_N0R, WSM6_N0S, WSM6_N0SMAX, WSM6_ALPHA = 8.0e6, 2.0e6, 1.0e11, 0.12   # :16, :41-43
WSM6_GRAUPEL = {"graupel": (4.0e6, 500.0), "hail": (4.0e4, 700.0)}      # :89-96 (n0g, deng)
WSM6_NC0 = 3.0e8                                             # mp_wsm6_effectRad.F nc0 (m-3)
WSM6_T0C = 273.15
#: (background, largest) radius, m, as mp_driver.F:1586-1607 passes them.
WSM6_BOUNDS = {"qc": (2.49e-6, 50.0e-6), "qi": (4.99e-6, 125.0e-6), "qs": (9.99e-6, 999.0e-6)}
WSM6_R1 = 1.0e-12

# ---------------------------------------------------------------- Thompson (thompson.F)
THOMPSON_NT_C = 100.0e6                                      # the droplet number ArWen's classic Thompson holds (m-3)
THOMPSON_RHO_W, THOMPSON_RHO_I, THOMPSON_RHO_G = 1000.0, 890.0, 500.0   # thompson.F:72-75
THOMPSON_AM_S, THOMPSON_BM_S = 0.069, 2.0                    # :128-129
THOMPSON_G_RATIO = (24, 60, 120, 210, 336, 504, 720, 990, 1320, 1716, 2184, 2730, 3360, 4080, 4896)
THOMPSON_SA = (5.065339, -0.062659, -3.032362, 0.029469, -0.000285, 0.31255, 0.000204, 0.003199, 0.0, -0.015952)
THOMPSON_SB = (0.476221, -0.015896, 0.165977, 0.007468, -0.000141, 0.060366, 0.000079, 0.000594, 0.0, -0.003577)
THOMPSON_BG = {"qc": 2.49e-6, "qi": 4.99e-6, "qs": 9.99e-6}  # :406-408
THOMPSON_R1, THOMPSON_R2 = 1.0e-12, 1.0e-6                    # :168-169


@dataclass
class Sizes:
    """How a frame's particles were sized: the scheme, the picture's words, where the rules are, and the
    mass-weighted mean radius (micrometres) of each species present."""

    scheme: str                       # "Morrison", "WSM6", "Thompson" or "fixed"
    words: str                        # the picture's words, e.g. "particle sizes: Morrison's own"
    source: str                       # where the rules are, in CM1's source
    mean_um: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def mean_radius_um(q: np.ndarray, radius: Any) -> float | None:
    """The mass-weighted mean of a radius (m) over the cells holding the species, in micrometres."""

    weight = np.maximum(np.asarray(q, np.float64), 0.0)
    total = float(weight.sum())
    if total <= 0.0:
        return None
    r = np.broadcast_to(np.asarray(radius, np.float64), weight.shape)
    return float((weight * r).sum() / total * 1.0e6)


# ---------------------------------------------------------------- the run's droplet number
def droplet_number(directory: Path) -> tuple[float, str]:
    """Morrison's fixed droplet number (per cm3) and where it was read: the deck the frame's segment ran from, the
    run's deck, or the run's namelist (``ndcnst``); else CM1's default."""

    import tomllib

    directory = Path(directory)
    for folder in [directory, *directory.parents][:5]:
        deck = folder / "deck.toml"
        if deck.is_file():
            try:
                value = (tomllib.loads(deck.read_text(encoding="utf-8")).get("physics") or {}).get("ndcnst")
            except (OSError, UnicodeError, tomllib.TOMLDecodeError):
                value = None
            if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
                return float(value), "the run's deck"
        namelist = folder / "namelist.input"
        if namelist.is_file():
            try:
                text = namelist.read_text(encoding="utf-8", errors="replace")
            except OSError:
                text = ""
            found = re.search(r"^\s*ndcnst\s*=\s*([0-9.eEdD+-]+)", text, re.M)
            if found:
                try:
                    return float(found.group(1).replace("d", "e").replace("D", "e")), "the run's namelist"
                except ValueError:
                    pass
    return NDCNST_DEFAULT, "CM1's default"


def dense_ice(config: Mapping[str, Any]) -> str:
    """'hail' or 'graupel', as the frame's publication record says (ihail)."""

    category = (config.get("dense_ice") or {}).get("category") if isinstance(config.get("dense_ice"), dict) else None
    return "hail" if category == "hail" else "graupel"


# ---------------------------------------------------------------- the schemes
def morrison(q: Mapping[str, np.ndarray], n: Mapping[str, np.ndarray], rho: np.ndarray, *, ndcnst: float,
             ice: str = "graupel") -> dict[str, np.ndarray]:
    """morrison.F:5001-5178: EFFI, EFFR, EFFS, EFFG = 3 / (2 lambda) from the number concentrations (per kg), and EFFC
    from the fixed droplet number ``ndcnst`` (per cm3).  Where a species is below QSMALL its radius is the scheme's
    25 micrometres (it carries no mass there)."""

    out: dict[str, np.ndarray] = {}
    rho_x = {"qi": MORRISON_RHOI, "qs": MORRISON_RHOSN, "qg": MORRISON_RHOG[ice]}
    number = {"qi": "nci", "qs": "ncs", "qr": "ncr", "qg": "ncg"}
    for name in ("qi", "qs", "qr", "qg"):
        if name not in q or number[name] not in n:
            continue
        qq = np.asarray(q[name], np.float64)
        nn = np.maximum(np.asarray(n[number[name]], np.float64), 0.0)
        present = qq >= MORRISON_QSMALL
        # lambda = (c Gamma(4) N / Q)^(1/3): c = rho_x pi / 6 (ice, snow, graupel: CONS12, CONS1, CONS2); rain pi rho_w
        coef = PI * MORRISON_RHOW if name == "qr" else math.gamma(4.0) * rho_x[name] * PI / 6.0
        with np.errstate(divide="ignore", invalid="ignore"):
            lam = np.cbrt(coef * nn / np.where(present, qq, 1.0))
        lo, hi = MORRISON_LIMITS[name]
        lam = np.clip(np.nan_to_num(lam, nan=lo, posinf=hi), lo, hi)
        out[name] = np.where(present, 1.5 / lam, 25.0e-6).astype(np.float32)
    if "qc" in q:
        qc = np.asarray(q["qc"], np.float64)
        present = qc >= MORRISON_QSMALL
        nc = float(ndcnst) * 1.0e6 / np.asarray(rho, np.float64)          # per kg (:5217)
        pgam = 0.0005714 * float(ndcnst) + 0.2714                          # nc / 1e6 * rho = ndcnst
        pgam = min(max(1.0 / pgam ** 2 - 1.0, 2.0), 10.0)
        cons26 = PI / 6.0 * MORRISON_RHOW
        ratio = math.gamma(pgam + 4.0) / math.gamma(pgam + 1.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            lam = np.cbrt(cons26 * nc * ratio / np.where(present, qc, 1.0))
        lam = np.clip(lam, (pgam + 1.0) / 60.0e-6, (pgam + 1.0) / 1.0e-6)
        effc = math.gamma(pgam + 4.0) / math.gamma(pgam + 3.0) / lam / 2.0
        out["qc"] = np.where(present, effc, 25.0e-6).astype(np.float32)
    return out


def wsm6(q: Mapping[str, np.ndarray], rho: np.ndarray, t: np.ndarray, *, ice: str = "graupel") -> dict[str, np.ndarray]:
    """mp_wsm6_effectRad.F:50-184 for cloud water, ice and snow (with mp_driver.F's background and largest radii);
    3 / (2 lambda) of WSM6's fixed-intercept distributions for rain and graupel or hail."""

    r = np.asarray(rho, np.float64)
    out: dict[str, np.ndarray] = {}
    if "qc" in q:
        rqc = np.maximum(WSM6_R1, np.asarray(q["qc"], np.float64) * r)
        lam = np.cbrt(PI * WSM6_DENR / 6.0 * WSM6_NC0 / rqc)                   # pidnc nc0 / rqc
        bg, top = WSM6_BOUNDS["qc"]
        re = np.where(rqc > WSM6_R1, np.maximum(2.51e-6, np.minimum(1.5 / lam, top)), bg)
        out["qc"] = np.maximum(bg, np.minimum(re, top)).astype(np.float32)
    if "qi" in q:
        qi = np.asarray(q["qi"], np.float64)
        rqi = np.maximum(WSM6_R1, qi * r)
        ni = np.clip(5.38e7 * (r * np.maximum(qi, WSM6_R1)) ** 0.75, 1.0e3, 1.0e6)
        diai = 11.9 * np.sqrt(rqi / ni)
        bg, top = WSM6_BOUNDS["qi"]
        re = np.where(rqi > WSM6_R1, np.maximum(10.01e-6, np.minimum(0.75 * 0.163 * diai, top)), bg)
        out["qi"] = np.maximum(bg, np.minimum(re, top)).astype(np.float32)
    if "qs" in q:
        rqs = np.maximum(WSM6_R1, np.asarray(q["qs"], np.float64) * r)
        supcol = WSM6_T0C - np.asarray(t, np.float64)
        n0sfac = np.maximum(np.minimum(np.exp(WSM6_ALPHA * supcol), WSM6_N0SMAX / WSM6_N0S), 1.0)
        lam = np.sqrt(np.sqrt(PI * WSM6_DENS * WSM6_N0S * n0sfac / rqs))
        bg, top = WSM6_BOUNDS["qs"]
        re = np.where(rqs > WSM6_R1, np.maximum(25.0e-6, np.minimum(0.5 / lam, top)), bg)
        out["qs"] = np.maximum(bg, np.minimum(re, top)).astype(np.float32)
    n0g, deng = WSM6_GRAUPEL[ice]
    for name, pidn0 in (("qr", PI * WSM6_DENR * WSM6_N0R), ("qg", PI * deng * n0g)):
        if name in q:
            rq = np.maximum(WSM6_R1, np.asarray(q[name], np.float64) * r)
            lam = np.sqrt(np.sqrt(pidn0 / rq))                                 # (pidn0 / (rho q))^(1/4), :1549-1551
            out[name] = (1.5 / lam).astype(np.float32)
    return out


def thompson(q: Mapping[str, np.ndarray], n: Mapping[str, np.ndarray], rho: np.ndarray,
             t: np.ndarray) -> dict[str, np.ndarray]:
    """calc_effectRad (thompson.F:5229-5333) for cloud water, ice and snow, with the background radii of
    thompson.F:1420-1422; rain and graupel 3 / (2 lambda) of the scheme's distributions (mu = 0)."""

    r = np.asarray(rho, np.float64)
    out: dict[str, np.ndarray] = {}
    if "qc" in q:
        rc = np.maximum(THOMPSON_R1, np.asarray(q["qc"], np.float64) * r)
        nc = THOMPSON_NT_C
        inu = min(15, int(round(1000.0e6 / nc)) + 2)
        am_r = PI * THOMPSON_RHO_W / 6.0
        lam = np.cbrt(nc * am_r * THOMPSON_G_RATIO[inu - 1] / rc)
        re = np.where(rc > THOMPSON_R1, np.clip(0.5 * (3.0 + inu) / lam, 2.51e-6, 50.0e-6), THOMPSON_BG["qc"])
        out["qc"] = np.maximum(THOMPSON_BG["qc"], np.minimum(re, 50.0e-6)).astype(np.float32)
    if "qi" in q and "nci" in n:
        ri = np.maximum(THOMPSON_R1, np.asarray(q["qi"], np.float64) * r)
        ni = np.maximum(THOMPSON_R2, np.asarray(n["nci"], np.float64) * r)
        am_i = PI * THOMPSON_RHO_I / 6.0
        lam = np.cbrt(am_i * math.gamma(4.0) * ni / ri)                      # cig(2) oig1 = Gamma(4) / Gamma(1)
        ok = (ri > THOMPSON_R1) & (ni > THOMPSON_R2)
        re = np.where(ok, np.clip(1.5 / lam, 5.01e-6, 125.0e-6), THOMPSON_BG["qi"])
        out["qi"] = np.maximum(THOMPSON_BG["qi"], np.minimum(re, 125.0e-6)).astype(np.float32)
    if "qs" in q:
        rs = np.maximum(THOMPSON_R1, np.asarray(q["qs"], np.float64) * r)
        tc0 = np.minimum(-0.1, np.asarray(t, np.float64) - 273.15)
        smob = rs / THOMPSON_AM_S
        smo2 = smob                                                          # bm_s = 2: the reference moment itself
        a, b = THOMPSON_SA, THOMPSON_SB
        c = THOMPSON_BM_S + 1.0                                              # cse(1)
        loga = (a[0] + a[1] * tc0 + a[2] * c + a[3] * tc0 * c + a[4] * tc0 * tc0 + a[5] * c * c
                + a[6] * tc0 * tc0 * c + a[7] * tc0 * c * c + a[8] * tc0 ** 3 + a[9] * c ** 3)
        bb = (b[0] + b[1] * tc0 + b[2] * c + b[3] * tc0 * c + b[4] * tc0 * tc0 + b[5] * c * c
              + b[6] * tc0 * tc0 * c + b[7] * tc0 * c * c + b[8] * tc0 ** 3 + b[9] * c ** 3)
        smoc = 10.0 ** loga * smo2 ** bb
        re = np.where(rs > THOMPSON_R1, np.clip(0.5 * smoc / smob, 10.01e-6, 999.0e-6), THOMPSON_BG["qs"])
        out["qs"] = np.maximum(THOMPSON_BG["qs"], np.minimum(re, 999.0e-6)).astype(np.float32)
    if "qr" in q and "ncr" in n:
        rr = np.maximum(THOMPSON_R1, np.asarray(q["qr"], np.float64) * r)
        nr = np.maximum(THOMPSON_R2, np.asarray(n["ncr"], np.float64) * r)
        lam = np.cbrt(PI * THOMPSON_RHO_W / 6.0 * math.gamma(4.0) * nr / rr)  # am_r crg(3) org2
        out["qr"] = (1.5 / lam).astype(np.float32)
    if "qg" in q:
        rg = np.maximum(THOMPSON_R1, np.asarray(q["qg"], np.float64) * r)
        zans = np.clip(3.0 + 2.0 / 7.0 * (np.log10(np.maximum(1.0e-9, rg)) + 8.0), 2.0, 6.0)   # :1980-1983
        am_g = PI * THOMPSON_RHO_G / 6.0
        lam = np.sqrt(np.sqrt(10.0 ** zans * am_g * math.gamma(4.0) / rg))     # :1984-1985 with mu_g = 0
        out["qg"] = (1.5 / lam).astype(np.float32)
    return out


# ---------------------------------------------------------------- one frame
SCHEMES = {"morrison-mp10": "Morrison", "wsm6-mp6": "WSM6", "thompson-mp8": "Thompson"}
SOURCES = {
    "Morrison": "morrison.F:5001-5178 (its radiation radii, from the saved number concentrations)",
    "WSM6": "mp_wsm6_effectRad.F:50-184 (cloud water, ice, snow) and WSM6's fixed intercepts (rain, graupel or hail)",
    "Thompson": "thompson.F:5229-5333 calc_effectRad (cloud water, ice, snow) and its rain and graupel distributions",
    "fixed": "the fixed sizes of the clouds-from-above look",
}


def optics(qf: Any, species: list[str], rho: np.ndarray,
           temperature: np.ndarray | None) -> tuple[np.ndarray, Sizes]:
    """The visible extinction (1/m) on the scalar levels, the sum of 3 rho q / (2 rho_w r_e) over ``species`` with
    each radius by the frame's own scheme (``config.microphysics``), and how they were sized.

    One species at a time, and only in the cells that hold it (most of a domain is clear air), so a fine grid holds
    one species' arrays at once."""

    config = qf.config
    component = str(config.get("microphysics") or "")
    scheme = SCHEMES.get(component) or "fixed"
    ice = dense_ice(config)
    numbers: tuple[str, ...] = ()
    if scheme in ("Morrison", "Thompson") and qf.has("nc3d"):
        try:
            numbers = tuple(qf.species_names("nc3d"))
        except (FrameUnreadable, ValueError, KeyError):
            numbers = ()
    if scheme in ("WSM6", "Thompson") and temperature is None:
        raise FrameUnreadable(f"{qf.directory.name}: {scheme}'s snow size needs the air temperature")
    count = {"qi": "nci", "qs": "ncs", "qr": "ncr", "qg": "ncg"}
    ndcnst, where = droplet_number(qf.directory) if scheme == "Morrison" else (None, "")
    found = Sizes(scheme=scheme, words="", source=SOURCES[scheme])
    fixed_ones: list[str] = []
    flat_rho = np.asarray(rho, np.float32).reshape(-1)
    flat_t = None if temperature is None else np.asarray(temperature, np.float32).reshape(-1)
    beta = np.zeros(flat_rho.size, np.float32)
    for name in species:
        q = qf.species(name).reshape(-1)
        cells = np.flatnonzero(q > 0.0)            # a cell without the species adds nothing, whatever its size
        if cells.size == 0:
            continue
        qq, rr = q[cells], flat_rho[cells]
        tt = None if flat_t is None else flat_t[cells]
        number = {}
        if name in count and count[name] in numbers:
            number = {count[name]: qf.species(count[name], carrier="nc3d").reshape(-1)[cells]}
        if scheme == "Morrison":
            radius = morrison({name: qq}, number, rr, ndcnst=float(ndcnst), ice=ice).get(name)
        elif scheme == "WSM6":
            radius = wsm6({name: qq}, rr, tt, ice=ice).get(name)
        elif scheme == "Thompson":
            radius = thompson({name: qq}, number, rr, tt).get(name)
        else:
            radius = None
        if radius is None:
            radius = np.float32(FIXED[name])
            if scheme != "fixed" and name not in fixed_ones:
                fixed_ones.append(name)
        beta[cells] += qq / np.asarray(radius, np.float32)
        mean = mean_radius_um(qq, radius)
        if mean is not None:
            found.mean_um[name] = mean
        del q, qq, rr, tt, number, radius, cells
    beta = beta.reshape(np.shape(rho)) * (np.asarray(rho, np.float32) * np.float32(1.5 / RHO_W))
    if scheme == "Morrison":
        found.words = "particle sizes: Morrison's own, from its saved number concentrations"
        if "qc" in species:
            found.notes.append(f"cloud droplets {ndcnst:g} per cm³ ({where})")
    elif scheme == "WSM6":
        found.words = "particle sizes: WSM6's own (its fixed intercepts and radiation radii)"
    elif scheme == "Thompson":
        found.words = "particle sizes: Thompson's own (100 droplets per cm³)"
    else:
        found.words = "particle sizes fixed (" + ", ".join(f"{WORD[n]} {FIXED[n] * 1e6:g} µm" for n in species) + ")"
    if fixed_ones:
        found.notes.append("no saved number for " + ", ".join(WORD[n] for n in fixed_ones) + ", so a fixed size")
    return beta, found


__all__ = ["FIXED", "NDCNST_DEFAULT", "SCHEMES", "Sizes", "dense_ice", "droplet_number", "mean_radius_um", "morrison",
           "optics", "thompson", "wsm6"]
