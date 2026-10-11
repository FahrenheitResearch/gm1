"""Make your own environment: a Weisman-Klemp sounding with a chosen instability and wind shear, as a CM1 file.

CM1 r21.1 has no sounding with a tunable CAPE and shear (its analytic ``isnd = 5`` is Weisman and Klemp's with
fixed numbers, and there is no McCaul-Weisman option; README.namelist:612-651).  So the guided "Make your own
environment" card writes the environment as an ordinary ``isnd = 7`` ``input_sounding``, which runs on the same
path as the observed Norman files.

**Temperature and moisture** are Weisman and Klemp (1982, MWR 110, 504-520) eqs. 1-2, as CM1's ``base.F``
``isnd = 5`` branch transcribes them (base.F:363-440; theta_0 = 300 K, theta_tr = 343 K, z_tr = 12 km,
T_tr = 213 K, surface pressure 1000 hPa, RH = 1 - 0.75 (z/z_tr)^1.25 below the tropopause and 0.25 above),
closed hydrostatically by the same 20-sweep iteration with CM1's ``rslf``.  The only change is the boundary-layer
mixing-ratio cap (CM1's ``qv_pbl`` = 14 g/kg), which the *instability* setting moves from 10 to 16 g/kg (plan
18.5 said 11-16; 10 lets the lowest setting reach "too stable", 300 J/kg, so the card can show that too).  At
14 g/kg the file's getcape CAPE and shear equal CM1's own ``isnd = 5, iwnd = 2`` base state's (1872 J/kg, 31.1 m/s,
measured).

**Wind** is 0-6 km shear of 2 to 40 m/s on the *shear* setting, ground-relative, calm at the ground:

* ``curved`` (the default): CM1's ``iwnd = 2`` shape (base.F:2321-2360), a quarter circle over 0-2 km and then
  straight to 6 km, scaled so the 0-6 km bulk shear is the chosen value (the quarter circle's radius keeps CM1's
  ratio of 7 to 31 m/s; at 31.8 m/s it is CM1's profile);
* ``straight``: one direction (west to east), linear over 0-6 km.

Above 6 km the wind is constant.

**The settings the card shows** are plain numbers the card can state: the instability setting is a CAPE target
(:data:`CAPE_RANGE_J_KG`, J/kg) and the shear setting a 0-6 km shear target (:data:`SHEAR_RANGE_M_S`, m/s).  The
mixing-ratio cap that gives the CAPE target is read from :data:`CAPE_TABLE`, the model's own getcape
(``sounding_report``) measured on the 1 km preset grid (40 levels of 500 m) for caps of 10 to 16 g/kg;
``tools/presets/calibrate_environment.py`` re-measures it and ``tests/test_analytic_sounding.py`` checks every
setting of a 5 x 5 grid lands within 10 % of its targets.  The numbers a page shows are always getcape's for the
written file (:func:`report`), never the target.

**The sentence** (:func:`regime`) is what Weisman and Klemp (1982, 1984) found across such environments, in plain
words; it is a rule of thumb, not a forecast.

Standard library and NumPy only; the getcape numbers need the installed initializer library.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

SCHEMA = "gpuwm-cm1.environment/v1"

# CM1 r21.1 constants (src/constants.F) and base.F's isnd = 5 numbers.
G = 9.81
RD = 287.04
CP = 1005.7
RV = 461.5
P00 = 1.0e5
REPS = RV / RD
Z_TROP = 12000.0
TH_TROP = 343.0
T_TROP = 213.0
TH_SFC = 300.0
P_SFC = 100000.0
#: CM1's own boundary-layer mixing-ratio cap (qv_pbl) and the range the instability setting moves it through.
QV_PBL_CM1 = 0.014
CAP_RANGE_G_KG = (10.0, 16.0)
#: What the two settings state, at 0 and at 1.
CAPE_RANGE_J_KG = (300.0, 3000.0)
#: The shear setting starts at 2 m/s, not 0: GM1's t=0 environment report (the Rust getcape transcription) refuses
#: air with no 0-6 km shear at all, because the Bunkers storm motion is undefined there, so a calm sounding would be
#: stopped at run start (measured 2026-09-24; CM1 itself runs it).  2 m/s is as good as calm for the storm.
SHEAR_RANGE_M_S = (2.0, 40.0)
#: iwnd = 2's quarter circle: radius 7 m/s against 31 m/s at 6 km (base.F:2325-2328).
CURVE_RATIO = 7.0 / 31.0
HODOGRAPHS = ("curved", "straight")
#: The levels written: every 100 m from 100 m to 24 km above the ground (above any shipped model top).
LEVEL_STEP_M = 100.0
TOP_M = 24000.0

#: getcape surface-based CAPE (J/kg) of this generator's sounding on the 1 km preset grid (nz = 40, dz = 500 m),
#: by the mixing-ratio cap (g/kg).  Measured with the initializer library's sounding_report
#: (tools/presets/calibrate_environment.py); the wind does not change CAPE.
CAPE_TABLE: tuple[tuple[float, float], ...] = (
    (10.0, 224.0), (10.25, 292.4), (10.5, 366.6), (10.75, 446.1), (11.0, 530.8), (11.25, 620.3), (11.5, 714.2),
    (11.75, 812.6), (12.0, 915.4), (12.25, 1022.1), (12.5, 1132.6), (12.75, 1247.0), (13.0, 1364.9),
    (13.25, 1486.6), (13.5, 1611.7), (13.75, 1740.3), (14.0, 1872.2), (14.25, 2007.5), (14.5, 2145.9),
    (14.75, 2285.7), (15.0, 2426.5), (15.25, 2568.6), (15.5, 2711.6), (15.75, 2855.7), (16.0, 3000.7),
)


class EnvironmentError(ValueError):
    """A setting outside 0-1, or an unknown hodograph."""


# ------------------------------------------------------------------ thermodynamics (CM1's forms)


def rslf(p: np.ndarray | float, t: np.ndarray | float) -> np.ndarray:
    """CM1's rslf (Bolton, with the min(es, p/2) clamp): saturation mixing ratio over liquid, kg/kg."""

    p = np.asarray(p, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    esl = np.minimum(611.2 * np.exp(17.67 * (t - 273.15) / (t - 29.65)), 0.5 * p)
    return (RD / RV) * esl / (p - esl)


def wk_theta(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    return np.where(z < Z_TROP, TH_SFC + (TH_TROP - TH_SFC) * (z / Z_TROP) ** 1.25,
                    TH_TROP * np.exp((G / (T_TROP * CP)) * (z - Z_TROP)))


def wk_rh(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    return np.where(z < Z_TROP, 1.0 - 0.75 * (z / Z_TROP) ** 1.25, 0.25)


def thermodynamics(z: np.ndarray, cap_kg_kg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(theta, qv, p) at heights ``z`` (m): base.F's isnd = 5 closure with the cap ``cap_kg_kg``."""

    z = np.asarray(z, dtype=np.float64)
    th = wk_theta(z)
    rh = wk_rh(z)
    pi_sfc = (P_SFC / P00) ** (RD / CP)
    qv_sfc = float(rslf(P_SFC, TH_SFC * pi_sfc))
    thv_sfc = TH_SFC * (1.0 + qv_sfc * REPS) / (1.0 + qv_sfc)
    qv = np.zeros_like(z)
    pi = np.empty_like(z)
    for _ in range(20):
        thv = th * (1.0 + REPS * qv) / (1.0 + qv)
        pi[0] = pi_sfc - G * z[0] / (CP * 0.5 * (thv_sfc + thv[0]))
        for k in range(1, len(z)):
            pi[k] = pi[k - 1] - G * (z[k] - z[k - 1]) / (CP * 0.5 * (thv[k] + thv[k - 1]))
        prs = P00 * pi ** (CP / RD)
        qv = np.minimum(rh * rslf(prs, th * pi), cap_kg_kg)
    return th, qv, prs


def wind(z: np.ndarray, shear_m_s: float, hodograph: str) -> tuple[np.ndarray, np.ndarray]:
    """Ground-relative (u, v) at ``z`` with 0-6 km bulk shear ``shear_m_s``."""

    z = np.asarray(z, dtype=np.float64)
    if hodograph == "straight":
        return shear_m_s * np.minimum(z, 6000.0) / 6000.0, np.zeros_like(z)
    top = shear_m_s / math.sqrt(1.0 + CURVE_RATIO ** 2)      # u at 6 km
    radius = CURVE_RATIO * top
    angle = 0.5 * math.pi * np.minimum(z, 2000.0) / 2000.0
    u = np.where(z <= 2000.0, radius - radius * np.cos(angle),
                 radius + (np.minimum(z, 6000.0) - 2000.0) * (top - radius) / 4000.0)
    v = np.where(z <= 2000.0, radius * np.sin(angle), radius)
    return u, v


# ------------------------------------------------------------------ the settings


def _check(name: str, value: float) -> float:
    value = float(value)
    if not (0.0 <= value <= 1.0) or math.isnan(value):
        raise EnvironmentError(f"--{name} must be between 0 and 1, not {value:g}")
    return value


def cape_target(instability: float) -> float:
    low, high = CAPE_RANGE_J_KG
    return low + (high - low) * _check("instability", instability)


def shear_target(shear: float) -> float:
    low, high = SHEAR_RANGE_M_S
    return low + (high - low) * _check("shear", shear)


def cap_for_cape(cape_j_kg: float) -> float:
    """The mixing-ratio cap (g/kg) whose getcape CAPE on the 1 km grid is ``cape_j_kg`` (from :data:`CAPE_TABLE`)."""

    caps = np.array([row[0] for row in CAPE_TABLE])
    capes = np.array([row[1] for row in CAPE_TABLE])
    if not np.all(np.diff(capes) > 0):
        raise EnvironmentError("the CAPE table is not calibrated (tools/presets/calibrate_environment.py)")
    return float(np.clip(np.interp(cape_j_kg, capes, caps), *CAP_RANGE_G_KG))


@dataclass(frozen=True)
class Environment:
    instability: float
    shear: float
    hodograph: str
    cap_g_kg: float
    cape_target_j_kg: float
    shear_target_m_s: float
    text: str

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.text.encode("ascii")).hexdigest()

    def command(self) -> list[str]:
        return ["preset", "environment", "--instability", f"{self.instability:g}", "--shear", f"{self.shear:g}",
                "--hodograph", self.hodograph]

    def record(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "instability": self.instability, "shear": self.shear, "hodograph": self.hodograph,
            "cape_target_j_kg": round(self.cape_target_j_kg, 1),
            "shear_0_6km_target_m_s": round(self.shear_target_m_s, 2),
            "boundary_layer_cap_g_kg": round(self.cap_g_kg, 4),
            "sha256": self.sha256,
            "method": ("Weisman and Klemp (1982) temperature and moisture as CM1 r21.1 base.F isnd = 5 builds them, "
                       "with the boundary-layer mixing-ratio cap set for the CAPE target; wind calm at the ground "
                       "with the chosen 0-6 km shear (curved: CM1's iwnd = 2 shape, scaled)"),
        }


def generate(instability: float, shear: float, hodograph: str = "curved") -> Environment:
    """The environment for two settings between 0 and 1, as CM1 ``isnd = 7`` text."""

    if hodograph not in HODOGRAPHS:
        raise EnvironmentError(f"--hodograph must be curved or straight, not {hodograph!r}")
    cape = cape_target(instability)
    shear_m_s = shear_target(shear)
    cap = cap_for_cape(cape)
    z = np.arange(LEVEL_STEP_M, TOP_M + 0.5 * LEVEL_STEP_M, LEVEL_STEP_M)
    th, qv, _ = thermodynamics(z, cap / 1000.0)
    u, v = wind(z, shear_m_s, hodograph)
    lines = ["%12.4f %14.6f %14.6f" % (P_SFC / 100.0, TH_SFC, cap)]
    for k in range(len(z)):
        lines.append("%12.4f %14.6f %14.6f %10.4f %10.4f" % (z[k], th[k], qv[k] * 1000.0, u[k], v[k]))
    return Environment(float(instability), float(shear), hodograph, cap, cape, shear_m_s, "\n".join(lines) + "\n")


# ------------------------------------------------------------------ numbers and words


def report(text: str, *, nz: int = 40, dz_m: float = 500.0, library: str | Path | None = None) -> dict[str, Any]:
    """getcape's numbers for the file on the model's own levels (the initializer library's sounding_report)."""

    from ..cartesian_door import resolve_initializer_library
    from ..errors import fail_fast
    from ..sounding_report import initial_sounding_report

    with fail_fast():
        path = resolve_initializer_library(None if library is None else Path(library))
        full = initial_sounding_report(path, text.encode("ascii"), isnd=7, iwnd=0, nz=nz, dz_m=dz_m)
    surface = full.get("surface_based") or {}
    return {
        "cape_j_kg": surface.get("cape_j_per_kg"),
        "cin_j_kg": surface.get("cin_j_per_kg"),
        "lcl_m": surface.get("lcl_m_agl"),
        "shear_0_6km_m_s": full.get("shear_0_6km_m_per_s"),
        "shear_0_1km_m_s": full.get("shear_0_1km_m_per_s"),
        "srh_0_1km_m2_s2": full.get("srh_0_1km_m2_per_s2"),
        "srh_0_3km_m2_s2": full.get("srh_0_3km_m2_per_s2"),
        "storm_motion_m_s": [full.get("bunkers_right_u_m_per_s"), full.get("bunkers_right_v_m_per_s")],
        "levels": {"nz": nz, "dz_m": dz_m},
        "basis": "measured: CM1 getcape (surface parcel) and the bulk shear on the model's own levels",
    }


#: The regime sentences, by id.
SENTENCES = {
    "stable": "Probably no storm: the air is too stable.",
    "single": "Short-lived single storms that rain themselves out.",
    "multicell": "Clusters of storms (multicells) along a spreading gust front.",
    "supercell": "Long-lived rotating storms (supercells) are likely.",
}
HODOGRAPH_SENTENCE = "A curved hodograph favours the right-moving storm; a straight one makes storms split in two."


def regime(cape_j_kg: float, shear_0_6km_m_s: float) -> str:
    """Weisman and Klemp's regimes in words: which kind of storm this CAPE and shear usually make."""

    if cape_j_kg < 500.0:
        return "stable"
    if shear_0_6km_m_s < 10.0:
        return "single"
    if shear_0_6km_m_s < 20.0 or cape_j_kg < 1000.0:
        return "multicell"
    return "supercell"


def describe(environment: Environment, numbers: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The card's words and numbers: the sentence from getcape's numbers when known, else from the targets."""

    cape = (numbers or {}).get("cape_j_kg")
    shear = (numbers or {}).get("shear_0_6km_m_s")
    measured = cape is not None and shear is not None
    if not measured:
        cape, shear = environment.cape_target_j_kg, environment.shear_target_m_s
    kind = regime(float(cape), float(shear))
    sentence = SENTENCES[kind]
    if kind == "supercell":
        sentence += " " + HODOGRAPH_SENTENCE
    return {"regime": kind, "sentence": sentence, "numbers_basis": "measured" if measured else "target"}


def write(environment: Environment, path: str | Path) -> Path:
    path = Path(path)
    path.write_text(environment.text, encoding="ascii")
    return path


__all__ = [
    "CAPE_RANGE_J_KG", "CAPE_TABLE", "CAP_RANGE_G_KG", "Environment", "EnvironmentError", "HODOGRAPHS",
    "HODOGRAPH_SENTENCE", "SENTENCES", "SHEAR_RANGE_M_S", "cap_for_cape", "cape_target", "describe", "generate",
    "regime", "report", "rslf", "shear_target", "thermodynamics", "wind", "write",
]
