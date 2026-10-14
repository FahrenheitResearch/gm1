"""Sounding files in each layout GM1 reads, made from GM1's own analytic sounding (no third-party data).

``measured()`` is the Make-your-own-environment sounding as a balloon would report it (pressure, height above sea
level, temperature, dewpoint, wind direction and speed), on a coarse set of levels; the writers below lay those
rows out as a University of Wyoming text listing, a SHARPpy file and tables with named columns.
"""

from __future__ import annotations

import math

from cm1core.cm1import import analytic_sounding as A
from cm1core.cm1import import sounding_formats as F

STATION_M = 300.0
KNOT = 0.514444
#: Heights above the ground of the levels kept (m): the ground, then coarser with height, to 24 km.
HEIGHTS = (0, 100, 300, 500, 800, 1000, 1400, 2000, 3000, 4000, 5000, 6000, 8000, 10000, 12000, 14000, 16000, 18000,
           20000, 22000, 24000)


def made(instability: float = 0.5, shear: float = 0.5) -> str:
    return A.generate(instability, shear).text


def profile(instability: float = 0.5, shear: float = 0.5) -> F.Profile:
    return F.parse_cm1(made(instability, shear))


def measured(top_m: float = 24000.0, instability: float = 0.5, shear: float = 0.5) -> list[dict[str, float]]:
    """One row per kept level: p (hPa), z (m above sea level), t and td (C), direction (deg), speed (m/s), u, v."""

    full = profile(instability, shear)
    pressures = F.pressures_pa(full)
    temperatures = F.temperatures_k(full)
    z = [0.0, *full.z_m]
    qv = [full.surface_qv_gkg / 1000.0, *(value / 1000.0 for value in full.qv_gkg)]
    u = [0.0, *full.u_ms]
    v = [0.0, *full.v_ms]
    rows = []
    for height in HEIGHTS:
        if height > top_m:
            break
        k = z.index(float(height))
        vapour = qv[k] * pressures[k] / (F.EPS + qv[k])
        log = math.log(vapour / 611.2)
        speed = math.hypot(u[k], v[k])
        rows.append({
            "p": pressures[k] / 100.0, "z": height + STATION_M, "t": temperatures[k] - 273.15,
            "td": 243.5 * log / (17.67 - log), "dir": (math.degrees(math.atan2(-u[k], -v[k])) + 360.0) % 360.0,
            "spd": speed, "u": u[k], "v": v[k], "agl": float(height), "theta": (full.surface_theta_k if k == 0
                                                                                else full.theta_k[k - 1]),
            "qv": qv[k] * 1000.0,
        })
    return rows


def wyoming(rows, knots: bool = False, page: bool = False) -> str:
    """The archive's listing: fixed columns seven characters wide; SKNT (knots) or SPED (m/s)."""

    rule = "-" * 77
    lines = [rule,
             "   PRES   HGHT   TEMP   DWPT   RELH   MIXR   DRCT   " + ("SKNT" if knots else "SPED")
             + "   THTA   THTE   THTV",
             "    hPa      m      C      C      %   g/kg    deg   " + ("knot" if knots else " m/s")
             + "      K      K      K ", rule]
    for row in rows:
        speed = row["spd"] / KNOT if knots else row["spd"]
        lines.append("%7.1f%7.0f%7.1f%7.1f%7.0f%7.2f%7.0f%7.1f%7.1f%7.1f%7.1f" % (
            row["p"], row["z"], row["t"], row["td"], 50.0, row["qv"], row["dir"], speed, row["theta"], 330.0,
            row["theta"] + 2.0))
    text = "\n".join(lines) + "\n"
    if page:
        text = ("<HTML><BODY><H2>Observations for Station 99999</H2>\n<PRE>\n" + text
                + "</PRE><H3>Station information and sounding indices</H3><PRE>\n"
                  "                         Station number: 99999\n</PRE></BODY></HTML>\n")
    return text


def sharppy(rows) -> str:
    lines = ["%TITLE%", " TEST   260101/0000", "",
             "   LEVEL       HGHT       TEMP       DWPT       WDIR       WSPD",
             "-------------------------------------------------------------------", "%RAW%"]
    for row in rows:
        lines.append("%8.2f,  %8.2f,  %8.2f,  %8.2f,  %8.2f,  %8.2f" % (
            row["p"], row["z"], row["t"], row["td"], row["dir"], row["spd"] / KNOT))
    return "\n".join([*lines, "%END%"]) + "\n"


def table(rows, header: str = "pressure_hPa,height_m,temperature_C,dewpoint_C,direction_deg,speed_kt",
          speed: float = 1.0 / KNOT, kelvin: bool = False, pascal: bool = False, delimiter: str = ",") -> str:
    lines = [header]
    for row in rows:
        lines.append(delimiter.join([
            "%.1f" % (row["p"] * (100.0 if pascal else 1.0)), "%.0f" % row["z"],
            "%.2f" % (row["t"] + (273.15 if kelvin else 0.0)), "%.2f" % (row["td"] + (273.15 if kelvin else 0.0)),
            "%.0f" % row["dir"], "%.2f" % (row["spd"] * speed)]))
    return "\n".join(lines) + "\n"


def table_uv(rows) -> str:
    lines = ["p [Pa]\tz [m]\tT [K]\tTd [K]\tu [m/s]\tv [m/s]"]
    for row in rows:
        lines.append("\t".join(["%.0f" % (row["p"] * 100.0), "%.0f" % row["z"], "%.2f" % (row["t"] + 273.15),
                                "%.2f" % (row["td"] + 273.15), "%.3f" % row["u"], "%.3f" % row["v"]]))
    return "\n".join(lines) + "\n"
