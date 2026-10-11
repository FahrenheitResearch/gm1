"""The storm's environment: the skew-T and hodograph of the air the run starts in, with its CAPE and shear.

Drawn once per run, from the run's own records:

* the profile is the earliest saved frame's undisturbed air: the median of a 4 x 4 block of columns in the
  domain's south-west corner (the storm starts in the middle), temperature from theta and pressure, the dewpoint
  from qv with CM1's getcape.F ``gettd``, winds ground-relative (u + umove, v + vmove);
* the numbers are the ones the run itself computed before it started (CM1's getcape.F three parcels, the bulk
  shear, the Bunkers storm motion and the storm-relative helicity: ``cm1core.sounding_report``), read from the
  importer's report (``import-report.json`` ``sounding.environment``) or, for a run made another way, from a
  segment's receipt (``initial_sounding_parameters``).  A run without either shows the profile with no numbers,
  and says so.

The skew-T panel is :mod:`.skewt`'s; the hodograph is drawn with the same colours.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from . import products as P
from . import style as S
from .frame import FrameUnreadable, QuickFrame
from .skewt import BASE_T, CP, EDGE, P00, RD, SkewT, TD_COLOUR, T_COLOUR, dewpoint_k

PARAMETERS_SCHEMA = "gm1.sounding-parameters/v1"
HODO_COLOURS = ((0.0, 1000.0, "#ff5f57"), (1000.0, 3000.0, "#ffb347"), (3000.0, 6000.0, "#34d399"),
                (6000.0, 9000.0, "#4cc3ff"), (9000.0, 12000.0, "#b9a8ff"))


def _read(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def parameters(root: Path, segments: list[Path] | None = None) -> tuple[dict[str, Any] | None, str]:
    """The run's own t=0 environment numbers and where they came from."""

    report = _read(root / "import-report.json")
    if isinstance(report, dict):
        env = (report.get("sounding") or {}).get("environment")
        if isinstance(env, dict) and env.get("schema") == PARAMETERS_SCHEMA:
            return env, "the importer's report"
    for segment in reversed(segments or []):
        receipt = _read(segment / "receipt.json")
        env = receipt.get("initial_sounding_parameters") if isinstance(receipt, dict) else None
        if isinstance(env, dict) and env.get("schema") == PARAMETERS_SCHEMA:
            return env, "the run's receipt"
    receipt = _read(root / "receipt.json")
    env = receipt.get("initial_sounding_parameters") if isinstance(receipt, dict) else None
    if isinstance(env, dict) and env.get("schema") == PARAMETERS_SCHEMA:
        return env, "the run's receipt"
    return None, "no record of the run's own numbers"


def air_varies(root: Path) -> bool:
    """Whether the run starts from a 3D state that differs from place to place (the ArWen/WRF chain), so no
    single column is "the" environment.  CM1's horizontally homogeneous base state (base.F) is the same in
    every column; the chain's initial state is not.

    A chain run is one whose deck has a ``[chain]`` table: ``gpuwm-cm1 chain new`` puts the prepared chain in
    the run's ``chain/`` folder, but a deck written by hand for tools/arwen_chain_prepare.py names its
    ``chain.prepared_directory`` anywhere, so the deck is the authority and the folder the fallback."""

    import tomllib

    root = Path(root)
    if (root / "chain" / "chain-receipt.json").is_file():
        return True
    try:
        deck = tomllib.loads((root / "deck.toml").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError):
        return False
    return isinstance(deck.get("chain"), dict)


def profile(qf: QuickFrame) -> dict[str, np.ndarray]:
    """The corner columns' median profile: z (m), p (Pa), T and Td (K), u and v (m/s, ground-relative)."""

    block = (slice(None), slice(0, min(4, qf.ny)), slice(0, min(4, qf.nx)))

    def med(values: Any) -> np.ndarray:
        a = np.asarray(values, dtype=np.float64)
        return np.median(a.reshape(a.shape[0], -1), axis=1)

    if not qf.has("prs"):
        raise FrameUnreadable(f"{qf.directory.name} has no pressure, so no skew-T")
    p = med(qf.view("prs")[block])
    if qf.has("th"):
        th = med(qf.view("th")[block])
    elif qf.base_column("th0") is not None and qf.has("th3d"):
        th = med(qf.view("th3d")[block]) + qf.base_column("th0")
    else:
        raise FrameUnreadable(f"{qf.directory.name} has neither theta nor theta' with the base state")
    qv = med(qf.species("qv", block)) if qf.has_species("qv") else np.full_like(p, 1.0e-10)
    u = med(0.5 * (np.asarray(qf.view("u3d")[:, 0:4, 0:4], np.float64)
                   + np.asarray(qf.view("u3d")[:, 0:4, 1:5], np.float64))) + qf.umove
    v = med(0.5 * (np.asarray(qf.view("v3d")[:, 0:4, 0:4], np.float64)
                   + np.asarray(qf.view("v3d")[:, 1:5, 0:4], np.float64))) + qf.vmove
    t = th * (p / P00) ** (RD / CP)
    return {"z": np.asarray(qf.zh, np.float64), "p": p, "t": t, "td": dewpoint_k(p, qv), "u": u, "v": v, "qv": qv}


def _num(env: Mapping[str, Any] | None, *path: str) -> float | None:
    value: Any = env
    for key in path:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _hodograph(fig: Any, rect: tuple[float, float, float, float], prof: Mapping[str, np.ndarray],
               env: Mapping[str, Any] | None, fs: float) -> None:
    from matplotlib.patches import Circle

    z, u, v = prof["z"], prof["u"], prof["v"]
    ax = fig.add_axes(rect)
    ax.set_facecolor("#060a12")
    keep = z <= 12000.0
    speed = float(np.nanmax(np.hypot(u[keep], v[keep]))) if keep.any() else 10.0
    for key in ("bunkers_right_u_m_per_s", "bunkers_left_u_m_per_s"):
        if _num(env, key) is not None:
            speed = max(speed, math.hypot(_num(env, key) or 0.0,
                                          _num(env, key.replace("_u_", "_v_")) or 0.0))
    lim = 10.0 * math.ceil((speed + 5.0) / 10.0)
    for r in range(10, int(lim) + 1, 10):
        ax.add_patch(Circle((0, 0), r, fill=False, color="#1d2740", lw=0.9, zorder=1))
        ax.text(r * 0.707 + 0.6, -r * 0.707 - 0.6, f"{r}", color=S.PALETTE["dim"], fontsize=fs * 0.85,
                ha="left", va="top", zorder=2)
    ax.axhline(0, color="#1d2740", lw=0.9, zorder=1)
    ax.axvline(0, color="#1d2740", lw=0.9, zorder=1)
    for z0, z1, colour in HODO_COLOURS:
        sel = (z >= z0) & (z <= z1)
        if sel.sum() >= 1:
            # join each band to the next band's first point, so the line is continuous
            idx = np.where(sel)[0]
            idx = np.append(idx, min(idx[-1] + 1, len(z) - 1))
            ax.plot(u[idx], v[idx], color=colour, lw=3.0, solid_capstyle="round", zorder=3)
    for hk in (1, 3, 6):
        if z[0] <= hk * 1000.0 <= z[-1]:
            uu, vv = float(np.interp(hk * 1000.0, z, u)), float(np.interp(hk * 1000.0, z, v))
            ax.plot([uu], [vv], "o", color="#cfd8e6", ms=4, zorder=4)
            ax.text(uu + 1.0, vv + 1.0, f"{hk} km", color=S.PALETTE["muted"], fontsize=fs * 0.85, zorder=5)
    ur, vr = _num(env, "bunkers_right_u_m_per_s"), _num(env, "bunkers_right_v_m_per_s")
    if ur is not None and vr is not None:
        ax.plot([ur], [vr], marker="+", color=S.PALETTE["gold"], ms=14, mew=2.4, zorder=6)
        ax.text(ur + 1.2, vr - 1.2, "right mover", color=S.PALETTE["gold"], fontsize=fs * 0.9, va="top", zorder=6)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(EDGE)


def rows(env: Mapping[str, Any] | None) -> list[tuple[str, str]]:
    """The readout: the parcel and wind numbers the run computed, in plain words."""

    if env is None:
        return []
    out = []
    ml = env.get("mixed_layer") if isinstance(env.get("mixed_layer"), Mapping) else {}
    mu = env.get("most_unstable") if isinstance(env.get("most_unstable"), Mapping) else {}
    if _num(ml, "cape_j_per_kg") is not None:
        out.append(("CAPE, mixed layer (energy for rising air)",
                    f"{_num(ml, 'cape_j_per_kg'):,.0f} J/kg"))
    if _num(mu, "cape_j_per_kg") is not None:
        out.append(("CAPE, most unstable", f"{_num(mu, 'cape_j_per_kg'):,.0f} J/kg"))
    if _num(ml, "cin_j_per_kg") is not None:
        out.append(("CIN, mixed layer (the lid)", f"{_num(ml, 'cin_j_per_kg'):,.0f} J/kg"))
    if _num(ml, "lcl_m_agl") is not None:
        out.append(("cloud base (LCL), mixed layer", S.fmt_height(_num(ml, "lcl_m_agl") or 0.0)))
    s1, s6 = _num(env, "shear_0_1km_m_per_s"), _num(env, "shear_0_6km_m_per_s")
    if s6 is not None:
        out.append(("wind shear 0–6 km (the wind's change with height)", f"{s6:.0f} m/s"))
    if s1 is not None:
        out.append(("wind shear 0–1 km", f"{s1:.0f} m/s"))
    h1, h3 = _num(env, "srh_0_1km_m2_per_s2"), _num(env, "srh_0_3km_m2_per_s2")
    if h1 is not None and h3 is not None:
        out.append(("helicity 0–1 | 0–3 km, storm-relative", f"{h1:.0f} | {h3:.0f} m² s⁻²"))
    ur, vr = _num(env, "bunkers_right_u_m_per_s"), _num(env, "bunkers_right_v_m_per_s")
    if ur is not None and vr is not None:
        direction = (math.degrees(math.atan2(-ur, -vr)) + 360.0) % 360.0
        out.append((f"right-moving storm, ground-relative, from {direction:03.0f}°",
                    f"{math.hypot(ur, vr):.0f} m/s"))
    return out


def draw(qf: QuickFrame, facts: Any, out: Path, env: Mapping[str, Any] | None, source: str,
         varies: bool = False) -> dict[str, Any]:
    prof = profile(qf)
    W, H = P.W, P.H
    fig = S.figure((W, H))
    fs = 10.5
    ax = fig.add_axes([0.042, 0.20, 0.36, 0.625])
    ax.set_facecolor("#060a12")
    sk = SkewT(ax)
    sk.background(fs)
    p_hpa = prof["p"] / 100.0
    keep = p_hpa >= 100.0
    for series, colour in ((prof["t"] - 273.15, T_COLOUR), (prof["td"] - 273.15, TD_COLOUR)):
        ax.plot(sk.x(series[keep], p_hpa[keep]), sk.y(p_hpa[keep]), color=colour, lw=3.0, zorder=4,
                solid_capstyle="round", solid_joinstyle="round")
    lcl = _num((env or {}).get("mixed_layer") if env else None, "lcl_m_agl")
    if lcl is not None and prof["z"][0] <= lcl <= prof["z"][-1]:
        pl = float(np.interp(lcl, prof["z"], p_hpa))
        ax.plot([sk.x(-40, pl), sk.x(50, pl)], [sk.y(pl), sk.y(pl)], color=BASE_T, lw=1.0, ls=(0, (2, 3)), zorder=3)
    side = 560  # px: a square hodograph
    _hodograph(fig, (0.425, 0.275, side / W, side / H), prof, env, fs)
    base = S.fig_to_rgb(fig)
    ov = S.Overlay((W, H))
    ov.title("The storm's environment", S.data_line(facts.name, "the air the storm starts in",
                                                    f"{S.fmt_grid(qf.dx)} grid"))
    ov.clock(qf.t)
    where = ("south-west corner columns (this run's air differs from place to place)" if varies
             else "undisturbed corner columns")
    notes = S.data_line(f"profile: the {S.fmt_clock(qf.t)} frame's {where}",
                        "dewpoint from qv (CM1 getcape.F gettd)",
                        f"numbers: {source}" if env is not None else "the run recorded no CAPE or shear")
    ov.footer(facts.footer(t=qf.t, level="the whole column", wind="ground-relative", notes=notes))
    ov.mark()
    sy = int(0.175 * H)
    ov.text((int(0.042 * W), sy - ov.U(6)), "skew-T log-p: temperature (red) and dewpoint (green)", "label",
            colour="muted", anchor="ls")
    ov.text((int(0.425 * W), sy - ov.U(6)), "hodograph: wind by height, ground-relative (m/s)", "label",
            colour="muted", anchor="ls")
    key_y = int(0.76 * H)
    kx = int(0.425 * W)
    for label_text, colour in (("0–1 km", "#ff5f57"), ("1–3", "#ffb347"), ("3–6", "#34d399"), ("6–9", "#4cc3ff"),
                               ("9–12", "#b9a8ff")):
        ov.rule(kx, key_y - ov.U(6), ov.U(26), thickness=3, colour=colour)
        ov.text((kx + ov.U(32), key_y), label_text, "micro", colour="data", anchor="ls")
        kx += ov.U(32) + ov.measure(label_text, "micro")[0] + ov.U(22)
    readout = rows(env)
    if readout:
        ov.readout((int(0.74 * W), int(0.19 * H)), readout[:9])
    else:
        ov.text((int(0.715 * W), int(0.24 * H)), "the run recorded no CAPE or shear numbers", "label",
                colour="muted", anchor="ls")
    P.save(ov, base, out)
    facts_out: dict[str, Any] = {"source": source, "profile_time_s": qf.t, "air_varies": bool(varies)}
    if env is not None:
        ml = env.get("mixed_layer") or {}
        facts_out.update({"cape_ml_j_kg": _num(ml, "cape_j_per_kg"), "cin_ml_j_kg": _num(ml, "cin_j_per_kg"),
                          "lcl_ml_m": _num(ml, "lcl_m_agl"),
                          "shear_0_1km_m_s": _num(env, "shear_0_1km_m_per_s"),
                          "shear_0_6km_m_s": _num(env, "shear_0_6km_m_per_s"),
                          "srh_0_1km": _num(env, "srh_0_1km_m2_per_s2"),
                          "srh_0_3km": _num(env, "srh_0_3km_m2_per_s2"),
                          "bunkers_right_m_s": [_num(env, "bunkers_right_u_m_per_s"),
                                                _num(env, "bunkers_right_v_m_per_s")]})
    return facts_out


__all__ = ["air_varies", "draw", "parameters", "profile", "rows"]
