"""The skew-T of one model column: its temperature and dewpoint over the base state, and its winds.

The panel grammar (skew-T log-p with isotherms, dry and moist adiabats and mixing-ratio lines, a barb strip) is the
viz 2.0 environment card's; the numbers are the frame's own:

* T = θ (the frame's ``th``, or ``th0 + th3d``) times the Exner function from the frame's pressure;
* the dewpoint from qv with CM1's getcape.F ``gettd`` formula;
* the base state: θ0 = th - th3d and p0 from pi0 = (prs/p00)^(rd/cp) - pp3d, both exact from the frame, or the
  frame's own base-state columns when it carries them; qv0 from the frame's ``qv0`` column, else the median qv of
  the run's first frame at each level (said on the picture), else no base-state dewpoint;
* winds ground-relative (u + umove, v + vmove), every 1 km.

The column is the one nearest the position asked for (km, run coordinates), or by default the strongest updraft on
the w level nearest 5 km.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from . import products as P
from . import style as S
from .facts import RunFacts
from .frame import FrameUnreadable, QuickFrame

P00, RD, CP, RV = 1.0e5, 287.04, 1005.7, 461.5
EPS = RD / RV
MS_TO_KT = 1.0 / 0.514444
T_COLOUR = "#ff5f57"
TD_COLOUR = "#34d399"
BASE_T = "#8b98ad"
BASE_TD = "#3d6b5a"
EDGE = "#1f2a40"


def dewpoint_k(p_pa: np.ndarray, qv: np.ndarray) -> np.ndarray:
    """getcape.F:583-594 (gettd)."""

    qv = np.maximum(np.asarray(qv, dtype=np.float64), 1.0e-10)
    el = np.log((qv / EPS) * np.asarray(p_pa, dtype=np.float64) / 100.0 / (1.0 + (qv / EPS)))
    return 273.15 + (243.5 * el - 440.8) / (19.48 - el)


def rslf(p, t):
    """cm1libs.F (Bolton, with the 171023 clamp)."""

    esl = 611.2 * np.exp(17.67 * (t - 273.15) / (t - 29.65))
    esl = np.minimum(esl, p * 0.5)
    return EPS * esl / (p - esl)


class SkewT:
    def __init__(self, ax, p_bot=1050.0, p_top=100.0, t_left=-40.0, t_right=50.0):
        self.ax = ax
        self.p_bot, self.p_top = p_bot, p_top
        self.y_bot, self.y_top = self.y(p_bot), self.y(p_top)
        bb = ax.get_position()
        fig = ax.figure
        w_px, h_px = bb.width * fig.get_figwidth(), bb.height * fig.get_figheight()
        self.skew = ((t_right - t_left) / w_px) / ((self.y_top - self.y_bot) / h_px)   # isotherms at 45 degrees
        self.t_left, self.t_right = t_left, t_right
        ax.set_xlim(t_left, t_right)
        ax.set_ylim(self.y_bot, self.y_top)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(EDGE)
            spine.set_linewidth(1.0)
            spine.set_visible(True)

    @staticmethod
    def y(p_hpa):
        return np.log(1000.0 / np.asarray(p_hpa, dtype=float))

    def x(self, t_c, p_hpa):
        return np.asarray(t_c, dtype=float) + self.skew * self.y(p_hpa)

    def background(self, fontsize: float) -> None:
        ax = self.ax
        p = np.geomspace(1050.0, self.p_top, 200)
        for t in range(-130, 60, 10):
            ax.plot(self.x(np.full_like(p, t), p), self.y(p), color="#18202f" if t else "#2b3650",
                    lw=0.8 if t else 1.0, ls="-" if t else (0, (5, 3)), zorder=1)
        for th in range(-30, 220, 10):
            tk = (th + 273.15) * (p / 1000.0) ** (RD / CP) - 273.15
            ax.plot(self.x(tk, p), self.y(p), color="#2a1f2c", lw=0.7, zorder=1)
        for t0 in range(-10, 40, 5):
            pp = np.geomspace(1050.0, 180.0, 160)
            tt = np.empty_like(pp)
            tt[0] = t0 + 273.15
            for i in range(1, pp.size):
                T, Pp = tt[i - 1], pp[i - 1] * 100.0
                rs = float(rslf(Pp, T))
                lv = 2.501e6
                gam = (RD * T + lv * rs) / (CP + lv * lv * rs * EPS / (RD * T * T)) / Pp
                tt[i] = T + gam * (pp[i] - pp[i - 1]) * 100.0
            ax.plot(self.x(tt - 273.15, pp), self.y(pp), color="#132b26", lw=0.8, ls=(0, (5, 3)), zorder=1)
        for w in (2, 4, 8, 12, 16, 20):
            pp = np.geomspace(1050.0, 600.0, 40)
            e = (w / 1000.0) * pp / (EPS + w / 1000.0)
            el = np.log(e)
            td = (243.5 * el - 440.8) / (19.48 - el)
            ax.plot(self.x(td, pp), self.y(pp), color="#1b2a38", lw=0.8, ls=(0, (1, 2.5)), zorder=1)
        for pl in (1000, 850, 700, 500, 400, 300, 250, 200, 150, 100):
            if not (self.p_top <= pl <= self.p_bot):
                continue
            ax.axhline(self.y(pl), color="#1d2740", lw=0.9, zorder=1)
            if self.p_top < pl < self.p_bot:
                ax.text(self.t_left + 0.8, self.y(pl) + 0.008, f"{pl} hPa" if pl == 850 else f"{pl}",
                        color=S.PALETTE["muted"], fontsize=fontsize, ha="left", va="bottom", zorder=6)
        for t in range(-30, 50, 10):
            xb = self.x(t, self.p_bot)
            if self.t_left + 3 < xb < self.t_right - 3:
                ax.text(xb, self.y(self.p_bot) + 0.008, f"{t}°", color=S.PALETTE["dim"], fontsize=fontsize * 0.9,
                        ha="center", va="bottom", zorder=6)


def _base_state(qf: QuickFrame, first: QuickFrame | None) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, str]:
    """(θ0, p0 (Pa), qv0 or None, where qv0 came from), columns on the scalar levels."""

    th0 = qf.base_column("th0")
    if th0 is None and qf.has("th") and qf.has("th3d"):
        th0 = np.median((np.asarray(qf.view("th")[:, :4, :4], np.float64)
                         - np.asarray(qf.view("th3d")[:, :4, :4], np.float64)).reshape(qf.nz, -1), axis=1)
    prs0 = qf.base_column("prs0")
    if prs0 is None:
        pi0 = qf.base_column("pi0")
        if pi0 is None and qf.has("prs") and qf.has("pp3d"):
            pi_total = (np.asarray(qf.view("prs")[:, :4, :4], np.float64) / P00) ** (RD / CP)
            pi0 = np.median((pi_total - np.asarray(qf.view("pp3d")[:, :4, :4], np.float64)).reshape(qf.nz, -1),
                            axis=1)
        prs0 = None if pi0 is None else P00 * pi0 ** (CP / RD)
    qv0 = qf.base_column("qv0")
    source = "the frame's qv0"
    if qv0 is None and first is not None and first.has_species("qv"):
        qv0 = np.median(first.species("qv").reshape(first.nz, -1), axis=1).astype(np.float64)
        source = f"the median qv of the {S.fmt_clock(first.t)} frame"
    elif qv0 is None:
        source = "not available"
    return th0, prs0, qv0, source


def column(qf: QuickFrame, facts: RunFacts, out: Path, *, x_km: float | None = None, y_km: float | None = None,
           first: QuickFrame | None = None) -> dict[str, Any]:
    ox, oy = facts.origin_m(qf.nx, qf.ny)
    k5 = qf.level_near(5000.0, faces=True)
    w5 = qf.get("w3d", k5)
    if x_km is None or y_km is None:
        j, i = np.unravel_index(int(np.argmax(w5)), w5.shape)
        chosen = f"the strongest w at {S.fmt_height(float(qf.zf[k5]))}"
    else:
        i = int(np.clip(math.floor((x_km * 1000.0 - ox) / qf.dx), 0, qf.nx - 1))
        j = int(np.clip(math.floor((y_km * 1000.0 - oy) / qf.dy), 0, qf.ny - 1))
        chosen = "the column asked for"
    xk = (ox + (i + 0.5) * qf.dx) / 1000.0
    yk = (oy + (j + 0.5) * qf.dy) / 1000.0
    zh = qf.zh
    prs = np.asarray(qf.view("prs")[:, j, i], dtype=np.float64)
    if qf.has("th"):
        th = np.asarray(qf.view("th")[:, j, i], dtype=np.float64)
    elif qf.base_column("th0") is not None:
        th = np.asarray(qf.view("th3d")[:, j, i], dtype=np.float64) + qf.base_column("th0")
    else:
        raise FrameUnreadable(f"{qf.directory.name} has neither th nor th3d with the base-state th0, so the "
                              "column's temperature is unknown")
    qv = qf.species("qv", (slice(None), j, i)).astype(np.float64)
    u = 0.5 * (np.asarray(qf.view("u3d")[:, j, i], np.float64) + np.asarray(qf.view("u3d")[:, j, i + 1], np.float64))
    v = 0.5 * (np.asarray(qf.view("v3d")[:, j, i], np.float64) + np.asarray(qf.view("v3d")[:, j + 1, i], np.float64))
    ug, vg = u + qf.umove, v + qf.vmove
    t = th * (prs / P00) ** (RD / CP)
    td = dewpoint_k(prs, qv)
    th0, prs0, qv0, qv0_source = _base_state(qf, first)

    W, H = P.W, P.H
    fig = S.figure((W, H))
    fs = 10.5
    ax = fig.add_axes([0.042, 0.20, 0.40, 0.625])
    ax.set_facecolor("#060a12")
    sk = SkewT(ax)
    sk.background(fs)
    keep = prs / 100.0 >= 100.0
    if th0 is not None and prs0 is not None:
        t0 = th0 * (prs0 / P00) ** (RD / CP)
        k0 = prs0 / 100.0 >= 100.0
        ax.plot(sk.x(t0[k0] - 273.15, prs0[k0] / 100.0), sk.y(prs0[k0] / 100.0), color=BASE_T, lw=1.6,
                ls=(0, (4, 3)), zorder=3)
        if qv0 is not None:
            ax.plot(sk.x(dewpoint_k(prs0, qv0)[k0] - 273.15, prs0[k0] / 100.0), sk.y(prs0[k0] / 100.0),
                    color=BASE_TD, lw=1.6, ls=(0, (4, 3)), zorder=3)
    for series, colour in ((t - 273.15, T_COLOUR), (td - 273.15, TD_COLOUR)):
        ax.plot(sk.x(series[keep], prs[keep] / 100.0), sk.y(prs[keep] / 100.0), color=colour, lw=3.0, zorder=4,
                solid_capstyle="round", solid_joinstyle="round")
    axb = fig.add_axes([0.445, 0.20, 0.045, 0.625])
    axb.set_xlim(0, 1)
    axb.set_ylim(*ax.get_ylim())
    axb.axis("off")
    bz = np.arange(0.0, 17000.0, 1000.0)
    bz = bz[(bz >= zh[0]) & (bz <= zh[-1])]
    bp = np.interp(bz, zh, prs / 100.0)
    ok = bp >= 102.0
    bz, bp = bz[ok], bp[ok]
    if bz.size:
        axb.plot([0.30, 0.30], [sk.y(bp[0]), sk.y(bp[-1])], color="#1d2740", lw=1.0, zorder=1)
        axb.barbs(np.full(bz.size, 0.30), sk.y(bp), np.interp(bz, zh, ug) * MS_TO_KT,
                  np.interp(bz, zh, vg) * MS_TO_KT, length=6.0, color="#cfd8e6", lw=1.1, zorder=3,
                  sizes=dict(emptybarb=0.15, spacing=0.17, height=0.42), pivot="middle")
    for hk in (1, 3, 6, 9, 12, 15):
        ph = float(np.interp(hk * 1000.0, zh, prs / 100.0))
        if ph >= 103.0 and hk * 1000.0 <= zh[-1]:
            axb.text(1.0, sk.y(ph), f"{hk} km", color=S.PALETTE["muted"], fontsize=fs * 0.9, ha="right",
                     va="center")
    base = S.fig_to_rgb(fig)

    ov = S.Overlay((W, H))
    where = f"x {S.fmt_num(xk, 1)} km, y {S.fmt_num(yk, 1)} km"
    ov.title("Column skew-T", S.data_line(facts.name, where, chosen, f"{S.fmt_grid(qf.dx)} grid"))
    ov.clock(qf.t)
    notes = S.data_line("T from the frame's θ and pressure; dewpoint from its qv (CM1 getcape.F gettd)",
                        f"dashed: the base state (qv0: {qv0_source})", "barbs: ground-relative wind in knots")
    ov.footer(facts.footer(t=qf.t, level="the whole column", wind="ground-relative", notes=notes))
    ov.mark()
    sy = int(0.175 * H)
    ov.text((int(0.042 * W), sy - ov.U(6)), "skew-T log-p: this column (solid) over the base state (dashed)", "label",
            colour="muted", anchor="ls")
    ov.text((int(0.4675 * W), sy - ov.U(6)), "wind (kt)", "micro", colour="muted", anchor="ms")
    key_y = int(0.86 * H)
    kx = int(0.044 * W)
    for label_text, colour in (("column temperature", T_COLOUR), ("column dewpoint", TD_COLOUR),
                               ("base-state temperature", BASE_T), ("base-state dewpoint", BASE_TD)):
        ov.rule(kx, key_y - ov.U(6), ov.U(34), thickness=3, colour=colour)
        ov.text((kx + ov.U(44), key_y), label_text, "micro", colour="data", anchor="ls")
        kx += ov.U(44) + ov.measure(label_text, "micro")[0] + ov.U(36)
    wcol = np.asarray(qf.view("w3d")[:, j, i], dtype=np.float64)
    kw = int(np.argmax(wcol))
    rows = [("column (x, y in the run's km)", where),
            (f"w at {S.fmt_height(float(qf.zf[k5]))}", f"{S.fmt_num(float(w5[j, i]), 1)} m/s"),
            ("strongest w in the column", f"{S.fmt_num(float(wcol.max()), 1)} m/s at {S.fmt_height(float(qf.zf[kw]))}")]
    vertical = [rows[1][1], rows[2][1]]
    if qf.has_species("qc"):
        qc = qf.species("qc", (slice(None), j, i))
        sat = np.where(qc > 1.0e-5)[0]
        rows.append(("cloud water above 0.01 g/kg",
                     f"{S.fmt_height(float(zh[sat[0]]))} to {S.fmt_height(float(zh[sat[-1]]))}" if sat.size else "none"))
    if th0 is not None:
        rows.append((f"θ{S.PRIME} at the lowest level ({S.fmt_height(float(zh[0]))})",
                     f"{S.fmt_num(float(th[0] - th0[0]), 1)} K"))
    rows.append(("wind at the lowest level", f"{math.hypot(ug[0], vg[0]):.0f} m/s ground-relative"))
    ov.readout((int(0.56 * W), int(0.20 * H)), rows, vertical=vertical)
    P.save(ov, base, out)
    return {"kind": "column", "time_s": qf.t, "frame": qf.key, "i": int(i) + 1, "j": int(j) + 1, "x_km": xk,
            "y_km": yk, "chosen": chosen, "w5": float(w5[j, i]), "w_max_column": float(wcol.max()),
            "qv0_source": qv0_source, "file": out.name}


__all__ = ["SkewT", "column", "dewpoint_k"]
