// Time-series small multiples as inline SVG: one y-axis per chart, GM1 solid, CM1's
// own curve dashed where the run is CM1's stock storm, a value at the end of each line, a gold line at the time
// shown, and a hover cursor with both values. The data is the quick-look worker's series.json
// (gm1.quicklook-series/v1): domain extremes at every saved frame.

import { h, esc, num, stormClock } from "./core.js";

const W = 320, H = 150, L = 38, R = 36, T = 8, B = 20;

function niceStep(span, n) {
  const raw = span / Math.max(1, n);
  const p = Math.pow(10, Math.floor(Math.log10(raw)));
  const m = raw / p;
  return (m < 1.5 ? 1 : m < 3.5 ? 2 : m < 7.5 ? 5 : 10) * p;
}

// fromZero false: the value axis keeps to the data (a pressure near 1000 hPa), not stretched down to zero.
function extent(series, fromZero = true) {
  let lo = Infinity, hi = -Infinity;
  for (const s of series) for (const [, v] of s) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
  if (!Number.isFinite(lo)) { lo = 0; hi = 1; }
  if (fromZero && lo > 0) lo = 0;
  if (fromZero && hi < 0) hi = 0;
  const step = niceStep(hi - lo || 1, 4);
  return { lo: Math.floor(lo / step) * step, hi: Math.ceil(hi / step) * step, step };
}

function digits(step) {
  return step >= 1 ? 0 : Math.min(4, Math.ceil(-Math.log10(step)));
}

// Minutes between labelled ticks on the time axis, for a span of `tmax` seconds.
function minuteStep(tmax) {
  for (const m of [5, 10, 15, 30, 60, 120, 180, 360]) if (tmax / 60 / m <= 5) return m;
  return 720;
}

// Hours between labelled ticks, for a long run charted in hours.
function hourStep(tmax) {
  for (const hr of [6, 12, 24, 48]) if (tmax / 3600 / hr <= 5) return hr;
  return 96;
}

export function legend(labelA, labelB) {
  return h("div", { class: "legend" },
    h("span", {}, h("span", { class: "sw a" }), labelA),
    labelB ? h("span", {}, h("span", { class: "sw b" }), labelB) : null);
}

// spec: {label, unit, gm1: [[t, v]], cm1: [[t, v]], zero}; opts: {tmax, now, labelA, labelB, hours}
export function lineChart(spec, opts = {}) {
  const a = spec.gm1 || [], b = spec.cm1 || [];
  const last = Math.max(0, ...a.map(([t]) => t), ...b.map(([t]) => t), opts.now || 0);
  const tmax = Math.max(60, opts.tmax || 0, last);
  const { lo, hi, step } = extent([a, b], spec.zero !== false);
  const nd = digits(step);
  const x = (t) => L + (W - L - R) * (t / tmax);
  const y = (v) => T + (H - T - B) * (1 - (v - lo) / (hi - lo || 1));
  const parts = [];
  for (let v = lo; v <= hi + step / 2; v += step) {
    parts.push(`<line class="grid" x1="${L}" x2="${W - R}" y1="${y(v).toFixed(1)}" y2="${y(v).toFixed(1)}"/>`);
    parts.push(`<text x="${L - 5}" y="${(y(v) + 3).toFixed(1)}" text-anchor="end">${esc(num(v, nd))}</text>`);
  }
  // A long run's time axis is in hours (series.time_unit "h"); any other in minutes.
  const every = opts.hours ? hourStep(tmax) * 3600 : minuteStep(tmax) * 60;
  for (let t = 0; t <= tmax + 1e-6; t += every) {
    parts.push(`<text x="${x(t).toFixed(1)}" y="${H - 5}" text-anchor="middle">${Math.round(t / (opts.hours ? 3600 : 60))}</text>`);
  }
  parts.push(`<line class="axis" x1="${L}" x2="${W - R}" y1="${H - B}" y2="${H - B}"/>`);
  const path = (s) => s.map(([t, v], i) => `${i ? "L" : "M"}${x(t).toFixed(1)},${y(v).toFixed(1)}`).join("");
  if (opts.now !== undefined && opts.now !== null) {
    parts.push(`<line class="now" x1="${x(opts.now).toFixed(1)}" x2="${x(opts.now).toFixed(1)}" y1="${T}" y2="${H - B}"/>`);
  }
  if (b.length) parts.push(`<path class="b" d="${path(b)}"/>`);
  if (a.length) parts.push(`<path class="a" d="${path(a)}"/>`);
  const ends = [];
  for (const [s, cls] of [[a, "a"], [b, "b"]]) {
    if (!s.length) continue;
    const [t, v] = s[s.length - 1];
    parts.push(`<circle class="dot-${cls}" cx="${x(t).toFixed(1)}" cy="${y(v).toFixed(1)}" r="3"/>`);
    ends.push({ x: x(t), y: y(v), text: num(v, nd) });
  }
  // end labels, kept 10 px apart
  ends.sort((p, q) => p.y - q.y);
  for (let i = 1; i < ends.length; i++) {
    if (Math.abs(ends[i].x - ends[i - 1].x) < 30 && ends[i].y - ends[i - 1].y < 10) ends[i].y = ends[i - 1].y + 10;
  }
  for (const e of ends) {
    parts.push(`<text class="end" x="${Math.min(e.x + 6, W - 2).toFixed(1)}" y="${(e.y + 3).toFixed(1)}">` +
      `${esc(e.text)}</text>`);
  }
  parts.push(`<line class="cursor" x1="0" x2="0" y1="${T}" y2="${H - B}" visibility="hidden"/>`);
  parts.push(`<rect x="${L}" y="${T}" width="${W - L - R}" height="${H - T - B}" fill="transparent"/>`);

  const box = h("div", { class: "chart" },
    h("div", { class: "title" }, `${spec.label} (${spec.unit})`),
    h("div", { html: `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(spec.label)}">${parts.join("")}</svg>` }));
  const tip = h("div", { class: "tip" });
  box.append(tip);
  const svg = box.querySelector("svg");
  const cursor = svg.querySelector(".cursor");
  const times = [...new Set([...a, ...b].map(([t]) => t))].sort((p, q) => p - q);
  const at = (s, t) => { const hit = s.find(([tt]) => tt === t); return hit ? num(hit[1], nd + 1) : "none"; };
  svg.addEventListener("mousemove", (ev) => {
    if (!times.length) return;
    const r = svg.getBoundingClientRect();
    const px = (ev.clientX - r.left) * (W / r.width);
    const t = times.reduce((best, tt) => (Math.abs(x(tt) - px) < Math.abs(x(best) - px) ? tt : best), times[0]);
    cursor.setAttribute("x1", x(t).toFixed(1));
    cursor.setAttribute("x2", x(t).toFixed(1));
    cursor.setAttribute("visibility", "visible");
    tip.textContent = `${stormClock(t, opts.hours ? 172800 : 0)}   ${opts.labelA || "GM1"} ${at(a, t)}` +
      (b.length ? `   ${opts.labelB || "CM1"} ${at(b, t)}` : "") + ` ${spec.unit}`;
    tip.style.display = "block";
    const left = (x(t) / W) * r.width;
    tip.style.left = `${Math.min(Math.max(0, left - 80), Math.max(0, r.width - tip.offsetWidth))}px`;
    tip.style.top = "18px";
  });
  svg.addEventListener("mouseleave", () => { cursor.setAttribute("visibility", "hidden"); tip.style.display = "none"; });
  return box;
}

// series: gm1.quicklook-series/v1. opts: {now, tmax, limit (how many charts)}
export function chartGrid(series, opts = {}) {
  const charts = (series.charts || []).slice(0, opts.limit || 9);
  const hasCm1 = charts.some((c) => (c.cm1 || []).length);
  const hours = series.time_unit === "h";
  const note = hasCm1 ? series.provenance : (series.cm1_reference ? `No CM1 curve: ${series.cm1_reference}.` : "");
  return h("div", {},
    legend(series.gm1_label || "GM1", hasCm1 ? series.cm1_label : null),
    h("div", { class: "charts" }, charts.map((c) => lineChart(c, { ...opts, hours, labelA: "GM1", labelB: "CM1" }))),
    h("p", { class: "note" }, note ? `${String(note).replace(/[.\s]*$/, "")}. ` : "",
      hours ? "Hours since the start on the horizontal axis." : "Minutes since the start on the horizontal axis."));
}
