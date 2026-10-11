// The Domain and time plan map (SVG): the domain in the namelist's own coordinates (iorigin), the bubble, the
// predicted storm track in the moving frame with its edge clearance, and an inset with the motion vectors.
// Drawn from geometry.json (the estimator's storm-track check), never from made-up numbers.

import { h, num, clock } from "../core.js";

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function niceStep(span) {
  const raw = span / 6;
  const pow = 10 ** Math.floor(Math.log10(raw));
  for (const m of [1, 2, 2.5, 5, 10]) if (m * pow >= raw) return m * pow;
  return 10 * pow;
}

// Clearance of a point to each open side, in km.
export function edgeClearance(g, point) {
  const [x0, y0] = g.origin_km, [w, d] = g.domain_km;
  const sides = g.open_sides || {};
  const out = {
    west: point[0] - x0, east: x0 + w - point[0], south: point[1] - y0, north: y0 + d - point[1],
  };
  for (const side of Object.keys(out)) if (sides[side] === false) out[side] = null;
  return out;
}

export function planMap(g, opts = {}) {
  const S = 540, pad = 36;
  const [x0, y0] = g.origin_km, [w, d] = g.domain_km;
  const span = Math.max(w, d);
  const k = (S - 2 * pad) / span;
  const X = (x) => pad + (x - x0) * k;
  const Y = (y) => pad + (y0 + d - y) * k;
  // A long, narrow domain (a squall line's) has no room for the motion inset inside it: it goes below.
  const insetBelow = d * k < 170;
  const H = pad * 2 + d * k + (insetBelow ? 170 : 0);
  const p = [];
  p.push(`<rect x="${pad}" y="${pad}" width="${(w * k).toFixed(1)}" height="${(d * k).toFixed(1)}" fill="#05070d" stroke="#465066"/>`);
  const step = niceStep(span);
  for (let v = Math.ceil(x0 / step) * step; v <= x0 + w + 1e-9; v += step) {
    p.push(`<line x1="${X(v).toFixed(1)}" x2="${X(v).toFixed(1)}" y1="${pad}" y2="${(pad + d * k).toFixed(1)}" stroke="#1d2436"/>`);
    p.push(`<text x="${X(v).toFixed(1)}" y="${(pad + d * k + 14).toFixed(1)}" text-anchor="middle">${esc(num(v, step % 1 ? 1 : 0))}</text>`);
  }
  for (let v = Math.ceil(y0 / step) * step; v <= y0 + d + 1e-9; v += step) {
    p.push(`<line y1="${Y(v).toFixed(1)}" y2="${Y(v).toFixed(1)}" x1="${pad}" x2="${(pad + w * k).toFixed(1)}" stroke="#1d2436"/>`);
    p.push(`<text x="${pad - 6}" y="${(Y(v) + 4).toFixed(1)}" text-anchor="end">${esc(num(v, step % 1 ? 1 : 0))}</text>`);
  }
  p.push(`<text x="${(pad + w * k).toFixed(1)}" y="${(pad + d * k + 28).toFixed(1)}" text-anchor="end">x (km)</text>`);
  p.push(`<text x="${pad}" y="${pad - 12}">y (km)</text>`);
  // repeating sides are drawn dashed
  const sides = g.open_sides || {};
  const edge = (x1, y1, x2, y2, open) => { if (!open) p.push(`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="#8e97a8" stroke-dasharray="6 4" stroke-width="2"/>`); };
  edge(pad, pad, pad, pad + d * k, sides.west !== false);
  edge(pad + w * k, pad, pad + w * k, pad + d * k, sides.east !== false);
  edge(pad, pad + d * k, pad + w * k, pad + d * k, sides.south !== false);
  edge(pad, pad, pad + w * k, pad, sides.north !== false);
  // bubble
  const b = g.bubble_km;
  if (b && opts.bubbleRadiusKm) {
    p.push(`<circle cx="${X(b[0]).toFixed(1)}" cy="${Y(b[1]).toFixed(1)}" r="${(opts.bubbleRadiusKm * k).toFixed(1)}" fill="rgba(255,179,71,0.10)" stroke="#ffb347" stroke-dasharray="3 3"/>`);
    p.push(`<text x="${X(b[0]).toFixed(1)}" y="${(Y(b[1]) + opts.bubbleRadiusKm * k + 13).toFixed(1)}" text-anchor="middle" style="fill:#ffb347">bubble</text>`);
  }
  const line = (pts, color, dash, width = 1.5) => `<polyline points="${pts.map((q) => `${X(q[1]).toFixed(1)},${Y(q[2]).toFixed(1)}`).join(" ")}" fill="none" stroke="${color}" stroke-width="${width}"${dash ? ` stroke-dasharray="${dash}"` : ""}/>`;
  const tracks = g.track_km || {};
  if (tracks.left && tracks.left.length > 1) p.push(line(tracks.left, "#5b6477", "4 3"));
  const right = tracks.right || [];
  let end = null;
  if (right.length > 1) {
    p.push(line(right, "#e6edf7", "2 3", 1.6));
    const every = Math.max(1, Math.round(900 / Math.max(1, (right[1][0] - right[0][0]))));
    right.forEach((q, i) => { if (i && i % every === 0) p.push(`<circle cx="${X(q[1]).toFixed(1)}" cy="${Y(q[2]).toFixed(1)}" r="2" fill="#e6edf7"><title>${clock(Math.round(q[0] / 60) * 60)}</title></circle>`); });
    end = right[right.length - 1];
    const endLeft = X(end[1]) > pad + w * k * 0.62;
    p.push(`<text x="${(X(end[1]) + (endLeft ? -8 : 8)).toFixed(1)}" y="${(Y(end[2]) - 8).toFixed(1)}" text-anchor="${endLeft ? "end" : "start"}" style="fill:#e6edf7">${clock(Math.round(end[0] / 60) * 60)}, predicted</text>`);
  }
  if (opts.recommended && right.length > 1) {
    const rb = opts.recommended.bubble_km;
    if (rb) p.push(`<circle cx="${X(rb[0]).toFixed(1)}" cy="${Y(rb[1]).toFixed(1)}" r="4" fill="none" stroke="#ffd166"><title>recommended bubble centre</title></circle>`);
  }
  // clearance at the end to the nearest open side
  if (end) {
    const c = edgeClearance(g, [end[1], end[2]]);
    const nearest = Object.entries(c).filter(([, v]) => v !== null).sort((a, b2) => a[1] - b2[1])[0];
    if (nearest) {
      const [side, km] = nearest;
      const ex = side === "west" ? x0 : side === "east" ? x0 + w : end[1];
      const ey = side === "south" ? y0 : side === "north" ? y0 + d : end[2];
      p.push(`<line x1="${X(end[1]).toFixed(1)}" y1="${Y(end[2]).toFixed(1)}" x2="${X(ex).toFixed(1)}" y2="${Y(ey).toFixed(1)}" stroke="#ffd166" stroke-dasharray="2 2"/>`);
      const across = side === "west" || side === "east";
      const lx = across ? (X(end[1]) + X(ex)) / 2 : X(end[1]) + 6;
      const ly = across ? Y(end[2]) + 15 : (Y(end[2]) + Y(ey)) / 2 + 4;
      p.push(`<text x="${lx.toFixed(1)}" y="${ly.toFixed(1)}" text-anchor="${across ? "middle" : "start"}" style="fill:#ffd166">${esc(num(km, 0))} km to the ${esc(side)} side</text>`);
    }
  }
  // inset: motion vectors, ground-relative, m/s
  const m = g.storm_motion || {};
  const vectors = [[g.frame_m_s, "#ffd166", "frame"], [m.right_mover_m_s, "#e6edf7", "right"],
    [m.left_mover_m_s, "#8e97a8", "left"], [m.mean_wind_m_s, "#5b6477", "mean"]].filter((v) => v[0]);
  if (vectors.length) {
    const iw = 150, ih = 150, ix = pad + w * k - iw - 6, iy = insetBelow ? pad + d * k + 22 : pad + 6;
    const vmax = Math.max(5, ...vectors.map(([v]) => Math.hypot(v[0], v[1])));
    const ks = 55 / vmax;
    const ic = [ix + iw / 2 - 10, iy + ih / 2 + 16];
    p.push(`<rect x="${ix}" y="${iy}" width="${iw}" height="${ih}" fill="#0a0e18" stroke="#1d2436"/>`);
    p.push(`<text x="${ix + 6}" y="${iy + 13}">motion (m/s)</text>`);
    p.push(`<text x="${ix + 6}" y="${iy + 26}">ground-relative</text>`);
    p.push(`<line x1="${ic[0] - 60}" x2="${ic[0] + 60}" y1="${ic[1]}" y2="${ic[1]}" stroke="#1d2436"/>`);
    p.push(`<line x1="${ic[0]}" x2="${ic[0]}" y1="${ic[1] + 50}" y2="${ic[1] - 60}" stroke="#1d2436"/>`);
    for (const [v, color, label] of vectors) {
      const x2 = ic[0] + v[0] * ks, y2 = ic[1] - v[1] * ks;
      const a = Math.atan2(y2 - ic[1], x2 - ic[0]);
      const hx = (t) => (x2 - 6 * Math.cos(a + t)).toFixed(1), hy = (t) => (y2 - 6 * Math.sin(a + t)).toFixed(1);
      p.push(`<line x1="${ic[0]}" y1="${ic[1]}" x2="${x2.toFixed(1)}" y2="${y2.toFixed(1)}" stroke="${color}" stroke-width="1.5"/>`);
      p.push(`<polyline points="${hx(0.45)},${hy(0.45)} ${x2.toFixed(1)},${y2.toFixed(1)} ${hx(-0.45)},${hy(-0.45)}" fill="none" stroke="${color}" stroke-width="1.5"/>`);
      p.push(`<text x="${(x2 + 4).toFixed(1)}" y="${(y2 + 3).toFixed(1)}" style="fill:#c7cfdb">${esc(label)}</text>`);
    }
  }
  return h("div", { class: "planmap", html: `<svg viewBox="0 0 ${S} ${H.toFixed(1)}" role="img" aria-label="Plan map of the domain, the bubble and the predicted storm track">${p.join("")}</svg>` });
}
