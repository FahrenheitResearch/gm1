// Screen 4, Sounding: where the run's atmosphere comes from, CM1's getcape parcels on the run's own levels, the
// hodograph numbers (Bunkers motion, shear, helicity) and the checks, with the hodograph drawn from the wind
// profile the run integrates. isnd = 7 or 17 read input_sounding; advanced mode can replace it.

import { h, num, table, checkline, button, kv } from "../core.js";
import * as api from "../api.js";
import { register, go, notice, runRoute } from "../router.js";
import { actionButton } from "../command.js";
import { runId, loadSetup, hint, dataLine, resultBox, act, ensureEstimate, notImported, lead } from "./common.js";

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function fromDegrees(u, v) {
  // meteorological direction the wind blows from, and speed
  const speed = Math.hypot(u, v);
  let dir = (Math.atan2(-u, -v) * 180) / Math.PI;
  if (dir < 0) dir += 360;
  return { dir: Math.round(dir), speed };
}

// The hodograph of the profile the run integrates (ground-relative), coloured by height, with the storm motions.
export function hodograph(profile, motion, frame) {
  const pts = (profile || []).filter((p) => p[0] <= 12000.5);
  if (pts.length < 2) return null;
  const all = pts.map((p) => [p[1], p[2]]);
  const extra = [];
  if (motion) for (const key of ["right_mover_m_s", "left_mover_m_s", "mean_wind_m_s"]) if (motion[key]) extra.push(motion[key]);
  if (frame) extra.push(frame);
  const lim = Math.max(10, ...[...all, ...extra].map(([u, v]) => Math.max(Math.abs(u), Math.abs(v)))) * 1.15;
  const ring = lim > 40 ? 20 : 10;
  const S = 420, pad = 26, c = S / 2, k = (S / 2 - pad) / lim;
  const X = (u) => c + u * k, Y = (v) => c - v * k;
  const p = [];
  p.push(`<rect x="0" y="0" width="${S}" height="${S}" fill="#05070d" stroke="#1d2436"/>`);
  for (let r = ring; r <= lim; r += ring) {
    p.push(`<circle cx="${c}" cy="${c}" r="${(r * k).toFixed(1)}" fill="none" stroke="#1d2436"/>`);
    p.push(`<text x="${(c - r * k * 0.707 - 3).toFixed(1)}" y="${(c + r * k * 0.707 + 11).toFixed(1)}" text-anchor="end">${r} m/s</text>`);
  }
  p.push(`<line x1="${pad}" x2="${S - pad}" y1="${c}" y2="${c}" stroke="#2a3246"/>`);
  p.push(`<line y1="${pad}" y2="${S - pad}" x1="${c}" x2="${c}" stroke="#2a3246"/>`);
  const bands = [[0, 1000, "#ffd166"], [1000, 3000, "#ffb347"], [3000, 6000, "#4cc3ff"], [6000, 12000.5, "#8e97a8"]];
  for (const [z0, z1, color] of bands) {
    const seg = pts.filter((q) => q[0] >= z0 - 1e-6 && q[0] <= z1 + 1e-6);
    if (seg.length < 2) continue;
    p.push(`<polyline points="${seg.map((q) => `${X(q[1]).toFixed(1)},${Y(q[2]).toFixed(1)}`).join(" ")}" fill="none" stroke="${color}" stroke-width="2"/>`);
  }
  const placed = [];
  for (const z of [1000, 3000, 6000, 9000, 12000]) {
    const q = pts.reduce((best, row) => (Math.abs(row[0] - z) < Math.abs(best[0] - z) ? row : best), pts[0]);
    if (Math.abs(q[0] - z) > 300) continue;
    const x = X(q[1]), y = Y(q[2]);
    // a height whose wind is where an earlier label is (a profile constant aloft) is not labelled again
    if (placed.some(([px, py]) => Math.hypot(px - x, py - y) < 16)) continue;
    placed.push([x, y]);
    p.push(`<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="2.5" fill="#e6edf7"/>`);
    p.push(`<text x="${(x - 5).toFixed(1)}" y="${(y - 6).toFixed(1)}" text-anchor="end" style="fill:#c7cfdb">${z / 1000} km</text>`);
  }
  const mark = (uv, label, color, shape, dy = 0) => {
    if (!uv) return;
    const x = X(uv[0]), y = Y(uv[1]);
    p.push(shape === "x" ?
      `<path d="M${x - 4},${y - 4}L${x + 4},${y + 4}M${x - 4},${y + 4}L${x + 4},${y - 4}" stroke="${color}" stroke-width="1.6"/>` :
      `<rect x="${x - 3.5}" y="${y - 3.5}" width="7" height="7" fill="none" stroke="${color}" stroke-width="1.4"/>`);
    p.push(`<text x="${x + 7}" y="${y + 4 + dy}" style="fill:${color}">${esc(label)}</text>`);
  };
  if (motion) {
    mark(motion.right_mover_m_s, "right mover", "#e6edf7", "sq");
    mark(motion.left_mover_m_s, "left mover", "#8e97a8", "sq");
    mark(motion.mean_wind_m_s, "0–6 km mean", "#8e97a8", "x", -9);
  }
  if (frame && (frame[0] || frame[1])) mark(frame, "frame (umove, vmove)", "#ffd166", "x", 9);
  p.push(`<text x="${S - pad}" y="${c - 4}" text-anchor="end">u</text><text x="${c + 4}" y="${pad + 8}">v</text>`);
  const legend = bands.map(([z0, z1, color]) => h("span", {}, h("span", { class: "sw", style: `border-color:${color}` }),
    `${z0 / 1000}–${Math.round(z1 / 1000)} km`));
  return h("div", { class: "hodo" },
    h("div", { class: "planmap", html: `<svg viewBox="0 0 ${S} ${S}" role="img" aria-label="Hodograph of the run's wind profile">${p.join("")}</svg>` }),
    h("div", { class: "legend" }, legend));
}

function parcelsTable(sv, guided) {
  if (!sv.parcels.length) return null;
  const f0 = (x) => (x === null || x === undefined ? "" : num(x, 0));
  const f1 = (x) => (x === null || x === undefined ? "" : num(x, 1));
  if (guided) {
    const sfc = sv.parcels[0];
    return kv([
      [h("span", {}, "Energy for rising air ", hint("surface-based CAPE")), `${f0(sfc.cape)} J/kg`],
      [h("span", {}, "Push needed to get it going ", hint("CIN")), `${f0(sfc.cin)} J/kg`],
      [h("span", {}, "Cloud base ", hint("LCL")), `about ${f0(sfc.lcl_m)} m up`],
      [h("span", {}, "Top of the storm ", hint("EL")), `about ${num((sfc.el_m || 0) / 1000, 1)} km up`],
    ]);
  }
  return table(["Parcel", { label: "Source", cls: "wide-only" }, { label: "LCL (m)", num: true },
    { label: "LFC (m)", num: true }, { label: "EL (m)", num: true }, { label: "CAPE (J/kg)", num: true },
    { label: "CIN (J/kg)", num: true }],
  sv.parcels.map((p) => h("tr", {},
    h("td", { class: "text" }, p.parcel),
    h("td", { class: "muted wide-only" }, p.source_hpa != null ?
      `${f1(p.source_hpa)} hPa, ${f1(p.source_t_k)} K, ${num(p.source_qv_gkg, 2)} g/kg` : ""),
    h("td", { class: "num" }, f0(p.lcl_m)), h("td", { class: "num" }, f0(p.lfc_m)), h("td", { class: "num" }, f0(p.el_m)),
    h("td", { class: "num text" }, f1(p.cape)), h("td", { class: "num" }, f1(p.cin)))), { compact: true });
}

function windTable(sv, guided) {
  if (!sv.wind.length) return null;
  const rows = sv.wind.map((w) => {
    let value;
    if (w.value !== undefined) value = `${num(w.value, w.unit.startsWith("m²") ? 0 : 1)} ${w.unit}`;
    else if (w.u !== null && w.u !== undefined) {
      const d = fromDegrees(w.u, w.v);
      value = `${num(w.u, 1)}, ${num(w.v, 1)} ${w.unit} (from ${d.dir}°, ${num(d.speed, 1)} ${w.unit})`;
    } else value = "";
    return h("tr", {}, h("td", {}, w.label), h("td", { class: "text" }, value));
  });
  return table(["Wind", "Value"], guided ? rows.slice(0, 1).concat(rows.slice(4, 5)) : rows, { compact: true });
}

async function render(body, args, page) {
  const id = runId(args);
  const guided = page.mode === "guided";
  let s = await loadSetup(id);
  page.setData(dataLine(s));
  if (!s.report.present) { notImported(body, s); return null; }
  if (!s.geometry || s.geometry_stale) s = await ensureEstimate(id, s);
  const sv = s.sounding || {};
  const box = resultBox();
  const surface = sv.surface;
  body.append(h("div", { class: "banner" }, sv.source || "",
    sv.winds ? h("span", { class: "sub" }, `Winds: ${sv.winds}.`) : null,
    surface ? h("span", { class: "sub" }, `Surface ${num(surface.p_hpa, 1)} hPa, θ ${num(surface.theta_k, 1)} K, ` +
      `${num(surface.qv_gkg, 2)} g/kg${surface.lowest_level_m ? ` · lowest model level ${num(surface.lowest_level_m, 0)} m` : ""}`) : null));
  if (guided) {
    body.append(lead("The air the storm grows in: how much energy a rising parcel of air gets, and how the wind ",
      "changes with height, which is what makes a storm rotate."));
  }
  const left = h("section", { class: "panel" },
    h("h2", { class: "cap" }, guided ? "Rising air" : "Parcels (CM1 getcape.F, pseudoadiabatic, liquid only)"),
    parcelsTable(sv, guided) || h("p", { class: "note" }, sv.environment_status === "computed" ? "" :
      "The parcel numbers need GM1's initializer library, which this installation has not admitted; the " +
      "checks say why."),
    h("div", { class: "section" }, h("h2", { class: "cap" }, guided ? "Wind" : "Hodograph (ground-relative, Bunkers et al. 2000)"),
      windTable(sv, guided) || h("p", { class: "note" }, "The storm motion comes from the estimate's storm-track check.")));
  const checks = h("section", { class: "panel" }, h("h2", { class: "cap" }, "Checks"),
    (sv.checks || []).map((c) => h("div", {}, checkline(c.state, c.text), c.fix ? h("div", { class: "note fixline" }, c.fix) : null)));
  body.append(h("div", { class: "grid cols-main section" }, left, checks));

  const values = s.values || {};
  const frame = values.imove === 1 ? [values.umove, values.vmove] : null;
  const hodo = hodograph(sv.hodograph, sv.storm_motion, frame);
  if (hodo) {
    body.append(h("section", { class: "panel section" }, h("h2", { class: "cap" }, "Hodograph of the run's wind profile"),
      hodo, h("p", { class: "note" }, "Ground-relative winds at the run's own levels, as base.F builds them; the ",
        "motions are Bunkers et al. (2000), which CM1 itself does not compute.")));
  }
  if (sv.card) {
    body.append(h("section", { class: "panel section" }, h("figure", {},
      h("a", { href: sv.card.url, target: "_blank" }, h("img", { src: sv.card.url, alt: `${sv.card.title}: skew-T and hodograph` })),
      h("figcaption", {}, h("b", {}, sv.card.title), h("span", {}, "drawn by the quick-look worker from this run")))));
  }
  if (!guided && sv.reads_file && sv.file) {
    body.append(h("section", { class: "panel section" }, h("h2", { class: "cap" }, "input_sounding"),
      kv([["File", sv.file.path || ""], ["sha256", h("span", { class: "mono" }, sv.file.sha256 || "")],
        ["Levels", String(sv.file.levels ?? "")], ["Top", sv.file.top_m != null ? `${num(sv.file.top_m, 0)} m` : ""]])));
  }
  if (!guided && (sv.isnd === 7 || sv.isnd === 17) && s.editable) {
    const path = h("input", { class: "input mono", placeholder: "/path/to/input_sounding", "aria-label": "new sounding path" });
    const file = h("input", { type: "file", "aria-label": "new sounding file" });
    // A chosen file is read once, so "Show command" can ask for the exact line (the upload's staged path).
    let uploaded = null;
    file.addEventListener("change", async () => {
      uploaded = file.files && file.files[0] ? { text: await file.files[0].text() } : null;
      if (replaceWrap.cmd) replaceWrap.cmd.refresh();
    });
    const request = () => {
      const sounding = uploaded || (path.value.trim() ? { path: path.value.trim() } : null);
      if (!sounding) return { error: "Give a path, or pick a file; the command shows here." };
      return { path: `${api.runPath(id)}/sounding`, body: { sounding } };
    };
    const replaceWrap = actionButton("Use this sounding", { action: "sounding", small: true, request, watch: path });
    const replace = replaceWrap.button;
    replace.addEventListener("click", async () => {
      if (file.files && file.files[0]) uploaded = { text: await file.files[0].text() };
      const req = request();
      if (req.error) { notice("Give a path, or pick a file.", "stop"); return; }
      const reply = await act(replace, req.path, req.body, box);
      if (reply && reply.command) setTimeout(() => go(runRoute("sounding", id)), 900);
    });
    body.append(h("section", { class: "panel section advanced-only" }, h("h2", { class: "cap" }, "Replace the sounding"),
      h("div", { class: "pathrow" }, path, file, replaceWrap), box));
  }
  if (!guided && sv.claim_limit) {
    body.append(h("p", { class: "note section" }, `${sv.claim_limit.charAt(0).toUpperCase()}${sv.claim_limit.slice(1)}.`));
  }
  body.append(h("div", { class: "section btns" },
    button("Continue", { kind: "primary", onclick: () => go(runRoute(guided ? "estimate" : "domain", id)) })));
  return null;
}

register("sounding", render, { title: "Sounding", runPage: true });
