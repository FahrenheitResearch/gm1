// The column route's steps, shared by guided mode's gallery card and advanced mode's New run panel:
// a history file, a time, the map where a click picks the spot, then the air above that spot as a skew-T and
// hodograph with its numbers and the plain note on what this is. Each step is one gpuwm-cm1 command:
//   Read the file       sounding times FILE --map-png ... --json       (POST /api/wrf/open)
//   a spot on the map   sounding from-wrf FILE ... --out ... --json    (POST /api/wrf/column)
// columnFlow({guided, onColumn}) returns the element; its .state holds what the next step (card or panel) sends.

import { h, num, readout } from "../core.js";
import * as api from "../api.js";
import { actionButton, commandSlot } from "../command.js";
import { ready, t, hint } from "./copy.js";
import { mapView } from "./map.js";

let flows = 0;

function km(m) {
  return m >= 1000 ? `${num(m / 1000, 1)} km` : `${num(m, 0)} m`;
}

function expectKey(n) {
  const cape = Math.max(Number(n.cape_sb || 0), Number(n.cape_mu || 0));
  const shear = Number(n.shear_0_6km || 0);
  if (cape < 300) return "expect.weak";
  if (shear < 10) return "expect.pulse";
  if (shear < 20) return "expect.multicell";
  return "expect.supercell";
}

function banner(reply) {
  return h("p", { class: "banner stop" }, reply.message || "", reply.fix ? h("span", { class: "sub" }, reply.fix) : null);
}

export async function columnFlow(opts = {}) {
  await ready();
  const guided = opts.guided === true;
  const id = ++flows;
  const state = { path: "", facts: null, when: {}, point: null, column: null, field: "shear" };
  const say = h("div", { class: "wrf-say", role: "status", "aria-live": "polite" });
  const slot = commandSlot();

  // 1. the file
  const path = h("input", { class: "input mono", placeholder: t("file.placeholder"), "aria-label": t("file.label"),
    spellcheck: "false", list: `wrf-files-${id}`, autocomplete: "off" });
  const listing = h("datalist", { id: `wrf-files-${id}` });
  const found = h("p", { class: "note wrf-found" });
  api.get("/api/wrf/files").then((d) => {
    for (const f of d.files || []) listing.append(h("option", { value: f.path }, f.folder ? `${f.folder}/${f.name}` : f.name));
    found.textContent = (d.files || []).length ? `${t("file.found")} ${d.files.length}` : t("file.none_found");
  }).catch(() => { found.textContent = ""; });
  const facts = h("p", { class: "wrf-facts" });

  // 2. the time, the map and the spot
  const time = h("select", { class: "input", "aria-label": t("time.label") });
  const fieldButtons = {};
  const fields = h("div", { class: "wrf-fields", role: "group", "aria-label": t("map.label") });
  for (const [field, key] of [["shear", "map.shear"], ["cape", "map.cape"]]) {
    fieldButtons[field] = h("button", { type: "button", class: "btn small", "aria-pressed": field === "shear" ? "true" : "false",
      dataset: { field } }, t(key));
    fields.append(fieldButtons[field]);
  }
  const lat = h("input", { class: "input num", inputmode: "decimal", "aria-label": t("point.lat"), placeholder: "35.2" });
  const lon = h("input", { class: "input num", inputmode: "decimal", "aria-label": t("point.lon"), placeholder: "-97.4" });
  const chosen = h("p", { class: "note wrf-chosen" });
  // An idealized run has no map (its latitudes and longitudes are one value): the two boxes take a grid column.
  const latLabel = h("label", {}, t("point.lat"));
  const lonLabel = h("label", {}, t("point.lon"));
  const noMapNote = h("p", { class: "note wrf-nomap" }, t("point.no_map"));
  noMapNote.hidden = true;
  const pointBoxes = (noMap) => {
    state.noMap = noMap;
    latLabel.textContent = t(noMap ? "point.i" : "point.lat");
    lonLabel.textContent = t(noMap ? "point.j" : "point.lon");
    lat.setAttribute("aria-label", t(noMap ? "point.i" : "point.lat"));
    lon.setAttribute("aria-label", t(noMap ? "point.j" : "point.lon"));
    lat.placeholder = noMap ? "1" : "35.2";
    lon.placeholder = noMap ? "1" : "-97.4";
    lat.inputMode = noMap ? "numeric" : "decimal";
    lon.inputMode = noMap ? "numeric" : "decimal";
    noMapNote.hidden = !noMap;
  };
  const map = mapView((ij) => takeColumn({ ij }));
  const where = h("div", { class: "wrf-where" });
  where.hidden = true;

  // 3. the column
  const preview = h("div", { class: "wrf-preview" });
  const about = h("div", { class: "wrf-about" }, h("h3", {}, t("coverage.title")), h("p", {}, t("coverage.text")));

  const openRequest = (extra = {}) => {
    const p = path.value.trim();
    if (!p) return { error: t("error.pick_file") };
    return { path: "/api/wrf/open", body: { path: p, map: state.field, ...state.when, ...extra } };
  };
  const columnRequest = () => {
    if (!state.path) return { error: t("error.pick_file") };
    // A typed minus sign may be the Unicode one.
    const la = Number(lat.value.trim().replace("\u2212", "-"));
    const lo = Number(lon.value.trim().replace("\u2212", "-"));
    if (lat.value.trim() === "" || lon.value.trim() === "" || Number.isNaN(la) || Number.isNaN(lo)) {
      return { error: t(state.noMap ? "error.pick_column" : "error.pick_point") };
    }
    if (state.noMap) {
      if (!Number.isInteger(la) || !Number.isInteger(lo) || la < 1 || lo < 1) return { error: t("error.pick_column") };
      return { path: "/api/wrf/column", body: { path: state.path, ...state.when, ij: [la, lo], map: state.field } };
    }
    return { path: "/api/wrf/column", body: { path: state.path, ...state.when, at: [la, lo], map: state.field } };
  };

  async function open(extra = {}) {
    const req = openRequest(extra);
    if (req.error) { say.replaceChildren(banner({ message: req.error })); return; }
    say.replaceChildren(h("p", { class: "note" }, t("file.reading")));
    let reply;
    try {
      reply = await api.post(req.path, req.body);
    } catch (err) {
      say.replaceChildren(banner({ message: err.message, fix: err.fix }));
      return;
    }
    slot.set(reply.command);
    if (!reply.ok) { say.replaceChildren(banner(reply)); return; }
    say.replaceChildren();
    const first = state.path !== reply.path;
    state.path = reply.path;
    state.facts = reply;
    facts.replaceChildren(t("file.facts", {
      producer: reply.producer, domain: reply.domain_id, nx: reply.nx, ny: reply.ny,
      dx: km(Number(reply.dx_m || 0)), records: reply.records,
    }));
    if (first) {
      time.replaceChildren(...reply.times.map((stamp, n) => h("option", { value: stamp }, (reply.times_display || [])[n] || stamp)));
      state.when = reply.records > 1 ? { time: reply.times[reply.record || 0] } : {};
      state.column = null;
      preview.replaceChildren();
      if (opts.onColumn) opts.onColumn(null);
    }
    time.disabled = reply.records < 2;
    const lr = reply.lat_range, lo = reply.lon_range;
    pointBoxes(!(Array.isArray(lr) && Array.isArray(lo) && lr[1] > lr[0] && lo[1] > lo[0]));
    map.show(reply.map);
    // A picture the server could not draw says why (the numbers and the typed spot still work).
    if (reply.map && !reply.map.url && reply.map.why) say.replaceChildren(h("p", { class: "note" }, reply.map.why));
    where.hidden = false;
  }

  async function takeColumn(point) {
    if (!state.path) return;
    const body = { path: state.path, ...state.when, map: state.field, ...point };
    chosen.textContent = t("column.working");
    let reply;
    try {
      reply = await api.post("/api/wrf/column", body);
    } catch (err) {
      chosen.replaceChildren(banner({ message: err.message, fix: err.fix }));
      return;
    }
    slot.set(reply.command);
    if (!reply.ok) {
      chosen.replaceChildren(banner(reply));
      return;
    }
    const p = reply.point || {};
    state.point = point.at ? { at: point.at } : { ij: [p.i, p.j] };
    state.column = reply;
    if (state.noMap) { lat.value = String(p.i); lon.value = String(p.j); }
    else if (p.lat !== undefined) { lat.value = p.lat.toFixed(3); lon.value = p.lon.toFixed(3); }
    chosen.textContent = state.noMap ? t("point.chosen_ij", { i: p.i, j: p.j }) : t("point.chosen", {
      i: p.i, j: p.j, lat: p.lat !== undefined ? p.lat.toFixed(3) : "", lon: p.lon !== undefined ? p.lon.toFixed(3) : "",
      distance: km(1000 * Number(p.distance_km || 0)),
    });
    if (reply.map) map.show(reply.map);
    drawColumn(reply);
    if (opts.onColumn) opts.onColumn(state);
  }

  function drawColumn(reply) {
    const n = reply.numbers || {};
    const fig = reply.preview && reply.preview.url ?
      h("figure", { class: "wrf-skewt" }, h("img", { src: reply.preview.url, alt: t("column.title"), width: "1920",
        height: "1080", loading: "lazy" })) : reply.preview && reply.preview.why ? h("p", { class: "note" }, reply.preview.why) : null;
    const value = (v, nd = 0) => (v === null || v === undefined ? "–" : num(Number(v), nd));
    const readouts = h("div", { class: "readouts" },
      readout([t("numbers.cape"), hint("numbers.cape")], value(n.cape_sb), "J/kg"),
      readout([t("numbers.cin"), hint("numbers.cin")], value(n.cin_sb), "J/kg"),
      readout(t("numbers.shear1"), value(n.shear_0_1km), "m/s"),
      readout(t("numbers.shear6"), value(n.shear_0_6km), "m/s"),
      readout([t("numbers.srh"), hint("numbers.srh")], value(n.srh_0_3km), "m²/s²"));
    const hydro = reply.hydrostatic ? h("p", { class: "note" }, t("numbers.hydro", { hpa: num(reply.hydrostatic.max_abs_hpa, 2) })) : null;
    preview.replaceChildren(
      h("h3", { class: "wrf-h" }, t("column.title")),
      fig,
      readouts,
      Object.keys(n).length ? h("p", { class: "wrf-expect" }, t(expectKey(n)), h("span", { class: "sub" }, t("expect.note"))) : null,
      h("p", { class: "note" }, t("numbers.levels", { levels: reply.levels, top: km(Number(reply.top_m || 0)) })),
      hydro,
      about);
  }

  const openButton = actionButton(t("file.read"), { action: "wrf-open", kind: guided ? "primary" : "",
    request: () => openRequest(), watch: [path] });
  openButton.button.addEventListener("click", async () => {
    openButton.button.disabled = true;
    try { await open(); } finally { openButton.button.disabled = false; }
  });
  path.addEventListener("keydown", (e) => { if (e.key === "Enter") openButton.button.click(); });
  const spotButton = actionButton(t("point.show"), { action: "wrf-column", small: true, request: columnRequest,
    watch: [lat, lon] });
  spotButton.button.addEventListener("click", () => {
    const req = columnRequest();
    if (req.error) { chosen.replaceChildren(banner({ message: req.error })); return; }
    takeColumn(req.body.ij ? { ij: req.body.ij } : { at: req.body.at });
  });
  time.addEventListener("change", () => {
    state.when = { time: time.value };
    state.column = null;
    preview.replaceChildren();
    if (opts.onColumn) opts.onColumn(null);
    open({ time: time.value }).then(() => { if (state.point) takeColumn(state.point); });
  });
  for (const [field, b] of Object.entries(fieldButtons)) {
    b.addEventListener("click", async () => {
      if (state.field === field) return;
      state.field = field;
      for (const [f, other] of Object.entries(fieldButtons)) other.setAttribute("aria-pressed", f === field ? "true" : "false");
      if (field === "cape") say.replaceChildren(h("p", { class: "note" }, t("map.cape_note")));
      if (state.point) await takeColumn(state.point);
      else await open();
    });
  }

  where.append(
    h("div", { class: "wrf-row" }, h("label", {}, t("time.label")), time, fields),
    h("div", { class: "wrf-mapwrap" }, map),
    noMapNote,
    h("div", { class: "wrf-row wrf-latlon" }, latLabel, lat, lonLabel, lon, spotButton),
    chosen);
  const root = h("div", { class: "wrf-flow", dataset: { flow: guided ? "guided" : "advanced" } },
    h("div", { class: "field wrf-file" }, h("label", {}, t("file.label"), hint("file.label")),
      h("div", {}, h("div", { class: "pathrow wrf-pathrow" }, path, openButton), listing, found,
        h("div", { class: "hint" }, t("file.help")))),
    say, facts, where, preview, slot);
  root.state = state;
  root.path = path;
  return root;
}
