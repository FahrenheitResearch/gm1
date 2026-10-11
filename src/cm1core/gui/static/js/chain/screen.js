// Advanced mode: "Downscale a WRF or ArWen run (3D)". A finer GM1 domain inside a WRF or ArWen
// run: its initial state is one saved time of the run and its four sides follow the saved times after it (the
// ArWen chain, records/GM1-ARWEN-CHAIN.md). Four steps, each one gpuwm-cm1 command:
//   1. Your run           chain inspect SRC --map-png P [--map-time T] --json        (POST /api/chain/open)
//   2. The box and time   a drag on the map, the ratio, the saved times (no command)
//   3. Can GM1 run it?    chain check --history SRC --window ... --ratio R --json   (POST /api/chain/check)
//   4. Make the run       chain new ... --out RUNDIR --json                          (POST /api/chain/new)
// The run folder it makes is an ordinary run: Check, Estimate and Start take it from there. Guided mode keeps only
// the column route's card; this screen is advanced mode's.

import { h, num, checkline, bytes, schemeName } from "../core.js";
import * as api from "../api.js";
import { register, go, notice } from "../router.js";
import { actionButton } from "../command.js";
import { boxMap } from "./boxmap.js";

const NAME_RE = /^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/;
let words = null;

async function copy() {
  if (!words) words = await api.get("/api/chain/copy");
  return words;
}

function fill(template, values) {
  return String(template || "").replace(/\{(\w+)\}/g, (_, k) => (values[k] === undefined || values[k] === null ? "" : String(values[k])));
}

function minutes(seconds) {
  const s = Number(seconds);
  if (!Number.isFinite(s)) return "";
  if (s >= 3600 && s % 3600 === 0) return `${s / 3600} h`;
  if (s >= 3600) return `${num(s / 3600, 1)} h`;
  if (s >= 60 && s % 60 === 0) return `${s / 60} min`;
  if (s >= 60) return `${num(s / 60, 1)} min`;
  return `${num(s, 0)} s`;
}

function length(m) {
  return m >= 1000 ? `${num(m / 1000, m % 1000 ? 1 : 0)} km` : `${num(m, 0)} m`;
}

function field(label, input, hintText) {
  return h("div", { class: "field" }, h("label", {}, label), input,
    hintText ? h("div", { class: "hint" }, hintText) : null);
}

function numberInput(label, value) {
  return h("input", { class: "input num", type: "number", min: "0", step: "1", inputmode: "numeric",
    "aria-label": label, value: String(value) });
}

function banner(text, kind, sub) {
  return h("p", { class: `banner${kind ? ` ${kind}` : ""}` }, text, sub ? h("span", { class: "sub" }, sub) : null);
}

async function render(body, args, page) {
  const W = await copy();
  const S = W.screen;
  const state = { path: "", facts: null, map: null, box: null, checked: null };

  // ---- what has been checked, and what this is
  // What this is first (full width), then what has been checked and the worked example side by side.
  const about = h("div", { class: "chain-about" },
    h("section", { class: "chain-limits" }, h("h2", { class: "cap" }, S.limits_title),
      h("p", {}, S.lead), h("p", {}, S.limits),
      h("p", { class: "note" }, h("a", { href: "#/new-run" }, S.column_route))),
    h("section", { class: "chain-verified" }, h("h2", { class: "cap" }, S.verified_title),
      h("p", {}, W.verified), S.verified_hint ? h("p", { class: "note" }, h("span", { class: "hint" }, S.verified_hint)) : null),
    h("section", { class: "chain-example" }, h("h2", { class: "cap" }, S.example_title),
      h("p", {}, S.example), S.example_hint ? h("p", { class: "note" }, h("span", { class: "hint" }, S.example_hint)) : null));

  // ---- 1. your run
  const path = h("input", { class: "input mono", placeholder: S.file_placeholder, "aria-label": S.file_label,
    spellcheck: "false", list: "chain-files", autocomplete: "off" });
  const listing = h("datalist", { id: "chain-files" });
  const found = h("p", { class: "note chain-found" });
  api.get("/api/chain/files").then((d) => {
    const rows = [...(d.folders || []).map((f) => [f.path, `${f.files} files`]), ...(d.files || []).map((f) => [f.path, f.name])];
    for (const [value, label] of rows) listing.append(h("option", { value }, label));
    found.textContent = rows.length ? `${S.found_files} ${rows.length}` : S.found_none;
  }).catch(() => { found.textContent = ""; });
  const facts = h("div", { class: "chain-facts", role: "status", "aria-live": "polite" });
  const openRequest = (extra = {}) => {
    const p = path.value.trim();
    if (!p) return { error: S.open_first };
    return { path: "/api/chain/open", body: { path: p, ...extra } };
  };
  const openBtn = actionButton(S.open, { action: "chain-open", request: () => openRequest(), watch: [path] });

  // ---- 2. the box and the time
  const where = h("div", { class: "chain-where" });
  where.hidden = true;
  const map = boxMap((b) => setBox(b, true), { alt: S.map_caption });
  const inputs = {
    i0: numberInput(S.i0, 0), j0: numberInput(S.j0, 0), ni: numberInput(S.ni, 1), nj: numberInput(S.nj, 1),
  };
  const ratio = h("select", { class: "input", "aria-label": S.ratio },
    ...(W.ratios || [3, 5, 7, 9]).map((r) => h("option", { value: String(r), selected: r === 5 ? "" : null }, String(r))));
  const start = h("select", { class: "input", "aria-label": S.start });
  const end = h("select", { class: "input", "aria-label": S.end });
  const every = h("select", { class: "input", "aria-label": S.every });
  // Moisture: every species of the run's scheme (GM1 must run it), or water vapour only (WRF's
  // have_bcs_moist = .false.), with the GM1 scheme a vapour-only run uses.
  const moisture = h("select", { class: "input", "aria-label": S.moisture },
    h("option", { value: "all-species" }, S.moisture_all), h("option", { value: "vapour-only" }, S.moisture_vapour));
  const scheme = h("select", { class: "input", "aria-label": S.scheme },
    ...(W.schemes || ["wsm6-mp6", "morrison-mp10"]).map((k) => h("option", { value: k }, schemeName(k))));
  const schemeField = field(S.scheme, scheme, S.scheme_help);
  const paintScheme = () => { schemeField.hidden = moisture.value !== "vapour-only"; };
  paintScheme();
  const gridLine = h("p", { class: "chain-grid" });
  const boxLine = h("p", { class: "note" });

  // ---- 3. the check
  const checks = h("div", { class: "chain-checks", role: "status", "aria-live": "polite" });
  const request = () => {
    if (!state.facts) return { error: S.open_first };
    const b = readBox();
    if (!b) return { error: S.check_first };
    const bodyOut = { path: state.path, window: b, ratio: Number(ratio.value), start: start.value || null,
      end: end.value || null };
    if (every.value) bodyOut.every = Number(every.value);
    if (moisture.value === "vapour-only") { bodyOut.moisture = "vapour-only"; bodyOut.scheme = scheme.value; }
    return bodyOut;
  };
  const checkBtn = actionButton(S.check, { action: "chain-check", request: () => {
    const b = request();
    return b.error ? b : { path: "/api/chain/check", body: b };
  }, watch: [path, ratio, start, end, every, moisture, scheme, ...Object.values(inputs)] });

  // ---- 4. make the run
  const name = h("input", { class: "input", placeholder: "my-downscaled-storm", "aria-label": S.name_label, spellcheck: "false" });
  const makeNote = h("p", { class: "note" }, S.make_needs_check);
  const makeRequest = () => {
    const b = request();
    if (b.error) return b;
    const n = name.value.trim();
    if (!n || !NAME_RE.test(n)) return { error: `${S.name_label}: ${S.name_hint}` };
    return { path: "/api/chain/new", body: { ...b, name: n } };
  };
  const makeBtn = actionButton(S.make, { action: "chain-new", kind: "primary", request: makeRequest,
    watch: [name, path, ratio, start, end, every, moisture, scheme, ...Object.values(inputs)] });
  makeBtn.button.disabled = true;

  const requestKey = () => JSON.stringify(request());
  const refreshMake = () => {
    const ok = state.checked && state.checked.key === requestKey() && state.checked.admitted;
    makeBtn.button.disabled = !ok;
    makeNote.hidden = !!ok;
  };

  function readBox() {
    const v = ["i0", "j0", "ni", "nj"].map((k) => Number(inputs[k].value));
    if (v.some((x) => !Number.isInteger(x)) || v[0] < 0 || v[1] < 0 || v[2] < 1 || v[3] < 1) return null;
    return v;
  }

  function paintGrid() {
    const b = readBox();
    const f = state.facts;
    if (!b || !f) { gridLine.textContent = ""; boxLine.textContent = ""; return; }
    const r = Number(ratio.value);
    // On a mapped run a cell of your run is DX/m on the ground; the check takes m over the box itself.
    const mf = f.grid.map_factor;
    const m = mf ? (mf.min + mf.max) / 2 : 1;
    const ground = f.grid.dx_m / m;
    const dx = ground / r;
    const scaled = Math.abs(m - 1) > 1e-4;
    gridLine.textContent = fill(scaled ? S.grid_line_scaled : S.grid_line, { nx: num(b[2] * r), ny: num(b[3] * r),
      dx: num(dx, dx % 1 ? 1 : 0), scale: num(m, 3),
      km_x: num((b[2] * ground) / 1000, 1), km_y: num((b[3] * ground) / 1000, 1) });
    boxLine.textContent = fill(S.box_cells, { ni: b[2], nj: b[3], i0: b[0], j0: b[1] });
  }

  function setBox(b, fromMap) {
    if (!b) return;
    state.box = b;
    if (fromMap) {
      inputs.i0.value = String(b[0]); inputs.j0.value = String(b[1]);
      inputs.ni.value = String(b[2]); inputs.nj.value = String(b[3]);
    }
    map.setBox(b);
    paintGrid();
    refreshMake();
  }
  for (const input of Object.values(inputs)) {
    input.addEventListener("input", () => { const b = readBox(); if (b) setBox(b, false); });
  }
  for (const el of [ratio, every, end, scheme]) el.addEventListener("change", () => { paintGrid(); refreshMake(); });
  moisture.addEventListener("change", () => { paintScheme(); refreshMake(); });
  name.addEventListener("input", refreshMake);

  function everyOptions() {
    const f = state.facts;
    every.replaceChildren(h("option", { value: "" }, S.every_all));
    const gap = (f.intervals_s || [])[0];
    if (!gap) return;
    for (const k of [2, 3, 4, 6]) every.append(h("option", { value: String(k * gap) }, minutes(k * gap)));
  }

  function paintFacts(reply) {
    const f = reply;
    const g = f.grid;
    const lines = [fill(f.count === 1 && S.facts_one ? S.facts_one : S.facts, { producer: f.producer, n: f.count, first: f.times[0], last: f.times[f.times.length - 1],
      interval: minutes((f.intervals_s || [])[0]), nx: g.west_east, ny: g.south_north, dx: length(g.dx_m),
      levels: g.bottom_top, projection: g.projection })];
    const m = f.microphysics || {};
    lines.push(fill(m.offered ? S.facts_scheme_ok : S.facts_scheme_no, { scheme: m.scheme }));
    // A scheme GM1 does not run can only give its water vapour: start there (the check says so too).
    moisture.value = m.offered ? "all-species" : "vapour-only";
    paintScheme();
    const t = g.terrain_m;
    if (t) lines.push(t.max - t.min > 0 ? fill(S.facts_ground, { lo: num(t.min, 0), hi: num(t.max, 0) }) : S.facts_flat);
    facts.replaceChildren(...lines.map((line, i) => h("p", { class: i === 1 && !m.offered ? "chain-warn" : null }, line)));
  }

  function showMap(reply) {
    if (!reply.map || !reply.map.url) return;
    state.map = reply.map;
    const parts = [reply.map.ground_drawn ? S.map_caption : S.map_caption_flat];
    if (reply.map.max_updraft_m_s !== null && reply.map.max_updraft_m_s !== undefined) parts.push(S.map_legend_updraft);
    map.show(reply.map, `${S.map_time} ${reply.map.time}. ${parts.join(" ")}`);
  }

  async function open(extra = {}) {
    const req = openRequest(extra);
    if (req.error) { notice(req.error, "stop"); return; }
    openBtn.button.disabled = true;
    if (!extra.map_time) notice(S.opening);
    try {
      const reply = await api.post(req.path, req.body);
      if (reply.ok === false) {
        facts.replaceChildren(banner(reply.message, "stop", reply.fix));
        notice(`${reply.message} ${reply.fix || ""}`.trim(), "stop");
        return;
      }
      if (!extra.map_time) {
        state.path = reply.path;
        state.facts = reply;
        paintFacts(reply);
        start.replaceChildren(...reply.times.map((t, i) => h("option", { value: t, selected: i === 0 ? "" : null }, t)));
        end.replaceChildren(...reply.times.map((t, i) => h("option", { value: t, selected: i === reply.times.length - 1 ? "" : null }, t)));
        everyOptions();
        const nx = reply.grid.west_east;
        const ny = reply.grid.south_north;
        const size = Math.max(3, Math.min(24, Math.floor(Math.min(nx, ny) / 4)));
        setBox([Math.floor((nx - size) / 2), Math.floor((ny - size) / 2), size, size], true);
        where.hidden = false;
        checks.replaceChildren();
        state.checked = null;
        notice(reply.message);
      }
      showMap(reply);
      refreshMake();
    } catch (err) {
      facts.replaceChildren(banner(err.message, "stop", err.fix));
      notice(`${err.message} ${err.fix || ""}`.trim(), "stop");
    } finally { openBtn.button.disabled = false; }
  }
  openBtn.button.addEventListener("click", () => open());
  path.addEventListener("keydown", (event) => { if (event.key === "Enter") open(); });
  start.addEventListener("change", () => { paintGrid(); refreshMake(); if (state.facts) open({ map_time: start.value }); });

  function paintChecks(reply) {
    const rows = (reply.checks || []).map((row) => {
      const st = row.status === "pass" ? "ok" : row.status === "refused" ? "refused" : "note";
      const mark = row.status === "pass" ? S.pass : row.status === "refused" ? S.fail : S.not_checked;
      return h("div", { class: `chain-check ${row.status}`, dataset: { check: row.id } },
        checkline(st, h("span", {}, h("b", {}, row.label), `: ${row.text}`,
          row.code ? h("span", { class: "hint", title: S.code_hint }, row.code) : null)),
        h("span", { class: "chain-mark" }, mark));
    });
    const out = [banner(reply.message || "", reply.admitted ? "" : "warn"), h("div", { class: "chain-rows" }, rows)];
    const e = reply.estimate;
    if (e && !e.error) {
      out.push(h("h3", { class: "chain-h" }, S.estimate_title),
        h("p", {}, fill(S.estimate_line, { cells: num(e.cells), dx: num(e.grid ? e.grid.dx_m : 0, 0), steps: num(e.steps),
          gpu: num((e.gpu_memory_bytes || 0) / 1e9, 2), wall: e.wall_s ? minutes(Math.round(e.wall_s)) : "?",
          disk: num((e.disk_bytes || 0) / 1e9, 1) })));
    } else if (e && e.error) {
      out.push(h("p", { class: "note" }, fill(S.estimate_none, { why: e.error })));
    }
    if (reply.prepare) out.push(h("p", { class: "note" }, fill(S.prepare_line, { prepare: bytes(reply.prepare.disk_bytes) })));
    checks.replaceChildren(...out);
    paintCheckedGrid(reply.choice);
  }

  // After a check, the grid line gives the spacing the check took: DX/(ratio*m) with m the box's own mean map
  // scale (before it, paintGrid's "about" line uses the middle of the run's range).
  function paintCheckedGrid(c) {
    const f = state.facts;
    if (!c || !f || !c.dx_m || !c.window || !c.ratio) return;
    const m = f.grid.dx_m / (c.ratio * c.dx_m);
    if (Math.abs(m - 1) <= 1e-4) return;
    const w = c.window;
    gridLine.textContent = fill(S.grid_line_checked, { nx: num(w.ni * c.ratio), ny: num(w.nj * c.ratio),
      dx: num(c.dx_m, 2), scale: num(m, 4),
      km_x: num((w.ni * c.ratio * c.dx_m) / 1000, 1), km_y: num((w.nj * c.ratio * c.dx_m) / 1000, 1) });
  }

  checkBtn.button.addEventListener("click", async () => {
    const b = request();
    if (b.error) { notice(b.error, "stop"); return; }
    const key = JSON.stringify(b);
    checkBtn.button.disabled = true;
    notice(S.checking);
    try {
      const reply = await api.post("/api/chain/check", b);
      if (reply.ok === false) {
        checks.replaceChildren(banner(reply.message, "stop", reply.fix));
        state.checked = null;
        notice(`${reply.message} ${reply.fix || ""}`.trim(), "stop");
      } else {
        paintChecks(reply);
        state.checked = { key, admitted: !!reply.admitted };
        notice(reply.message, reply.admitted ? "" : "warn");
        if (reply.admitted && !name.value) {
          name.value = `downscale-${b.window.join("-")}-x${b.ratio}`;
        }
      }
    } catch (err) {
      checks.replaceChildren(banner(err.message, "stop", err.fix));
      notice(`${err.message} ${err.fix || ""}`.trim(), "stop");
    } finally {
      checkBtn.button.disabled = false;
      refreshMake();
    }
  });

  makeBtn.button.addEventListener("click", async () => {
    const req = makeRequest();
    if (req.error) { notice(req.error, "stop"); return; }
    makeBtn.button.disabled = true;
    notice(S.making);
    try {
      const reply = await api.post(req.path, req.body);
      if (reply.ok === false || !reply.url) {
        notice(`${reply.message || ""} ${reply.fix || ""}`.trim(), "stop");
        return;
      }
      notice(reply.message);
      setTimeout(() => go(`check/${reply.url}`), 300);
    } catch (err) {
      notice(`${err.message} ${err.fix || ""}`.trim(), "stop");
    } finally { refreshMake(); }
  });

  where.append(
    h("section", { class: "panel section" }, h("h2", { class: "cap" }, S.step_box),
      h("div", { class: "chain-mapwrap" }, map),
      h("div", { class: "chain-form" },
        h("div", { class: "chain-boxfields" },
          h("p", { class: "note" }, S.box_hint),
          h("div", { class: "chain-cells" }, ...["i0", "j0", "ni", "nj"].map((k) =>
            h("label", { class: "chain-cell" }, h("span", {}, S[k]), inputs[k]))),
          boxLine),
        field(S.ratio, ratio, S.ratio_help),
        field(S.start, start),
        field(S.end, end),
        field(S.every, every),
        field(S.moisture, moisture, S.moisture_help),
        schemeField,
        gridLine)),
    h("section", { class: "panel section" }, h("h2", { class: "cap" }, S.step_check),
      h("div", { class: "btns" }, checkBtn), checks),
    h("section", { class: "panel section" }, h("h2", { class: "cap" }, S.step_make),
      field(S.name_label, name, S.name_hint), makeNote, h("div", { class: "btns" }, makeBtn)));

  body.append(
    about,
    h("section", { class: "panel section" }, h("h2", { class: "cap" }, S.step_file),
      field(S.file_label, h("div", { class: "chain-pathrow" }, path, openBtn)), listing, found, facts),
    where);
  return null;
}

register("downscale-3d", render, { label: "Downscale (3D)", nav: true, mode: "advanced", order: 25,
  title: "Downscale a WRF or ArWen run (3D)" });

export { render };
