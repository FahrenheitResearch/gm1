// Screen 9, Results (guided: 7.2). One screen, two views of the same run folder:
//
//   advanced  a time slider over every quick-look; slices of any cm1out field (horizontal at a level, vertical
//             along a line drawn on it, the skew-T of a clicked column); the cm1out files with their sizes and
//             ready-to-copy ncview, xarray and GrADS lines; the receipt, the disk and the segments; Continue or
//             extend, Convert again and Reproduce bundle, each with its gpuwm-cm1 command.
//   guided    guided mode's view: the four favourite looks in order (reflectivity, spin near the ground, updraft at 5 km,
//             the wind swath) with a slider and Play, "What am I looking at" beside the picture, then Open the
//             files, Run it longer and Show everything. A "More looks" row under the four tabs (the cold
//             pool, rain, clouds seen from above, the updraft cut open, the hail and spin tracks, the environment,
//             the air parcels, the swirl film), and "Save as film" under the picture.
//
// A plain gpuwm-cm1 cartesian output folder opens view-only: its pictures and slices, and the commands that
// convert it, with no button that changes it.

import { h, num, clock, parseClock, bytes, button, copyline, panel, table } from "../core.js";
import * as api from "../api.js";
import { register, go, notice, runRoute } from "../router.js";
import { showCommand } from "../command.js";
import { folderLine } from "../folder.js";
import { ready as copyReady, t, looks as favouriteCopy } from "../../guided/copy.js";
import {
  FAVOURITES, LIVE_STATES, actionButton, continueClock, runExists, runNotFound, untilProblem, diskPanel, filesPanel, filmBar, filmLook,
  gridWords,
  headerClock, moreLookRows, moreLooksRow, noFrames, put, receiptPanel, segmentsTable, seriesPanel, stormLine, viewer,
  wallWords, words,
} from "./parts.js";

const REFRESH_MS = 5000;

// ------------------------------------------------------------------ guided

// netCDF with one file per saved time, and nothing else (what the presets write).
function perTimeNetcdf(files) {
  const names = (files || []).map((row) => row.name);
  return names.some((name) => /^cm1out_\d+\.nc$/.test(name)) && !names.some((name) => /\.(ctl|dat)$|^cm1out\.nc$/.test(name));
}

async function guidedResults(body, id, page) {
  await copyReady();
  const runUrl = encodeURIComponent(id);
  const clockEl = headerClock(t("watch.storm_time"));
  let data = await api.get(`${api.runPath(id)}/results?mode=guided`);
  const titles = Object.fromEntries(favouriteCopy().map((l) => [l.id, l.title]));
  const what = h("div", {});
  const strongest = h("p", { class: "g-strongest text" });
  // A picture with no storm in it yet says so here too, in place of its strongest value (which would be a number
  // from quiet air).
  const noStorm = h("p", { class: "nostorm", role: "note", hidden: true });
  let lookNow = FAVOURITES[0];
  const moreSlot = h("div", { class: "more-looks-slot" });
  const filmSlot = h("div", { class: "film-slot" });
  const moreRow = (id) => (data.more_looks || []).find((row) => row.id === id) || null;
  const view = viewer({
    order: [...FAVOURITES, ...(data.more_looks || []).map((row) => row.id)], tabIds: FAVOURITES, slider: true,
    empty: t("results.no_pictures"),
    emptyFor: (look) => { const row = moreRow(look.id); return row && row.why_not ? row.why_not : null; },
    caption: (look) => [h("span", {}, look.footer || "")],
    note: (look, f) => stormLine(f),
    onTime: (time, frame, look) => {
      if (frame && frame.time_s !== null && frame.time_s !== undefined) clockEl.set(time);
      const quiet = frame && frame.storm ? frame.storm : "";
      noStorm.textContent = quiet;
      noStorm.hidden = !quiet;
      strongest.textContent = frame && frame.strongest && !quiet ? t("results.strongest", { value: frame.strongest }) : "";
      if (look && look.id !== lookNow) { lookNow = look.id; paintWhat(); paintMore(); }
    },
    onLook: (look) => { if (look && look.id !== lookNow) { lookNow = look.id; paintWhat(); paintMore(); } },
  });
  view.setPlayLabels(t("results.play"), t("results.pause"));
  view.tabs.after(moreSlot);
  view.append(filmSlot);
  const whatBox = h("section", { class: "panel g-what" }, h("h2", { class: "cap" }, t("results.what")), what);
  // A run imported but not started: the way to start it, under the picture that says so.
  const startSlot = h("div", { class: "btns", hidden: true },
    button(t("watch.start_it"), { kind: "primary", onclick: () => go(runRoute("estimate", id)) }));
  view.append(startSlot);
  const lower = h("div", { class: "grid cols-3 section" });
  // Pictures still being drawn on the processor (a fast GPU ends before they do): said above the slider, so a
  // slider that ends early is not read as the storm ending there.
  const drawingNote = h("p", { class: "note", role: "status", hidden: true });
  body.append(h("div", { class: "grid cols-main" }, h("section", { class: "panel" }, drawingNote, view), whatBox), lower);

  const looksOf = () => [...(data.looks || []).filter((l) => FAVOURITES.includes(l.id))
    .map((l) => ({ ...l, title: titles[l.id] || l.title })), ...moreLookRows(data)];

  function paintWhat() {
    const g = data.guided || {};
    const more = moreRow(lookNow);
    const panelText = more || (g.looks || {})[lookNow];
    put(what,
      panelText ? h("h3", { class: "g-look" }, panelText.title) : null,
      panelText ? h("p", { class: "text" }, panelText.text) : null,
      more && more.note ? h("p", { class: "note" }, more.note) : null,
      more && !more.available && more.why_not ? h("p", { class: "note" }, more.why_not,
        more.hint ? h("span", { class: "hint" }, ` ${more.hint}`) : null) : null,
      more ? null : noStorm,
      more ? null : strongest,
      g.this_run ? h("p", { class: "note" }, g.this_run) : null);
  }

  function paintMore() {
    // A run with no saved frame has no look to draw and nothing to film.
    if (noFrames(data)) { put(moreSlot); put(filmSlot); return; }
    put(moreSlot, moreLooksRow(data, lookNow, (id) => { view.select(id); }));
    put(filmSlot, filmBar(id, data, lookNow, { always: true, refresh: () => refresh(true) }));
  }

  function paintLower() {
    const g = data.guided || {};
    const files = data.copy || [];
    // The run's folder as this computer's file manager names it (under Windows, the \\wsl.localhost name), with
    // Open folder; then the lines that open the files with a program.
    const folder = folderLine({ label: t("results.folder"), path: data.folder, windows: data.folder_windows,
      canOpen: data.can_open_folder, run: id });
    const trimmed = data.trimmed ? h("p", { class: "note" }, t("results.trimmed")) : null;
    const filesBody = files.length
      ? [folder, h("p", { class: "note haspath" }, t(perTimeNetcdf(data.cm1out) ? "results.files_where" : "results.files_where_other",
        { folder: data.cm1out_folder || "cm1out" })),
        files.map((c) => h("div", {}, h("div", { class: "muted" }, c.label), copyline(c.text, c.label)))]
      : [folder, trimmed || h("p", { class: "muted" }, t({ ended: "results.no_files_ended", not_started: "results.no_files_not_started" }[
        noFrames(data)] || "results.no_files"))];
    const longer = g.run_longer;
    const longerButton = button(t("results.run_longer"), {});
    const st = data.status || {};
    const actions = new Set(st.actions || []);
    longerButton.disabled = !longer || !actions.has("resume");
    // A greyed button says why: a run still going is offered it at its end; a run stopped before its end has no
    // saved point to go on from (its one restart is written at the end).
    let longerWhy = null;
    if (longer && longerButton.disabled) {
      if (data.trimmed) longerWhy = t("results.run_longer_trimmed");
      else if ([...LIVE_STATES, "stale"].includes(st.state)) longerWhy = t("results.run_longer_later");
      else if (["stopped", "failed"].includes(st.state)) longerWhy = t("results.run_longer_none");
    }
    longerButton.addEventListener("click", async () => {
      longerButton.disabled = true;
      try {
        const reply = await api.post(`${api.runPath(id)}/resume`, { until_s: longer.until_s });
        notice(reply.ok ? t("results.run_longer_started") : `${reply.message} ${reply.fix || ""}`.trim());
        if (reply.ok) go(`guided/watch/${runUrl}`);
      } catch (err) {
        notice(`${err.message} ${err.fix || ""}`.trim());
        longerButton.disabled = false;
      }
    });
    const everything = button(t("results.show_everything"), {});
    everything.addEventListener("click", async () => {
      try { await api.setMode("advanced"); } catch (err) { notice(err.message); return; }
      go(runRoute("results", id));
    });
    put(lower,
      panel(t("results.open_files"), filesBody),
      panel(t("results.run_longer"),
        longer && longer.about && !longerWhy ? h("p", { class: "note" }, t("results.run_longer_about", { wall: longer.about })) : null,
        longerWhy ? h("p", { class: "note" }, longerWhy) : null,
        h("div", { class: "btns" }, longerButton),
        longer ? showCommand(longer.command, { always: true }) : null),
      panel(t("results.show_everything"), h("p", { class: "note" }, t("results.show_everything_about")),
        h("div", { class: "btns" }, everything),
        showCommand((data.commands || {}).show_everything, { always: true })));
  }

  function paintAll(keep) {
    const g = data.guided || {};
    const st = data.status || {};
    page.setData([g.case || id, st.label].filter(Boolean).join(" · "));
    const none = noFrames(data);
    view.setEmpty(none ? t(none === "ended" ? "results.no_frames" : "results.not_started") : t("results.no_pictures"));
    startSlot.hidden = none !== "not_started";
    const behind = data.pictures_behind;
    drawingNote.hidden = !(behind && behind.behind > 0);
    drawingNote.textContent = drawingNote.hidden ? "" : t("watch.drawing",
      { drawn: behind.drawn, saved: behind.saved, wall: wallWords(behind.about_s) });
    view.update(looksOf(), keep);
    paintWhat();
    paintMore();
    paintLower();
  }
  paintAll(false);

  let closed = false;
  let timer = null;
  const busy = () => Object.values(data.films || {}).some((f) => f.state === "running") || data.drawing_looks
    || Boolean(data.pictures_behind);
  async function refresh(now) {
    if (closed) return;
    clearTimeout(timer);
    const live = LIVE_STATES.includes((data.status || {}).state) || busy();
    try {
      data = await api.get(`${api.runPath(id)}/results?mode=guided`);
      paintAll(true);
    } catch (_) { /* the next refresh reads it */ }
    if (!closed) timer = setTimeout(refresh, live || busy() ? (now ? 1500 : REFRESH_MS) : REFRESH_MS * 4);
  }
  timer = setTimeout(refresh, REFRESH_MS);
  return () => { closed = true; clearTimeout(timer); view.stop(); };
}

// ------------------------------------------------------------------ advanced: slices

// A click on a horizontal slice is a position in km (the slice's "map": where the map is in the picture).
function clickKm(img, map, ev) {
  if (!map) return null;
  const r = img.getBoundingClientRect();
  const [pw, ph] = map.picture_px || [1920, 1080];
  const px = (ev.clientX - r.left) * (pw / r.width);
  const py = (ev.clientY - r.top) * (ph / r.height);
  if (px < map.left || px > map.left + map.width || py < map.top || py > map.top + map.height) return null;
  const x = map.x_km[0] + ((px - map.left) / map.width) * (map.x_km[1] - map.x_km[0]);
  const y = map.y_km[1] - ((py - map.top) / map.height) * (map.y_km[1] - map.y_km[0]);
  return [Number(x.toFixed(3)), Number(y.toFixed(3))];
}

function slicesPanel(id, getTime) {
  // The field list is asked for (gpuwm-cm1 quicklook --fields, one line in commands.log) the first time the list
  // is opened, not on every visit; until then it offers w, which every run has.
  const fieldSel = h("select", { class: "input", style: "width:auto", "aria-label": "Field" },
    h("option", { value: "w" }, "w · vertical velocity"));
  const levelIn = h("input", { class: "input num", type: "number", min: "0", step: "50", value: "5000", style: "width:100px",
    "aria-label": "Height above ground (m)" });
  const drawH = button("Draw horizontal slice", { kind: "primary" });
  const drawV = button("Draw vertical section A–B", { disabled: true });
  const drawC = button("Draw column skew-T", {});
  const pointsNote = h("span", { class: "note" }, "Click the horizontal slice twice for A and B; once for a column.");
  const hImg = h("img", { alt: "horizontal slice" });
  const vImg = h("img", { alt: "vertical section" });
  const cImg = h("img", { alt: "column skew-T" });
  const hCap = h("figcaption", {});
  const vCap = h("figcaption", {});
  const cCap = h("figcaption", {});
  const hFig = h("figure", { hidden: true }, hImg, hCap);
  const vFig = h("figure", { hidden: true }, h("a", { target: "_blank", rel: "noopener" }, vImg), vCap);
  const cFig = h("figure", { hidden: true }, h("a", { target: "_blank", rel: "noopener" }, cImg), cCap);
  const cmd = h("div", {});
  let map = null;
  let points = [];
  let fields = [];
  const box = h("div", {},
    h("div", { class: "btns", style: "margin-bottom:10px" },
      h("label", { class: "muted" }, "Field "), fieldSel,
      h("label", { class: "muted" }, "Height "), levelIn, h("span", { class: "unit" }, "m"),
      drawH, drawV, drawC),
    h("p", { class: "note" }, pointsNote),
    h("div", { class: "grid cols-3 slices" }, hFig, vFig, cFig), cmd);

  async function loadFields() {
    try {
      const reply = await api.post(`${api.runPath(id)}/fields`, getTime() != null ? { time_s: getTime() } : {});
      fields = (reply.result && reply.result.fields) || [];
      const keep = fieldSel.value || "w";
      fieldSel.replaceChildren(...fields.map((f) => h("option", { value: f.name, selected: f.name === keep ? true : null },
        `${f.name}${f.description ? ` · ${f.description}` : ""}`)));
      if (!fields.length) fieldSel.append(h("option", { value: "w" }, "w"));
    } catch (err) {
      fieldSel.replaceChildren(h("option", { value: "w" }, "w"));
      notice(`${err.message} ${err.fix || ""}`.trim(), "warn");
    }
  }

  async function draw(action, payload, img, fig, cap, label) {
    const b = { slice: drawH, section: drawV, column: drawC }[action];
    b.disabled = true;
    try {
      const reply = await api.post(`${api.runPath(id)}/${action}`, payload);
      cmd.replaceChildren(showCommand(reply.command) || "");
      if (!reply.ok) { notice(`${reply.message} ${reply.fix || ""}`.trim(), "stop"); return null; }
      img.src = `${reply.picture_url}?v=${Date.now()}`;
      if (img.parentElement.tagName === "A") img.parentElement.href = reply.picture_url;
      fig.hidden = false;
      const r = reply.result || {};
      cap.replaceChildren(h("b", {}, label(r)), h("span", {}, clock(r.time_s || 0)));
      return r;
    } catch (err) {
      if (err.command) cmd.replaceChildren(showCommand(err.command) || "");
      notice(`${err.message} ${err.fix || ""}`.trim(), "stop");
      return null;
    } finally {
      b.disabled = action === "section" ? points.length < 2 : false;
    }
  }

  drawH.addEventListener("click", async () => {
    const payload = { field: fieldSel.value || "w", height_m: Number(levelIn.value || 0) };
    if (getTime() != null) payload.time_s = getTime();
    if (points.length === 2) payload.line_km = [...points[0], ...points[1]];
    const r = await draw("slice", payload, hImg, hFig, hCap, (rec) =>
      `${rec.field || payload.field} at ${rec.height_m != null ? `${num(rec.height_m)} m` : "the level asked"}` +
      (rec.min != null ? `, ${num(rec.min, 2)} to ${num(rec.max, 2)} ${rec.units || ""}` : ""));
    if (r) map = r.map || null;
  });
  drawV.addEventListener("click", () => {
    if (points.length < 2) return;
    const payload = { field: fieldSel.value || "w", from_km: points[0], to_km: points[1] };
    if (getTime() != null) payload.time_s = getTime();
    draw("section", payload, vImg, vFig, vCap, (rec) => `${rec.field || payload.field} along A–B`);
  });
  drawC.addEventListener("click", () => {
    const payload = {};
    if (points.length) payload.at_km = points[points.length - 1];
    if (getTime() != null) payload.time_s = getTime();
    draw("column", payload, cImg, cFig, cCap, (rec) =>
      rec.x_km != null ? `Column at ${num(rec.x_km, 1)}, ${num(rec.y_km, 1)} km` : "Column at the strongest updraft");
  });
  hImg.addEventListener("click", (ev) => {
    const at = clickKm(hImg, map, ev);
    if (!at) return;
    points = points.length >= 2 ? [at] : [...points, at];
    drawV.disabled = points.length < 2;
    pointsNote.textContent = points.length === 1
      ? `A at ${num(at[0], 1)}, ${num(at[1], 1)} km. Click B for a section, or Draw column skew-T for this column.`
      : `A–B from ${num(points[0][0], 1)}, ${num(points[0][1], 1)} to ${num(points[1][0], 1)}, ${num(points[1][1], 1)} km. ` +
        "Draw vertical section A–B; Draw horizontal slice again shows the line.";
  });
  let asked = null;
  const ask = () => { if (!asked) asked = loadFields(); return asked; };
  for (const kind of ["focus", "pointerdown", "keydown"]) fieldSel.addEventListener(kind, ask);
  box.load = ask;
  return box;
}

// ------------------------------------------------------------------ advanced

// Which of CM1's formats the run's cm1out folder holds, from its files.
function formatWords(files) {
  const names = (files || []).map((row) => row.name);
  const kinds = [];
  if (names.some((name) => /\.nc$/.test(name) && !/_stats|_pdata/.test(name))) kinds.push("netCDF");
  if (names.some((name) => /\.ctl$/.test(name))) kinds.push("GrADS");
  return kinds.length ? `${kinds.join(" and ")}, CM1's layout` : "";
}

async function advancedResults(body, id, page) {
  const clockEl = headerClock("model time");
  let data = await api.get(`${api.runPath(id)}/results?mode=advanced`);
  const cmds = data.commands || {};
  let shownTime = null;

  // Quick-looks with the time slider; every look is a tab, films play in place, and the film row is under it
  let lookNow = null;
  const filmSlot = h("div", { class: "film-slot" });
  const view = viewer({ slider: true, thumbs: false, empty: "No quick-looks yet: they are drawn as frames land.",
    emptyFor: (look) => { const row = (data.more_looks || []).find((r) => r.id === look.id); return row ? row.why_not : null; },
    caption: (look, f) => [h("b", {}, look.title || look.id), h("span", {}, f.time_s === null || f.time_s === undefined ?
      "a film picture · a click opens it" : `${clock(f.time_s || 0)} · a click opens the full picture`)],
    note: (look, f) => stormLine(f),
    onTime: (time, f) => {
      if (f && f.time_s !== null && f.time_s !== undefined) { shownTime = time; clockEl.set(time, frameWords()); }
    },
    onLook: (look) => { lookNow = look ? look.id : null; paintFilmBar(); } });
  view.append(filmSlot);
  const filmsBox = h("section", { class: "panel section" });
  const seriesBox = h("div", {});
  const slices = slicesPanel(id, () => shownTime);
  const filesBox = h("section", { class: "panel" });
  const receiptBox = h("section", { class: "panel" });
  const actionsBox = h("div", { class: "section" });
  const bundleBox = h("div", {});
  const viewOnly = h("div", {});
  const seriesSection = h("section", { class: "panel section" }, h("h2", { class: "cap" }, "Over time",
    h("span", { class: "n" }, "domain extremes at every saved frame")), seriesBox);
  const slicesSection = h("section", { class: "panel section" }, h("h2", { class: "cap" }, "Slices",
    h("span", { class: "n" }, "any cm1out field, at the time the slider shows")), slices);
  body.append(viewOnly,
    h("section", { class: "panel" }, h("h2", { class: "cap" }, "Quick-looks"), view),
    filmsBox, seriesSection, slicesSection,
    h("div", { class: "grid cols-main section" }, filesBox, receiptBox),
    actionsBox, bundleBox);

  function frameWords() {
    const frames = data.frames || [];
    const i = frames.findIndex((f) => Math.abs((f.time_s || 0) - (shownTime || 0)) < 0.5);
    return frames.length ? `frame ${i >= 0 ? i + 1 : frames.length} of ${frames.length}` : "model time";
  }

  let actionsKey = null;
  function paintActions() {
    const st = data.status || {};
    const actions = new Set(st.actions || []);
    // Redrawn only when what it offers changes, so a refresh never clears what is being typed.
    const none = noFrames(data);
    const key = JSON.stringify([st.state, st.actions, data.restarts, data.converting, data.view_only, none]);
    if (key === actionsKey) return;
    actionsKey = key;
    actionsBox.replaceChildren();
    if (data.view_only) return;
    const restartSel = h("select", { class: "input", style: "width:auto", "aria-label": "Continue from" },
      h("option", { value: "" }, "the last restart"),
      (data.restarts || []).map((r) => h("option", { value: String(r.number) },
        `restart ${r.number} (${clock(r.model_time_s || 0)}, segment ${String(r.segment).padStart(3, "0")})`)));
    const until = h("input", { class: "input num", style: "width:100px", placeholder: "T+2:30:00",
      "aria-label": "to (storm time, T+H:MM:SS)", value: continueClock(st) });
    const resume = actionButton(id, "resume", "Continue", {
      kind: "primary", command: cmds.resume,
      body: () => {
        const b = {};
        if (restartSel.value) b.from_restart = Number(restartSel.value);
        if (parseClock(until.value)) b.until_s = parseClock(until.value);
        return b;
      },
      watch: [restartSel, until],
      check: () => untilProblem(until.value),
      done: (reply) => { if (reply.ok) setTimeout(() => go(runRoute("run", id)), 600); },
    });
    resume.button.disabled = !actions.has("resume");
    resume.button.title = actions.has("resume") ? "" : (LIVE_STATES.includes(st.state) ? "offered once the run ends" :
      "no restart to continue from: this run stopped before its first restart");
    const formatSel = h("select", { class: "input", style: "width:auto", "aria-label": "Format" },
      h("option", { value: "" }, "as the namelist asks"), h("option", { value: "nc" }, "netCDF"),
      h("option", { value: "grads" }, "GrADS"));
    const convert = actionButton(id, "convert", "Convert", {
      command: cmds.convert, body: () => (formatSel.value ? { format: formatSel.value } : {}), watch: [formatSel],
      done: () => setTimeout(refresh, 1500),
    });
    const running = LIVE_STATES.includes(st.state) || st.state === "stale";
    convert.button.disabled = running || Boolean(none) || (data.converting || []).length > 0;
    convert.button.title = running ? "while the run goes, the cm1out sidecar converts each frame as it lands" :
      none ? "no saved frame to convert" : "";
    const reproduce = actionButton(id, "reproduce", "Write the bundle", {
      command: cmds.reproduce, title: "namelist, sounding, report, settings, deck, receipts and versions in one folder",
      done: (reply) => { data.reproduce_reply = reply; refresh(); },
    });
    actionsBox.append(h("section", { class: "panel" }, h("h2", { class: "cap" }, "Go on, convert, bundle"),
      h("div", { class: "actrow" }, h("div", { class: "what" }, h("b", {}, "Continue or extend"),
        h("span", { class: "note" }, "from the run's restarts; to a new end time. ",
          h("a", { href: `#/${runRoute("continue", id)}` }, "Every branch and a cluster job script")),
        h("div", { class: "btns" }, h("label", { class: "muted" }, "from ", restartSel),
          h("label", { class: "muted" }, "to storm time ", until))), resume),
      h("div", { class: "actrow" }, h("div", { class: "what" }, h("b", {}, "Convert again"),
        h("span", { class: "note" }, "every frame to cm1out files again, on the CPU"),
        h("div", { class: "btns" }, h("label", { class: "muted" }, "as ", formatSel),
          (data.converting || []).length ? h("span", { class: "note" }, "Converting now…") : null)), convert),
      h("div", { class: "actrow" }, h("div", { class: "what" }, h("b", {}, "Reproduce bundle"),
        h("span", { class: "note" }, "the namelist, sounding, report, settings, deck, receipts and versions in one " +
          "folder and a .tar.gz, with the commands that run it again")), reproduce)));
  }

  function paintBundle() {
    bundleBox.replaceChildren();
    const r = data.reproduce;
    if (!r) return;
    bundleBox.append(h("section", { class: "panel", style: "margin-top:20px" }, h("h2", { class: "cap" }, "Reproduce bundle",
      h("span", { class: "n" }, r.folder)),
    r.tarball_url ? h("p", {}, h("a", { href: r.tarball_url, download: "" }, "Download the bundle (.tar.gz)"),
      r.readme_url ? [" · ", h("a", { href: r.readme_url, target: "_blank", rel: "noopener" }, "README.txt")] : null) : null,
    table(["File", { label: "Size", num: true }], (r.files || []).map((f) => h("tr", {},
      h("td", { class: "mono" }, f.name), h("td", { class: "num" }, bytes(f.bytes || 0)))), { compact: true })));
  }

  function paintAll(keep) {
    const st = data.status || {};
    const run = data.quicklook_run || {};
    page.setData([id, run.grid || gridWords(st.grid), st.label].filter(Boolean).join(" · "));
    viewOnly.replaceChildren();
    if (data.view_only) {
      viewOnly.append(h("div", { class: "banner warn section" },
        "A plain gpuwm-cm1 cartesian output folder: it opens view-only. Convert it or draw its quick-looks from a terminal:",
        copyline(cmds.convert || "", "convert"), copyline(cmds.quicklook || "", "quick-looks")));
    }
    // A run with no saved frame: no film, chart or slice can be made, so they are not offered.
    const none = noFrames(data);
    view.setEmpty(none === "ended" ? "No quick-looks: this run ended before its first saved frame, so there is " +
      "nothing to show, film, slice or convert." : none === "not_started" ? "No quick-looks: this run has not started." :
      "No quick-looks yet: they are drawn as frames land.");
    filmsBox.hidden = Boolean(none);
    seriesSection.hidden = Boolean(none);
    slicesSection.hidden = Boolean(none);
    const swirl = filmLook(data, "swirl", ((data.more_looks || []).find((r) => r.id === "swirl") || {}).title || "Swirl near the ground");
    view.update([...(data.looks || []), ...(swirl ? [swirl] : [])], keep);
    paintFilmBar();
    paintFilms();
    put(seriesBox, seriesPanel(data.series_doc, { now: shownTime, tmax: st.duration_s,
      empty: "The charts appear once the quick-look worker has drawn a frame." }));
    put(filesBox, h("h2", { class: "cap" }, "cm1out", h("span", { class: "n" }, formatWords(data.cm1out))),
      data.trimmed ? h("p", { class: "note" }, "This run's big files (cm1out, the saved frames and the restarts) were " +
        "deleted to free disk space; its quick-looks are kept. It cannot be continued.") : null,
      ...filesPanel(data, { open: true }),
      folderLine({ label: "The run's folder", path: data.folder, windows: data.folder_windows,
        canOpen: data.can_open_folder, run: id }));
    put(receiptBox, h("h2", { class: "cap" }, "Receipt"), receiptPanel(data.receipt_summary),
      h("h3", { class: "cap section" }, "Disk"), diskPanel(data.disk),
      (data.segments || []).length ? [h("h3", { class: "cap section" }, "Segments"), segmentsTable(data.segments)] : null,
      data.receipt ? h("p", { class: "note" }, "The full receipt: ",
        h("a", { href: data.receipt_url || "#", target: "_blank", rel: "noopener" }, "receipt.json")) : null);
    paintActions();
    paintBundle();
  }

  function paintFilmBar() {
    if (noFrames(data)) { put(filmSlot); return; }
    put(filmSlot, filmBar(id, data, lookNow, { always: false, refresh }));
  }

  // Films: any look, any format, with its live Show command; the films made, and the ones being made.
  let filmsKey = null;
  function paintFilms() {
    const key = JSON.stringify([data.film_looks, data.films, data.mp4_available, (data.more_looks || []).map((r) => [r.id, r.action])]);
    if (key === filmsKey) return;
    filmsKey = key;
    const looks = data.film_looks || [];
    const titleOf = (lid) => ((data.looks || []).find((l) => l.id === lid) || (data.more_looks || []).find((r) => r.id === lid) || {}).title || lid;
    const lookSel = h("select", { class: "input", style: "width:auto", "aria-label": "Film of" },
      looks.map((lid) => h("option", { value: lid, selected: lid === "swath" ? true : null }, titleOf(lid))));
    const formatSel = h("select", { class: "input", style: "width:auto", "aria-label": "Film format" },
      h("option", { value: "" }, data.mp4_available ? "animated PNG and MP4" : "animated PNG and GIF"),
      h("option", { value: "apng" }, "animated PNG only"),
      h("option", { value: "gif" }, "GIF too"),
      h("option", { value: "mp4", disabled: data.mp4_available ? null : true }, data.mp4_available ? "MP4 too" : "MP4 (needs the film extra)"));
    const fpsIn = h("input", { class: "input num", type: "number", min: "0.5", max: "60", step: "1", style: "width:80px",
      "aria-label": "Pictures per second", placeholder: "auto" });
    const betweenIn = h("input", { class: "input num", type: "number", min: "0", max: "30", step: "1", value: "8",
      style: "width:70px", "aria-label": "Blended pictures between saved times (swirl)" });
    const make = actionButton(id, "film", "Make the film", {
      kind: "primary",
      body: () => {
        const b = { look: lookSel.value };
        if (formatSel.value) b.format = formatSel.value;
        if (fpsIn.value) b.fps = Number(fpsIn.value);
        if (lookSel.value === "swirl" && betweenIn.value !== "") b.between = Number(betweenIn.value);
        return b;
      },
      watch: [lookSel, formatSel, fpsIn, betweenIn],
      done: () => setTimeout(refresh, 800),
    });
    const rows = Object.values(data.films || {}).filter((f) => f.state !== "none").map((f) => {
      const p = f.progress || {};
      const state = f.state === "running" ? (p.total ? `making: ${p.done || 0} of ${p.total} pictures` : "making") :
        f.state === "done" ? `${f.pictures || 0} pictures at ${f.fps || ""} per second` : f.state === "failed" ? `not made: ${p.message || ""}` :
          `stopped after ${p.done || 0} pictures`;
      const urls = f.urls || {};
      const links = ["mp4", "gif", "apng"].filter((k) => urls[k]).map((k) =>
        h("a", { href: urls[k], target: k === "apng" ? "_blank" : null, rel: "noopener", download: k === "apng" ? null : "" },
          { mp4: "MP4", gif: "GIF", apng: "animated PNG" }[k]));
      const cells = [h("td", {}, titleOf(f.look)), h("td", {}, state), h("td", {}, ...links.flatMap((a, n) => (n ? [" · ", a] : [a])))];
      return h("tr", {}, ...cells);
    });
    const draws = (data.more_looks || []).filter((r) => r.action === "all" || r.action === "look");
    put(filmsBox, h("h2", { class: "cap" }, "Films", h("span", { class: "n" },
      "any look over time, on the CPU at nice 19; written to quicklook/films/")),
    h("div", { class: "btns" }, h("label", { class: "muted" }, "of ", lookSel), h("label", { class: "muted" }, "as ", formatSel),
      h("label", { class: "muted" }, "pictures per second ", fpsIn),
      h("label", { class: "muted" }, "swirl: blended pictures between saved times ", betweenIn), make),
    data.mp4_available ? null : h("p", { class: "note" }, words(data, "no_mp4"), " ",
      h("code", {}, data.mp4_fix || 'pip install "gpuwm-cm1[film]"')),
    rows.length ? table(["Look", "Film", "Files"], rows, { compact: true }) : null,
    draws.length ? h("p", { class: "note" }, `Not drawn for this run yet: ${draws.map((r) => r.title).join(", ")}. ` +
      "Open the look above to draw it.") : null);
  }

  async function loadSeries() {
    if (!data.series || !data.series.url) { data.series_doc = null; return; }
    try { data.series_doc = await api.get(data.series.url); } catch (_) { data.series_doc = null; }
  }
  await loadSeries();
  paintAll(false);

  let closed = false;
  let timer = null;
  async function refresh() {
    clearTimeout(timer);
    if (closed) return;
    try {
      data = await api.get(`${api.runPath(id)}/results?mode=advanced`);
      await loadSeries();
      paintAll(true);
    } catch (_) { /* the next refresh reads it */ }
    const live = LIVE_STATES.includes((data.status || {}).state) || (data.converting || []).length ||
      Object.values(data.films || {}).some((f) => f.state === "running") || data.drawing_looks;
    if (!closed) timer = setTimeout(refresh, live ? REFRESH_MS : REFRESH_MS * 6);
  }
  timer = setTimeout(refresh, REFRESH_MS);
  return () => { closed = true; clearTimeout(timer); view.stop(); };
}

// ------------------------------------------------------------------ the screen

export async function renderResults(body, args, page = {}) {
  const id = decodeURIComponent(args[0] || "");
  const mode = page.mode || api.currentMode();
  const frame = { setData: page.setData || ((text) => { const d = document.getElementById("data"); if (d) d.textContent = text; }) };
  if (!id) {
    body.append(h("p", { class: "banner warn" }, "No run named.", h("span", { class: "sub" }, "Open one from Runs.")));
    return null;
  }
  const known = await runExists(id);
  if (!known.ok) return runNotFound(body, id, known.err);
  return mode === "guided" ? guidedResults(body, id, frame) : advancedResults(body, id, frame);
}

register("results", (body, args, page) => renderResults(body, args, page), { title: "Results", runPage: true });
