// Guided screen 2: How long can it run?. Four times (the last has no limit); under them "Customize this storm" (closed until opened), the
// fitted run in one sentence, what you get (plain words, CM1's names as grey hints), what you changed, each change
// the fit made, and one Start button with its command. A preset that runs a sounding file may be given one in the
// route's query: #/guided/fit/ID/BUDGET?sounding=FILE (the column the WRF card keeps; with &start=push|bubble, the
// file the Use my own sounding card keeps and how its storm is started) or
// ?instability=I&shear=S&hodograph=H (the Make-your-own-environment card, or the air in Customize); the fit and
// Start then carry it, and "The air" says what it is. The route also keeps the Customize choices
// (?grid=500&area=96&...; customize.js) and ?customize=1 opens the panel, so a reload or Back shows the same fit.

import { h } from "../js/core.js";
import * as api from "../js/api.js";
import { go, notice, runRoute } from "../js/router.js";
import { showCommand } from "../js/command.js";
import { t, term, wallWords, spanWords, gb, metres, shortGpu } from "./copy.js";
import { startOptions } from "../js/sounding/options.js";
import { customizePanel, customFrom, customQuery, changedKnobs, changedWords, changesStorm } from "./customize.js";

// "any" is "As long as it takes": no limit on the time, for a run that needs more than overnight's 10 hours.
const BUDGETS = ["10m", "1h", "overnight", "any"];
const REFIT_MS = 350;
// Why a fit is not offered (fit.json reason_kind), and which of them a longer time can help.
const BADGES = ["time", "gpu", "host", "disk", "edge", "limits", "unpriced", "drift", "withheld", "check"];
const LONGER_HELPS = ["time", "gpu", "host", "disk", "edge", "limits", "unpriced"];
// At most this many fits at once when the screen looks for a remedy.
const PARALLEL = 4;

// "All settings (Advanced)" shows the new folder in Advanced for this visit only: the saved mode is not changed,
// and a guided screen reached again (Back) is guided again.
const HOP_KEY = "gm1.guided-hop";
let hopHere = false;

function markHop() {
  hopHere = true;
  try { window.sessionStorage.setItem(HOP_KEY, "1"); } catch (_) { /* the flag above is enough for this tab */ }
}

export function takeAdvancedHop() {
  let hop = hopHere;
  hopHere = false;
  try {
    hop = hop || window.sessionStorage.getItem(HOP_KEY) === "1";
    window.sessionStorage.removeItem(HOP_KEY);
  } catch (_) { /* no storage: the flag above */ }
  return hop;
}

// "10m?instability=0.5&shear=0.5" -> ["10m", {environment: {...}}]; "?sounding=FILE" -> {sounding: FILE}.
export function givenFrom(value) {
  const [head, query = ""] = String(value || "").split("?");
  const params = new URLSearchParams(query);
  const given = {};
  if (params.get("sounding")) {
    given.sounding = params.get("sounding");
    if (params.get("start")) given.start = params.get("start");
  } else if (params.has("instability") && params.has("shear")) {
    given.environment = {
      instability: Number(params.get("instability")), shear: Number(params.get("shear")),
      hodograph: params.get("hodograph") || "curved",
    };
  }
  return [head, given, query];
}

// The fit's reason without its own "Not offered ...:" opening (the badge says that).
export function reasonWords(reason) {
  const text = String(reason || "").replace(/^Not offered( yet| in [^:]+)?:\s*/i, "");
  return text ? text[0].toUpperCase() + text.slice(1) : "";
}

function airRow(fit) {
  const snd = fit.sounding;
  if (!snd) return null;
  const made = snd.environment;
  let text;
  if (made && made.sentence) {
    const n = made.numbers;
    text = n && typeof n.cape_j_kg === "number"
      ? t("budget.air_made", { sentence: made.sentence, cape: Math.round(n.cape_j_kg), shear: Math.round(n.shear_0_6km_m_s) })
      : made.sentence;
  } else if (snd.given && snd.own && snd.source) {
    // A file of the user's own: its words already begin "your sounding NAME" (own_sounding.source_words).
    text = `${snd.source.charAt(0).toUpperCase()}${snd.source.slice(1)}`;
  } else if (snd.given) {
    text = t("budget.air_file", { source: snd.source || snd.file || "" });
  } else {
    text = snd.observation ? t("budget.air_observed", { observation: snd.observation }) : "";
  }
  return text ? row("air", text) : null;
}

function row(key, value, cm1) {
  const [label, hintEl] = term(key, cm1);
  return h("tr", {}, h("td", { class: "muted" }, label, hintEl), h("td", { class: "text" }, value));
}

function youGet(fit, p) {
  const f = fit.facts || {};
  const est = fit.estimate || {};
  const gpu = fit.gpu;
  const rows = [];
  // The area as it is simulated: whole km as they are, anything else to a tenth, with its cells.
  const km = (v) => (Math.abs(v - Math.round(v)) < 0.05 ? String(Math.round(v)) : v.toFixed(1));
  // A typed 82.9 m is held as 82.9000015258789 (the model's 4-byte word for it): seven figures say the same number.
  const real = (v) => Number(Number(v).toPrecision(7));
  if (f.dx_m) rows.push(row("grid_spacing", metres(f.dx_m), real(f.dx_m)));
  if (f.domain_km) {
    rows.push(row("area", `${km(f.domain_km[0])} × ${km(f.domain_km[1])} km (${t("budget.cells", { nx: f.nx, ny: f.ny })})`, `${f.nx}, ${f.ny}`));
  }
  // A stretched column has no one spacing: it says its thinnest and thickest level.
  if (f.nz && f.stretch_z && f.dz_bottom_m && f.dz_top_m) {
    rows.push(row("levels", t("budget.levels_stretched", { count: f.nz, bottom: metres(f.dz_bottom_m), top: metres(f.dz_top_m) }),
      `${f.nz}, ${f.dz_bottom_m}, ${f.dz_top_m}`));
  } else if (f.nz) rows.push(row("levels", `${f.nz} × ${metres(f.dz_m)}`, `${f.nz}, ${real(f.dz_m)}`));
  if (f.dt_s) rows.push(row("time_step", `${Number(Number(f.dt_s).toPrecision(7))} s`, Number(Number(f.dt_s).toPrecision(7))));
  if (f.duration_s) rows.push(row("storm_time", spanWords(f.duration_s), f.duration_s));
  if (f.output_interval_s) rows.push(row("picture_every", spanWords(f.output_interval_s), f.output_interval_s));
  if (p.facts && p.facts.scheme) rows.push(row("scheme", p.facts.scheme, p.facts.ptype));
  const air = airRow(fit);
  if (air) rows.push(air);
  if (est.gpu_memory_bytes) {
    const free = gpu ? ` of ${gb(gpu.free_bytes)} free` : "";
    rows.push(row("gpu_memory", `about ${gb(est.gpu_memory_bytes)}${free} (${t("common.estimated")})`));
  }
  if (est.disk_bytes) rows.push(row("disk", `about ${gb(est.disk_bytes)} (${t("common.estimated")})`));
  if (est.wall_s) {
    const where = gpu ? shortGpu(gpu.name) : "RTX 3080";
    rows.push(row("wall", `about ${wallWords(est.wall_s)} on ${where} (${t("common.estimated")}; ` +
      `${wallWords(est.wall_low_s)} to ${wallWords(est.wall_high_s)})`));
  }
  return h("div", { class: "tablewrap" }, h("table", { class: "compact g-got" }, h("tbody", {}, rows)));
}

// The live line under Customize: "About N min on your RTX 4090 · X GB of GPU memory · Y GB of disk (estimated)".
function liveWords(reply) {
  const fit = reply.fit || {};
  const est = fit.estimate || {};
  const values = { wall: wallWords(est.wall_s), memory: gb(est.gpu_memory_bytes), disk: gb(est.disk_bytes) };
  return fit.gpu ? t("custom.live", { ...values, gpu: shortGpu(fit.gpu.name) }) : t("custom.live_nogpu", values);
}

// A few of `values`, spread from the first to the last (the last always kept).
function spread(values, count) {
  if (values.length <= count) return values;
  const out = [];
  for (let i = 1; i <= count; i += 1) out.push(values[Math.round((i * values.length) / count) - 1]);
  return [...new Set(out)];
}

// Runs the calls `jobs` (functions returning promises) at most PARALLEL at a time; failures become null.
async function pooled(jobs) {
  const out = new Array(jobs.length).fill(null);
  let next = 0;
  const worker = async () => {
    while (next < jobs.length) {
      const i = next;
      next += 1;
      try { out[i] = await jobs[i](); } catch (_) { out[i] = null; }
    }
  };
  await Promise.all(Array.from({ length: Math.min(PARALLEL, jobs.length) }, worker));
  return out;
}

export async function budget(body, args) {
  const [idPart, givenInId, queryInId] = givenFrom(args[0]);
  const [budgetPart, givenInBudget, queryInBudget] = givenFrom(args[1]);
  const presetId = decodeURIComponent(idPart || "");
  const given = Object.keys(givenInBudget).length ? givenInBudget : givenInId;
  const params = new URLSearchParams(queryInBudget || queryInId || "");
  let chosen = BUDGETS.includes(budgetPart) ? budgetPart : null;
  const list = await api.get("/api/presets");
  let p = (list.presets || []).find((x) => x.id === presetId);
  if (!p) {
    body.append(h("p", { class: "banner stop" }, t("budget.back")));
    go("guided");
    return null;
  }
  // The user's own sounding (the Use my own sounding card): the storm's settings as they are with the start that
  // card chose, and the sounding's own picture in place of the preset's.
  const ownStart = given.sounding && given.start ? given.start : null;
  const ownKey = ownStart ? (String(given.sounding).match(/[\\/]soundings[\\/]([0-9a-f]{20})[\\/]input_sounding$/) || [])[1] : null;
  if (ownStart) {
    const own = await startOptions(p.id, ownStart);
    p = { ...p, customize: own || p.customize, note: t("budget.own_note"),
      picture: ownKey ? { url: `/api/sounding/files/${ownKey}/preview.png`, caption: t("budget.own_caption") } : null };
  }
  // A sounding file is the air: the two sliders of Make your own environment would not change it, so the row says
  // where the air comes from instead of offering them.
  if (given.sounding && p.customize && p.customize.air) {
    p = { ...p, customize: { ...p.customize, air: { available: false, why: t("budget.file_air") } } };
  }
  chosen = chosen || p.budget;
  document.getElementById("data").textContent = t("budget.data", { name: p.name, line: p.line });
  const knobsById = {};
  for (const knob of (p.customize && p.customize.knobs) || []) knobsById[knob.id] = knob;

  // What Customize holds: the knobs the user changed and the made air. A sounding file (the WRF card's) is not
  // Customize's: it stays as it came.
  const state = {
    custom: customFrom(params),
    environment: given.environment || null,
    open: params.get("customize") === "1",
  };
  const sounding = given.sounding || null;
  const requestWith = (custom, environment) => {
    const out = {};
    if (sounding) out.sounding = sounding;
    else if (environment) out.environment = environment;
    if (sounding && ownStart) out.start = ownStart;
    if (Object.keys(custom).length) out.custom = { ...custom };
    return out;
  };
  const request = () => requestWith(state.custom, state.environment);
  const routeFor = (b) => {
    const query = sounding ? new URLSearchParams({ sounding, ...(ownStart ? { start: ownStart } : {}) }).toString()
      : customQuery(state.custom, state.environment);
    const extra = sounding && Object.keys(state.custom).length ? `&${customQuery(state.custom, null)}` : "";
    const open = state.open ? `${query || extra ? "&" : ""}customize=1` : "";
    const all = `${query}${extra}${open}`;
    return `#/guided/fit/${encodeURIComponent(presetId)}/${b}${all ? `?${all}` : ""}`;
  };

  const buttons = {};
  const choice = h("div", { class: "g-budgets", role: "group", "aria-label": t("budget.title") });
  for (const b of BUDGETS) {
    buttons[b] = h("button", { type: "button", class: "btn", "aria-pressed": "false" }, t(`budget.${b}`));
    buttons[b].addEventListener("click", () => pick(b));
    choice.append(buttons[b]);
  }
  function pick(b) {
    if (b === chosen) return;
    chosen = b;
    // The route keeps the time, the sounding and the choices, so a reload or Back shows the same fit.
    history.replaceState(null, "", routeFor(b));
    paint();
    fitNow();
  }
  const summary = h("p", { class: "g-summary text", tabindex: "-1" });
  // The settings of the fitted storm (the preset's and the user's own) that are not yet compared with CM1.
  const unchecked = h("p", { class: "note g-unchecked", hidden: true });
  const help = h("div", { class: "g-cz-help" });
  const got = h("div", {});
  // What an offered run should be read with (the storm near the edge, an untried size or spacing): it still runs.
  const notes = h("div", { class: "g-notes" });
  const yours = h("div", {});
  const changesHead = h("h2", { class: "cap", hidden: true }, t("budget.changes"));
  const changes = h("div", {});
  const start = h("button", { class: "btn primary g-start", type: "button", disabled: true }, t("budget.start"));
  const cmd = h("div", {});
  const caption = h("figcaption", {}, (p.picture && p.picture.caption) || "");
  const pic = p.picture && p.picture.url
    ? h("figure", { class: "g-pic" }, h("img", { src: p.picture.url, alt: p.name }), caption)
    : null;

  let current = null;
  let serial = 0;
  let timer = null;
  // Customize is for a storm guided mode offers: a withheld one has nothing in it to change that.
  const panel = p.customize && p.offered ? customizePanel(p, p.customize, state, {
    onChange: ({ refit }) => {
      history.replaceState(null, "", routeFor(chosen));
      paintCaption();
      if (!refit) return;
      // As the environment card does: a short pause, and a counter so a slow reply never overwrites a newer one.
      serial += 1;
      current = null;
      start.disabled = true;
      hold();
      panel.showFit(null);
      clearTimeout(timer);
      timer = setTimeout(fitNow, REFIT_MS);
    },
    onAdvanced: makeForAdvanced,
    onSeeWhy: () => {
      summary.scrollIntoView({ block: "start", behavior: "smooth" });
      summary.focus({ preventScroll: true });
    },
  }) : null;

  const left = h("section", { class: "panel" }, choice, panel ? panel.el : null, summary, notes, unchecked, help, got,
    yours, changesHead, changes,
    h("div", { class: "btns g-go" }, start, h("a", { href: "#/guided" }, t("budget.back"))), cmd,
    h("p", { class: "note" }, t("budget.estimated")));
  body.append(h("div", { class: "grid cols-main" }, left, h("section", { class: "panel" }, pic, p.note ? h("p", { class: "note" }, p.note) : null)));

  // While a fit is redrawn the column keeps its height, so the page under the reader never shrinks and jumps.
  function hold() { if (!left.style.minHeight) left.style.minHeight = `${left.offsetHeight}px`; }
  function release() { left.style.minHeight = ""; }

  function paint() {
    for (const [b, el] of Object.entries(buttons)) {
      el.classList.toggle("on", b === chosen);
      el.setAttribute("aria-pressed", b === chosen ? "true" : "false");
    }
  }

  // The preset's picture is of the preset as it ships: say so once something is changed.
  function paintCaption() {
    if (!pic) return;
    const base = (p.picture && p.picture.caption) || "";
    const own = p.card === "environment";
    if (ownStart) { caption.textContent = base; return; }
    const what = changedWords(state.custom, own ? null : state.environment, knobsById);
    if (!what) { caption.textContent = base; return; }
    let text = t("custom.caption", { name: p.name, what });
    if (changesStorm(state.custom, own ? null : state.environment, knobsById)) {
      // A hurricane has no air or push to change: its one storm setting is the sea temperature.
      text += ` ${t(knobsById.sea ? "custom.caption_storm_sea" : "custom.caption_storm")}`;
    }
    caption.textContent = text;
  }

  const airIsMine = () => Boolean(state.environment && p.customize && p.customize.air && p.customize.air.weisman_klemp);
  function customized() {
    return changedKnobs(state.custom, knobsById).length > 0 || airIsMine();
  }

  function refitWith(change) {
    change();
    if (panel) panel.paint();
    history.replaceState(null, "", routeFor(chosen));
    paintCaption();
    fitNow();
  }

  // A setting that is taken instead of the one turned down ({grid: 23}, {levels: 78}, {area: 36.4}), as one-tap
  // buttons. The grid is never finer than this card has room for (the panel's own "finest grid").
  function suggestButtons(suggest) {
    const out = [];
    const add = (knob, value, words) => {
      const b = h("button", { type: "button", class: "btn primary g-suggest", dataset: { use: knob } }, words);
      b.addEventListener("click", () => refitWith(() => { state.custom[knob] = value; }));
      out.push(b);
    };
    if (!suggest || !panel) return out;
    if (typeof suggest.grid === "number" && suggest.grid > 0) {
      const finest = panel.finestGrid() || 0;
      // A spacing above the card's finest is used as it is; one below it becomes the card's finest.
      const value = suggest.grid >= finest ? suggest.grid : finest;
      add("grid", value, t("custom.use_grid", { value: metres(value) }));
    }
    if (typeof suggest.levels === "number" && suggest.levels > 0) {
      add("levels", suggest.levels, t("custom.use_levels", { value: metres(suggest.levels) }));
    }
    if (typeof suggest.area === "number" && suggest.area > 0) {
      add("area", suggest.area, t("custom.use_area", { value: `${Number(suggest.area.toFixed(1))} km` }));
    }
    return out;
  }

  // Not offered with the user's own choices: what would take them (a longer time, a larger area, one change
  // undone), and a button per choice to hand back to GM1.
  async function helpFor(reply, mine) {
    const fit = reply.fit || {};
    const kind = fit.reason_kind || "";
    if (kind === "withheld" || kind === "check") return;
    const pinned = ["grid", "area", "time", "pictures"].filter((knob) => (fit.pinned || []).includes(knob));
    // What the fit says would be offered comes first (a coarser grid that fits), then handing choices back.
    const handBack = suggestButtons((fit.needs || {}).suggest);
    for (const knob of pinned) {
      const b = h("button", { type: "button", class: "btn" }, t("custom.let_choose", { what: t(`custom.what.${knob}`) }));
      b.addEventListener("click", () => refitWith(() => { delete state.custom[knob]; }));
      handBack.push(b);
    }
    if (handBack.length) help.append(h("div", { class: "btns" }, handBack));
    if (kind === "drift") { await keepInside(mine); return; }
    if (!LONGER_HELPS.includes(kind)) return;
    // The longer times, all at once, until one takes these choices.
    const later = BUDGETS.slice(BUDGETS.indexOf(chosen) + 1);
    if (!later.length) return;
    const checking = h("p", { class: "note" }, t("custom.checking_longer"));
    help.prepend(checking);
    const replies = await pooled(later.map((b) => () => api.post("/api/fit", { preset: p.id, budget: b, ...request() })));
    if (mine !== serial) return;
    checking.remove();
    const i = replies.findIndex((r) => r && r.offered);
    if (i < 0) return;
    const ofit = replies[i].fit || {};
    const wall = wallWords((ofit.estimate || {}).wall_s);
    // The longer time is for the area that time fits, which can be larger than the one the reason above is for.
    const side = Math.max(...(((ofit.facts || {}).domain_km) || [0]));
    const values = { wall, budget: t(`budget.${later[i]}`), area: `${Number(side.toFixed(1))} km` };
    const words = ofit.gpu ? t("custom.need_longer", { ...values, gpu: shortGpu(ofit.gpu.name) }) : t("custom.need_longer_nogpu", values);
    const use = h("button", { type: "button", class: "btn primary" }, t("custom.use_budget", { budget: t(`budget.${later[i]}`) }));
    use.addEventListener("click", () => pick(later[i]));
    help.prepend(h("p", { class: "text" }, words, " ", use));
  }

  // The storm leaves the area: the smallest larger area that keeps it in, and which one change, undone, would.
  async function keepInside(mine) {
    const areas = spread(panel ? panel.largerAreas() : [], 5);
    // The fit's own levers already have their "Let GM1 choose" button above; the area is tried on its own.
    const undoable = changedKnobs(state.custom, knobsById).filter((k) => !["grid", "area", "time", "pictures"].includes(k));
    if (airIsMine()) undoable.push("air");
    const tries = undoable.slice(0, 6);
    if (!areas.length && !tries.length) return;
    const checking = h("p", { class: "note" }, t("custom.checking_inside"));
    help.append(checking);
    const inside = (r) => Boolean(r && (r.offered || ((r.fit || {}).reason_kind && r.fit.reason_kind !== "drift")));
    const ask = (custom, environment) => () => api.post("/api/fit", { preset: p.id, budget: chosen, ...requestWith(custom, environment) });
    const without = (knob) => { const c = { ...state.custom }; delete c[knob]; return c; };
    const replies = await pooled([
      ...areas.map((a) => ask({ ...state.custom, area: a }, state.environment)),
      ...tries.map((k) => (k === "air" ? ask(state.custom, null) : ask(without(k), state.environment))),
    ]);
    if (mine !== serial) return;
    checking.remove();
    const area = areas.find((a, i) => inside(replies[i]));
    if (area !== undefined) {
      const use = h("button", { type: "button", class: "btn primary" }, t("custom.use_area", { value: `${Math.round(area)} km` }));
      use.addEventListener("click", () => refitWith(() => { state.custom.area = area; }));
      help.append(h("p", { class: "text" }, t("custom.inside_area", { value: `${Math.round(area)} km` }), " ", use));
    } else if (areas.length) {
      help.append(h("p", { class: "text" }, t("custom.inside_none", { value: `${Math.round(areas[areas.length - 1])} km` })));
    }
    const fixes = tries.filter((k, i) => inside(replies[areas.length + i]));
    if (fixes.length) {
      const undo = fixes.map((k) => {
        const b = h("button", { type: "button", class: "btn" }, t("custom.undo_what", { what: t(`custom.what.${k}`) }));
        b.addEventListener("click", () => refitWith(() => { if (k === "air") state.environment = null; else delete state.custom[k]; }));
        return b;
      });
      help.append(h("p", { class: "text" }, t(fixes.length === 1 ? "custom.inside_undo_one" : "custom.inside_undo")),
        h("div", { class: "btns" }, undo));
    }
  }

  // The storm comes close to the edge of an area the user chose: the smallest larger area with no such note, as
  // one button (the area slider's own steps are tried, a few at a time).
  async function clearOfEdge(mine, li, setUpForKm) {
    const larger = panel ? panel.largerAreas() : [];
    // The first step at least as wide as the area the storm was set up for is the likeliest smallest area without
    // the note, and a spread of six can step over it: it is tried with them.
    const own = larger.find((a) => a >= Number(setUpForKm) - 1e-6);
    const areas = [...new Set([...(own === undefined ? [] : [own]), ...spread(larger, 6)])].sort((a, b) => a - b);
    if (!areas.length) return;
    const replies = await pooled(areas.map((a) => () => api.post("/api/fit", { preset: p.id, budget: chosen, ...requestWith({ ...state.custom, area: a }, state.environment) })));
    if (mine !== serial || !li.isConnected) return;
    const clear = (r) => Boolean(r && r.offered && !((r.fit || {}).warning_records || []).some((w) => w.kind === "edge"));
    const area = areas.find((a, i) => clear(replies[i]));
    if (area === undefined) return;
    const use = h("button", { type: "button", class: "btn g-follow" }, t("custom.use_area", { value: `${Math.round(area)} km` }));
    use.addEventListener("click", () => refitWith(() => { state.custom.area = area; }));
    li.append(" ", t("custom.edge_area", { value: `${Math.round(area)} km` }), " ", use);
  }

  async function fitNow() {
    const mine = ++serial;
    current = null;
    start.disabled = true;
    hold();
    try {
      summary.textContent = t("budget.fitting");
      unchecked.hidden = true;
      help.replaceChildren();
      notes.replaceChildren();
      got.replaceChildren();
      yours.replaceChildren();
      changes.replaceChildren();
      changesHead.hidden = true;
      cmd.replaceChildren();
      if (panel) panel.showFit(null);
      let reply;
      try {
        reply = await api.post("/api/fit", { preset: p.id, budget: chosen, ...request() });
      } catch (err) {
        if (mine !== serial) return;
        // A choice this storm cannot take (the fit names it): the badge says it is the settings.
        summary.replaceChildren(h("span", { class: "st refused" }, t("budget.badge.setting")), " ",
          `${err.message} ${err.fix || ""}`.trim());
        if (panel) panel.showFit({ offered: false });
        // The value the message names as taken, one tap away, and the choice handed back to GM1.
        const suggest = (err.body && err.body.suggest) || {};
        const use = suggestButtons(suggest);
        for (const knob of Object.keys(suggest)) {
          if (state.custom[knob] === undefined || !["grid", "area", "levels"].includes(knob)) continue;
          const words = knob === "levels" ? t("custom.as_preset_what", { what: t("custom.what.levels") }) : t("custom.let_choose", { what: t(`custom.what.${knob}`) });
          const b = h("button", { type: "button", class: "btn" }, words);
          b.addEventListener("click", () => refitWith(() => { delete state.custom[knob]; }));
          use.push(b);
        }
        if (use.length) help.replaceChildren(h("div", { class: "btns" }, use));
        return;
      }
      if (mine !== serial) return;
      const fit = reply.fit || {};
      if (ownStart && fit.sounding && fit.sounding.label) {
        document.getElementById("data").textContent = t("budget.own_data", { name: fit.sounding.label });
      }
      if (panel) panel.showFit(reply, liveWords);
      const mine2 = fit.customized || [];
      if (mine2.length) {
        yours.replaceChildren(h("h2", { class: "cap" }, t("custom.you_changed")),
          h("ul", { class: "g-changes" }, mine2.map((c) => h("li", {}, c.sentence))));
      }
      if (!reply.offered) {
        const kind = fit.reason_kind;
        const badge = reply.blocked || !BADGES.includes(kind) ? t("budget.not_offered") : t(`budget.badge.${kind}`);
        const why = [reasonWords(fit.reason || reply.message || ""), reply.blocked ? reply.fix : ""].filter(Boolean).join(" ");
        summary.replaceChildren(h("span", { class: "st refused" }, badge), " ", why);
        if (customized() && !reply.blocked) helpFor(reply, mine);
        return;
      }
      current = reply;
      summary.textContent = fit.summary || "";
      const warned = fit.warning_records || [];
      if (warned.length) {
        const items = warned.map((w) => {
          const li = h("li", {}, w.lead || w.sentence);
          // Keeping the area as the user's own choice makes Customize move the view with the storm.
          if (w.kind === "edge" && w.follow_area_km && panel && state.custom.area === undefined) {
            const follow = h("button", { type: "button", class: "btn g-follow" }, t("custom.follow_storm"));
            follow.addEventListener("click", () => refitWith(() => { state.custom.area = w.follow_area_km; }));
            li.append(" ", follow);
          } else if (w.kind === "edge" && panel && state.custom.area !== undefined) {
            clearOfEdge(mine, li, w.set_up_for_km);
          }
          return li;
        });
        notes.replaceChildren(h("h2", { class: "cap" }, t("budget.notes")), h("ul", { class: "g-changes g-notes-list" }, items));
      }
      const notCompared = fit.not_compared || [];
      unchecked.textContent = notCompared.length ? `${t("gallery.not_compared")} ${notCompared.join(" ")}` : "";
      unchecked.hidden = !notCompared.length;
      got.replaceChildren(h("h2", { class: "cap" }, t("budget.you_get")), youGet(fit, p));
      const sentences = fit.sentences || [];
      changesHead.hidden = false;
      changes.replaceChildren(sentences.length
        ? h("ul", { class: "g-changes" }, sentences.map((s) => h("li", {}, s)))
        : h("p", { class: "muted" }, t(mine2.length ? "budget.nothing_else" : "budget.nothing_changed")));
      // Start is two commands: the fitted run folder, then the run. One line runs both.
      if (reply.start && reply.start.commands) cmd.replaceChildren(showCommand(reply.start.commands.join(" && "), { always: true }) || "");
      start.disabled = false;
    } finally {
      if (mine === serial) release();
    }
  }

  // "All settings (Advanced)": the run folder with these choices (and a greyed choice, when one was picked: WSM6,
  // Thompson or ground friction), opened on its namelist; nothing runs yet.
  async function makeForAdvanced(greyed) {
    let reply;
    try {
      reply = await api.post("/api/presets/new", { preset: p.id, budget: chosen, ...request(), ...(greyed ? { greyed } : {}) });
    } catch (err) {
      notice(`${err.message} ${err.fix || ""}`.trim());
      return;
    }
    if (!reply || !reply.run) {
      notice(`${(reply && reply.message) || ""}`.trim());
      return;
    }
    markHop();
    go(runRoute(greyed ? "check" : "namelist", reply.run));
    api.applyMode("advanced");
    notice(reply.message || "");
  }

  start.addEventListener("click", async () => {
    if (!current) return;
    try { sessionStorage.removeItem("gm1.sounding.back"); } catch (_) { /* nothing kept */ }
    start.disabled = true;
    start.textContent = t("budget.starting");
    try {
      const reply = await api.post("/api/guided/start", {
        preset: p.id, budget: chosen, name: current.start ? current.start.name : undefined, ...request(),
      });
      notice(reply.ok ? "" : `${reply.message} ${reply.fix || ""}`.trim());
      go(`guided/watch/${reply.url}`);
    } catch (err) {
      notice(`${err.message} ${err.fix || ""}`.trim());
      start.textContent = t("budget.start");
      start.disabled = false;
      if (err.status === 409) fitNow();
    }
  });
  paint();
  paintCaption();
  fitNow();
  return () => { serial += 1; clearTimeout(timer); };
}
