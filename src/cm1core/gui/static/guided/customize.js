// "Customize this storm": the budget screen's panel. Each row is one of the preset's settings in plain words, CM1's
// name as a grey hint, and its choices; the choices come from the preset list (`gpuwm-cm1 preset options ID`, the
// row's "customize"). What the user changes is sent as {custom: {KNOB: VALUE}} with every fit and with Start
// (`--custom KNOB=VALUE`), and the fit never changes it again. The air is the Make-your-own-environment sliders
// (`--instability I --shear S --hodograph H`). Nothing here decides what runs: a choice guided mode would turn
// down is greyed with the guided check's own reason, and the fit says whether the rest fits. The rows the user
// leaves to GM1 show what the last fit chose for them.

import { h } from "../js/core.js";
import { t, term, metres, spanWords } from "./copy.js";

// The knobs the route keeps, in the panel's order (#/guided/fit/ID/BUDGET?grid=500&area=96&...).
export const KNOB_QUERY = ["grid", "levels", "area", "time", "pictures", "push", "bubble", "bumps", "rotation", "parcels"];
const PINNABLE = ["grid", "area", "time", "pictures"];
// The choices guided mode greys that "All settings (Advanced)" can still carry into the run folder.
const TO_ADVANCED = ["scheme", "friction"];
const TERM = {
  grid: "grid_spacing", levels: "levels", area: "area", time: "storm_time", pictures: "picture_every",
  bubble: "bubble_size", bumps: "bumps", scheme: "scheme", friction: "friction", rotation: "rotation",
  parcels: "parcels",
};
const AIR_STEPS = 20;

// The custom choices a route's query carries: {grid: 500, bumps: "seed:12", parcels: "on", ...}.
export function customFrom(params) {
  const custom = {};
  for (const knob of KNOB_QUERY) {
    const value = params.get(knob);
    if (value === null || value === "") continue;
    custom[knob] = /^-?\d+(\.\d+)?$/.test(value) ? Number(value) : value;
  }
  return custom;
}

// The route's query for custom choices and the made air (the inverse of customFrom and budget.givenFrom).
export function customQuery(custom, environment) {
  const params = new URLSearchParams();
  for (const knob of KNOB_QUERY) {
    if (custom[knob] !== undefined) params.set(knob, String(custom[knob]));
  }
  if (environment) {
    params.set("instability", String(environment.instability));
    params.set("shear", String(environment.shear));
    params.set("hodograph", environment.hodograph || "curved");
  }
  return params.toString();
}

// The time step a grid and its levels take (customize.time_step: CM1's 6 s per 1 km and per 500 m up).
export function timeStep(dx, dz) {
  return Math.min(0.006 * dx, 0.012 * dz);
}

function whole(x) { return Math.abs(x - Math.round(x)) < 1e-6; }

function minutesWords(minutes) { return spanWords(minutes * 60); }

function same(a, b) {
  if (typeof a === "number" || typeof b === "number") return Math.abs(Number(a) - Number(b)) < 1e-6;
  return a === b;
}

function valueWords(knob, value) {
  if (value === undefined || value === null) return "";
  switch (knob.id) {
    case "grid": return metres(value);
    case "levels": return t("custom.every", { value: metres(value) });
    case "area": return `${Math.round(value)} km`;
    case "time": return minutesWords(value);
    case "pictures": return t("custom.every", { value: minutesWords(value) });
    case "push": return knob.kind === "bubble" ? `${value} K` : `${value} m/s`;
    case "bubble": return `${value} km`;
    case "scheme": return { morrison: "Morrison", wsm6: "WSM6", thompson: "Thompson" }[value] || String(value);
    default:
      if (value === "on") return t("custom.on");
      if (value === "off") return t("custom.off");
      if (typeof value === "string" && value.startsWith("seed:")) return t("custom.draw", { seed: value.slice(5) });
      return String(value);
  }
}

// The knobs that change something: a choice equal to the preset's own value is not a change (the run is named,
// labelled and captioned as the preset), though a pinned one still holds the fit to it.
export function changedKnobs(custom, knobs) {
  return KNOB_QUERY.filter((knob) => custom[knob] !== undefined && !(knobs && knobs[knob] && same(custom[knob], knobs[knob].default)));
}

// What changed, in words, for the picture's caption ("grid spacing, area").
export function changedWords(custom, environment, knobs) {
  const out = changedKnobs(custom, knobs).map((knob) => t(`custom.what.${knob}`));
  if (environment) out.push(t("custom.what.air"));
  return out.join(", ");
}

// Changes that make another storm (so the preset's picture is only a likeness).
export function changesStorm(custom, environment, knobs) {
  const changed = changedKnobs(custom, knobs);
  return Boolean(environment) || changed.includes("push") || changed.includes("bubble");
}

// A row of buttons, one pressed. Pressing the pressed one does nothing (no redraw, no refit).
function segmented(label, items, current, onPick) {
  const wrap = h("div", { class: "g-cz-seg", role: "group", "aria-label": label });
  for (const item of items) {
    const on = item.value === current;
    const b = h("button", { type: "button", class: `btn${on ? " on" : ""}`, "aria-pressed": on ? "true" : "false",
      disabled: item.available === false ? true : null, title: item.why || null,
      dataset: { value: String(item.value) } }, item.words);
    b.addEventListener("click", () => { if (item.value !== current) onPick(item.value); });
    wrap.append(b);
  }
  return wrap;
}

// Where the keyboard focus is inside the panel, so a redraw can put it back: the row, the kind of control, its
// value (a button) or its place in the row.
function focusKey(body) {
  const el = document.activeElement;
  if (!el || !body.contains(el)) return null;
  const row = el.closest("[data-knob]");
  if (!row) return null;
  const all = [...row.querySelectorAll(el.tagName)];
  return { knob: row.dataset.knob, tag: el.tagName, value: el.dataset ? el.dataset.value ?? null : null, index: all.indexOf(el) };
}

function refocus(body, key) {
  if (!key) return;
  const row = body.querySelector(`[data-knob="${key.knob}"]`);
  if (!row) return;
  const all = [...row.querySelectorAll(key.tag)];
  const byValue = key.value !== null ? all.find((el) => el.dataset.value === key.value) : null;
  const target = [byValue, all[key.index], row.querySelector("button:not([disabled]), select, input")]
    .find((el) => el && !el.disabled);
  if (target) target.focus({ preventScroll: true });
}

// The panel. state = {custom, environment, open}; onChange() is called after every change (the screen refits).
export function customizePanel(p, opts, state, { onChange, onAdvanced, onSeeWhy }) {
  const knobs = {};
  for (const knob of (opts && opts.knobs) || []) knobs[knob.id] = knob;
  const air = (opts && opts.air) || { available: false };
  const grid = (opts && opts.grid) || {};
  const panelId = `g-cz-${p.id}`;
  // What the last offered fit chose for the rows left to GM1 ({grid, area, time, pictures}), and why a pinned area
  // was moved (shown on its row).
  let fitted = null;
  let areaNote = null;
  const toggle = h("button", { type: "button", class: "btn g-cz-toggle", "aria-expanded": "false", "aria-controls": panelId },
    t("custom.open"));
  const live = h("p", { class: "g-cz-live", "aria-live": "polite" });
  const body = h("div", { class: "g-cz-body" });
  const reset = h("button", { type: "button", class: "btn" }, t("custom.reset"));
  const advanced = h("button", { type: "button", class: "btn" }, t("custom.advanced"));
  const foot = h("div", { class: "g-cz-foot" },
    h("div", { class: "btns" }, reset, advanced),
    h("p", { class: "note" }, t("custom.advanced_note")));
  const panel = h("div", { class: "g-cz", id: panelId, hidden: true },
    live, h("p", { class: "note g-cz-intro" }, t("custom.intro")), body, foot);
  const el = h("div", { class: "g-cz-wrap" }, toggle, panel);

  function setOpen(open) {
    state.open = open;
    panel.hidden = !open;
    toggle.setAttribute("aria-expanded", open ? "true" : "false");
    toggle.classList.toggle("on", open);
  }
  toggle.addEventListener("click", () => { setOpen(!state.open); onChange({ refit: false }); });
  reset.addEventListener("click", () => {
    state.custom = {};
    state.environment = null;
    areaNote = null;
    paint();
    onChange({ refit: true });
  });
  async function toAdvanced(button, greyed) {
    button.disabled = true;
    const was = button.textContent;
    button.textContent = t("custom.advanced_making");
    try { await onAdvanced(greyed); } finally { button.disabled = false; button.textContent = was; }
  }
  advanced.addEventListener("click", () => toAdvanced(advanced, null));

  function set(knob, value) {
    const knobDef = knobs[knob];
    if (value === undefined || (knobDef && value === knobDef.default && !PINNABLE.includes(knob))) delete state.custom[knob];
    else state.custom[knob] = value;
    // A picture interval that no longer splits the storm time (or lands between time steps) goes back to GM1.
    if (state.custom.pictures !== undefined && !pictureChoices().includes(Number(state.custom.pictures))) delete state.custom.pictures;
    // An area outside what the grid now allows is brought inside it, and its row says so.
    if (knob === "area") areaNote = null;
    if (state.custom.area !== undefined) {
      const [low, high, , cap] = areaBounds();
      const chosen = Number(state.custom.area);
      const now = Math.min(high, Math.max(low, chosen));
      if (knob !== "area" && Math.abs(now - chosen) > 1e-6) {
        const key = now < chosen ? (cap === "run" ? "custom.area_cut_run" : "custom.area_cut_offer") : "custom.area_raised";
        areaNote = t(key, { value: `${Math.round(now)} km`, was: `${Math.round(chosen)} km` });
      }
      state.custom.area = now;
    } else areaNote = null;
    paint();
    onChange({ refit: true });
  }

  const dxNow = () => Number(state.custom.grid ?? grid.dx_m);
  const dzNow = () => Number(state.custom.levels ?? grid.dz_m);
  const dtNow = () => (state.custom.grid === undefined && state.custom.levels === undefined ? Number(grid.dt_s) : timeStep(dxNow(), dzNow()));
  const timeNow = () => Number(state.custom.time ?? (fitted && fitted.time) ?? (knobs.time ? knobs.time.default : 0));

  function pictureChoices() {
    const knob = knobs.pictures;
    if (!knob) return [];
    const seconds = Number(state.custom.time ?? (knobs.time ? knobs.time.default : 0)) * 60;
    const dt = dtNow();
    return (knob.all || []).filter((m) => m * 60 <= seconds + 1e-6 && whole(seconds / (m * 60)) && whole((m * 60) / dt));
  }

  // The area slider's ends, in km of the longer side, in steps of 8 cells of the grid in use, and what caps the top
  // ("run": the largest domain GM1 has run, "offer": the largest this storm offers).
  function areaBounds() {
    const area = knobs.area;
    const dx = dxNow();
    const aspect = area.aspect || 1;
    const parcels = (state.custom.parcels ?? knobs.parcels.default) === "on";
    const floor = Math.max(area.floor_km, parcels ? area.parcels_floor_km : 0);
    const nz = Math.round((grid.top_m || area.nz * grid.dz_m) / dzNow());
    const maxCells = Math.floor(Math.sqrt(area.largest_cells / (nz * aspect)));
    const step = (area.step_cells * dx) / 1000;
    const base = Math.round((grid.nx >= grid.ny ? grid.nx : grid.ny) * grid.dx_m / dx);
    const baseKm = (base * dx) / 1000;
    const low = baseKm - step * Math.floor((baseKm - floor / aspect) / step + 1e-9);
    const runKm = (maxCells * dx) / 1000;
    const high = Math.min(area.max_km, runKm);
    const top = baseKm + step * Math.floor((high - baseKm) / step + 1e-9);
    return [low, Math.max(low, top), step, runKm < area.max_km ? "run" : "offer"];
  }

  // The area slider's steps above the area in use, for "Make the area larger" (budget.js tries some of them).
  function largerAreas() {
    if (!knobs.area) return [];
    const [low, high, step] = areaBounds();
    const now = Number(state.custom.area ?? (fitted && fitted.area) ?? knobs.area.default);
    const out = [];
    for (let v = low; v <= high + 1e-6; v += step) if (v > now + 1e-6) out.push(Math.round(v * 1000) / 1000);
    return out;
  }

  // The term a row is labelled with: the push is a warm bubble's strength or a push of rising air.
  const termOf = (knob) => (knob.id === "push" ? (knob.kind === "bubble" ? "push_bubble" : "push_updraft") : TERM[knob.id] || knob.id);

  // "Let GM1 choose", with what the last fit chose ("now 88 km", and the preset's when it differs).
  function autoWords(knob) {
    const preset = valueWords(knob, knob.default);
    const now = fitted ? fitted[knob.id] : undefined;
    if (now === undefined || now === null) return t("custom.auto_preset", { preset });
    if (same(now, knob.default)) return t("custom.auto_now", { value: valueWords(knob, now) });
    return t("custom.auto_now_preset", { value: valueWords(knob, now), preset });
  }

  function row(knob, control, extra) {
    const changed = state.custom[knob.id] !== undefined;
    const undo = changed ? h("button", { type: "button", class: "linkbtn g-cz-undo" }, t("custom.undo")) : null;
    if (undo) undo.addEventListener("click", () => set(knob.id, undefined));
    const [text, hintEl] = term(termOf(knob));
    return h("div", { class: `g-cz-row${changed ? " changed" : ""}`, dataset: { knob: knob.id } },
      h("div", { class: "g-cz-label" }, text, hintEl),
      h("div", { class: "g-cz-control" }, control,
        changed ? h("span", { class: "g-cz-was" }, t("custom.preset_was", { value: valueWords(knob, knob.default) }), " · ", undo) : null,
        extra || null));
  }

  function choiceRow(knob, values, extra) {
    const select = h("select", { class: "input", "aria-label": term(termOf(knob))[0] });
    const pinnable = PINNABLE.includes(knob.id);
    select.append(h("option", { value: "" }, pinnable ? autoWords(knob) : `${t("custom.as_preset")} (${valueWords(knob, knob.default)})`));
    for (const v of values) {
      // The preset's own value is "As the preset" (or, for the fit's own levers, kept as a choice to hold it).
      if (v === knob.default && !pinnable) continue;
      select.append(h("option", { value: String(v) }, valueWords(knob, v)));
    }
    select.value = state.custom[knob.id] !== undefined ? String(state.custom[knob.id]) : "";
    select.addEventListener("change", () => set(knob.id, select.value === "" ? undefined : Number(select.value)));
    return row(knob, select, extra);
  }

  function areaRow(knob) {
    const [low, high, step] = areaBounds();
    const pinned = state.custom.area !== undefined;
    // Left to GM1, the thumb sits where the last fit put the area.
    const value = Number(pinned ? state.custom.area : ((fitted && fitted.area) ?? knob.default));
    const input = h("input", { type: "range", min: String(low), max: String(high), step: String(step),
      value: String(Math.min(high, Math.max(low, value))), "aria-label": term("area")[0] });
    const shown = h("span", { class: "g-cz-value" },
      pinned ? t("custom.area_value", { value: valueWords(knob, value) }) : autoWords(knob));
    input.addEventListener("input", () => {
      shown.textContent = t("custom.area_value", { value: valueWords(knob, Number(input.value)) });
    });
    input.addEventListener("change", () => set("area", Math.round(Number(input.value) * 1000) / 1000));
    const ends = h("div", { class: "g-cz-ends", "aria-hidden": "true" },
      h("span", {}, `${Math.round(low)} km`), h("span", {}, `${Math.round(high)} km`));
    return row(knob, h("div", { class: "g-cz-slider" }, shown, input, ends),
      areaNote ? h("p", { class: "note g-cz-clamp" }, areaNote) : null);
  }

  function switchRow(knob, note) {
    const items = knob.choices.map((c) => ({ value: c.value, words: valueWords(knob, c.value), available: c.available, why: c.why }));
    const greyed = knob.choices.filter((c) => c.available === false && c.why);
    const current = state.custom[knob.id] ?? knob.default;
    const whyRows = greyed.map((c) => {
      if (!TO_ADVANCED.includes(knob.id)) return h("p", { class: "note g-cz-why" }, c.why);
      // Still a choice: "All settings (Advanced)" makes the folder with it set, one click.
      const words = knob.id === "friction" ? t("custom.greyed_go_on") : t("custom.greyed_go", { what: valueWords(knob, c.value) });
      const go = h("button", { type: "button", class: "linkbtn g-cz-adv" }, words);
      go.addEventListener("click", () => toAdvanced(go, { [knob.id]: c.value }));
      return h("p", { class: "note g-cz-why" }, `${c.why} `, go);
    });
    return row(knob, segmented(term(termOf(knob))[0], items, current, (v) => set(knob.id, v)),
      [note ? h("p", { class: "note" }, note) : null, ...whyRows]);
  }

  function bumpsRow(knob) {
    const current = state.custom.bumps ?? knob.default;
    const seeded = typeof current === "string" && current.startsWith("seed:");
    const items = [
      { value: "off", words: t("custom.off") },
      { value: "on", words: t("custom.on") },
      { value: seeded ? current : "seed", words: seeded ? t("custom.draw", { seed: current.slice(5) }) : t("custom.another_draw") },
    ];
    const seg = segmented(term("bumps")[0], items, current, (v) => {
      if (v === "seed" || (typeof v === "string" && v.startsWith("seed:"))) {
        // Another draw: a new seed each press (GM1's own random numbers).
        set("bumps", `seed:${1 + Math.floor(Math.random() * 999999)}`);
      } else set("bumps", v);
    });
    if (seeded) {
      const again = h("button", { type: "button", class: "btn", dataset: { value: "again" } }, t("custom.another_draw"));
      again.addEventListener("click", () => set("bumps", `seed:${1 + Math.floor(Math.random() * 999999)}`));
      seg.append(again);
    }
    return row(knob, seg, h("p", { class: "note" }, t("custom.bumps_note")));
  }

  function airRow() {
    const [text, hintEl] = term("air");
    if (!air.available) {
      return h("div", { class: "g-cz-row", dataset: { knob: "air" } },
        h("div", { class: "g-cz-label" }, text, hintEl), h("div", { class: "g-cz-control" }, h("p", { class: "note" }, air.why || "")));
    }
    const env = state.environment || { instability: air.instability ?? 0.5, shear: air.shear ?? 0.5, hodograph: air.hodograph || "curved" };
    const slider = (label, value, low, high, key) => {
      const input = h("input", { type: "range", min: "0", max: String(AIR_STEPS), step: "1", value: String(Math.round(value * AIR_STEPS)), "aria-label": label });
      input.addEventListener("change", () => {
        state.environment = { ...env, [key]: Number(input.value) / AIR_STEPS };
        paint();
        onChange({ refit: true });
      });
      return h("div", { class: "g-cz-slider" }, h("span", { class: "g-cz-value" }, label), input,
        h("div", { class: "g-cz-ends", "aria-hidden": "true" }, h("span", {}, low), h("span", {}, high)));
    };
    const shape = segmented(text, [
      { value: "curved", words: t("custom.air_curved") }, { value: "straight", words: t("custom.air_straight") },
    ], env.hodograph, (v) => {
      state.environment = { ...env, hodograph: v };
      paint();
      onChange({ refit: true });
    });
    const changed = Boolean(state.environment);
    const undo = changed ? h("button", { type: "button", class: "linkbtn g-cz-undo" }, t("custom.undo")) : null;
    if (undo) {
      undo.addEventListener("click", () => {
        state.environment = null;
        paint();
        onChange({ refit: true });
      });
    }
    return h("div", { class: `g-cz-row${changed ? " changed" : ""}`, dataset: { knob: "air" } },
      h("div", { class: "g-cz-label" }, text, hintEl),
      h("div", { class: "g-cz-control" },
        slider(t("env.instability"), env.instability, t("env.instability_low"), t("env.instability_high"), "instability"),
        slider(t("env.shear"), env.shear, t("env.shear_low"), t("env.shear_high"), "shear"),
        shape,
        h("p", { class: "note" }, changed ? t("custom.air_note") : `${t("custom.air_note")} ${t("custom.air_own")}`),
        undo));
  }

  // Draws the rows again. The page stays where it is and the focus stays on the control that had it (a pressed
  // button, a list), so a change never moves the reader.
  function paint() {
    const keep = focusKey(body);
    const y = window.scrollY;
    const groups = {};
    const add = (group, rowEl) => { (groups[group] = groups[group] || []).push(rowEl); };
    if (knobs.grid) add("size", choiceRow(knobs.grid, knobs.grid.choices.map((c) => c.value)));
    if (knobs.levels) {
      const count = Math.round(knobs.levels.top_m / dzNow());
      add("size", choiceRow(knobs.levels, knobs.levels.choices.map((c) => c.value),
        h("p", { class: "note" }, t("custom.levels_note", { top: metres(knobs.levels.top_m), count }))));
    }
    if (knobs.area) add("size", areaRow(knobs.area));
    if (knobs.time) add("size", choiceRow(knobs.time, knobs.time.choices.map((c) => c.value)));
    if (knobs.pictures) add("size", choiceRow(knobs.pictures, pictureChoices()));
    add("storm", airRow());
    if (knobs.push) add("storm", choiceRow(knobs.push, knobs.push.choices.map((c) => c.value)));
    if (knobs.bubble) add("storm", choiceRow(knobs.bubble, knobs.bubble.choices.map((c) => c.value)));
    if (knobs.bumps) add("storm", bumpsRow(knobs.bumps));
    if (knobs.scheme) add("physics", switchRow(knobs.scheme));
    if (knobs.friction) add("physics", switchRow(knobs.friction));
    if (knobs.rotation) {
      add("physics", switchRow(knobs.rotation, t(timeNow() < 60 ? "custom.rotation_note_short" : "custom.rotation_note")));
    }
    if (knobs.parcels) add("extras", switchRow(knobs.parcels, t("custom.parcels_note")));
    body.replaceChildren(...((opts && opts.groups) || ["size", "storm", "physics", "extras"])
      .filter((g) => groups[g])
      .map((g) => h("fieldset", { class: "g-cz-group" }, h("legend", { class: "cap" }, t(`custom.group.${g}`)), groups[g])));
    reset.disabled = !Object.keys(state.custom).length && !state.environment;
    refocus(body, keep);
    if (Math.abs(window.scrollY - y) > 1) window.scrollTo(window.scrollX, y);
  }

  // The live line: what the fit says these choices take, that it is checking, or that they are not offered here
  // (with a link down to why and what to change).
  function showFit(reply, wordsFor) {
    let now = null;
    if (reply && reply.offered) {
      const f = (reply.fit && reply.fit.facts) || {};
      now = {
        grid: f.dx_m, area: f.domain_km ? Math.max(...f.domain_km) : undefined,
        time: f.duration_s ? f.duration_s / 60 : undefined, pictures: f.output_interval_s ? f.output_interval_s / 60 : undefined,
      };
    }
    if (reply && JSON.stringify(now) !== JSON.stringify(fitted)) {
      fitted = now;
      paint();
    }
    if (!reply) { live.textContent = t("custom.live_fitting"); return; }
    if (!reply.offered) {
      const see = h("button", { type: "button", class: "linkbtn g-cz-see" }, t("custom.live_see"));
      see.addEventListener("click", () => { if (onSeeWhy) onSeeWhy(); });
      live.replaceChildren(t("custom.live_not"), " ", see);
      return;
    }
    live.textContent = wordsFor(reply);
  }

  setOpen(Boolean(state.open));
  paint();
  return { el, setOpen, paint, showFit, largerAreas };
}
