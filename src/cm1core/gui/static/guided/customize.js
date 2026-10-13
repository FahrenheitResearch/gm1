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
export const KNOB_QUERY = ["grid", "levels", "area", "time", "pictures", "push", "bubble", "bumps", "sea", "friction", "rotation", "parcels"];
const PINNABLE = ["grid", "area", "time", "pictures"];
// The choices guided mode greys that "All settings (Advanced)" can still carry into the run folder.
const TO_ADVANCED = ["scheme"];
const TERM = {
  grid: "grid_spacing", levels: "levels", area: "area", time: "storm_time", pictures: "picture_every",
  bubble: "bubble_size", bumps: "bumps", scheme: "scheme", friction: "friction", rotation: "rotation",
  parcels: "parcels", sea: "sea",
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
    if (custom[knob] === undefined) continue;
    // An area is kept to the metre, so the route never carries 239.59199999999998.
    const value = knob === "area" && typeof custom[knob] === "number" ? Math.round(custom[knob] * 1000) / 1000 : custom[knob];
    params.set(knob, String(value));
  }
  if (environment) {
    params.set("instability", String(environment.instability));
    params.set("shear", String(environment.shear));
    params.set("hodograph", environment.hodograph || "curved");
  }
  return params.toString();
}

// Whether an interval is a whole number of time steps, by the runtime's own arithmetic (customize.lands).
export function lands(interval, dt) {
  if (!(dt > 0) || !(interval > 0)) return false;
  const steps = interval / dt;
  return Number.isFinite(steps) && steps >= 1 && Number.isInteger(steps);
}

// The cells a side takes at a spacing: the exact count when that is whole, otherwise the next whole number up,
// never fewer than 8 (customize.cells_across).
export function cellsAcross(lengthM, dx) {
  const exact = lengthM / dx;
  const whole = Math.round(exact);
  return Math.max(8, Math.abs(exact - whole) < 1e-6 ? whole : Math.ceil(exact));
}

// The cells one saved picture of the storm holds (customize.FRAME_CELLS_MAX): an area past it cannot be saved on
// any card, so the area slider stops there.
const FRAME_CELLS = 150e6;

// A slid spacing as a number worth keeping: whole metres under 200 m, 5 m steps to 1 km, 50 m steps above.
function slidMetres(m) {
  if (m < 200) return Math.max(1, Math.round(m));
  if (m < 1000) return 5 * Math.round(m / 5);
  return 50 * Math.round(m / 50);
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
    case "sea": return `${value} °C`;
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
  return Boolean(environment) || changed.includes("push") || changed.includes("bubble") || changed.includes("sea");
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
  // What the last fit said this card holds: the cells along the side of the largest square area, at how many levels.
  let card = null;
  // A redraw replaces every control, so it waits while one is in use: a number box with text not yet entered, or a
  // slider held down. It is drawn once that control is let go (pendingPaint).
  let painting = false;
  let paintAgain = false;
  let pendingPaint = false;
  let dragging = false;
  let nowLine = null;
  const toggle = h("button", { type: "button", class: "btn g-cz-toggle", "aria-expanded": "false", "aria-controls": panelId },
    t("custom.open"));
  const live = h("p", { class: "g-cz-live", "aria-live": "polite" });
  const body = h("div", { class: "g-cz-body" });
  // Whether a control is in the middle of being used (see pendingPaint).
  function busy() {
    if (dragging) return true;
    const el = document.activeElement;
    return Boolean(el && body.contains(el) && el.matches("input.g-cz-num") && el.value !== el.dataset.committed);
  }
  function flushPaint() { if (pendingPaint && !painting && !busy()) paint(); }
  body.addEventListener("pointerdown", (event) => {
    if (event.target instanceof Element && event.target.matches('input[type="range"]')) dragging = true;
  });
  // The slider's own "change" comes after the pointer is let go: it is handled first, then anything still waiting.
  const letGo = () => { if (dragging) { dragging = false; setTimeout(flushPaint, 0); } };
  window.addEventListener("pointerup", letGo);
  window.addEventListener("pointercancel", letGo);
  body.addEventListener("focusout", () => setTimeout(flushPaint, 0));
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
      state.custom.area = Math.round(now * 1000) / 1000;
    } else areaNote = null;
    paint();
    onChange({ refit: true });
  }

  const dxNow = () => Number(state.custom.grid ?? grid.dx_m);
  const dzNow = () => Number(state.custom.levels ?? grid.dz_m);
  const timeNow = () => Number(state.custom.time ?? (fitted && fitted.time) ?? (knobs.time ? knobs.time.default : 0));

  function pictureChoices() {
    const knob = knobs.pictures;
    if (!knob) return [];
    const seconds = Number(state.custom.time ?? (knobs.time ? knobs.time.default : 0)) * 60;
    // With a grid spacing or level height of the user's own the time step is made to land on the picture interval
    // chosen (customize.snapped_time_step), so every interval that splits the storm time is a choice; otherwise the
    // preset's own step decides.
    const own = !same(dxNow(), grid.dx_m) || !same(dzNow(), grid.dz_m);
    return (knob.all || []).filter((m) => m * 60 <= seconds + 1e-6 && whole(seconds / (m * 60))
      && (own || lands(m * 60, Number(grid.dt_s))));
  }

  // The area slider's ends, in km of the longer side, in steps of 8 cells of the grid in use, and what caps the top
  // ("run": the largest domain GM1 has run, "offer": the largest this storm offers).
  function areaBounds() {
    const area = knobs.area;
    const dx = dxNow();
    const aspect = area.aspect || 1;
    const parcels = (state.custom.parcels ?? knobs.parcels.default) === "on";
    // The smallest area a storm starts in (16 km, or the bubble's own width), and never fewer than 8 cells.
    const floor = Math.max(area.floor_km, parcels ? area.parcels_floor_km : 0, (8 * dx) / 1000);
    const step = (area.step_cells * dx) / 1000;
    const base = cellsAcross((grid.nx >= grid.ny ? grid.nx : grid.ny) * grid.dx_m, dx);
    const baseKm = (base * dx) / 1000;
    const low = baseKm - step * Math.floor((baseKm - floor / aspect) / step + 1e-9);
    // An area larger than any GM1 has run is offered with a note; one past what a saved picture holds at this grid
    // and these levels is not offered, since it cannot be saved (cells = long side x short side x levels).
    const levels = Math.max(1, Math.round((grid.top_m || grid.nz * grid.dz_m) / dzNow()) || grid.nz || 1);
    const runKm = (Math.sqrt(FRAME_CELLS / (levels * aspect)) * dx) / 1000;
    const high = Math.min(area.max_km, runKm);
    const top = baseKm + step * Math.floor((high - baseKm) / step + 1e-9);
    const km = (v) => Math.round(v * 1000) / 1000;
    return [km(low), km(Math.max(low, top)), step, runKm < area.max_km ? "run" : "offer"];
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
      // A choice with a note of its own says it beside the value ("5 days (not run yet)").
      const noted = (knob.choices || []).find((c) => same(c.value, v) && c.note);
      select.append(h("option", { value: String(v) }, noted ? `${valueWords(knob, v)} (${noted.note})` : valueWords(knob, v)));
    }
    select.value = state.custom[knob.id] !== undefined ? String(state.custom[knob.id]) : "";
    select.addEventListener("change", () => {
      if (painting || !select.isConnected) return;
      set(knob.id, select.value === "" ? undefined : Number(select.value));
    });
    return row(knob, select, extra);
  }

  // The grid slider's ends in metres: from the finest spacing this card can run this storm at, on its smallest
  // area (once a fit has said what the card holds), up to a few km. A typed spacing outside them leaves the ends
  // where they are, with the thumb at the nearer one.
  function gridEnds(knob) {
    let [low, high] = knob.slider_m || [20, 4000];
    const finest = finestGrid();
    if (finest) low = Math.min(finest, high / 2);
    return [Math.max(1, low), high];
  }

  // A short note under a number box for a value that is not a spacing (0, a minus, letters), gone after a moment.
  function badValue(noteEl, key) {
    noteEl.textContent = t(key);
    setTimeout(() => { if (noteEl.isConnected) noteEl.textContent = ""; }, 3000);
  }

  // The finest spacing whose smallest area fits this card's free memory (the last fit's largest square area, at
  // the levels in use), or null before a fit has said.
  function finestGrid() {
    if (!card || !card.side || !knobs.area) return null;
    const levels = Math.round((grid.top_m || card.nz * grid.dz_m) / dzNow());
    const side = card.side * Math.sqrt(card.nz / Math.max(1, levels));
    return Math.max(1, Math.ceil((knobs.area.floor_km * 1000) / side));
  }

  const nowWords = (f) => t("custom.grid_now", { exact: f.exact, dt: Number(Number(f.dt).toPrecision(7)) });

  // The grid spacing: any value. Quick picks, a slider on a log scale and a box to type metres in; left alone, GM1
  // chooses it. The time step and the cells follow (the fit says what they come to).
  function gridRow(knob) {
    const pinned = state.custom.grid !== undefined;
    const now = Number(pinned ? state.custom.grid : ((fitted && fitted.grid) ?? knob.default));
    const [low, high] = gridEnds(knob);
    // Left to GM1, the first button says what the fit chose; with a spacing of the user's own it hands it back.
    const items = [{ value: "", words: pinned ? t("custom.auto") : autoWords(knob) },
      ...knob.choices.map((c) => ({ value: Number(c.value), words: valueWords(knob, c.value) }))];
    const seg = segmented(term("grid_spacing")[0], items, pinned ? Number(state.custom.grid) : "",
      (v) => set("grid", v === "" ? undefined : Number(v)));
    const span = Math.log(high / low) || 1;
    const toPlace = (m) => Math.min(1000, Math.max(0, Math.round((1000 * Math.log(m / low)) / span)));
    const fromPlace = (place) => slidMetres(low * Math.exp((span * place) / 1000));
    const slider = h("input", { type: "range", min: "0", max: "1000", step: "1", value: String(toPlace(now)),
      "aria-label": t("custom.grid_slider") });
    const box = h("input", { type: "number", class: "input num g-cz-num", min: "1", step: "any", value: String(now),
      "aria-label": t("custom.grid_box"), dataset: { value: "box", committed: String(now) } });
    // The buttons above say "Let GM1 choose: now 1 km"; the slider's own label is the spacing alone.
    const shown = h("span", { class: "g-cz-value" }, metres(now));
    const bad = h("span", { class: "note g-cz-bad", "aria-live": "polite" });
    slider.addEventListener("input", () => {
      const m = fromPlace(Number(slider.value));
      box.value = String(m);
      shown.textContent = metres(m);
    });
    slider.addEventListener("change", () => {
      // A slider taken off the page by a redraw is no longer the one in use: its last event changes nothing.
      if (painting || !slider.isConnected) return;
      // An arrow key moves the thumb less than a metre at the fine end: the spacing then moves one step that way.
      const place = Number(slider.value);
      let m = fromPlace(place);
      if (pinned && m === now && place !== toPlace(now)) {
        const step = now < 200 ? 1 : now < 1000 ? 5 : 50;
        m = Math.max(1, now + (place > toPlace(now) ? step : -step));
      }
      set("grid", m);
    });
    box.addEventListener("change", () => {
      // A box taken off the page by a redraw is no longer the one in use: what it held is not entered for it.
      if (painting || !box.isConnected) return;
      const m = Number(box.value);
      if (box.value.trim() !== "" && Number.isFinite(m) && m > 0) set("grid", m);
      else { box.value = String(now); badValue(bad, "custom.grid_bad"); flushPaint(); }
    });
    const ends = h("div", { class: "g-cz-ends", "aria-hidden": "true" }, h("span", {}, metres(low)), h("span", {}, metres(high)));
    const typed = h("label", { class: "g-cz-type" }, t("custom.grid_type"), " ", box, " ", t("custom.metres"));
    const notes = [bad, h("p", { class: "note" }, t("custom.grid_any"))];
    nowLine = null;
    if (fitted && fitted.exact) {
      nowLine = h("p", { class: "note g-cz-now" }, nowWords(fitted));
      notes.push(nowLine);
    }
    const finest = finestGrid();
    if (finest) {
      notes.push(h("p", { class: "note" }, t("custom.grid_finest", { value: metres(finest), area: `${Math.round(knobs.area.floor_km)} km` })));
    }
    return row(knob, h("div", { class: "g-cz-free" }, seg, h("div", { class: "g-cz-slider" }, shown, slider, ends), typed), notes);
  }

  // The height of a level: any value that gives the model's 8 to 256 levels under the preset's top.
  function levelsRow(knob) {
    const chosen = state.custom.levels !== undefined;
    const now = Number(chosen ? state.custom.levels : knob.default);
    const items = [{ value: "", words: `${t("custom.as_preset")} (${valueWords(knob, knob.default)})` },
      ...knob.choices.filter((c) => !same(c.value, knob.default)).map((c) => ({ value: Number(c.value), words: metres(c.value) }))];
    const seg = segmented(term("levels")[0], items, chosen ? Number(state.custom.levels) : "",
      (v) => set("levels", v === "" ? undefined : Number(v)));
    const box = h("input", { type: "number", class: "input num g-cz-num", min: "1", step: "any", value: String(now),
      "aria-label": t("custom.levels_box"), dataset: { value: "box", committed: String(now) } });
    const bad = h("span", { class: "note g-cz-bad", "aria-live": "polite" });
    box.addEventListener("change", () => {
      if (painting || !box.isConnected) return;
      const m = Number(box.value);
      if (box.value.trim() !== "" && Number.isFinite(m) && m > 0) set("levels", same(m, knob.default) ? undefined : m);
      else { box.value = String(now); badValue(bad, "custom.levels_bad"); flushPaint(); }
    });
    const typed = h("label", { class: "g-cz-type" }, t("custom.levels_type"), " ", box, " ", t("custom.metres"));
    const count = Math.round(knob.top_m / dzNow());
    const [fewest, most] = knob.levels_range || [8, 256];
    // More levels than the model takes (or fewer) is said here, before the fit refuses it.
    const over = count > most ? t("custom.levels_over", { count, most }) : count < fewest ? t("custom.levels_under", { count, fewest }) : null;
    return row(knob, h("div", { class: "g-cz-free" }, seg, typed), [
      bad,
      h("p", { class: `note${over ? " g-cz-clamp" : ""}` }, over || t("custom.levels_note", { top: metres(count * dzNow()), count })),
      h("p", { class: "note" }, t("custom.levels_any", { low: metres(Math.ceil((knob.min_m || knob.top_m / most) * 10) / 10), count: most })),
    ]);
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
    input.addEventListener("change", () => {
      if (painting || !input.isConnected) return;
      set("area", Math.round(Number(input.value) * 1000) / 1000);
    });
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
      const words = t("custom.greyed_go", { what: valueWords(knob, c.value) });
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
        if (painting || !input.isConnected) return;
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
    // Taking a focused control off the page makes the browser send its last "change"; that must not start a
    // second redraw inside this one (the handlers above also ignore it).
    if (painting) { paintAgain = true; return; }
    painting = true;
    pendingPaint = false;
    try { paintNow(); } finally { painting = false; }
    if (paintAgain) { paintAgain = false; paint(); }
  }

  function paintNow() {
    const keep = focusKey(body);
    const y = window.scrollY;
    const groups = {};
    const add = (group, rowEl) => { (groups[group] = groups[group] || []).push(rowEl); };
    // Any spacing for the thunderstorms; a hurricane keeps its own grid choices (no slider, no typed value).
    if (knobs.grid) add("size", knobs.grid.free ? gridRow(knobs.grid) : choiceRow(knobs.grid, knobs.grid.choices.map((c) => c.value)));
    if (knobs.levels) add("size", levelsRow(knobs.levels));
    if (knobs.area) add("size", areaRow(knobs.area));
    if (knobs.time) add("size", choiceRow(knobs.time, knobs.time.choices.map((c) => c.value),
      knobs.time.note ? h("p", { class: "note" }, knobs.time.note) : undefined));
    if (knobs.pictures) add("size", choiceRow(knobs.pictures, pictureChoices()));
    add("storm", airRow());
    if (knobs.push) add("storm", choiceRow(knobs.push, knobs.push.choices.map((c) => c.value)));
    if (knobs.bubble) add("storm", choiceRow(knobs.bubble, knobs.bubble.choices.map((c) => c.value)));
    if (knobs.bumps) add("storm", bumpsRow(knobs.bumps));
    // A hurricane's sea surface temperature (its only storm setting: CM1 keeps the starting swirl in its code).
    if (knobs.sea) add("storm", choiceRow(knobs.sea, knobs.sea.choices.map((c) => c.value), h("p", { class: "note" }, t("custom.sea_note"))));
    if (knobs.scheme) add("physics", switchRow(knobs.scheme));
    if (knobs.friction) add("physics", switchRow(knobs.friction, t("custom.friction_note")));
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
        exact: f.exact, dt: f.dt_s,
      };
    }
    // What the card holds, from an offered fit's estimate or a not-offered one's needs.
    const fit = (reply && reply.fit) || {};
    const side = (fit.estimate || {}).largest_square_side_cells || (fit.needs || {}).largest_square_side_cells;
    const levels = (fit.facts || {}).nz || Math.round((grid.top_m || 0) / dzNow());
    const held = side && levels ? { side, nz: levels } : card;
    if (reply && (JSON.stringify(now) !== JSON.stringify(fitted) || JSON.stringify(held) !== JSON.stringify(card))) {
      fitted = now;
      card = held;
      // Not while a box holds text not yet entered or a slider is held: the redraw would enter half a number or
      // drop the thumb. The line that says what the fit came to is brought up to date in place meanwhile.
      if (busy()) {
        pendingPaint = true;
        if (nowLine && now && now.exact) nowLine.textContent = nowWords(now);
      } else paint();
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
  return { el, setOpen, paint, showFit, largerAreas, finestGrid };
}
