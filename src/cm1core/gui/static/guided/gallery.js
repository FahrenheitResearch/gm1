// Guided screen 1: Pick a storm. The cards sit in four groups: Start here (open), Explore,
// Other storm types and Famous environments (last while its cards are all greyed), each of the last three
// showing one row and "Show all N" until it is opened, in place. A card per preset: its picture from a real run, a plain name, one line of what it makes,
// and "About N min on your GPU" from the fit. A preset GM1 cannot offer here stays, greyed, with one line saying
// why. A storm with a setting not yet compared with CM1 runs and says so on its card ("Runs, not yet compared with
// CM1: ..."). A card that is not a preset folder (the WRF or ArWen column) joins through registerCard.

import { h } from "../js/core.js";
import * as api from "../js/api.js";
import { go } from "../js/router.js";
import { t, wallWords, shortGpu } from "./copy.js";
import { environmentCard } from "./environment.js";

const GROUPS = ["start", "explore", "other", "famous"];
const OPEN_KEY = "gm1.gallery.open";
const extra = [];

// registerCard({group, order, id, render(el, ctx)}): a card that is not a preset folder. render draws the card
// into el (a grid cell of its group) and may return a promise; ctx carries {go, fit}.
export function registerCard(card) {
  if (!card || !GROUPS.includes(card.group) || typeof card.render !== "function") return;
  if (extra.some((c) => c.id === card.id)) return;
  extra.push({ order: 50, ...card });
}

function picture(p) {
  const pic = p.picture || {};
  if (!pic.available || !pic.url) {
    return h("div", { class: "g-pic none" }, h("span", {}, t("gallery.picture_none")));
  }
  return h("figure", { class: "g-pic" },
    h("img", { src: pic.url, alt: p.name, loading: "lazy" }),
    h("figcaption", {}, pic.caption || ""));
}

function aboutLine(fit) {
  const est = fit.estimate || {};
  const gpu = fit.gpu;
  return gpu ? t("gallery.about", { wall: wallWords(est.wall_s), gpu: shortGpu(gpu.name) })
    : t("gallery.about_nogpu", { wall: wallWords(est.wall_s) });
}

// Fit at the preset's own time; if it does not fit that, at the longest time before the card is greyed.
export async function fitCard(p, extraBody = {}) {
  const tries = [p.budget, "overnight"].filter((b, i, all) => all.indexOf(b) === i);
  let last = null;
  for (const budget of tries) {
    const reply = await api.post("/api/fit", { preset: p.id, budget, ...extraBody });
    last = reply;
    if (reply.offered) return reply;
  }
  return last;
}

// Say what a fit reply means on a card: About N min, or greyed with the reason. own: the time the card opens on
// by itself; when the storm only fits a longer one here, the card says which one it opens on.
export function showFit(card, fitLine, why, choose, reply, own) {
  const fit = reply.fit || {};
  if (reply.offered) {
    fitLine.textContent = aboutLine(fit);
    // Its own line under the time: the card opens on a longer time than its own, and says which.
    why.textContent = own && fit.budget && fit.budget !== own.budget && fit.budget_words
      ? t("gallery.needs_budget", { own: String(own.budget_words || "").toLowerCase(), budget: fit.budget_words }) : "";
    card.classList.remove("off");
    choose.disabled = false;
    return fit.budget;
  }
  card.classList.add("off");
  fitLine.textContent = t("gallery.not_offered");
  // No GPU that can run it (reply.blocked): the fix says what to do; otherwise the reason is enough.
  why.textContent = [fit.reason || reply.message || "", reply.blocked ? reply.fix : ""].filter(Boolean).join(" ");
  choose.disabled = true;
  return null;
}

function presetCard(p, alive) {
  const fitLine = h("p", { class: "g-fit" }, p.offered ? t("gallery.checking") : "");
  const why = h("p", { class: "g-why" }, p.offered ? "" : (p.why_not || t("gallery.not_offered")));
  // The settings this storm runs that are not yet compared with CM1, in the guided check's own sentences.
  const unchecked = (p.not_compared || []).length
    ? h("p", { class: "note g-unchecked" }, `${t("gallery.not_compared")} ${p.not_compared.join(" ")}`) : null;
  const choose = h("button", { class: "btn primary", type: "button", disabled: true }, t("gallery.choose"));
  // "Customize": the same budget screen with its Customize this storm panel open (not on a withheld storm: nothing
  // in the panel could make it offered).
  const customize = p.customize && p.offered ? h("button", { class: "btn g-cz-link", type: "button", disabled: true }, t("gallery.customize")) : null;
  const card = h("article", { class: `g-card${p.offered ? "" : " off"}`, dataset: { preset: p.id } },
    picture(p),
    h("div", { class: "g-body" },
      h("h2", {}, p.name),
      h("p", { class: "g-line" }, p.line),
      p.group === "famous" ? h("p", { class: "g-idealized" }, t("gallery.famous_note")) : null,
      p.facts ? h("p", { class: "g-facts" }, `${p.facts.words} · ${p.facts.scheme}`) : null,
      fitLine, why, unchecked,
      h("div", { class: "btns" }, choose, customize)));
  let budget = p.budget;
  choose.addEventListener("click", () => go(`guided/fit/${encodeURIComponent(p.id)}/${budget}`));
  if (customize) customize.addEventListener("click", () => go(`guided/fit/${encodeURIComponent(p.id)}/${budget}?customize=1`));
  // Fitted once the card is shown (a folded group fits its cards when it is opened).
  card.fitNow = () => {
    if (!p.offered || card.fitted) return;
    card.fitted = true;
    fitCard(p).then((reply) => {
      if (!alive() || !reply) return;
      const got = showFit(card, fitLine, why, choose, reply, p);
      if (got) budget = got;
      if (customize) customize.disabled = choose.disabled;
    }).catch((err) => {
      if (!alive()) return;
      card.classList.add("off");
      fitLine.textContent = t("gallery.not_offered");
      why.textContent = `${err.message} ${err.fix || ""}`.trim();
    });
  };
  return card;
}

function readOpen() {
  try {
    const stored = JSON.parse(window.localStorage.getItem(OPEN_KEY) || "[]");
    return new Set(Array.isArray(stored) ? stored : []);
  } catch (err) {
    return new Set();
  }
}

function writeOpen(open) {
  try { window.localStorage.setItem(OPEN_KEY, JSON.stringify([...open])); } catch (err) { /* private window */ }
}

// How many cards one row of this grid holds at its current width.
function perRow(grid) {
  const columns = getComputedStyle(grid).gridTemplateColumns.split(" ").filter((c) => c && c !== "none");
  return Math.max(1, columns.length);
}

export async function gallery(body) {
  document.getElementById("data").textContent = t("gallery.data");
  const list = await api.get("/api/presets");
  if (!list.presets || !list.presets.length) {
    body.append(h("p", { class: "banner warn" }, t("gallery.empty")));
    return null;
  }
  let alive = true;
  const isAlive = () => alive;
  const open = readOpen();
  const sections = [];
  for (const group of GROUPS) {
    const cards = [];
    for (const p of list.presets.filter((row) => (row.group || "start") === group)) {
      cards.push({ order: p.order ?? 99, make: () => (p.card === "environment" ? environmentCard(p, isAlive) : presetCard(p, isAlive)) });
    }
    for (const c of extra.filter((row) => row.group === group)) {
      cards.push({
        order: c.order,
        make: () => {
          const cell = h("div", { class: "g-cell", dataset: { card: c.id } });
          Promise.resolve(c.render(cell, { go, fitCard })).catch(() => null);
          return cell;
        },
      });
    }
    if (!cards.length) continue;
    cards.sort((a, b) => a.order - b.order);
    const grid = h("div", { class: "g-cards" });
    const elements = cards.map((c) => c.make());
    grid.append(...elements);
    const always = group === "start";
    const more = h("button", { class: "btn g-more", type: "button", "aria-expanded": "false" });
    const head = h("div", { class: "g-group-head" },
      h("h2", { class: "cap", id: `g-group-${group}` }, t(`gallery.group.${group}`)),
      h("p", { class: "g-group-line" }, t(`gallery.group.${group}_line`)));
    const section = h("section", { class: "g-group", dataset: { group }, "aria-labelledby": `g-group-${group}` },
      head, grid, always ? null : h("div", { class: "g-more-row" }, more));
    body.append(section);
    const state = { group, grid, elements, more, open: always || open.has(group), always };
    sections.push(state);
    more.addEventListener("click", () => {
      state.open = !state.open;
      if (state.open) open.add(group); else open.delete(group);
      writeOpen(open);
      layout(state);
      if (state.open) {
        const firstNew = state.elements[perRow(state.grid)];
        const focusable = firstNew && firstNew.querySelector("button:not([disabled]), a[href], input:not([disabled])");
        if (focusable) focusable.focus({ preventScroll: false });
      }
    });
  }

  function layout(state) {
    const row = perRow(state.grid);
    const folded = !state.open && state.elements.length > row;
    state.elements.forEach((el, i) => {
      el.hidden = folded && i >= row;
      if (!el.hidden && typeof el.fitNow === "function") el.fitNow();
    });
    if (!state.always) {
      state.more.hidden = state.elements.length <= row;
      state.more.textContent = state.open ? t("gallery.show_fewer") : t("gallery.show_all", { n: state.elements.length });
      state.more.setAttribute("aria-expanded", state.open ? "true" : "false");
    }
  }

  // After the first layout the grids know their widths.
  requestAnimationFrame(() => { if (alive) sections.forEach(layout); });
  let resizeTimer = null;
  const onResize = () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => { if (alive) sections.forEach(layout); }, 120);
  };
  window.addEventListener("resize", onResize);

  body.append(h("p", { class: "note g-foot" }, t("gallery.why_offered"), " ", t("gallery.own_namelist")));
  return () => { alive = false; window.removeEventListener("resize", onResize); };
}
