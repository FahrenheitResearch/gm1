// The live GUI's frame: the left rail (Runs, New run, System, and an open run's pages), the page head with the
// Guided | Advanced switch, the numbered setup steps and a hash router. Every screen registers itself in
// router.js: Runs (runs.js), Run and Results (run/), System (system.js), the setup screens (setup/) and guided
// mode's (../guided/).

import { h } from "./core.js";
import * as api from "./api.js";
import {
  SCREENS, NAV, RUN_PAGES, register, go, notice, setNoticeBox, titleOf, runRoute, homeRoute, onOpenRun, openRunRoute,
} from "./router.js";
import { showCommand as sharedShowCommand } from "./command.js";
import { landingRoute, runningCard } from "./running.js";
import "./setup/index.js";
import "../guided/index.js";
import "./runs.js";
import "./run/run.js";
import "./run/results.js";
import "./system.js";
import "./advanced/index.js";
import "./wrf/index.js";
import "./sounding/index.js";
import "./chain/index.js";

export { register, go, notice, homeRoute, onOpenRun, openRunRoute };

const MARK = '<svg viewBox="0 0 100 100" width="18" height="18" aria-hidden="true"><defs><linearGradient id="gm1g" ' +
  'x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#fff4dc"/><stop offset="1" stop-color="#f09a3a"/>' +
  '</linearGradient></defs><polyline points="50.0,46.7 51.3,45.9 53.2,45.6 55.3,46.1 57.4,47.6 58.9,50.0 59.7,53.1 ' +
  '59.2,56.7 57.4,60.2 54.3,63.2 50.0,65.2 44.9,65.7 39.6,64.4 34.5,61.2 30.6,56.3 28.2,50.0 28.0,42.9 30.2,35.6 ' +
  '34.8,29.1 41.6,24.1 50.0,21.4 59.3,21.4 68.5,24.5 76.6,30.7 82.6,39.4 85.7,50.0 85.4,61.5 81.3,72.7 73.6,82.4 ' +
  '62.8,89.5 50.0,93.0 36.2,92.3 23.0,87.2" fill="none" stroke="url(#gm1g)" stroke-width="8" ' +
  'stroke-linecap="round"/><circle cx="50" cy="50" r="7" fill="#fff4dc"/></svg>';

// "Show command" (command.js). `always`: shown in guided mode too (Start and Results); otherwise advanced only.
export function showCommand(command, always = false) {
  return sharedShowCommand(command, { always });
}

let unpaintMode = null;

function modeSwitch() {
  const wrap = h("div", { class: "modes", role: "group", "aria-label": "View" });
  const buttons = {};
  for (const [mode, label] of [["guided", "Guided"], ["advanced", "Advanced"]]) {
    // The mode is remembered per OS user (~/.config/gpuwm-cm1/gui.json); the same switch from a terminal:
    buttons[mode] = h("button", { type: "button", "aria-pressed": "false", title: `gpuwm-cm1 gui --mode ${mode}`,
      dataset: { action: "mode", command: `gpuwm-cm1 gui --mode ${mode}` } }, label);
    buttons[mode].addEventListener("click", async () => {
      if (api.currentMode() === mode) return;
      try { await api.setMode(mode); } catch (err) { notice(err.message, "stop"); }
    });
    wrap.append(buttons[mode]);
  }
  const paint = (mode) => {
    for (const [m, b] of Object.entries(buttons)) {
      b.classList.toggle("on", m === mode);
      b.setAttribute("aria-pressed", m === mode ? "true" : "false");
    }
  };
  paint(api.currentMode());
  // Every screen draws a new switch; only the one on the page listens.
  if (unpaintMode) unpaintMode();
  unpaintMode = api.onMode(paint);
  return wrap;
}

// The commands this page has run, newest first: every action is one gpuwm-cm1 command (advanced only).
const ranCommands = [];
api.onCommand((reply) => {
  if (!reply || typeof reply.command !== "string") return;
  ranCommands.unshift({ command: reply.command, action: reply.action || "", ok: reply.ok !== false });
  ranCommands.length = Math.min(ranCommands.length, 12);
  const box = document.getElementById("ran");
  if (box) paintRan(box);
});

function paintRan(box) {
  box.replaceChildren();
  if (!ranCommands.length) return;
  const last = ranCommands[0];
  box.append(h("div", { class: "cap" }, "Commands this page ran"),
    ...ranCommands.slice(0, 5).map((row, i) => h("div", { class: `ranline${i ? " old" : ""}` },
      h("code", {}, row.command))));
  box.dataset.last = last.command;
}

// The run pages a folder of CM1 output opened for pictures has (Runs: Open CM1 output).
const CM1_OUTPUT_PAGES = ["results", "commands"];

function frame(page, sessionInfo) {
  const mode = api.currentMode();
  const nav = h("nav", { class: "nav", "aria-label": "Screens" });
  for (const item of NAV) {
    const cls = [page.screen === item.name ? "on" : "", item.mode ? `${item.mode}-only` : ""].join(" ").trim();
    nav.append(h("a", { href: `#/${item.name}`, class: cls || null }, item.label));
    if (item.name === "runs" && page.run) {
      nav.append(h("div", { class: "group" }, "Open run"), h("div", { class: "runname" }, page.run));
      for (const p of RUN_PAGES) {
        if (mode === "guided" && !p.guided) continue;
        if (!SCREENS.has(p.name)) continue;
        const label = mode === "guided" && p.guidedLabel ? p.guidedLabel : p.label;
        nav.append(h("a", { href: `#/${runRoute(p.name, page.run)}`, class: `step${page.screen === p.name ? " on" : ""}`,
          dataset: { page: p.name } }, label));
      }
      // CM1's own output opened for pictures has nothing to set up or run: its pages are Results and Commands.
      api.get(`${api.runPath(page.run)}/status`).then((st) => {
        if (!st || !st.cm1_output) return;
        for (const a of nav.querySelectorAll("a.step")) a.hidden = !CM1_OUTPUT_PAGES.includes(a.dataset.page);
      }).catch(() => { /* the screen itself says what went wrong */ });
    }
  }
  const rail = h("aside", { class: "rail" }, h("div", { class: "rail-inner" },
    h("div", { class: "mark", html: `${MARK}<b>GM1</b><span>simulated</span>` }),
    nav,
    h("div", { class: "foot" },
      h("div", { class: "mono" }, `${sessionInfo.bind}:${sessionInfo.port}`),
      h("div", {}, `gpuwm-cm1 ${sessionInfo.version}`),
      sessionInfo.runner === "fake" ? h("div", {}, "Fake runner: every result is made up.") :
        sessionInfo.runner === "mixed" ? h("div", {}, "Fake GPU: runs and speed measurements are made up.") : null)));
  const dataLine = h("div", { class: "data", id: "data" }, page.data || "");
  const head = h("header", { class: "head" },
    h("div", {}, h("h1", {}, page.title), dataLine),
    h("div", { class: "right" }, h("div", { class: "modebox" }, modeSwitch(),
      sharedShowCommand(`gpuwm-cm1 gui --mode ${mode === "guided" ? "advanced" : "guided"}`,
        { title: "Switch the view from a terminal; it is remembered for this user" }))));
  const noticeBox = h("div", { id: "notice", role: "status", "aria-live": "polite" });
  setNoticeBox(noticeBox);
  const body = h("div", { id: "body" });
  const main = h("main", { id: "main" }, head);
  const current = RUN_PAGES.findIndex((p) => p.name === page.screen);
  if (page.run && current >= 0 && RUN_PAGES[current].setup && mode === "advanced") {
    const steps = h("nav", { class: "steps", id: "steps", "aria-label": "Setup steps" });
    RUN_PAGES.filter((p) => p.setup && p.step !== false).forEach((p, i) => {
      steps.append(h("a", { href: `#/${runRoute(p.name, page.run)}`, class: p.name === page.screen ? "on" : null,
        dataset: { step: p.name } }, h("b", {}, String(i + 1)), p.label));
    });
    main.append(steps);
  }
  const ran = h("div", { class: "ran advanced-only", id: "ran" });
  paintRan(ran);
  main.append(noticeBox, body, ran);
  document.body.replaceChildren(h("div", { class: "app" }, rail, main));
  document.title = `${page.title} · GM1`;
  return { body, setData: (text) => { dataLine.textContent = text || ""; } };
}

// ---- router

let cleanup = null;
let routing = 0;
// A page that was just opened (no screen named in its address) goes straight to the one storm that is running.
// Asked once per page load, so coming back to the first screen afterwards stays there (the card leads it).
let landed = false;
let card = null;
// The screens that lead with the "A storm is running" card: each mode's first screens, and Runs.
const FIRST_SCREENS = ["guided", "new-run", "runs"];

async function route() {
  const ticket = ++routing;
  if (cleanup) { try { cleanup(); } catch (_) { /* a screen that failed to clean up */ } cleanup = null; }
  if (card) { card.stop(); card = null; }
  const [name, ...args] = (location.hash.replace(/^#\/?/, "") || "").split("/");
  const s = await api.session();
  if (!landed) {
    landed = true;
    if (!name) {
      const to = await landingRoute(api.currentMode());
      if (ticket !== routing) return;
      if (to) {
        // Replaced, not pushed: Back leaves GM1 instead of coming back to this hop.
        history.replaceState(null, "", `#/${to}`);
        return route();
      }
    }
  }
  let screen = name;
  if (!SCREENS.has(screen)) {
    // Guided mode opens on the storm gallery; without it, a first launch with no runs opens New run.
    screen = "runs";
    if (!name && api.currentMode() === "guided") {
      if (homeRoute("guided") !== "runs") screen = homeRoute("guided");
      else if (SCREENS.has("new-run")) {
        try {
          const runs = await api.get("/api/runs");
          if (!runs.runs.length) screen = "new-run";
        } catch (_) { /* the Runs screen says what went wrong */ }
      }
    }
  }
  if (ticket !== routing) return;
  const entry = SCREENS.get(screen);
  const mode = api.currentMode();
  const run = entry.runPage && args[0] ? decodeURIComponent(args[0]) : null;
  const page = frame({ screen, title: titleOf(screen, mode), run }, s);
  if (FIRST_SCREENS.includes(screen) && !(screen === "guided" && args[0])) {
    card = runningCard(go);
    page.body.before(card.el);
  }
  document.body.dataset.screen = screen;
  document.body.dataset.route = String(ticket);
  delete document.body.dataset.ready;
  try {
    cleanup = (await entry.render(page.body, args, { ...page, mode, run })) || null;
    if (ticket === routing) document.body.dataset.ready = "1";
  } catch (err) {
    page.body.append(h("p", { class: "banner stop" }, `This screen failed to draw: ${err.message}`,
      err.fix ? h("span", { class: "sub" }, err.fix) : null));
    document.body.dataset.ready = "error";
  }
}

let started = false;

export async function start() {
  if (started) return;
  started = true;
  try {
    const s = await api.session();
    api.applyMode(s.mode);
  } catch (err) {
    document.body.replaceChildren(h("p", { class: "banner stop", style: "margin:24px" },
      `${err.message} ${err.fix || "Start GM1 again and use the page it opens."}`));
    return;
  }
  window.addEventListener("hashchange", route);
  // The mode is a view: switching it draws the same screen again, for the other mode.
  let shown = api.currentMode();
  api.onMode((mode) => { if (mode !== shown) { shown = mode; route(); } });
  await route();
}

// Every module index.html loads registers its screens first: module scripts all run before DOMContentLoaded
// (document.readyState is "interactive" while they run), so the first route waits for it.
if (document.readyState === "complete") start();
else document.addEventListener("DOMContentLoaded", start, { once: true });
