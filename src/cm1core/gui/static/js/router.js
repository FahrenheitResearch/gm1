// The screen table, the hash router's helpers and the notice line. Screen modules import this (never app.js), so
// there is no import cycle: app.js imports the screens, the screens register themselves here.

export const SCREENS = new Map();
export const NAV = [];

// The pages of one run, in order (screens 3-9). Guided mode shows fewer of them: its setup is
// "Ready to start" (the Estimate screen drawn in plain words), then Run and Results. Advanced mode adds the
// namelist editor beside Check (a setup page, not a numbered step), Continue or extend, and the run's commands
//; guided mode lists only the pages marked guided.
export const RUN_PAGES = [
  { name: "check", label: "Check", setup: true },
  { name: "namelist", label: "Namelist", setup: true, step: false },
  { name: "sounding", label: "Sounding", setup: true },
  { name: "domain", label: "Domain and time", setup: true },
  { name: "settings", label: "Settings", setup: true },
  { name: "estimate", label: "Estimate", guidedLabel: "Ready to start", setup: true, guided: true },
  { name: "run", label: "Run", guidedLabel: "Watching", guided: true },
  { name: "results", label: "Results", guided: true },
  { name: "continue", label: "Continue or extend" },
  { name: "commands", label: "Commands" },
];

// register("check", render, {title: "Check" | (mode) => "...", label, nav, runPage, mode: "guided"|"advanced", order}).
// render(body, args, page) draws into body and may return a cleanup function. page.setData(text) sets the line
// under the title; page.mode is the view it was drawn for. A nav item with a mode is drawn in that mode only (the
// mode is a view: the screen itself stays reachable from both).
export function register(name, render, opts = {}) {
  SCREENS.set(name, { render, ...opts });
  if (opts.label && opts.nav && !NAV.some((item) => item.name === name)) {
    NAV.push({ name, label: opts.label, mode: opts.mode || null, order: opts.order ?? 50 });
    NAV.sort((a, b) => a.order - b.order);
  }
}

// The first screen of each mode: guided opens on the storm gallery when one is registered.
export function homeRoute(mode = document.body.dataset.mode || "guided") {
  return mode === "guided" && SCREENS.has("guided") ? "guided" : "runs";
}

// Where a run opens from the Runs list. Guided mode registers a hook (Watching while it runs, Results after);
// the first hook that names a route wins, else `fallback` (the run's setup pages, or its live view).
const openRunHooks = [];

export function onOpenRun(fn) {
  openRunHooks.push(fn);
}

export function openRunRoute(row, mode, fallback) {
  for (const fn of openRunHooks) {
    const route = fn(row, mode);
    if (route) return route;
  }
  return fallback || `run/${row.url}`;
}

export function titleOf(name, mode) {
  const entry = SCREENS.get(name);
  if (!entry) return name;
  const title = entry.title ?? entry.label ?? name;
  return typeof title === "function" ? title(mode) : title;
}

export function go(route) {
  const target = `#/${route}`;
  if (location.hash === target) window.dispatchEvent(new HashChangeEvent("hashchange"));
  else location.hash = target;
}

export function runRoute(page, id) {
  return `${page}/${encodeURIComponent(id)}`;
}

let noticeBox = null;
let lastNotice = null;
//: A notice given this soon before the screen is drawn again (a Save or Measure speed that reloads its page) is
//: shown on the new drawing too, so what just happened is not wiped by the redraw.
const NOTICE_CARRY_MS = 2500;

export function setNoticeBox(el) {
  noticeBox = el;
  const carried = lastNotice;
  lastNotice = null;
  if (carried && Date.now() - carried.at < NOTICE_CARRY_MS) notice(carried.text, carried.kind);
}

// One line under the title: what just happened, and what to do (kind: "", "warn" or "stop").
export function notice(text, kind = "") {
  lastNotice = text ? { text, kind, at: Date.now() } : null;
  if (!noticeBox) return;
  noticeBox.replaceChildren();
  if (!text) return;
  const p = document.createElement("p");
  p.className = `notice${kind ? ` ${kind}` : ""}`;
  p.textContent = text;
  noticeBox.append(p);
}
