// "A storm is running": the card the first screens lead with while a run is going, and where a page that was just
// opened lands.
//
// GM1's window can be closed while a storm runs, and the storm keeps going. Opening GM1 again must show it at
// once: with exactly one run going, a freshly opened page goes straight to it (Watching in guided mode, Run in
// advanced); with more than one, or when the person comes back to the first screen, the card names each run with
// its share done and a "Watch it" button. The card is drawn over Pick a storm, New run and Runs, in both modes.

import { h, button } from "./core.js";
import * as api from "./api.js";

const EVERY_MS = 3000;
// A run that is going: running, or waiting for the GPU. (A run on another computer that went quiet is "stale":
// the Runs list says so; it is not offered here.)
const GOING = ["running", "waiting"];
// How many runs the card names before it says how many more there are.
const NAMED = 3;

export function goingRuns(runs) {
  return (runs || []).filter((r) => GOING.includes((r.status || {}).state) && !r.view_only);
}

// Where a run that is going is watched: guided mode's Watching, advanced mode's Run.
export function watchRoute(run, mode) {
  return mode === "guided" ? `guided/watch/${run.url}` : `run/${run.url}`;
}

// The route a freshly opened page goes to, or null: the one run that is going.
export async function landingRoute(mode) {
  let reply;
  try { reply = await api.get("/api/runs"); } catch (_) { return null; }
  const going = goingRuns(reply.runs);
  return going.length === 1 ? watchRoute(going[0], mode) : null;
}

// "A storm is running: NAME, 42%" (or "is starting", or "is waiting for the GPU").
export function runningWords(run) {
  const st = run.status || {};
  const starting = /^Starting/.test(st.label || "");
  if (st.state === "waiting") return `A storm is waiting for the GPU: ${run.name}`;
  if (starting || st.percent === null || st.percent === undefined) return `A storm is starting: ${run.name}`;
  return `A storm is running: ${run.name}, ${Math.round(st.percent)}%`;
}

// The card. go(route) opens a route; returns {el, stop()}: el is empty (and hidden) while nothing is going.
export function runningCard(go) {
  const el = h("div", { class: "running-card", id: "running", hidden: true, role: "status", "aria-live": "polite" });
  let closed = false;
  let timer = null;
  async function draw() {
    if (closed) return;
    try {
      const reply = await api.get("/api/runs");
      const going = goingRuns(reply.runs);
      const mode = api.currentMode();
      el.hidden = !going.length;
      el.dataset.count = String(going.length);
      el.replaceChildren(...going.slice(0, NAMED).map((run) => {
        const watch = button("Watch it", { kind: "primary", onclick: () => go(watchRoute(run, mode)) });
        watch.dataset.run = run.id;
        return h("p", { class: "banner running", dataset: { run: run.id } },
          h("span", { class: "running-words" }, runningWords(run)), " ", watch);
      }), going.length > NAMED ? h("p", { class: "note" },
        `and ${going.length - NAMED} more: `, h("a", { href: "#/runs" }, "Runs"), " lists them all.") : "");
    } catch (_) { /* the next tick tries again */ }
    if (!closed) timer = setTimeout(draw, EVERY_MS);
  }
  draw();
  return { el, stop: () => { closed = true; clearTimeout(timer); } };
}
