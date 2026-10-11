// Guided mode: four screens over the same engine and run folders as Advanced.
//
//   #/guided                          Pick a storm (the gallery)
//   #/guided/fit/PRESET[/BUDGET]      How long can it run?
//   #/guided/watch/RUN                Watching
//   #/guided/results/RUN              Results, on the four favourite looks
//
// The mode is a view: every screen here is reachable in Advanced too, and a guided run opens in Advanced with its
// namelist, its check against CM1 and fit.json.

import { register, onOpenRun } from "../js/router.js";
import { ready, t } from "./copy.js";
import { gallery } from "./gallery.js";
import { budget, takeAdvancedHop } from "./budget.js";
import { applyMode } from "../js/api.js";
import { watch } from "./watch.js";
import { results } from "./results.js";

const TITLES = { "": "gallery.title", fit: "budget.title", watch: "watch.title", results: "results.title" };

async function guided(body, args, page = {}) {
  // Back from the folder "All settings (Advanced)" opened: guided again (that hop never saved the mode).
  if (page.mode === "advanced" && takeAdvancedHop()) {
    applyMode("guided");
    return null;
  }
  await ready();
  const [screen = "", ...rest] = args;
  const head = document.querySelector(".head h1");
  if (head) head.textContent = t(TITLES[screen] || TITLES[""]);
  document.title = `${head ? head.textContent : ""} · GM1`;
  if (screen === "fit") return budget(body, rest);
  if (screen === "watch") return watch(body, rest, page);
  if (screen === "results") return results(body, rest, page);
  return gallery(body, rest);
}

register("guided", guided, { label: "New storm", nav: true, mode: "guided", title: "Pick a storm", order: 10 });

// A run in the Runs list opens on Watching while it runs and on Results after, in guided mode.
onOpenRun((row, mode) => {
  if (mode !== "guided" || !row || !row.url) return null;
  const state = (row.status || {}).state;
  if (["running", "waiting", "stale"].includes(state)) return `guided/watch/${row.url}`;
  if (["finished", "stopped", "paused", "failed"].includes(state)) return `guided/results/${row.url}`;
  return null;
});

