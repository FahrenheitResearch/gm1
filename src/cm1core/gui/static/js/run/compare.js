// Results: Compare with other runs. Pick one to five runs from the Runs list; the server draws them beside this one
// (gpuwm-cm1 quicklook RUNDIR --compare OTHER ..., on the computer's processor at nice 19 with no GPU) into
// quicklook/compare/: the storms side by side at one storm time, every run over time, and the soundings each
// started in beside what it grew. The pictures show here when they are drawn; earlier comparisons stay listed.

import { h, stormClock, parseClock } from "../core.js";
import * as api from "../api.js";
import { notice } from "../router.js";
import { actionButton, put } from "./parts.js";

const POLL_MS = 2000;

function runMeta(row) {
  const st = row.status || {};
  // CM1 output opened for pictures has the same words for both: said once
  return [...new Set([row.case, st.label].filter(Boolean))].join(" · ");
}

// A comparison's storm time, as its pictures say it (a hurricane's in hours and days).
function when(cmp) {
  return stormClock(cmp.time_s || 0, cmp.kind === "hurricane" ? 172800 : 0);
}

// One comparison: its runs (letter, colour, name), its time and its pictures, each opening full size.
function comparisonView(cmp) {
  const runs = (cmp.runs || []).map((r) => h("li", { class: "cmp-run" },
    h("span", { class: "cmp-chip", style: `background:${r.colour || "#888"}`, "aria-hidden": "true" }),
    h("b", {}, r.letter || ""), " ", r.name || "",
    r.saved_frames === false ? h("span", { class: "note" }, " · drawn from its own pictures (its saved frames were deleted)") : null));
  const differ = (cmp.differ || []).length ? `These runs differ in: ${cmp.differ.join(", ")}.` :
    "No setting that changes the storm differs between these runs.";
  const pics = (cmp.pictures || []).map((p) => h("figure", { class: "cmp-pic" },
    h("a", { href: p.url, target: "_blank", rel: "noopener" }, h("img", { src: `${p.url}?v=${encodeURIComponent(cmp.stamp || "")}`, alt: p.title, loading: "lazy" })),
    h("figcaption", {}, p.title)));
  const notDrawn = Object.entries(cmp.not_drawn || {}).map(([k, why]) => h("p", { class: "note" }, `Not drawn: ${k}: ${why}`));
  return h("div", { class: "cmp-one" },
    h("ul", { class: "cmp-runs-shown" }, runs),
    h("p", { class: "note" }, cmp.time_s != null ? `Side by side at ${when(cmp)} (each run's save nearest it). ` : "", differ),
    notDrawn, h("div", { class: "cmp-pics" }, pics));
}

export function comparePanel(id) {
  const list = h("div", { class: "cmp-pick", role: "group", "aria-label": "Runs to compare with" });
  const timeIn = h("input", { class: "input num", style: "width:110px", placeholder: "latest",
    "aria-label": "Storm time to compare at (T+H:MM:SS)" });
  const status = h("p", { class: "note", role: "status", hidden: true });
  const shown = h("div", { class: "cmp-shown" });
  const earlierSel = h("select", { class: "input", style: "width:auto", "aria-label": "Earlier comparisons", hidden: true });
  let boxes = [];
  let blocked = new Set();
  let comparisons = [];
  let maxRuns = 6;
  let drawing = false;
  let timer = null;
  // After a press: the newest comparison as it was then, and the reply's words. Until the new drawing shows up
  // (running, or a newer row), the status keeps those words rather than an earlier comparison's refusal.
  let pressed = null;
  const signature = (c) => (c ? `${c.folder}|${(c.progress || {}).updated_utc || ""}|${c.state}` : "");

  const picked = () => boxes.filter((b) => b.checked).map((b) => b.value);
  const body = () => {
    const b = { with: picked() };
    const t = parseClock(timeIn.value);
    if (t != null) b.time_s = t;
    return b;
  };
  const draw = actionButton(id, "compare", "Draw the comparison", {
    kind: "primary", always: true, body, watch: [list, timeIn],
    check: () => {
      if (!picked().length) return "Pick at least one other run to compare with.";
      if (picked().length > maxRuns - 1) return `Pick at most ${maxRuns - 1} other runs.`;
      if (timeIn.value.trim() && parseClock(timeIn.value) == null) return "Write the storm time as T+H:MM:SS, or leave it empty for the latest time every run reached.";
      return null;
    },
    done: (reply) => {
      if (reply.ok) {
        pressed = { before: signature(comparisons[0]), message: reply.message };
        status.hidden = false; status.textContent = reply.message; poll(true);
      }
    },
  });

  async function loadRuns() {
    let rows = [];
    let cannot = {};
    try { rows = (await api.get("/api/runs")).runs || []; } catch (err) { notice(err.message, "warn"); }
    // runs that can only be refused (a hurricane beside a thunderstorm, or nothing of its storm left) are greyed
    // out with the reason, before Draw is pressed
    try { cannot = (await api.get(`${api.runPath(id)}/compare?choices=1`)).cannot || {}; } catch (_) { cannot = {}; }
    const others = rows.filter((r) => r.id !== id);
    if (!others.length) {
      put(list, h("p", { class: "note" }, "No other run in this runs folder yet. Runs you make or open from the Runs list can be compared here."));
      return;
    }
    blocked = new Set(others.filter((r) => cannot[r.id]).map((r) => r.id));
    boxes = others.map((r) => h("input", { type: "checkbox", value: r.id, disabled: blocked.has(r.id) ? true : null }));
    put(list, others.map((r, n) => h("label", { class: "cmp-choice" }, boxes[n],
      h("span", { class: "cmp-name" }, r.name || r.id),
      h("span", { class: "note" }, cannot[r.id] ? `Cannot be compared with this run: ${cannot[r.id]}.` : runMeta(r)))));
    for (const b of boxes) b.addEventListener("change", () => {
      const full = picked().length >= maxRuns - 1;
      for (const o of boxes) o.disabled = blocked.has(o.value) || (full && !o.checked);
    });
  }

  function paint() {
    const running = comparisons.find((c) => c.state === "running");
    if (pressed && (running || !drawing || signature(comparisons[0]) !== pressed.before)) pressed = null;
    if (running) {
      const p = running.progress || {};
      status.hidden = false;
      status.textContent = `Drawing the comparison: ${p.step || "starting"}${p.total ? ` (${p.done || 0} of ${p.total})` : ""}.`;
    } else if (pressed) {
      status.hidden = false;
      status.textContent = pressed.message;
    } else {
      const failed = comparisons[0] && comparisons[0].state === "failed" ? comparisons[0] : null;
      status.hidden = !failed;
      status.textContent = failed ? `Not drawn: ${(failed.progress || {}).message || "the command stopped."}` : "";
    }
    const done = comparisons.filter((c) => c.state === "done");
    earlierSel.hidden = done.length < 2;
    const keep = earlierSel.value;
    earlierSel.replaceChildren(...done.map((c, n) => h("option", { value: c.folder, selected: c.folder === keep ? true : null },
      `${n === 0 ? "Newest: " : ""}${(c.runs || []).map((r) => r.letter + " " + r.name).join(", ")} · ${when(c)}`)));
    const chosen = done.find((c) => c.folder === earlierSel.value) || done[0];
    put(shown, chosen ? comparisonView(chosen) : null);
  }

  async function poll(soon) {
    clearTimeout(timer);
    try {
      const reply = await api.get(`${api.runPath(id)}/compare`);
      comparisons = reply.comparisons || [];
      maxRuns = reply.max_runs || maxRuns;
      drawing = !!reply.drawing;
      paint();
      if (reply.drawing || comparisons.some((c) => c.state === "running") || soon === true) {
        timer = setTimeout(poll, POLL_MS);
      }
    } catch (_) { /* the next press asks again */ }
  }
  earlierSel.addEventListener("change", paint);

  const section = h("section", { class: "panel section compare" },
    h("h2", { class: "cap" }, "Compare with other runs",
      h("span", { class: "n" }, "side by side at one storm time, over time, and the soundings each started in")),
    h("p", { class: "note" }, "Pick one to five runs from your Runs list. Each storm is shown in a box of one size centred on it, " +
      "at the save nearest one storm time; runs with another grid, area or save times are drawn as they are, and the " +
      "pictures say so. Drawn on the computer's processor (nice 19, no GPU) into quicklook/compare/ in this run's folder."),
    list,
    h("div", { class: "btns" }, h("label", { class: "muted" }, "at storm time ", timeIn), draw),
    status, earlierSel, shown);
  section.load = () => { loadRuns(); poll(false); };
  section.stop = () => clearTimeout(timer);
  section.load();
  return section;
}

