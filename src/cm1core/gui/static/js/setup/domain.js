// Screen 5, Domain and time: the plan map with the bubble, the motion vectors and the predicted track with its
// edge clearance; "Use the recommended frame and bubble" (one gpuwm-cm1 edit, whose diff is shown and kept);
// the time step, the derived sound-wave substeps and the Courant checks. Advanced also changes the grid and times.

import { h, num, table, button, checkline, append, clock } from "../core.js";
import * as api from "../api.js";
import { register, go, notice, runRoute } from "../router.js";
import { actionButton } from "../command.js";
import {
  runId, loadSetup, hint, dataLine, resultBox, act, ensureEstimate, notImported, lead, spacing, stormTime,
  areaWords, levelState,
} from "./common.js";
import { planMap, edgeClearance } from "./planmap.js";

// A predicted storm time, to the minute, in the one clock form.
function minutes(s) {
  return clock(Math.round(s / 60) * 60);
}

function trackWords(g) {
  if (!g) return ["The storm-track check has not run yet."];
  if (g.status !== "evaluated") return [g.reason || `The storm-track check could not run (${g.status}).`];
  const out = [];
  if (g.stays_inside === false && g.exit_time_s != null) {
    out.push(`The storm is predicted to leave the area at ${minutes(g.exit_time_s)}.`);
  } else if (g.minimum_edge_clearance_km != null) {
    out.push(`The storm stays inside the area: it comes no closer than ${num(g.minimum_edge_clearance_km, 0)} km to ` +
      "an open side.");
  } else {
    out.push("The storm cannot leave: every side repeats.");
  }
  if (g.left_mover_leaves_at_s != null) out.push(`The left-moving storm leaves at ${minutes(g.left_mover_leaves_at_s)}.`);
  return out;
}

function timeRows(s) {
  const v = s.values || {};
  const run = (s.estimate && s.estimate.run) || {};
  const rows = [];
  const add = (label, cm1, value, note) => rows.push(h("tr", {},
    h("td", { class: "muted" }, label, hint(cm1)), h("td", { class: "text" }, value), h("td", { class: "muted" }, note || "")));
  add("Time step", "dtl", `${num(v.dtl, v.dtl % 1 ? 2 : 0)} s`, v.adapt_dt ? "adaptive (adapt_dt = 1)" : "");
  if (run.acoustic_substeps) {
    add("Sound-wave steps per step", "nsound, derived", `${run.acoustic_substeps} of ${num(v.dtl / run.acoustic_substeps, 2)} s`,
      "derived as CM1 derives it");
  }
  add("Storm time", "timax", `${num(v.timax, 0)} s (${stormTime(v.timax)})`, run.steps ? `${num(run.steps)} steps` : "");
  add("Files and pictures every", "tapfrq", `${num(v.tapfrq, 0)} s`, run.frames ? `${run.frames} saved times, with T+0` : "");
  add("Restarts", "rstfrq", v.rstfrq > 0 ? `every ${num(v.rstfrq, 0)} s` : "one at the end",
    v.rstfrq > 0 ? "" : "CM1 writes none for a negative rstfrq; GM1 keeps one so the run can be continued");
  return rows;
}

function changeForm(s, id, box) {
  const v = s.values || {};
  const fields = [
    ["nx", "Points across (x)", "nx", ""], ["ny", "Points across (y)", "ny", ""],
    ["dx", "Grid spacing, x", "dx", "m"], ["dy", "Grid spacing, y", "dy", "m"],
    ["dtl", "Time step", "dtl", "s"], ["timax", "Storm time", "timax", "s"], ["tapfrq", "Files every", "tapfrq", "s"],
    ["imove", "Moving frame on (1) or off (0)", "imove", ""], ["umove", "Frame speed east", "umove", "m/s"],
    ["vmove", "Frame speed north", "vmove", "m/s"],
  ];
  const inputs = {};
  const grid = h("div", { class: "formgrid" });
  for (const [key, label, cm1, unit] of fields) {
    const input = h("input", { class: "input num", value: v[key] ?? "", "aria-label": label, inputmode: "decimal",
      dataset: { key } });
    inputs[key] = input;
    grid.append(h("label", {}, label, hint(cm1)), h("div", {}, input, unit ? h("span", { class: "unit" }, unit) : null));
  }
  const collect = () => {
    const set = {};
    for (const [key] of fields) {
      const raw = inputs[key].value.trim();
      if (raw === "" || Number(raw) === Number(v[key])) continue;
      const n = Number(raw);
      if (!Number.isFinite(n)) return { error: `${key}: give a number.` };
      set[key] = ["nx", "ny", "imove"].includes(key) ? Math.round(n) : n;
    }
    if (!Object.keys(set).length) return null;
    return { path: `${api.runPath(id)}/edit`, body: { set } };
  };
  const saveWrap = actionButton("Save and check again", { action: "edit", kind: "primary", small: true,
    disabled: !s.editable, request: collect, watch: grid });
  const save = saveWrap.button;
  save.addEventListener("click", async () => {
    const req = collect();
    if (!req) { notice("Nothing changed."); return; }
    if (req.error) { notice(req.error, "stop"); return; }
    const reply = await act(save, req.path, req.body, box);
    if (reply && reply.command) {
      try { await api.post(`${api.runPath(id)}/estimate`); } catch (_) { /* the screen says it is stale */ }
      setTimeout(() => go(runRoute("domain", id)), 500);
    }
  });
  return h("section", { class: "panel advanced-only", id: "change-domain" },
    h("h2", { class: "cap" }, "Change the grid and times"),
    grid,
    h("div", { class: "btns", style: "margin-top:8px" }, saveWrap,
      h("span", { class: "note" }, "Written into the run's namelist.input; the estimate runs again.")));
}

async function render(body, args, page) {
  const id = runId(args);
  const guided = page.mode === "guided";
  let s = await loadSetup(id);
  if (!s.report.present) { page.setData(dataLine(s)); notImported(body, s); return null; }
  s = await ensureEstimate(id, s);
  const v = s.values || {};
  const g = s.geometry;
  page.setData(`${areaWords(v)}, dx ${spacing(v.dx)}, iorigin ${v.iorigin} · ` +
    (v.imove === 1 ? `moving frame ${num(v.umove, 3)}, ${num(v.vmove, 3)} m/s` : "no moving frame") + ` · ${stormTime(v.timax)}`);
  const box = resultBox();
  if (guided) {
    body.append(lead("The storm starts from a warm bubble and drifts with the wind. GM1 predicts its path so it stays ",
      "inside the area for the whole run."));
  }
  const bubble = ((s.setup.fortran || []).find((grp) => grp.id === "bubble") || {});
  const radius = bubble.active ? ((bubble.fields || []).find((f) => f.key === "initialization.bhrad_m") || {}).value : null;
  const pair = g && g.recommended_pair;
  const mapPanel = h("section", { class: "panel" }, h("h2", { class: "cap" }, "Plan view"),
    g && g.domain_km ? planMap(g, { bubbleRadiusKm: radius, recommended: pair && !pair.uses_it ? pair : null }) :
      h("p", { class: "note" }, "The map appears once the storm-track check has run (Estimate)."),
    h("div", { class: "legend", style: "margin-top:8px" },
      h("span", {}, h("span", { class: "sw", style: "border-color:#e6edf7;border-top-style:dotted" }), "predicted track, right mover"),
      h("span", {}, h("span", { class: "sw", style: "border-color:#5b6477;border-top-style:dashed" }), "left mover"),
      radius ? h("span", {}, h("span", { class: "sw", style: "border-color:#ffb347;border-top-style:dashed" }), "warm bubble") : null,
      h("span", {}, h("span", { class: "sw", style: "border-color:#8e97a8;border-top-style:dashed" }), "repeating side")));

  const recBox = h("div", {});
  // When the run as it is already keeps the predicted storm inside by the edge rule, the recommendation is optional:
  // advanced mode offers it as a plain button, and guided mode keeps the namelist's own frame and bubble (CM1's own
  // setup stays CM1's). The track is a prediction from the wind profile, and the storm GM1 runs can sit several km
  // from it, so a small predicted gain is not worth moving a run away from its namelist.
  const meets = !!g && g.status === "evaluated" && g.stays_inside !== false && g.meets_edge_rule !== false;
  if (pair && guided && meets && !pair.uses_it) {
    recBox.append(h("p", { class: "note", style: "margin:0 0 10px" }, "As it is set up, the storm is predicted to stay ",
      "inside the area with room to spare, so the moving frame and the bubble stay as the namelist has them.",
      hint("umove, vmove and the bubble centre")));
  } else if (pair) {
    const words = pair.uses_it ? "The run already uses the recommended frame and bubble." :
      `Recommended: moving frame ${num(pair.frame_m_s[0], 3)}, ${num(pair.frame_m_s[1], 3)} m/s` +
      (v.iinit === 1 ? `, bubble at ${num(pair.bubble_km[0], 1)}, ${num(pair.bubble_km[1], 1)} km` : "") +
      (pair.minimum_edge_clearance_km != null ? `. The storm is then predicted to keep at least ${num(pair.minimum_edge_clearance_km, 0)} km ` +
        "from every open side" + (meets && g.minimum_edge_clearance_km != null ?
        ` (${num(g.minimum_edge_clearance_km, 0)} km as it is now)` : "") + "." : ".");
    // A recommendation worked out before the namelist last changed is not offered (the server refuses it too).
    const stale = s.geometry_stale === true;
    const useWrap = actionButton("Use the recommended frame and bubble", {
      action: "recommended", kind: pair.uses_it || stale || meets ? "" : "primary",
      disabled: pair.uses_it || !s.editable || stale,
      title: !s.editable ? "This run has already run." :
        stale ? "The namelist changed since this was worked out: press Estimate again." : null,
      request: () => ({ path: `${api.runPath(id)}/recommended`, body: {} }) });
    const use = useWrap.button;
    use.addEventListener("click", async () => {
      const reply = await act(use, `${api.runPath(id)}/recommended`, {}, box);
      if (reply && reply.ok && reply.verdict !== "unchanged") {
        try { await api.post(`${api.runPath(id)}/estimate`); } catch (_) { /* the screen says it is stale */ }
        setTimeout(() => go(runRoute("domain", id)), 700);
      }
    });
    // core.append, not Element.append: the DOM's own append writes a null part as the word "null".
    append(recBox, [h("p", { class: "note", style: "margin:0 0 10px" }, words, guided ? hint("umove, vmove and the bubble centre") : null),
      stale ? h("p", { class: "banner warn" }, "This was worked out before the namelist last changed.",
        h("span", { class: "sub" }, "Estimate again to see the recommendation for the namelist as it is now.")) : null,
      h("div", { class: "btns" }, useWrap),
      meets && !pair.uses_it ? h("p", { class: "note" }, "Optional: the run as it is already keeps the predicted storm ",
        "inside by the edge rule. The track is predicted from the wind profile and can be several km off.") : null,
      h("p", { class: "note" }, pair.why ? `${pair.why.charAt(0).toUpperCase()}${pair.why.slice(1)}.` : "",
        " The change is written into the run's namelist.input and settings.toml; the difference from your file is shown ",
        "here and kept in the run folder.")]);
  }
  const trackPanel = h("section", { class: "panel", id: "track" }, h("h2", { class: "cap" }, "Storm track"),
    trackWords(g).map((t) => h("p", { style: "margin:0 0 6px" }, t)),
    g && g.model && !guided ? h("p", { class: "note" }, `Model: ${g.model}.`) : null,
    recBox, box);

  const checks = ((s.estimate && s.estimate.checks) || []).filter((c) => ["time-step", "courant"].includes(c.id));
  const timePanel = h("section", { class: "panel" }, h("h2", { class: "cap" }, "Time"),
    table(["", "", ""], timeRows(s), { compact: true }),
    checks.length ? h("div", { class: "section" }, checks.map((c) => checkline(levelState(c.level), c.text))) : null);

  let clearPanel = null;
  const right = g && g.track_km && g.track_km.right;
  if (right && right.length && g.domain_km) {
    const end = right[right.length - 1];
    const c = edgeClearance(g, [end[1], end[2]]);
    const nearest = Object.entries(c).filter(([, x]) => x !== null).sort((a, b) => a[1] - b[1])[0];
    clearPanel = h("section", { class: "panel" }, h("h2", { class: "cap" }, `Room to each side at ${minutes(end[0])} (predicted)`),
      table(["Side", { label: "km", num: true }], Object.entries(c).map(([side, km]) =>
        h("tr", {}, h("td", {}, side), h("td", { class: `num${nearest && side === nearest[0] ? " text" : ""}` },
          km === null ? "repeats" : num(km, 0)))), { compact: true }));
  }

  body.append(h("div", { class: "grid cols-main" }, mapPanel,
    h("div", { class: "grid", style: "align-content:start" }, trackPanel, timePanel, clearPanel)));
  if (!guided) body.append(h("div", { class: "section" }, changeForm(s, id, box)));
  if (s.diff && !guided) {
    body.append(h("details", { class: "section" }, h("summary", {}, "This run's namelist differs from your original file"),
      h("pre", { class: "diff" }, s.diff)));
  }
  body.append(h("div", { class: "section btns" },
    button("Continue", { kind: "primary", onclick: () => go(runRoute(guided ? "estimate" : "settings", id)) })));
  return null;
}

register("domain", render, { title: "Domain and time", runPage: true });
