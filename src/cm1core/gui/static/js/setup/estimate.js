// Screen 7, Estimate: needed against available for GPU memory, locked host memory and disk; the wall time with its
// range and basis; the estimator's checks in plain words; Measure speed, Start, Copy job script and Show command.
// Guided mode draws the same numbers as "Ready to start": the run in one sentence, three checks and one Start.

import { h, num, table, status, button, readout, kv, meter, checkline } from "../core.js";
import * as api from "../api.js";
import { register, go, notice, runRoute } from "../router.js";
import { showCommand, actionButton } from "../command.js";
import { jobscriptPanel } from "../advanced/jobscript.js";
import {
  runId, loadSetup, hint, dataLine, resultBox, act, ensureEstimate, notImported, lead, gb, durationWords,
  levelState, toAdvanced,
} from "./common.js";

function resourceRow(r) {
  const total = r.total_bytes || Math.max(r.need_bytes || 0, r.have_bytes || 0) || 1;
  const other = Math.max(0, Math.min(total, r.other_bytes || 0));
  const need = r.need_bytes || 0;
  const over = r.fits === false;
  const parts = [{ cls: "other", from: 0, to: other / total },
    { cls: over ? "over" : "need", from: other / total, to: Math.min(1, (other + need) / total) }];
  return h("tr", { dataset: { resource: r.id } },
    h("td", { class: "text nowrap" }, r.what),
    h("td", { class: "num text" }, gb(r.need_bytes),
      r.low_bytes && r.high_bytes ? h("span", { class: "sub" }, `${gb(r.low_bytes)} to ${gb(r.high_bytes)}`) : null),
    h("td", { class: "num" }, r.have_text || ""),
    h("td", { style: "min-width:180px" }, meter(parts)),
    h("td", {}, r.fits === false ? status("failed", "does not fit") : r.fits === true ? status("ok", "fits") :
      status("note", "not known"),
    h("span", { class: "sub" }, `${r.label || ""}${r.basis_label ? ` (basis ${r.basis_label})` : ""}: ${r.basis || ""}`),
    r.note ? h("span", { class: "sub" }, r.note) : null));
}

// GM1's own run setting: the most GPU memory the run's device registry may take (runtime.memory_budget_mib).
function budgetPanel(s, id, box) {
  const gpu = (s.setup && s.setup.gpu) || {};
  const rec = s.estimate && s.estimate.recommendation ? s.estimate.recommendation.memory_budget_mib : null;
  const current = gpu.memory_budget_mib;
  const input = h("input", { class: "input num", value: current ?? "", style: "width:110px", inputmode: "numeric",
    "aria-label": "GPU memory budget in MiB", disabled: s.editable ? null : true });
  const save = (value, b) => act(b, `${api.runPath(id)}/edit`, { settings: { "gpu.memory_budget_mib": value } }, box)
    .then((reply) => { if (reply && reply.ok) setTimeout(() => go(runRoute("estimate", id)), 500); });
  const typed = () => {
    const n = Math.round(Number(input.value));
    if (!Number.isFinite(n) || n < 256) return { error: "Give the budget in MiB, 256 or more." };
    return { path: `${api.runPath(id)}/edit`, body: { settings: { "gpu.memory_budget_mib": n } } };
  };
  const set = actionButton("Save", { action: "edit", small: true, disabled: !s.editable, request: typed, watch: input,
    onclick: () => {
      const req = typed();
      if (req.error) { notice(req.error, "stop"); return; }
      save(req.body.settings["gpu.memory_budget_mib"], set.button);
    } });
  const useRec = actionButton(rec ? `Use the recommended ${num(rec)} MiB` : "No recommendation yet", {
    action: "edit", small: true, kind: "primary", disabled: !s.editable || !rec || rec === current,
    request: rec ? () => ({ path: `${api.runPath(id)}/edit`, body: { settings: { "gpu.memory_budget_mib": rec } } }) : null,
    onclick: () => save(rec, useRec.button) });
  const resetBody = { reset: ["gpu.memory_budget_mib"] };
  const reset = actionButton("Back to GM1's default", { action: "edit", small: true,
    disabled: !s.editable || gpu.chosen == null, request: () => ({ path: `${api.runPath(id)}/edit`, body: resetBody }),
    onclick: () => act(reset.button, `${api.runPath(id)}/edit`, resetBody, box)
      .then((reply) => { if (reply && reply.ok) setTimeout(() => go(runRoute("estimate", id)), 500); }) });
  return h("section", { class: "panel section advanced-only", id: "budget" },
    h("h2", { class: "cap" }, "GPU memory budget", h("span", { class: "n" }, "GM1 only: runtime.memory_budget_mib")),
    h("div", { class: "btns" }, input, h("span", { class: "unit" }, "MiB"), set, useRec, reset),
    h("p", { class: "note" }, gpu.note || "",
      rec ? ` The estimate recommends ${num(rec)} MiB: this run's priced peak with room.` : ""));
}

async function measure(id, b, box, onDone) {
  const reply = await act(b, `${api.runPath(id)}/measure`, {}, box);
  if (!reply || !reply.ok) return;
  b.disabled = true;
  const started = Date.now();
  const poll = async () => {
    let st;
    try { st = await api.get(`${api.runPath(id)}/measure`); } catch (_) { st = { state: "failed" }; }
    if (st.state === "running" && Date.now() - started < 30 * 60 * 1000) {
      b.textContent = `Measuring… ${Math.round((Date.now() - started) / 1000)} s`;
      setTimeout(poll, 1500);
      return;
    }
    b.disabled = false;
    b.textContent = "Measure speed";
    if (st.state === "done") {
      const m = st.measured || {};
      notice(m.seconds_per_step != null ?
        `Measured ${num(m.seconds_per_step, 4)} s per step over ${m.steps} steps${m.gpu_name ? ` on the ${m.gpu_name}` : ""}. ` +
          "The estimate below now uses this GPU's own step time." :
        "Measured. The estimate below now uses this GPU's own step time.");
      onDone();
    } else {
      notice("The speed measurement stopped; its output is below.", "stop");
      box.show({ ok: false, message: "The speed measurement stopped.", fix: (st.log_tail || "").split("\n").slice(-3).join(" "),
        command: st.command });
    }
  };
  setTimeout(poll, 1000);
}

async function startRun(id, b, box, guided) {
  const reply = await act(b, `${api.runPath(id)}/start`, {}, box, { always: true });
  if (reply && reply.ok !== false) setTimeout(() => go(runRoute("run", id)), guided ? 1200 : 800);
}

function guidedView(body, s, id) {
  const e = s.estimate || {};
  const rows = e.rows || [];
  const byId = Object.fromEntries(rows.map((r) => [r.id, r]));
  const box = resultBox();
  body.append(h("p", { class: "lead big", id: "summary" }, (s.plain && s.plain.text) || ""));
  const checks = [];
  if (byId.gpu) {
    checks.push(checkline(byId.gpu.fits === false ? "fail" : "ok", byId.gpu.fits === false ?
      `It needs ${gb(byId.gpu.need_bytes)} of GPU memory and your GPU has ${gb(byId.gpu.have_bytes)} free.` :
      `Fits in your GPU's memory: needs about ${gb(byId.gpu.need_bytes)}, and ${gb(byId.gpu.have_bytes)} is free.`));
  }
  if (byId.disk) {
    checks.push(checkline(byId.disk.fits === false ? "fail" : "ok", byId.disk.fits === false ?
      `It writes about ${gb(byId.disk.need_bytes)}, more than the disk has free (${gb(byId.disk.have_bytes)}).` :
      `Writes about ${gb(byId.disk.need_bytes)} of files; ${gb(byId.disk.have_bytes)} is free.`));
  }
  const track = (e.checks || []).find((c) => c.id === "storm-track");
  if (track) checks.push(checkline(levelState(track.level), track.text));
  // Anything the estimate says would stop the run is shown, and Start waits for it to be fixed.
  const stops = (e.checks || []).filter((c) => c.level === "stop" && c.id !== "storm-track");
  for (const c of stops) checks.push(checkline("fail", c.text));
  const refused = s.report.exit === 2;
  const offered = !refused && e.fits !== false && !stops.length;
  const guidedOk = s.report.guided ? s.report.guided.offered : true;
  body.append(h("section", { class: "panel section" }, h("h2", { class: "cap" }, "Before it starts"), checks,
    !guidedOk ? checkline("warn", `Not one of the setups GM1 offers in Guided mode: ${s.report.guided.why_not || ""}`) : null));
  const start = button("Start", { kind: "primary big", disabled: !offered || s.has_run || s.report.exit === 3,
    title: offered ? null : "It does not fit on this computer now." });
  start.addEventListener("click", () => startRun(id, start, box, true));
  body.append(h("div", { class: "section btns" }, start,
    s.has_run ? h("a", { href: `#/${runRoute("run", id)}` }, "This run has started: watch it") : null,
    h("span", { class: "note" }, "Closing this page does not stop the run.")));
  body.append(showCommand(s.commands.start, { always: true }), box);
  body.append(h("p", { class: "note section" }, "Want every number, or to run it on a cluster? ", toAdvanced("Show everything"), "."));
}

async function render(body, args, page) {
  const id = runId(args);
  const guided = page.mode === "guided";
  let s = await loadSetup(id);
  if (!s.report.present) { page.setData(dataLine(s)); notImported(body, s); return null; }
  s = await ensureEstimate(id, s);
  const e = s.estimate;
  const gpuName = e && e.gpu && e.gpu.name ? e.gpu.name : "no GPU seen";
  const run = (e && e.run) || {};
  const scheme = s.report.component && s.report.component.scheme;
  page.setData(guided ? `${s.id} · your GPU: ${gpuName.replace("NVIDIA GeForce ", "")}` :
    `${s.id} · ${gpuName}${run.cells ? ` · ${num(run.cells)} cells` : ""}` +
    `${scheme ? ` · ${scheme}` : ""}${run.steps ? ` · ${num(run.steps)} steps` : ""}`);
  if (s.report.exit === 2) {
    body.append(h("p", { class: "banner stop" }, "GM1 cannot run this namelist as it is, so there is nothing to price.",
      h("span", { class: "sub" }, "Open Check: each refused setting says what to change.")),
    h("div", { class: "btns section" }, button("Open Check", { kind: "primary", onclick: () => go(runRoute("check", id)) })));
    return null;
  }
  if (!e) {
    body.append(h("p", { class: "banner warn" }, "The estimate did not finish.", h("span", { class: "sub" },
      "Press Estimate again; if it repeats, the notice above names the problem.")));
  }
  if (guided) { if (e) guidedView(body, s, id); return null; }

  const box = resultBox();
  if (e) {
    body.append(h("section", { class: "panel" }, h("h2", { class: "cap" }, "Needed against available"),
      table(["", { label: "Needs", num: true }, { label: "Available", num: true }, "grey: in use by others · white: this run", ""],
        e.rows.map(resourceRow), { stack: true })));
    const w = e.wall || {};
    const lowHigh = w.low_s && w.high_s ? `${durationWords(w.low_s).replace("about ", "")} to ${durationWords(w.high_s).replace("about ", "")}` : "";
    body.append(h("div", { class: "grid cols-main section" },
      h("section", { class: "panel" }, h("h2", { class: "cap" }, "Wall time"),
        h("div", { class: "grid cols-2" },
          readout("Wall time on this GPU", (w.words || "").replace("about ", ""), null,
            `${lowHigh}${w.label ? ` (${w.label})` : ""}`),
          kv([["Steps", run.steps ? `${num(run.steps)} of ${num(run.dt_s, run.dt_s % 1 ? 2 : 0)} s` : ""],
            ["Sound-wave steps", run.acoustic_substeps ? `${run.acoustic_substeps} per step` : ""],
            ["Cells", run.cells ? num(run.cells) : ""],
            ["Step time", e.step && e.step.seconds ? `${e.step.seconds.toFixed(4)} s (${e.step.label || ""})` : ""],
            ["Saved times", run.frames ? String(run.frames) : ""]])),
        h("p", { class: "note" }, w.basis || ""),
        e.step && e.step.basis ? h("p", { class: "note" }, `Step time: ${e.step.basis}`) : null),
      h("section", { class: "panel" }, h("h2", { class: "cap" }, "Checks"),
        (e.checks || []).map((c) => checkline(levelState(c.level), c.text)),
        (e.assumptions || []).map((a) => h("p", { class: "note" }, a)))));
  }
  const measureWrap = actionButton("Measure speed", { action: "measure", command: s.commands.measure,
    title: "Runs a few steps on your GPU (about a minute with start-up) and keeps the measured step time for the " +
      "next estimate." });
  const measureB = measureWrap.button;
  measureB.addEventListener("click", () => measure(id, measureB, box, () => go(runRoute("estimate", id))));
  const againWrap = actionButton("Estimate again", { action: "estimate", command: s.commands.estimate });
  const again = againWrap.button;
  again.addEventListener("click", async () => {
    const reply = await act(again, `${api.runPath(id)}/estimate`, {}, box);
    if (reply && reply.ok) setTimeout(() => go(runRoute("estimate", id)), 300);
  });
  const startWrap = actionButton("Start", { action: "start", kind: "primary", command: s.commands.start, always: true,
    disabled: s.has_run || s.report.exit === 3, title: s.report.exit === 3 ? "Tick the box on Check first." : null });
  const start = startWrap.button;
  start.addEventListener("click", () => startRun(id, start, box, false));
  const jobPanel = jobscriptPanel(id);
  // Opens (or closes) the job script panel below, where the script is written and copied.
  const job = button("Job script for a cluster", { onclick: () => {
    jobPanel.classList.toggle("hidden");
    if (!jobPanel.classList.contains("hidden")) jobPanel.scrollIntoView({ behavior: "smooth", block: "start" });
  } });
  const stops = ((e && e.checks) || []).filter((c) => c.level === "stop");
  const fitsNote = e && e.fits === false ? h("span", { class: "st warn" }, "does not fit now: gpuwm-cm1 run waits or refuses") :
    stops.length ? h("span", { class: "st fail" }, `${stops.length} check${stops.length > 1 ? "s" : ""} above would stop the run`) : null;
  body.append(h("div", { class: "section btns", id: "actions" }, measureWrap, startWrap, job, againWrap, fitsNote,
    s.has_run ? h("a", { href: `#/${runRoute("run", id)}` }, "This run has started: open Run") : null));
  body.append(h("div", { class: "section" }, box));
  if (s.setup && s.setup.gpu) body.append(budgetPanel(s, id, box));
  body.append(jobPanel);
  return null;
}

register("estimate", render, { title: (mode) => (mode === "guided" ? "Ready to start" : "Estimate"), runPage: true });
