// New run, "Start from a preset": a preset fitted to this GPU and a time, made into a run folder that
// opens in the namelist editor; nothing runs until Start. It is guided mode's gallery and budget choice as one
// form (the same gpuwm-cm1 preset fit and preset new commands), so no preset is only reachable in guided mode.

import { h, num, append } from "../core.js";
import * as api from "../api.js";
import { go, notice, runRoute } from "../router.js";
import { actionButton } from "../command.js";
import { resultBox, act, hint } from "../setup/common.js";

const NAME_RE = /^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/;

function stamp() {
  const d = new Date();
  const two = (n) => String(n).padStart(2, "0");
  return `${d.getFullYear()}${two(d.getMonth() + 1)}${two(d.getDate())}-${two(d.getHours())}${two(d.getMinutes())}`;
}

export async function presetPanel() {
  const box = resultBox();
  const panel = h("section", { class: "panel", id: "from-preset" }, h("h2", { class: "cap" }, "Start from a preset"));
  let list;
  try {
    list = await api.get("/api/presets");
  } catch (err) {
    panel.append(h("p", { class: "banner stop" }, err.message, h("span", { class: "sub" }, err.fix || "")));
    return panel;
  }
  const presets = list.presets || [];
  const budgets = list.budgets || [];
  const pick = h("select", { class: "input", "aria-label": "preset" },
    presets.map((p) => h("option", { value: p.id }, `${p.name}${p.offered ? "" : " (not offered yet)"}`)));
  const budget = h("select", { class: "input", "aria-label": "time" },
    budgets.map((b) => h("option", { value: b.id }, b.words)));
  const name = h("input", { class: "input", "aria-label": "run name", spellcheck: "false" });
  const about = h("p", { class: "note" });
  const fitOut = h("div", { class: "fitout" });
  const current = () => presets.find((p) => p.id === pick.value) || null;
  const syncName = () => { if (!name.dataset.typed) name.value = `${pick.value}-${stamp()}`; };
  const describe = () => {
    const p = current();
    if (!p) { about.replaceChildren(); return; }
    const f = p.facts || {};
    about.replaceChildren();
    append(about, [h("span", { class: "text" }, p.line || ""), " ",
      f.words ? h("span", { class: "sub" }, `As the preset is: ${f.words}; ${num((f.cells || 0) / 1e6, 2)} M cells` +
        `${f.scheme ? `, ${f.scheme}` : ""}.`) : null,
      p.offered ? null : h("span", { class: "sub warn" }, p.why_not || "Not offered yet.")]);
    if (!budget.dataset.touched && p.budget) budget.value = p.budget;
  };
  name.addEventListener("input", () => { name.dataset.typed = "1"; });
  budget.addEventListener("change", () => { budget.dataset.touched = "1"; });
  pick.addEventListener("change", () => { describe(); syncName(); fitOut.replaceChildren(); });
  describe();
  syncName();

  const form = h("div", { class: "formgrid" },
    h("label", {}, "Preset"), pick,
    h("label", {}, "Time on this GPU", hint("--budget")), budget,
    h("label", {}, "Run name"), name);

  const fitRequest = () => ({ path: "/api/fit", body: { preset: pick.value, budget: budget.value } });
  const newRequest = () => {
    const runName = name.value.trim();
    if (!NAME_RE.test(runName)) return { error: "Use letters, digits, '.', '_' or '-' for the name, up to 64." };
    return { path: "/api/presets/new", body: { preset: pick.value, budget: budget.value, name: runName } };
  };
  const fitButton = actionButton("Fit it to this GPU", {
    action: "fit", small: true, request: fitRequest, watch: form,
    onclick: async () => {
      const reply = await act(fitButton.button, "/api/fit", fitRequest().body, null, { notice: false });
      fitOut.replaceChildren();
      if (!reply) return;
      const fit = reply.fit || {};
      append(fitOut, [h("p", { class: reply.offered ? "text" : "banner warn" }, reply.message || fit.summary || ""),
        (fit.sentences || []).length ? h("ul", { class: "plainlist" }, fit.sentences.map((s) => h("li", {}, s))) : null]);
    },
  });
  const make = actionButton("Make the run folder and open its namelist", {
    action: "preset-new", kind: "primary", request: newRequest, watch: form,
    onclick: async () => {
      const req = newRequest();
      if (req.error) { notice(req.error, "stop"); return; }
      const reply = await act(make.button, req.path, req.body, box);
      if (reply && reply.ok && reply.run) setTimeout(() => go(runRoute("namelist", reply.run)), 400);
    },
  });
  panel.append(
    h("p", { class: "note", style: "margin-top:0" }, "A preset is a CM1 r21.1 namelist GM1 runs. It is ",
      "fitted to this GPU's free memory and the time picked (a smaller area, then a coarser grid, then a shorter ",
      "storm time), and every change is written to fit.json. The run folder opens in the namelist editor."),
    form, about,
    h("div", { class: "btns", style: "margin-top:10px" }, fitButton, make),
    fitOut, box,
    h("p", { class: "note" }, "Or pick from ", h("a", { href: "#/guided" }, "the preset gallery"),
      ", the guided view of the same presets."));
  return panel;
}
