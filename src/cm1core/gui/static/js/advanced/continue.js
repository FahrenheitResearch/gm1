// Continue or extend: go on from a restart of a paused, stopped or finished run, to the namelist's
// timax or to a later storm time, in a new segment. The choice of restart and end time is one gpuwm-cm1 resume
// command, shown before it is pressed; the same run can go on on a cluster with the job script (--until).

import { h, num, table, clock, parseClock, status as chip } from "../core.js";
import * as api from "../api.js";
import { register, go, notice, runRoute } from "../router.js";
import { actionButton, showCommand } from "../command.js";
import { resultBox, act, hint } from "../setup/common.js";
import { jobscriptPanel } from "./jobscript.js";
import { UNTIL_HELP } from "../run/parts.js";

function stormClock(seconds) {
  if (seconds === null || seconds === undefined) return "";
  return clock(seconds);
}

async function render(body, args, page) {
  const id = decodeURIComponent(args[0] || "");
  const d = await api.get(`${api.runPath(id)}/restarts`);
  page.setData(`${id} · ${d.label || ""}${d.model_time_s != null ? ` · reached ${stormClock(d.model_time_s)}` : ""}` +
    `${d.timax_s ? ` · timax ${stormClock(d.timax_s)}` : ""}`);
  const box = resultBox();

  if (!d.can_resume) {
    body.append(h("p", { class: "banner warn", id: "why-not" }, d.why_not || "This run cannot be continued now.",
      h("span", { class: "sub" }, d.state === "running" || d.state === "waiting" ?
        "Its Run page shows how it is going." : "")));
  }

  // The restart to go on from: the run's own line first, newest first.
  let chosen = d.restarts.find((r) => r.newest) || d.restarts.find((r) => r.on_line) || d.restarts[0] || null;
  const radios = [];
  const rows = d.restarts.map((r) => {
    const radio = h("input", { type: "radio", name: "restart", checked: r === chosen ? true : null,
      "aria-label": `restart ${r.number}`, disabled: d.can_resume ? null : true });
    radio.addEventListener("change", () => { if (radio.checked) { chosen = r; changed(); } });
    radios.push(radio);
    return h("tr", { dataset: { restart: String(r.number), segment: String(r.segment) } },
      h("td", {}, radio),
      h("td", { class: "num text" }, String(r.number)),
      h("td", { class: "num" }, stormClock(r.model_time_s)),
      h("td", { class: "num" }, r.step != null ? num(r.step) : ""),
      h("td", { class: "num" }, String(r.segment).padStart(3, "0")),
      h("td", {}, r.on_line ? chip("ok", r.newest ? "newest on this run's line" : "on this run's line") :
        chip("note", "a branch the run has left")));
  });
  const restartsPanel = h("section", { class: "panel", id: "restarts" },
    h("h2", { class: "cap" }, "Go on from", h("span", { class: "n" }, "restarts, as rstnum")),
    d.restarts.length ? table(["", { label: "Restart", num: true }, { label: "Storm time", num: true },
      { label: "Step", num: true }, { label: "Segment", num: true }, ""], rows, { compact: true }) :
      h("p", { class: "note" }, "No restart has been saved yet."),
    h("p", { class: "note" }, d.note || ""));

  // The storm time to end at.
  const options = d.until_options || [];
  const select = h("select", { class: "input", style: "width:auto", "aria-label": "end at", disabled: d.can_resume ? null : true },
    options.map((o, i) => h("option", { value: String(i), selected: o.default ? true : null },
      `${stormClock(o.until_s)} (${o.label})`)),
    h("option", { value: "custom" }, "another storm time…"));
  const minutes = h("input", { class: "input num", style: "width:110px", "aria-label": "storm time to end at",
    placeholder: "T+2:30:00", hidden: true });
  select.addEventListener("change", () => { minutes.hidden = select.value !== "custom"; changed(); });
  minutes.addEventListener("input", () => changed());
  const untilS = () => {
    if (select.value === "custom") return parseClock(minutes.value);
    const o = options[Number(select.value)];
    return o ? o.until_s : null;
  };

  const request = () => {
    if (!d.can_resume) return { error: d.why_not || "This run cannot be continued now." };
    if (!chosen) return { error: "Pick a restart to go on from." };
    const until = untilS();
    if (select.value === "custom" && !until) return { error: UNTIL_HELP };
    if (until && chosen.model_time_s != null && until <= chosen.model_time_s) {
      return { error: `End after the restart's storm time (${stormClock(chosen.model_time_s)}).` };
    }
    const body = { from_restart: chosen.number };
    if (chosen.needs_segment) body.from_segment = chosen.segment;
    if (until) body.until_s = until;
    return { path: `${api.runPath(id)}/resume`, body };
  };
  const endPanel = h("div", { class: "btns", id: "until" }, h("span", { class: "muted" }, "End at", hint("--until")), select, minutes);
  const cont = actionButton("Continue", {
    action: "resume", kind: "primary", disabled: !d.can_resume, request, watch: [endPanel, restartsPanel],
    onclick: async () => {
      const req = request();
      if (req.error) { notice(req.error, "stop"); return; }
      const reply = await act(cont.button, req.path, req.body, box, { notice: false });
      if (reply && reply.ok !== false) {
        notice(reply.message || "Started.");
        setTimeout(() => go(runRoute("run", id)), 800);
      }
    },
  });
  const jobs = jobscriptPanel(id, {
    id: "jobscript-continue", title: "Continue on a cluster",
    note: "The job script runs gpuwm-cm1 run RUNDIR --continue on the GPU node: it goes on from the newest restart " +
      "on this run's line (the choice above is for this computer), to the end time picked above.",
    until: () => (d.can_resume ? untilS() : null),
  });
  function changed() {
    if (cont.cmd) cont.cmd.refresh();
    jobs.refreshCommand();
  }
  const clusterButton = h("button", { type: "button", class: "btn", onclick: () => jobs.classList.toggle("hidden") },
    "Job script for a cluster");

  body.append(h("div", { class: "grid cols-main" }, restartsPanel,
    h("section", { class: "panel" }, h("h2", { class: "cap" }, "Continue or extend"),
      h("p", { class: "note", style: "margin-top:0" }, "A new segment, from the restart picked, to the end time picked. ",
        "The run's earlier segments are kept as they are."),
      endPanel,
      h("div", { class: "btns section" }, cont, clusterButton),
      box)));
  body.append(jobs);
  body.append(h("section", { class: "panel section" }, h("h2", { class: "cap" }, "Also"),
    h("p", { class: "note", style: "margin-top:0" },
      h("a", { href: `#/${runRoute("run", id)}` }, "Run"), " · ",
      h("a", { href: `#/guided/results/${encodeURIComponent(id)}` }, "The four favourite looks"), " · ",
      h("a", { href: `#/guided/watch/${encodeURIComponent(id)}` }, "Watching, the simple view"), " · ",
      h("a", { href: `#/${runRoute("commands", id)}` }, "Every command this run has run")),
    showCommand(d.commands.status)));
  return null;
}

register("continue", render, { title: "Continue or extend", runPage: true });
