// Screen 10, System: is this computer ready to run GM1, in plain words, with one fix line per failure.
// The same check runs from a terminal as gpuwm-cm1 doctor --plain; it never starts a run and never holds the GPU.

import { h, button, table, copyline, status as statusChip } from "./core.js";
import * as api from "./api.js";
import { register, notice } from "./router.js";
import { showCommand } from "./command.js";

const WORD = { ok: "ok", note: "note", fail: "fix" };
// A fix that is one command is drawn as a copy line; a fix in words, as words.
const COMMAND = /^(pip|uv|cargo|gpuwm-cm1|sudo|apt|conda) /;

async function render(body, args, page) {
  const holder = h("div", {});
  const again = button("Check again", { small: true });
  body.append(holder);
  async function draw(refresh) {
    again.disabled = true;
    let d;
    try {
      d = await api.get(`/api/system${refresh ? "?refresh=1" : ""}`);
    } catch (err) {
      holder.replaceChildren(h("p", { class: "banner stop" }, err.message, h("span", { class: "sub" }, err.fix || "")));
      again.disabled = false;
      return;
    }
    again.disabled = false;
    const r = d.report;
    page.setData([d.command, "this computer", r ? `exit ${r.exit}` : `exit ${d.rc}`].join(" · "));
    if (!r) {
      holder.replaceChildren(h("p", { class: "banner stop" }, d.message || "The check did not finish.",
        h("span", { class: "sub" }, d.fix || "")), showCommand(d.command, { always: true }) || "");
      return;
    }
    const fails = r.rows.filter((row) => row.state === "fail");
    const rows = r.rows.map((row) => h("tr", { dataset: { row: row.id } },
      h("td", { class: "text nowrap" }, row.area),
      h("td", {}, statusChip(row.state === "fail" ? "fail" : row.state === "note" ? "note" : "ok", WORD[row.state] || row.state)),
      h("td", {}, row.text),
      h("td", { class: "fix" }, row.fix ? (COMMAND.test(row.fix) ? copyline(row.fix, "fix") :
        h("span", { class: row.state === "fail" ? "text" : "muted" }, row.fix)) : "")));
    holder.replaceChildren(
      h("div", { class: fails.length ? "banner stop" : "banner" }, r.summary),
      h("section", { class: "panel section" }, table(["", "State", "What the check found", "Fix"], rows, { stack: true })),
      h("div", { class: "section btns" }, again,
        d.cached_s ? h("span", { class: "note" }, `checked ${Math.round(d.cached_s)} s ago`) : null),
      h("p", { class: "note" }, "The same check from a terminal (exit 0 when ready). It never starts a run and never ",
        "holds the GPU."), copyline(d.command, "the check"),
      r.wsl2 ? h("p", { class: "note" }, "Windows with WSL2: open the link gpuwm-cm1 gui prints in a Windows browser; ",
        "localhost reaches it.") : "");
  }
  again.addEventListener("click", () => draw(true).then(() => notice("Checked again.")));
  await draw(false);
  return null;
}

register("system", render, { label: "System", nav: true, title: "System", order: 90 });
