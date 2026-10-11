// Commands: every gpuwm-cm1 command the GUI has run on this run, as the server wrote them to
// RUNDIR/commands.log. The file is a shell script: bash RUNDIR/commands.log does the same from a terminal.

import { h, button, copyline, copyButton, table } from "../core.js";
import * as api from "../api.js";
import { register } from "../router.js";

async function render(body, args, page) {
  const id = decodeURIComponent(args[0] || "");
  const d = await api.get(`${api.runPath(id)}/commands`);
  page.setData(`${id} · ${d.count} ${d.count === 1 ? "command" : "commands"} · ${d.path}`);
  if (!d.exists) {
    body.append(h("p", { class: "banner" }, "Nothing has been run on this run from the GUI yet.",
      h("span", { class: "sub" }, "A run started from a terminal or a job script keeps its own record.")));
    return null;
  }
  const rows = d.entries.map((e) => h("tr", { dataset: { action: e.action || "" } },
    h("td", { class: "nowrap muted" }, e.time_utc ? e.time_utc.replace("T", " ").replace("Z", " UTC") : ""),
    h("td", { class: "nowrap text" }, e.action || ""),
    h("td", {}, e.lines.map((line) => h("code", { class: "cmdline" }, line)),
      e.note ? h("span", { class: "sub" }, e.note) : null)));
  const copyAll = copyButton("Copy the whole script", d.text);
  body.append(
    h("p", { class: "lead" }, d.note),
    h("section", { class: "panel" }, h("h2", { class: "cap" }, "Replay from a terminal"),
      copyline(d.replay, "run every command again"),
      h("div", { class: "btns" }, copyAll,
        d.url ? h("a", { class: "btn small", href: d.url, download: "commands.log" }, "Download commands.log") : null),
      d.truncated ? h("p", { class: "note" }, "The file is longer than this page shows; the download has all of it.") : null),
    h("section", { class: "panel section", id: "commands-log" }, h("h2", { class: "cap" }, "commands.log"),
      table(["Time", "Action", "Command"], rows, { compact: true, stack: true })));
  return null;
}

register("commands", render, { title: "Commands", runPage: true });
