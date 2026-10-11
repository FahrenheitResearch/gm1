// Namelist: the run's CM1 namelist, edited as text or word by word, with the six-category
// check against CM1 r21.1 drawn beside it. Every Save is one gpuwm-cm1 edit command, which imports the run again:
// the report on the right is the new one, and the difference from your original file is kept in namelist.diff.
// A run made from a preset shows its fit.json here too (what the fit changed, and why).

import { h, table, status as chip } from "../core.js";
import * as api from "../api.js";
import { register, notice, runRoute } from "../router.js";
import { actionButton, showCommand } from "../command.js";
import { runId, loadSetup, dataLine, resultBox, act, notImported } from "../setup/common.js";

const CHIPS = {
  refused: ["refused", "Refused"], not_read: ["warn", "Not read"], differs: ["warn", "Runs differently"],
  not_checked: ["note", "Not yet checked"], not_needed: ["note", "Not needed"], same: ["ok", "Same as CM1"],
  default: ["", "CM1's own value"],
};
const OPEN = new Set(["refused", "not_read", "differs", "not_checked"]);

// ---- where a key is in the text

function lineOf(text, key) {
  const re = new RegExp(`^[ \\t]*${key.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}[ \\t]*(\\(|=)`, "im");
  const m = re.exec(text);
  if (!m) return null;
  return text.slice(0, m.index).split("\n").length;
}

function selectLine(area, line) {
  const lines = area.value.split("\n");
  let start = 0;
  for (let i = 0; i < line - 1 && i < lines.length; i += 1) start += lines[i].length + 1;
  const end = start + (lines[line - 1] || "").length;
  area.focus();
  area.setSelectionRange(start, end);
  const lh = parseFloat(getComputedStyle(area).lineHeight) || 16;
  area.scrollTop = Math.max(0, (line - 4) * lh);
}

// ---- the report, drawn beside the editor

function reportPanel(ctx) {
  const holder = h("section", { class: "panel nlreport", id: "report" });
  holder.draw = (report) => {
    holder.replaceChildren(h("h2", { class: "cap" }, "Checked against CM1 r21.1", h("span", { class: "n" }, "six categories")));
    if (!report || !report.sections) {
      holder.append(h("p", { class: "note" }, "No coverage report yet."));
      return;
    }
    const exit = report.exit;
    const cls = exit === 2 ? "banner stop" : exit === 3 ? "banner warn" : "banner";
    holder.append(h("div", { class: cls, id: "nl-verdict" }, h("span", { class: "text" }, report.verdict_text || ""),
      h("span", { class: "muted" }, `   exit ${exit}`)));
    const counts = h("div", { class: "nlcounts" });
    for (const sec of report.sections) {
      const [cls2, label] = CHIPS[sec.id] || ["note", sec.label];
      counts.append(h("span", { class: "nlcount", dataset: { cat: sec.id } }, chip(cls2, sec.label || label),
        h("b", {}, String(sec.count ?? (sec.rows || []).length))));
    }
    holder.append(counts);
    for (const sec of report.sections) {
      const rows = sec.rows || [];
      const n = sec.count ?? rows.length;
      const det = h("details", { class: "cat", open: OPEN.has(sec.id) && n ? true : null, dataset: { cat: sec.id } },
        h("summary", {}, h("span", {}, sec.label), h("span", { class: "count" }, `(${n})`)));
      const inner = h("div", { class: "body" });
      if (!rows.length && !(sec.families || []).length) inner.append(h("p", { class: "note" }, "none"));
      for (const r of rows) {
        const keys = r.keys || [r.key];
        // A link to each word's line in the text; a row of many words (stat_*) links its first few.
        const found = keys.map((k) => [k, lineOf(ctx.text(), k)]).filter(([, line]) => line);
        const jumps = found.slice(0, 4);
        inner.append(h("div", { class: "nlrow", dataset: { key: r.key } },
          h("div", {}, h("span", { class: "key" }, r.key), h("span", { class: "mono muted" }, ` = ${r.value ?? ""}`),
            jumps.length ? h("span", { class: "jumps" }, jumps.map(([k, line]) => {
              const b = h("button", { type: "button", class: "linkbtn", title: `Show ${k} in the text` }, `line ${line}`);
              b.addEventListener("click", () => ctx.jump(line));
              return b;
            }), found.length > jumps.length ? h("span", { class: "muted" }, ` and ${found.length - jumps.length} more`) : null) : null),
          h("div", { class: "note" }, r.text || "", r.after ? " (after fixing the above)" : ""),
          r.fix ? h("div", { class: "fix" }, r.fix) : null));
      }
      if ((sec.families || []).length) {
        inner.append(table(["Family", { label: "Settings", num: true }], sec.families.map((f) =>
          h("tr", {}, h("td", {}, f.family, h("span", { class: "sub mono" }, (f.keys || []).slice(0, 10).join(", ") +
            ((f.keys || []).length > 10 ? ", …" : ""))), h("td", { class: "num" }, String(f.n)))), { compact: true }));
      }
      det.append(inner);
      holder.append(det);
    }
    if (exit === 3) {
      const needed = [...new Set(report.sections.flatMap((s) => (s.rows || []).map((r) => r.acknowledge)).filter(Boolean))];
      const given = new Set(((report.acknowledgments || {}).given) || []);
      if (needed.length && !needed.every((a) => given.has(a))) {
        const body = { output_keys: needed.includes("output-keys"), thompson: needed.includes("thompson") };
        const ack = actionButton(needed.includes("thompson") && needed.length === 1 ? "Run Thompson as checked so far" :
          "Run without the settings GM1 does not read", {
          action: "acknowledge", small: true, disabled: !ctx.editable,
          request: () => ({ path: `${api.runPath(ctx.id)}/acknowledge`, body }),
          onclick: async () => {
            const reply = await act(ack.button, `${api.runPath(ctx.id)}/acknowledge`, body, ctx.box);
            if (reply) ctx.after(reply);
          },
        });
        holder.append(h("div", { class: "btns section" }, ack));
      }
    }
    holder.append(h("p", { class: "note" }, "The Morrison and WSM6 rows' ratios on CM1's own storm, the filters and the ",
      "report file are on ", h("a", { href: `#/${runRoute("check", ctx.id)}` }, "Check"), "."));
  };
  return holder;
}

// ---- every setting, word by word

function wordsTab(ctx) {
  const holder = h("div", { id: "words" });
  let data = null;
  const changes = new Map();
  const search = h("input", { class: "input", placeholder: "find a word, e.g. dx or ptype", "aria-label": "find a word",
    spellcheck: "false" });
  const filter = h("select", { class: "input", style: "width:auto", "aria-label": "show" },
    [["all", "Every word"], ["in_file", "In your file"], ["not_in_file", "Not in your file (CM1's value)"],
      ["changed", "Different from CM1's value"], ["refused", "Refused"], ["not_read", "Not read"],
      ["differs", "Runs differently"], ["not_checked", "Not yet checked"], ["not_needed", "Not needed"],
      ["same", "Same as CM1"]].map(([v, l]) => h("option", { value: v }, l)));
  const count = h("span", { class: "note" });
  const list = h("div", { class: "nlwords" });
  const box = resultBox();
  const request = () => {
    if (!changes.size) return null;
    return { path: `${api.runPath(ctx.id)}/edit`, body: { set: Object.fromEntries(changes) } };
  };
  const save = actionButton("Save the changed words and check again", {
    action: "edit", kind: "primary", small: true, disabled: true, request, empty: "Change a value first; the command shows here.",
    onclick: async () => {
      const req = request();
      if (!req) { notice("Nothing changed."); return; }
      const reply = await act(save.button, req.path, req.body, box);
      if (reply && reply.command) {
        // The words are in the file whenever the edit ran, a refused value included (the report says why).
        if (reply.verdict && reply.verdict !== "error") changes.clear();
        await ctx.after(reply);
        await load();
      }
    },
  });
  const logical = (v) => ({ ".true.": "t", ".t.": "t", t: "t", ".false.": "f", ".f.": "f", f: "f" }[String(v).toLowerCase()]);
  const differs = (w) => {
    if (!w.in_file || w.default === null || w.value === null) return false;
    const a = Number(w.value), b = Number(w.default);
    if (Number.isFinite(a) && Number.isFinite(b)) return a !== b;
    return (logical(w.value) || String(w.value).toLowerCase()) !== (logical(w.default) || String(w.default).toLowerCase());
  };
  const draw = () => {
    list.replaceChildren();
    if (!data) return;
    const q = search.value.trim().toLowerCase();
    const f = filter.value;
    let shown = 0;
    for (const g of data.groups) {
      const rows = [];
      for (const w of g.words) {
        if (q && !w.key.includes(q) && !(w.text || "").toLowerCase().includes(q)) continue;
        if (f === "in_file" && !w.in_file) continue;
        if (f === "not_in_file" && w.in_file) continue;
        if (f === "changed" && !differs(w)) continue;
        if (!["all", "in_file", "not_in_file", "changed"].includes(f) && w.category !== f) continue;
        shown += 1;
        const [cls, label] = CHIPS[w.category] || ["note", w.category_label || ""];
        const pending = changes.get(w.key);
        let value;
        if (data.editable && w.in_r21 && !w.key.includes("(")) {
          value = h("input", { class: `input mono${pending !== undefined ? " changed" : ""}`, value: pending ?? w.value ?? "",
            placeholder: w.default ?? "", "aria-label": w.key, spellcheck: "false" });
          value.addEventListener("input", () => {
            const v = value.value.trim();
            if (v === "" || v === (w.value ?? "")) changes.delete(w.key); else changes.set(w.key, v);
            value.classList.toggle("changed", changes.has(w.key));
            save.button.disabled = !changes.size;
            if (save.cmd) save.cmd.refresh();
          });
        } else {
          value = h("span", { class: "mono" }, w.value ?? "");
        }
        rows.push(h("tr", { dataset: { key: w.key, cat: w.category || "" } },
          h("td", { class: "key" }, w.key, w.in_file ? null : h("span", { class: "sub" }, "not in your file"),
            w.in_r21 ? null : h("span", { class: "sub warn" }, "not a CM1 r21.1 word")),
          h("td", {}, value),
          h("td", { class: "mono muted", title: w.source ? `CM1 r21.1 ${w.source}` : null }, w.default ?? "",
            w.basis === "static-zero" ? h("span", { class: "sub" }, "not set in CM1's code; reads as 0") : null),
          h("td", {}, label ? chip(cls, w.category_label || label) : null,
            w.text ? h("span", { class: "sub" }, w.text) : null, w.fix ? h("span", { class: "sub fix" }, w.fix) : null)));
      }
      if (!rows.length) continue;
      list.append(h("details", { class: "nlgroup", open: q || f !== "all" || g.in_file ? true : null, dataset: { group: g.group } },
        h("summary", {}, h("span", { class: "mono" }, `&${g.group}`), h("span", { class: "count" }, `${g.stated} in your file, ${g.total} words`)),
        table(["Word", "Your value", "CM1 uses when left out", "What GM1 does"], rows, { compact: true, cls: "nltable" })));
    }
    count.textContent = `${shown} of ${data.total} words`;
  };
  async function load() {
    data = await api.get(`${api.runPath(ctx.id)}/words`);
    save.button.disabled = !changes.size || !data.editable;
    draw();
  }
  search.addEventListener("input", draw);
  filter.addEventListener("change", draw);
  holder.append(
    h("div", { class: "btns" }, search, filter, count),
    h("p", { class: "note" }, "Every CM1 r21.1 namelist word. A word not in your file takes CM1 r21.1's own value, as ",
      "CM1 does; a value typed here is written into the run's namelist.input with gpuwm-cm1 edit --set."),
    list, h("div", { class: "btns section" }, save), box);
  holder.load = load;
  return holder;
}

// ---- the screen

async function render(body, args, page) {
  const id = runId(args);
  const s = await loadSetup(id);
  // A run made from a preset keeps the preset's namelist as its original; the words say so.
  const yours = s.fit ? "the preset's namelist" : "your file";
  page.setData(dataLine(s, s.report.namelist && s.report.namelist.edited ? `edited from ${yours}` : null));
  if (!s.report.present) { notImported(body, s); return null; }
  const [nml, full] = await Promise.all([api.get(`${api.runPath(id)}/namelist`),
    api.get(`${api.runPath(id)}/report`).catch(() => null)]);
  const editable = Boolean(nml.editable && s.editable);
  const box = resultBox();
  const area = h("textarea", { class: "input mono nltext", rows: "30", spellcheck: "false", "aria-label": "namelist text",
    readonly: editable ? null : true }, nml.text);
  const lines = h("span", { class: "note" });
  const countLines = () => { lines.textContent = `${area.value.split("\n").length} lines`; };
  countLines();

  const tabs = h("div", { class: "tabs", role: "tablist" });
  const panes = {};
  const showTab = (name) => {
    for (const [n, el] of Object.entries(panes)) el.hidden = n !== name;
    tabs.querySelectorAll("button").forEach((b) => b.classList.toggle("on", b.dataset.tab === name));
    if (name === "words" && !panes.words.dataset.loaded) { panes.words.dataset.loaded = "1"; panes.words.load(); }
  };
  const addTab = (name, label) => {
    tabs.append(h("button", { type: "button", dataset: { tab: name }, onclick: () => showTab(name) }, label));
  };

  const ctx = {
    id, editable, box, text: () => area.value,
    jump: (line) => { showTab("text"); selectLine(area, line); },
    after: async (reply) => {
      if (reply.report) report.draw(reply.report);
      if (reply.stored_diff !== undefined) drawDiff(reply.stored_diff);
      try {
        const again = await api.get(`${api.runPath(id)}/namelist`);
        if (!again.editable) area.readOnly = true;
        if (ctx.namelistChanged) ctx.namelistChanged(again);
        // Text typed and not yet saved stays as typed (Save and revert clear the mark before they get here).
        if (area.value !== again.text && !area.dataset.dirty) area.value = again.text;
        countLines();
      } catch (_) { /* the text stays as typed */ }
    },
  };
  const report = reportPanel(ctx);
  report.draw(full ? full.report : null);

  // Text
  area.addEventListener("input", () => { area.dataset.dirty = "1"; countLines(); });
  const saveRequest = () => {
    if (!area.value.trim()) return { error: "The namelist is empty." };
    return { path: `${api.runPath(id)}/namelist`, body: { text: area.value } };
  };
  const save = actionButton("Save and check again", {
    action: "namelist", kind: "primary", small: true, disabled: !editable, request: saveRequest, watch: area,
    onclick: async () => {
      const req = saveRequest();
      if (req.error) { notice(req.error, "stop"); return; }
      const reply = await act(save.button, req.path, req.body, box);
      if (reply && reply.command) { area.dataset.dirty = ""; await ctx.after(reply); }
    },
  });
  const revertBody = { revert: true };
  const revert = actionButton("Put my original file back", {
    action: "namelist", small: true, disabled: !editable || !nml.edited,
    request: () => ({ path: `${api.runPath(id)}/namelist`, body: revertBody }),
    onclick: async () => {
      const reply = await act(revert.button, `${api.runPath(id)}/namelist`, revertBody, box);
      if (reply && reply.command) { area.dataset.dirty = ""; await ctx.after(reply); }
    },
  });
  // After each Save: "Put my original file back" is offered while the run's namelist differs from the user's file.
  ctx.namelistChanged = (again) => {
    revert.button.disabled = !(editable && again.editable && again.edited);
    save.button.disabled = !(editable && again.editable);
  };
  panes.text = h("div", { dataset: { pane: "text" } }, area,
    h("div", { class: "btns", style: "margin-top:8px" }, save, revert,
      h("a", { class: "btn small", href: `${api.runPath(id)}/files/namelist.input`, download: "namelist.input" },
        "Download namelist.input"),
      lines),
    h("p", { class: "note" }, editable ? "Saving imports the run again from this text, exactly as gpuwm-cm1 import does. " +
      "Your original file stays as namelist.input.original, and the difference is kept in namelist.diff." :
      "This run has already run, so its namelist is fixed. Import it into a new run to change it."));

  // Every setting
  panes.words = wordsTab(ctx);

  // Difference from your file
  const diffBody = h("div", {});
  function drawDiff(text) {
    diffBody.replaceChildren(text ? h("pre", { class: "diff", id: "nl-diff" }, text) :
      h("p", { class: "note" }, `The run's namelist is ${yours}, unchanged.`));
  }
  drawDiff(nml.diff);
  panes.diff = h("div", { dataset: { pane: "diff" } }, diffBody,
    nml.original !== null && nml.original !== undefined ? h("details", { class: "section" },
      h("summary", {}, s.fit ? "The preset's namelist (namelist.input.original)" : "Your original file (namelist.input.original)"),
      h("pre", { class: "log" }, nml.original)) : null,
    h("p", { class: "note" }, "namelist.diff in the run folder holds this difference; gpuwm-cm1 edit rewrites it on every Save."));

  addTab("text", "Text");
  addTab("words", "Every setting");
  addTab("diff", s.fit ? "Difference from the preset" : "Difference from your file");

  // A run made from a preset: what the fit chose, and why (fit.json).
  if (s.fit) {
    const f = s.fit;
    const sentences = f.sentences || [];
    panes.fit = h("div", { dataset: { pane: "fit" }, id: "fit" },
      h("p", { class: "text" }, f.summary || ""),
      sentences.length ? h("ul", { class: "plainlist" }, sentences.map((x) => h("li", {}, x))) :
        h("p", { class: "note" }, "The fit changed nothing: the preset runs as it is."),
      h("div", { class: "kvline" }, h("span", { class: "muted" }, "Preset "), h("span", { class: "text" }, f.preset_name || f.preset || ""),
        h("span", { class: "muted" }, "   time "), h("span", { class: "text" }, f.budget_words || f.budget || "")),
      h("details", { class: "section" }, h("summary", {}, "fit.json"), h("pre", { class: "log" }, JSON.stringify(f, null, 1))),
      h("div", { class: "btns" }, h("a", { class: "btn small", href: `${api.runPath(id)}/files/fit.json`, download: "fit.json" },
        "Download fit.json")));
    addTab("fit", "Fit");
  }

  body.append(h("div", { class: "grid cols-main" },
    h("section", { class: "panel" }, tabs, ...Object.values(panes), box),
    report));
  showTab(args[1] === "words" || args[1] === "diff" || (args[1] === "fit" && s.fit) ? args[1] : "text");
  const reportCommand = showCommand(s.report.command);
  if (reportCommand) body.append(reportCommand);
  if (!editable) {
    body.prepend(h("p", { class: "banner warn" }, "This run has already run, so its namelist is fixed.",
      h("span", { class: "sub" }, "Import it into a new run to change it: New run, From a CM1 namelist, with this path: ",
        h("span", { class: "mono" }, `${s.path}/namelist.input`))));
  }
  return null;
}

register("namelist", render, { title: "Namelist", runPage: true });

export { lineOf, CHIPS };
