// Screen 3, Check: the coverage report for the run's CM1 namelist. Advanced: CM1's words, the six
// categories with a filter row, refused rows with their fix, the Morrison or WSM6 row opening the stock-namelist
// ratio table, the output line, "Download report" and "Edit namelist", which opens the namelist editor (the
// Namelist page: Save re-imports, the report is redrawn beside the text, and the diff is kept).
// Guided: the verdict and each difference in plain words, CM1's names as grey hints.

import { h, num, status, button, table, kv } from "../core.js";
import * as api from "../api.js";
import { register, go, notice, runRoute } from "../router.js";
import { showCommand, liveCommand } from "../command.js";
import { runId, loadSetup, hint, dataLine, resultBox, act, toAdvanced, notImported } from "./common.js";

const OPEN = new Set(["refused", "not_read", "differs", "not_checked"]);

function ratioBar(r) {
  const lo = 0.7, hi = 1.3;
  const pos = (v) => (100 * (Math.min(hi, Math.max(lo, v)) - lo)) / (hi - lo);
  const inside = r.inside_band !== false;
  return h("div", { class: "ratio-bar", title: `band ±${Math.round(r.band * 100)}%` },
    h("span", { class: "band", style: `left:${pos(1 - r.band).toFixed(1)}%;right:${(100 - pos(1 + r.band)).toFixed(1)}%` }),
    h("span", { class: "one", style: `left:${pos(1).toFixed(1)}%` }),
    h("span", { class: `val${inside ? "" : " out"}`, style: `left:${pos(r.ratio).toFixed(1)}%` }));
}

function sig(x) {
  if (x === null || x === undefined) return "";
  const a = Math.abs(x);
  if (a >= 100) return num(x, 1);
  if (a >= 1) return num(x, 2);
  if (a === 0) return "0";
  return x.toPrecision(3).replace("-", "−");
}

function comparison(c) {
  const rows = c.rows.map((r) => h("tr", {},
    h("td", {}, r.label, h("span", { class: "sub" }, (c.families || {})[r.family] || r.family)),
    h("td", { class: "num" }, `${sig(r.cm1)} ${r.unit}`),
    h("td", { class: "num" }, `${sig(r.gm1)} ${r.unit}`),
    h("td", { class: "num text" }, r.ratio.toFixed(3)),
    h("td", {}, ratioBar(r)),
    h("td", { class: "num" }, `±${Math.round(r.band * 100)}%`),
    h("td", {}, r.inside_band ? status("ok", "inside") : status("failed", "outside")),
    h("td", { class: "num muted" }, r.twin != null ? r.twin.toFixed(3) : ""),
    h("td", { class: "num muted" }, r.ramp0 != null ? r.ramp0.toFixed(2) : "")));
  const loc = c.location || {};
  const worst = (loc.frames || []).reduce((m, f) => Math.max(m, f.cells || 0), 0);
  const onset = c.onset || {};
  return h("div", { class: "section" },
    h("h3", { class: "cap" }, c.title),
    table(["Statistic", { label: "CM1", num: true }, { label: "GM1", num: true }, { label: "GM1 / CM1", num: true },
      "0.7 – 1.3", { label: "Band", num: true }, "", { label: "CM1 twin / CM1", num: true },
      { label: "CM1 ramp_sgs 0 / CM1", num: true }], rows, { compact: true }),
    h("div", { class: "section" }, kv([
      ["Where the storm is", `${loc.rule || ""}${loc.frames ? `: worst ${worst} cell${worst === 1 ? "" : "s"}` : ""}`],
      ["Onset", onset.rule ? `${onset.rule}: CM1 ${num(onset.cm1, 1)} m/s, GM1 ${num(onset.gm1, 1)} m/s (${(onset.ratio || 0).toFixed(2)})` : ""],
      ["Target", c.target || ""],
      ["Basis", c.basis || ""],
      ["Namelist", c.deck || ""],
      ["Evidence", c.evidence || ""],
    ])));
}

function sectionEl(sec, report, guided, onAck, ackRequest) {
  const rows = sec.rows || [];
  const n = sec.count ?? rows.length;
  const after = rows.filter((r) => r.after).length;
  const countText = after ? `(${n - after}, and ${after} after fixing the above)` : `(${n})`;
  const det = h("details", {
    class: `cat${sec.id === "refused" && n ? " has-refused" : ""}`,
    open: OPEN.has(sec.id) && !sec.collapsed && n ? true : null,
    dataset: { cat: sec.id },
  }, h("summary", {}, h("span", {}, sec.label), h("span", { class: "count" }, countText)));
  const inner = h("div", { class: "body" });
  if (!rows.length && !(sec.families || []).length) inner.append(h("p", { class: "note" }, "none"));
  if (rows.length) {
    const hasFix = rows.some((r) => r.fix || r.plain_fix);
    const trs = [];
    for (const r of rows) {
      if (guided) {
        trs.push(h("tr", { dataset: { key: r.key } },
          h("td", {}, h("span", { class: "text" }, r.plain || r.text), hint(`${r.key} = ${r.value}`),
            r.after ? h("span", { class: "sub" }, "after fixing the above") : null),
          hasFix ? h("td", { class: "fix" }, r.plain_fix || r.fix || "") : null));
      } else {
        trs.push(h("tr", { dataset: { key: r.key } },
          h("td", { class: "key" }, r.key, r.defaulted ? h("span", { class: "sub" }, "not in your file") : null),
          h("td", { class: "key" }, r.value),
          h("td", {}, r.text, r.after ? h("span", { class: "sub" }, "after fixing the above") : null,
            r.evidence ? h("span", { class: "sub mono" }, r.evidence) : null),
          hasFix ? h("td", { class: "fix" }, r.fix || "") : null));
      }
      if (r.expand === "ratios" && report.comparison) {
        trs.push(h("tr", {}, h("td", { colspan: guided ? (hasFix ? 2 : 1) : (hasFix ? 4 : 3) },
          h("details", { class: "inner", open: guided ? null : true },
            h("summary", {}, guided ? "How close it is to CM1 on CM1's own storm" : "Ratios on CM1's supercell namelist"),
            h("div", { class: "body" }, comparison(report.comparison))))));
      }
    }
    const heads = guided ? ["What GM1 does", hasFix ? "What to change" : null] :
      ["CM1 key", "Your value", sec.id === "refused" ? "Why" : "What GM1 does", hasFix ? "What to change" : null];
    inner.append(table(heads.filter(Boolean), trs, { stack: true }));
  }
  if ((sec.families || []).length) {
    const heads = ["Family", { label: "Settings", num: true }, guided ? null : "Evidence"].filter(Boolean);
    const familyRows = sec.families.map((f) => {
      const keys = f.keys || [];
      const shown = keys.slice(0, 12).join(", ") + (keys.length > 12 ? ", …" : "");
      return h("tr", {},
        h("td", {}, f.family, guided ? null : h("span", { class: "sub mono" }, shown)),
        h("td", { class: "num" }, String(f.n)),
        guided ? null : h("td", { class: "mono muted" }, f.evidence));
    });
    inner.append(h("div", { class: rows.length ? "section" : "" }, table(heads, familyRows, { compact: true })));
  }
  if (sec.ack && n) {
    const given = new Set(((report.acknowledgments || {}).given) || []);
    const needed = new Set(rows.map((r) => r.acknowledge).filter(Boolean));
    const done = [...needed].every((a) => given.has(a)) && needed.size > 0;
    const box = h("input", { type: "checkbox", checked: done ? true : null, disabled: done ? true : null,
      dataset: { action: "acknowledge" } });
    box.addEventListener("change", () => { if (box.checked) onAck([...needed], box); });
    inner.append(h("label", { class: "check", style: "margin-top:10px" }, box, sec.ack,
      done ? h("span", { class: "sub" }, "done: stored in run.json") : null),
      done ? null : liveCommand(() => ackRequest([...needed])));
  }
  det.append(inner);
  return det;
}

function verdictBanner(report, guided) {
  const exit = report.exit;
  const cls = exit === 2 ? "banner stop" : exit === 3 ? "banner warn" : "banner";
  if (guided) {
    const words = exit === 2 ? "GM1 cannot run this yet." : exit === 3 ? "GM1 runs this once you say OK to the list below." :
      (report.guided && report.guided.offered) ? "GM1 runs this as CM1 does." : "GM1 runs this, with the differences below.";
    return h("div", { class: cls, id: "verdict" }, h("span", { class: "text" }, words),
      h("span", { class: "sub" }, report.verdict_text || ""));
  }
  return h("div", { class: cls, id: "verdict" }, h("span", { class: "text" }, report.verdict_text || report.verdict || ""),
    h("span", { class: "muted" }, `   exit ${exit}`));
}

async function render(body, args, page) {
  const id = runId(args);
  const guided = page.mode === "guided";
  const [s, full] = await Promise.all([loadSetup(id), api.get(`${api.runPath(id)}/report`).catch(() => null)]);
  page.setData(dataLine(s, s.report.not_r21 && s.report.not_r21 !== "none" ? `not in CM1 r21.1: ${s.report.not_r21}` : null));
  if (!s.report.present || !full) { notImported(body, s); return null; }
  const report = full.report;
  const box = resultBox();
  const redraw = () => go(runRoute("check", id));

  const onAck = async (needed, input) => {
    const reply = await act(null, `${api.runPath(id)}/acknowledge`,
      { output_keys: needed.includes("output-keys"), thompson: needed.includes("thompson") }, box);
    if (reply && reply.ok) setTimeout(redraw, 400);
    else if (input) input.checked = false;
  };

  body.append(verdictBanner(report, guided));
  // What the run will do that the user should know before it starts (a push centred outside the domain).
  for (const w of report.warnings || []) {
    body.append(h("p", { class: "banner warn", dataset: { warning: w.code || "" } }, w.text));
  }
  if (report.aliases && report.aliases.length && !guided) {
    body.append(h("p", { class: "note" }, report.aliases.map((a) => a.text || `${a.old} read as ${a.new}`).join(" · ")));
  }
  if (report.error) body.append(h("pre", { class: "log" }, report.error));

  const sections = report.sections || [];
  if (!guided) {
    const filters = h("div", { class: "tabs", style: "margin-top:14px", role: "tablist" });
    const ids = [["all", "All"], ...sections.map((sec) => [sec.id, sec.label])];
    for (const [fid, label] of ids) {
      filters.append(h("button", { type: "button", class: fid === "all" ? "on" : null, dataset: { filter: fid },
        onclick: (ev) => {
          filters.querySelectorAll("button").forEach((b) => b.classList.toggle("on", b === ev.currentTarget));
          body.querySelectorAll("details.cat").forEach((d) => {
            d.classList.toggle("hidden", fid !== "all" && d.dataset.cat !== fid);
            if (fid !== "all" && d.dataset.cat === fid) d.open = true;
          });
        } }, label));
    }
    body.append(filters);
  } else {
    body.append(h("p", { class: "lead section" }, "Every setting in the namelist, checked against what CM1 r21.1 does ",
      "with it. Most are the same; the lists below say what is not."));
  }
  const ackRequest = (needed) => ({ path: `${api.runPath(id)}/acknowledge`,
    body: { output_keys: needed.includes("output-keys"), thompson: needed.includes("thompson") } });
  body.append(h("div", { id: "sections" }, sections.map((sec) => sectionEl(sec, report, guided, onAck, ackRequest))));

  const out = report.output || {};
  body.append(h("section", { class: "section panel", id: "output-line" },
    h("h2", { class: "cap" }, guided ? "Files it writes" : "Output"),
    h("div", { class: "text" }, guided ? `CM1-format files: ${out.format || ""}, as the namelist asks.` : (out.text || "")),
    guided ? hint(`output_format = ${out.output_format}, output_filetype = ${out.output_filetype}`) : null,
    out.also_netcdf === false ? (((s.settings_toml || {}).output || {}).also_netcdf === true ?
      h("p", { class: "note" }, "Also write netCDF is on: the cm1out writer makes netCDF files beside the GrADS ones (",
        h("a", { href: `#/${runRoute("settings", id)}` }, "Settings"), ").") :
      h("p", { class: "note" }, "To also get netCDF files beside the GrADS ones, tick ",
        h("a", { href: `#/${runRoute("settings", id)}` }, "Also write netCDF"), " on Settings.")) : null));

  const refused = (s.report.counts || {}).refused || 0;
  const needsOk = s.report.exit === 3;
  const cont = button(guided ? "Continue" : "Continue", {
    kind: "primary",
    disabled: refused > 0 || needsOk,
    title: refused ? "Change the refused settings first." : needsOk ? "Tick the box above first." : null,
    onclick: () => go(runRoute(guided ? "estimate" : "sounding", id)),
  });
  const reportText = s.files.report_text ? h("a", { href: s.files.report_text, download: "import-report.txt",
    class: "btn small" }, "Download report") : null;
  const edit = button(s.editable ? "Edit namelist" : "Show namelist", { small: true,
    onclick: () => go(runRoute("namelist", id)) });
  body.append(h("div", { class: "section btns" }, cont,
    guided ? null : h("span", { class: "advanced-only btns" }, reportText, edit),
    guided ? toAdvanced("Show every setting") : null,
    h("span", { class: "note" }, refused ? "Change the refused settings, in your file or with Edit namelist, then import it again." : "")));
  if (!guided && s.fit) {
    body.append(h("p", { class: "note section", id: "fit-line" }, `Made from the preset ${s.fit.preset_name || s.fit.preset || ""}, `,
      `fitted for ${(s.fit.budget_words || s.fit.budget || "").toLowerCase()} on this GPU. `,
      h("a", { href: `#/${runRoute("namelist", id)}/fit` }, "What the fit changed"), " (fit.json)."));
  }
  body.append(showCommand(s.report.command), box);
  if (s.diff && !guided) {
    body.append(h("details", { class: "section" }, h("summary", {}, "This run's namelist differs from your original file"),
      h("pre", { class: "diff" }, s.diff)));
  }
  return null;
}

register("check", render, { title: "Check", runPage: true });
