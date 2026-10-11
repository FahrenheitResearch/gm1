// Screen 6, Settings: the values CM1 keeps in Fortran (the warm bubble, cold pool, line thermal and parcel seed box;
// init3d.F and parcel.F), the output format and every &param9 word with what GM1 does with it (words GM1 cannot
// write are greyed with the reason), and the intervals and seed. Each Save is one gpuwm-cm1 edit command.
// Guided shows the same values in plain sentences.

import { h, num, numText, table, status, button } from "../core.js";
import * as api from "../api.js";
import { register, go, notice, runRoute } from "../router.js";
import { showCommand, actionButton } from "../command.js";
import { runId, loadSetup, hint, dataLine, resultBox, act, notImported, lead, toAdvanced, stormTime } from "./common.js";

const STATE = {
  always: ["ok", "always written"], written: ["ok", "written"], off: ["note", "off"], refused: ["refused", "not available"],
  differs: ["warn", "runs differently"],
};

function groupPanel(g, s, id, box) {
  const editable = g.active && s.editable;
  const inputs = [];
  const rows = g.fields.map((f) => {
    let input;
    if (f.kind === "bool") {
      input = h("input", { type: "checkbox", checked: f.value ? true : null, disabled: editable ? null : true,
        "aria-label": f.label });
    } else {
      input = h("input", { class: "input num", value: numText(f.value), disabled: editable ? null : true,
        style: "width:14ch;min-width:120px", "aria-label": f.label, inputmode: "decimal" });
    }
    inputs.push([f, input]);
    const differs = f.changed && f.cm1_value !== null && f.cm1_value !== undefined;
    return h("div", { class: "field" },
      h("label", {}, f.label, hint(f.cm1)),
      h("div", {}, input, f.unit && f.unit !== "parcels" ? h("span", { class: "unit" }, f.unit) : null,
        f.unit === "parcels" ? h("span", { class: "unit" }, "parcels") : null),
      differs ? h("div", { class: "hint" }, `CM1's own value: ${f.kind === "bool" ? (f.cm1_value ? "on" : "off") :
        trim(f.cm1_value)}${f.unit && f.unit !== "parcels" ? ` ${f.unit}` : ""}`) : null);
  });
  // The request Save makes: the changed values, or null (nothing changed), or {error}.
  const collect = () => {
    const settings = {};
    for (const [f, input] of inputs) {
      let value;
      if (f.kind === "bool") value = input.checked;
      else {
        const raw = input.value.trim();
        if (raw === "") continue;
        // The box shows the stored value rounded for reading; left as shown, the stored value stands.
        if (raw === numText(f.value)) continue;
        const n = Number(raw);
        if (!Number.isFinite(n)) return { error: `${f.label}: give a number.` };
        value = f.kind === "int" ? Math.round(n) : n;
      }
      const same = f.kind === "bool" ? value === Boolean(f.value) : Math.abs(value - Number(f.value)) < 1e-9;
      if (same) continue;
      settings[f.key] = f.kind === "float" ? Number((value / (f.factor || 1)).toPrecision(12)) : value;
    }
    // CM1's ric and rjc are set together.
    const cx = "initialization.bubble_center_x_m", cy = "initialization.bubble_center_y_m";
    if ((cx in settings) !== (cy in settings)) {
      const other = cx in settings ? cy : cx;
      const f = g.fields.find((row) => row.key === other);
      if (f) settings[other] = Number((Number(f.value) / (f.factor || 1)).toPrecision(12));
    }
    if (!Object.keys(settings).length) return null;
    return { path: `${api.runPath(id)}/edit`, body: { settings } };
  };
  const fieldsBox = h("div", {}, rows);
  const saveWrap = actionButton("Save", { action: "edit", small: true, kind: "primary", disabled: !editable,
    request: collect, watch: fieldsBox });
  const save = saveWrap.button;
  save.addEventListener("click", async () => {
    const req = collect();
    if (!req) { notice("Nothing changed."); return; }
    if (req.error) { notice(req.error, "stop"); return; }
    const reply = await act(save, req.path, req.body, box);
    if (reply && reply.command) setTimeout(() => go(runRoute("settings", id)), reply.ok ? 600 : 1500);
  });
  const changed = g.fields.filter((f) => f.changed).map((f) => f.key);
  const resetBody = { reset: changed };
  const resetWrap = actionButton("Back to CM1's values", { action: "edit", small: true,
    disabled: !editable || !changed.length,
    request: () => (changed.length ? { path: `${api.runPath(id)}/edit`, body: resetBody } : null),
    empty: "Every value here is CM1's own already." });
  const reset = resetWrap.button;
  reset.addEventListener("click", async () => {
    const reply = await act(reset, `${api.runPath(id)}/edit`, resetBody, box);
    if (reply && reply.command) setTimeout(() => go(runRoute("settings", id)), 600);
  });
  return h("section", { class: `panel${g.active ? "" : " off"}`, dataset: { group: g.id } },
    h("h2", { class: "cap" }, g.title, h("span", { class: "n" }, `${g.selector}${g.active ? "" : ": not used by this namelist"}`)),
    fieldsBox,
    g.active ? h("div", { class: "btns", style: "margin-top:8px" }, saveWrap, resetWrap) : null);
}

function wordsTable(words, editable, picks) {
  return table(["&param9 word", { label: "Namelist", num: true }, "GM1"], words.map((w) => {
    const [cls, label] = STATE[w.state] || ["note", w.state];
    let value = h("span", { class: "mono" }, String(w.value));
    if (w.editable && editable) {
      const box = h("input", { type: "checkbox", checked: w.value ? true : null, "aria-label": w.word });
      picks.push([w, box]);
      value = box;
    }
    return h("tr", { class: w.state === "refused" ? "off" : null, dataset: { word: w.word } },
      h("td", { class: "key" }, w.word, w.fields && w.fields.length ? h("span", { class: "sub fields", title: w.fields.join(", ") },
        w.fields.slice(0, 4).join(", ") + (w.fields.length > 4 ? `, … (${w.fields.length})` : "")) : null),
      h("td", { class: "num" }, value),
      h("td", {}, status(cls, w.label || label), w.why && w.state !== "written" ?
        h("span", { class: "sub", title: w.why }, w.why.length > 140 ? `${w.why.slice(0, 137).replace(/\s+\S*$/, "")} …` : w.why) : null));
  }), { compact: true, cls: "words" });
}

function trim(x) {
  if (x === null || x === undefined) return "";
  return numText(x);
}

function outputPanel(s, id, box) {
  const v = s.values || {};
  const editable = s.editable;
  const words = (s.setup.output_words || []);
  const picks = [];
  const half = Math.ceil(words.length / 2);
  const also = ((s.settings_toml || {}).output || {}).also_netcdf === true;
  const fmt = v.output_format;
  const grads = h("input", { type: "radio", name: "fmt", value: "1", checked: fmt === 1 ? true : null, disabled: editable ? null : true });
  const nc = h("input", { type: "radio", name: "fmt", value: "2", checked: fmt === 2 ? true : null, disabled: editable ? null : true });
  const alsoBox = h("input", { type: "checkbox", checked: also ? true : null, disabled: editable && fmt === 1 ? null : true });
  const filetype = h("select", { class: "input", style: "width:auto", disabled: editable ? null : true, "aria-label": "output_filetype" },
    // CM1's third layout (one file per time and MPI rank) is listed when the namelist has it, so a Save of
    // anything else on this panel never changes it silently.
    [[1, "every time in one file (output_filetype = 1)"], [2, "one file per time (output_filetype = 2)"],
      ...(v.output_filetype === 3 ? [[3, "one file per time; one GPU writes what CM1's MPI ranks would split (output_filetype = 3)"]] : [])].map(([val, label]) =>
      h("option", { value: String(val), selected: v.output_filetype === val ? true : null }, label)));
  const collect = () => {
    const set = {};
    const chosenFmt = nc.checked ? 2 : 1;
    if (chosenFmt !== fmt) set.output_format = chosenFmt;
    const ft = Number(filetype.value);
    if (ft !== v.output_filetype) set.output_filetype = ft;
    for (const [w, input] of picks) {
      const want = input.checked ? 1 : 0;
      if (want !== w.value) set[w.word] = want;
    }
    const body = { set };
    if (chosenFmt === 1 && alsoBox.checked !== also) {
      if (alsoBox.checked) body.settings = { "output.also_netcdf": true };
      else body.reset = ["output.also_netcdf"];
    }
    if (!Object.keys(set).length && !body.settings && !body.reset) return null;
    return { path: `${api.runPath(id)}/edit`, body };
  };
  const saveWrap = actionButton("Save output", { action: "edit", small: true, kind: "primary", disabled: !editable,
    request: collect });
  const save = saveWrap.button;
  save.addEventListener("click", async () => {
    const req = collect();
    if (!req) { notice("Nothing changed."); return; }
    const reply = await act(save, req.path, req.body, box);
    if (reply && reply.command) setTimeout(() => go(runRoute("settings", id)), reply.ok ? 600 : 1500);
  });
  const refused = words.filter((w) => w.state === "refused").length;
  const section = h("section", { class: "panel section", id: "output" },
    h("h2", { class: "cap" }, "Output"),
    h("div", { class: "btns", style: "margin-bottom:10px" },
      h("label", { class: "check" }, grads, "GrADS", hint("output_format = 1")),
      h("label", { class: "check" }, nc, "netCDF", hint("output_format = 2")),
      h("label", { class: "check" }, alsoBox, "Also write netCDF"),
      filetype),
    h("p", { class: "note", style: "margin-top:0" }, `${words.length} words; GM1 cannot write ${refused} of them. `,
      "A word GM1 cannot write is listed on Check and not saved; the run goes ahead. With Also write netCDF, ",
      "the cm1out writer makes netCDF files beside the GrADS ones in the same pass."),
    h("div", { class: "grid cols-2" }, wordsTable(words.slice(0, half), editable, picks),
      wordsTable(words.slice(half), editable, picks)),
    h("div", { class: "btns", style: "margin-top:10px" }, saveWrap));
  section.addEventListener("change", () => { if (saveWrap.cmd) saveWrap.cmd.refresh(); });
  return section;
}

function intervalsPanel(s, id, box) {
  const v = s.values || {};
  const editable = s.editable;
  const tap = h("input", { class: "input num", value: numText(v.tapfrq), style: "width:120px", disabled: editable ? null : true, "aria-label": "tapfrq" });
  const rst = h("input", { class: "input num", value: numText(v.rstfrq), style: "width:120px", disabled: editable ? null : true, "aria-label": "rstfrq" });
  const seedOn = v.irandp === 1;
  const seed = h("input", { class: "input num", value: s.setup.random_seed ?? "", style: "width:160px",
    disabled: editable && seedOn ? null : true, "aria-label": "random seed" });
  const collect = () => {
    const set = {};
    const t = Number(tap.value), r = Number(rst.value);
    if (tap.value.trim() && !Number.isFinite(t)) return { error: "tapfrq: give a number of seconds." };
    if (rst.value.trim() && !Number.isFinite(r)) return { error: "rstfrq: give a number of seconds." };
    if (tap.value.trim() && tap.value.trim() !== numText(v.tapfrq) && t !== v.tapfrq) set.tapfrq = t;
    if (rst.value.trim() && rst.value.trim() !== numText(v.rstfrq) && r !== v.rstfrq) set.rstfrq = r;
    const body = { set };
    if (seedOn && seed.value.trim() && Number(seed.value) !== s.setup.random_seed) body.seed = Math.round(Number(seed.value));
    if (!Object.keys(set).length && body.seed === undefined) return null;
    return { path: `${api.runPath(id)}/edit`, body };
  };
  const saveWrap = actionButton("Save", { action: "edit", small: true, kind: "primary", disabled: !editable,
    request: collect, watch: [tap, rst, seed] });
  const save = saveWrap.button;
  save.addEventListener("click", async () => {
    const req = collect();
    if (!req) { notice("Nothing changed."); return; }
    if (req.error) { notice(req.error, "stop"); return; }
    const reply = await act(save, req.path, req.body, box);
    if (reply && reply.command) setTimeout(() => go(runRoute("settings", id)), reply.ok ? 600 : 1500);
  });
  const row = (label, cm1, input, unit, note) => h("tr", {}, h("td", { class: "muted" }, label, hint(cm1)),
    h("td", {}, input, unit ? h("span", { class: "unit" }, unit) : null), h("td", { class: "muted" }, note || ""));
  return h("section", { class: "panel section", id: "intervals" }, h("h2", { class: "cap" }, "Intervals and seed"),
    table(["", "Value", ""], [
      row("Files and pictures every", "tapfrq", tap, "s", ""),
      row("Restarts every", "rstfrq", rst, "s", v.rstfrq < 0 ? "negative: CM1 writes none; GM1 writes one at the end" : ""),
      h("tr", {}, h("td", { class: "muted" }, "Statistics every", hint("statfrq")), h("td", { class: "mono text" }, `${v.statfrq ?? ""} s`),
        h("td", { class: "muted" }, "GM1 works these out from the saved frames after the run, not during it")),
      row("Random seed", "irandp = 1; CM1 has no seed word", seed, "", seedOn ? (s.setup.random_generator === "cm1" ?
        "CM1's own random numbers, the same as CM1's; a seed here switches to GM1's stream, a different realization" :
        "GM1's own random stream: the details differ from CM1's as with another seed") :
        "used only with irandp = 1"),
    ], { compact: true }),
    h("div", { class: "btns", style: "margin-top:8px" }, saveWrap));
}

function guidedView(s) {
  const active = (s.setup.fortran || []).filter((g) => g.active && g.id !== "parcels");
  const v = s.values || {};
  const sentences = [];
  for (const g of active) {
    const f = Object.fromEntries(g.fields.map((row) => [row.key.split(".")[1], row.value]));
    if (g.id === "bubble") {
      sentences.push(h("li", {}, `The storm starts from a warm bubble ${num(f.bptpert_k, 1)} K warmer than the air around it, `,
        `${num(2 * f.bhrad_m, 0)} km across and centred ${num(f.zc_m, 1)} km up.`, hint("iinit = 1: bptpert, bhrad, zc")));
    } else if (g.id === "coldpool") {
      sentences.push(h("li", {}, `The storm starts from a block of air ${num(Math.abs(f.bptpert_k), 1)} K colder than its `,
        `surroundings, ${num(f.zdep_m, 1)} km deep.`, hint("iinit = 2")));
    } else if (g.id === "line") {
      sentences.push(h("li", {}, `The storms start from a line of air ${num(f.bptpert_k, 1)} K warmer than its surroundings, `,
        `with random noise of ${num(f.random_line_amplitude_k, 2)} K along it.`, hint("iinit = 8")));
    } else if (g.id === "nudging") {
      sentences.push(h("li", {}, `The storm is started by a gentle push of rising air, as CM1's updraft nudging does: `,
        `up to ${num(f.wmax_wnudge, 0)} m/s, ${num(2 * f.xr_wnudge, 0)} km across, centred ${num(f.zc_wnudge, 1)} km up, `,
        `fading out by minute ${num(f.t2_wnudge / 60, 0)}.`, hint("iinit = 12: wmax_wnudge, xr_wnudge, zc_wnudge, t2_wnudge")));
    } else if (g.id === "convergence") {
      sentences.push(h("li", {}, `The storm is started by air pushed together near the ground, as CM1's forced convergence does, `,
        `for the first ${num(f.convtime / 60, 0)} minutes.`, hint("iinit = 9: dmax, zdeep, convtime")));
    } else if (g.id === "benchmark") {
      sentences.push(h("li", {}, `The run starts from Bryan and Fritsch's moist benchmark bubble: saturated air as light `,
        `as a dry bubble ${num(f.bptpert_k, 1)} K warmer, ${num(2 * f.bhrad_m, 0)} km across, centred ${num(f.zc_m, 1)} km up, `,
        `the same all along y. GM1 shows this start only; its run from it does not match CM1's yet.`,
        hint("iinit = 4: bptpert, bhrad, zc; iconly = 1")));
    } else if (g.id === "benchmark_sounding") {
      sentences.push(h("li", {}, `The air is saturated and neutrally stable: wet equivalent potential temperature `,
        `${num(f.thec_mb_k, 1)} K, total water ${num(1000 * f.qt_mb, 1)} g/kg, with cloud water everywhere. `,
        `GM1 shows this sounding only; its run from it does not match CM1's yet.`,
        hint("isnd = 4: thec_mb, qt_mb; iconly = 1")));
    } else if (g.id === "uforce") {
      sentences.push(h("li", {}, `The storms are started by a push of wind along a line, as CM1's momentum forcing does, `,
        `fading out by minute ${num(f.t2_uforce / 60, 0)}.`, hint("iinit = 10: alpha_uforce, t2_uforce")));
    }
  }
  if (!active.length) sentences.push(h("li", {}, "The storm starts the way the namelist's iinit says.", hint(`iinit = ${v.iinit}`)));
  // GM1 runs only the soundings whose runs matched CM1's (isnd = 5, 7, 9, 17); CM1's other built-in profiles are
  // shown only (config.py CM1-CFG-BASE-CLOUD-INTEGRATION, CM1-CFG-ANALYTIC-SOUNDING-INTEGRATION). isnd = 4 says so
  // in its own group above.
  const isnd = Number(v.isnd);
  if (Number.isInteger(isnd) && ![4, 5, 7, 9, 17].includes(isnd) && isnd > 0 && isnd <= 23) {
    const why = [2, 3].includes(isnd) ? "it is dry, and GM1 runs moist air only" :
      [10, 11].includes(isnd) ? "its run from it does not match CM1's yet" :
      "its run from it has not been checked against CM1's yet";
    sentences.push(h("li", {}, `The air is CM1's built-in profile number ${isnd}. GM1 shows how it starts only: ${why}.`,
      hint(`isnd = ${isnd}; iconly = 1`)));
  }
  sentences.push(h("li", {}, `Files and pictures every ${stormTime(v.tapfrq)} of storm time.`, hint(`tapfrq = ${v.tapfrq}`)));
  sentences.push(h("li", {}, `CM1-format files in ${v.output_format === 2 ? "netCDF" : "GrADS"}.`,
    hint(`output_format = ${v.output_format}`)));
  return h("section", { class: "panel" }, h("h2", { class: "cap" }, "What starts the storm, and what is saved"),
    h("ul", { class: "plainlist" }, sentences),
    h("p", { class: "note" }, "These are CM1's own values for this setup. ", toAdvanced("Change them in Advanced"), "."));
}

async function render(body, args, page) {
  const id = runId(args);
  const guided = page.mode === "guided";
  const s = await loadSetup(id);
  page.setData(dataLine(s, "values CM1 keeps in Fortran · output · intervals"));
  if (!s.report.present) { notImported(body, s); return null; }
  const box = resultBox();
  if (guided) {
    body.append(guidedView(s));
    body.append(h("div", { class: "section btns" },
      button("Continue", { kind: "primary", onclick: () => go(runRoute("estimate", id)) })));
    return null;
  }
  if (!s.editable) body.append(h("p", { class: "banner warn" }, "This run has already run, so its settings are fixed. ",
    h("span", { class: "sub" }, "Import the namelist into a new run to change them.")));
  body.append(h("h2", { class: "cap" }, "Initial perturbation and parcels (CM1 sets these in init3d.F and parcel.F)"));
  body.append(h("div", { class: "grid cols-2" }, (s.setup.fortran || []).map((g) => groupPanel(g, s, id, box))));
  body.append(h("p", { class: "note" }, s.setup.note || ""));
  body.append(box);
  body.append(outputPanel(s, id, box));
  body.append(intervalsPanel(s, id, box));
  body.append(showCommand(`gpuwm-cm1 edit ${s.path} [--set KEY=VALUE] [--setting TABLE.KEY=VALUE] [--reset-setting TABLE.KEY] --json`));
  body.append(h("div", { class: "section btns" },
    button("Continue", { kind: "primary", onclick: () => go(runRoute("estimate", id)) })));
  return null;
}

register("settings", render, { title: "Settings", runPage: true });
