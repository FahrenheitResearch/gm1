// Screen 2, New run. Advanced: a CM1 namelist (a path on the machine running GM1, or an upload) with its sounding;
// a preset fitted to this GPU and a time, opened in the namelist editor; one column of a WRF or ArWen run as
// the sounding (../wrf/tab.js); or one of CM1's own examples with
// its size, smallest card and wall time on this GPU. Guided: "Pick a storm", a gallery of the examples in plain
// words, each with what it makes and about how long it takes here; one button sets it up.

import { h, button, table, num, schemeName } from "../core.js";
import * as api from "../api.js";
import { register, go, notice, runRoute, homeRoute } from "../router.js";
import { actionButton } from "../command.js";
import { gb, durationWords, stormTime, spacing, gridWords, hint, resultBox, toAdvanced } from "./common.js";
import { presetPanel } from "../advanced/presets.js";
import { wrfPanel } from "../wrf/tab.js";

const NAME_RE = /^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/;

function field(label, input, hintText, cm1) {
  return h("div", { class: "field" }, h("label", {}, label, cm1 ? hint(cm1) : null), input,
    hintText ? h("div", { class: "hint" }, hintText) : null);
}

async function freeName(base) {
  let names = new Set();
  try {
    const d = await api.get("/api/runs");
    names = new Set(d.runs.map((r) => r.id));
  } catch (_) { /* the import itself refuses a name in use */ }
  const clean = (base || "run").replace(/[^A-Za-z0-9._-]/g, "-").replace(/^[^A-Za-z0-9]+/, "").slice(0, 56) || "run";
  if (!names.has(clean)) return clean;
  for (let i = 2; i < 1000; i += 1) if (!names.has(`${clean}-${i}`)) return `${clean}-${i}`;
  return `${clean}-${Date.now()}`;
}

// A file chosen in the browser, read as text (it goes to the server in the request; nothing else is read).
function fileText(input) {
  return new Promise((resolve, reject) => {
    const file = input.files && input.files[0];
    if (!file) { resolve(null); return; }
    const reader = new FileReader();
    reader.onload = () => resolve({ name: file.name, text: String(reader.result) });
    reader.onerror = () => reject(new Error(`Could not read ${file.name}.`));
    reader.readAsText(file);
  });
}

function pathOrUpload(labelText, placeholder, cm1, hintText) {
  const path = h("input", { class: "input mono", placeholder, "aria-label": `${labelText} path`, spellcheck: "false" });
  const upload = h("input", { type: "file", class: "hidden", "aria-label": `${labelText} file` });
  const chosen = h("span", { class: "note" });
  const pick = button("Upload…", { small: true, onclick: () => upload.click() });
  upload.addEventListener("change", () => {
    const f = upload.files && upload.files[0];
    chosen.textContent = f ? `${f.name} (from this computer)` : "";
    if (f) path.value = "";
  });
  path.addEventListener("input", () => { if (path.value) { upload.value = ""; chosen.textContent = ""; } });
  const row = h("div", { class: "pathrow" }, path, pick, upload);
  const el = field(labelText, h("div", {}, row, chosen), hintText, cm1);
  el.read = async () => {
    const f = await fileText(upload);
    if (f) return { text: f.text };
    return path.value.trim() ? { path: path.value.trim() } : null;
  };
  el.pathSpec = () => (path.value.trim() ? { path: path.value.trim() } : null);
  return el;
}

async function importExample(example, name, box, guided) {
  const body = { name, namelist: { path: example.path }, example: true };
  if (example.sounding) body.sounding = { path: example.sounding };
  const reply = await api.post("/api/import", body);
  if (box) box.show(reply, { always: guided });
  if (!reply.ok) {
    notice(`${reply.message} ${reply.fix || ""}`.trim(), "stop");
    return null;
  }
  return reply;
}

// ---- advanced

function namelistPanel(root) {
  const box = resultBox();
  const nml = pathOrUpload("Namelist", "~/cm1/run/namelist.input", "namelist.input",
    "A path on the computer running GM1, or a file from this one.");
  const snd = pathOrUpload("Sounding", "only for isnd = 7 or 17", "input_sounding",
    "The WRF-format file SounderPy's to_file('cm1') writes. Left empty, GM1 looks beside the namelist.");
  const name = h("input", { class: "input", placeholder: "my-run", "aria-label": "run name", spellcheck: "false" });
  const folder = h("input", { class: "input mono", placeholder: "(none)", "aria-label": "folder", spellcheck: "false" });
  // An upload is read when it is chosen, so "Show command" can ask for the exact line; a path is sent as typed.
  let texts = { namelist: null, sounding: null };
  const request = () => {
    const namelist = texts.namelist || nml.pathSpec();
    if (!namelist) return { error: "Give the namelist's path, or upload it; the command shows here." };
    const runName = name.value.trim();
    if (!runName) return { error: "Give the run a name; the command shows here." };
    if (!NAME_RE.test(runName)) return { error: "Use letters, digits, '.', '_' or '-' for the name." };
    const bodyJson = { name: runName, namelist };
    if (folder.value.trim()) bodyJson.folder = folder.value.trim();
    const sounding = texts.sounding || snd.pathSpec();
    if (sounding) bodyJson.sounding = sounding;
    return { path: "/api/import", body: bodyJson };
  };
  const importButton = actionButton("Import and check", { action: "import", kind: "primary", request });
  const go1 = importButton.button;
  go1.addEventListener("click", async () => {
    const namelist = await nml.read().catch((err) => { notice(err.message, "stop"); return undefined; });
    if (namelist === undefined) return;
    if (!namelist) { notice("Give the namelist's path, or upload it.", "stop"); return; }
    const runName = name.value.trim() || await freeName("run");
    if (!NAME_RE.test(runName)) {
      notice("Use letters, digits, '.', '_' or '-' for the name (up to 64, starting with a letter or digit).", "stop");
      return;
    }
    const sounding = await snd.read().catch(() => null);
    const bodyJson = { name: runName, namelist };
    if (folder.value.trim()) bodyJson.folder = folder.value.trim();
    if (sounding) bodyJson.sounding = sounding;
    go1.disabled = true;
    try {
      const reply = await api.post("/api/import", bodyJson);
      box.show(reply);
      if (reply.mode_notice) notice(reply.mode_notice);
      if (reply.url) {
        notice(reply.message + (reply.fix ? ` ${reply.fix}` : ""), reply.ok ? "" : "warn");
        setTimeout(() => go(`check/${reply.url}`), 300);
      } else {
        notice(`${reply.message} ${reply.fix || ""}`.trim(), "stop");
      }
    } catch (err) {
      box.show({ ok: false, message: err.message, fix: err.fix, command: err.command });
      notice(`${err.message} ${err.fix || ""}`.trim(), "stop");
    } finally { go1.disabled = false; }
  });
  const panel = h("section", { class: "panel", id: "from-namelist" },
    h("h2", { class: "cap" }, "From a CM1 namelist"),
    nml, snd,
    field("Run name", name, `Letters, digits, '.', '_' and '-'. The run's folder is ${root}/NAME.`),
    field("Folder", folder, "Optional: a folder under the runs folder to keep it in."),
    h("div", { class: "btns", style: "margin-top:12px" }, importButton,
      h("span", { class: "note" }, "Reads the namelist against CM1 r21.1's 583 settings. Nothing runs yet.")),
    box);
  const reread = async () => {
    const upload = async (el) => el.read().then((v) => (v && v.text !== undefined ? v : null)).catch(() => null);
    texts = { namelist: await upload(nml), sounding: await upload(snd) };
    if (importButton.cmd) importButton.cmd.refresh();
  };
  panel.addEventListener("change", reread);
  panel.addEventListener("input", () => { if (importButton.cmd) importButton.cmd.refresh(); });
  return panel;
}

// The examples are priced in the background: /api/examples answers at once, each example saying whether
// its numbers are here ("priced"), kept from an earlier pricing on this GPU ("kept"), being checked again
// ("checking") or on their way ("pricing"), and the screen asks again while "pricing" is true.
// Asked again quickly at first (kept prices are confirmed within a moment of the GPU answering), then every second.
const PRICE_POLL_MS = [250, 250, 500, 1000];
const PRICE_POLL_LIMIT = 300;

function priceWords(e) {
  const state = (e.price || {}).state;
  if (state === "pricing") return "Working it out on this GPU…";
  if (state === "kept") return "From the last check on this GPU; checking again…";
  if (state === "checking") return "Checking again…";
  return null;
}

// Ask for the examples until the background pricing is done, handing each answer to show(); stop when the
// screen is left (el is no longer in the page) or stop() says so.
async function watchExamples(el, show, stop) {
  for (let i = 0; i < PRICE_POLL_LIMIT; i += 1) {
    let d;
    try {
      d = await api.get("/api/examples");
    } catch (err) {
      show(null, err);
      return;
    }
    if (i > 0 && !el.isConnected) return;
    if (stop && stop()) return;
    show(d, null);
    if (!d.pricing) return;
    await new Promise((resolve) => setTimeout(resolve, PRICE_POLL_MS[Math.min(i, PRICE_POLL_MS.length - 1)]));
  }
}

function exampleRows(d, box, onUse) {
  const examples = d.examples.filter((e) => e.kind !== "preset");
  return examples.map((e) => {
    const est = e.estimate || {};
    const fits = est.fits !== false;
    // An example guided mode does not offer is not offered here either, with the same reason.
    const useWrap = actionButton("Use", { action: "import", small: true, disabled: !e.path || Boolean(e.not_offered),
      command: e.commands && e.commands.import });
    const use = useWrap.button;
    use.addEventListener("click", async () => {
      use.disabled = true;
      onUse();
      try {
        const name = await freeName(e.suggested_name);
        const reply = await importExample(e, name, box, false);
        if (reply && reply.url) go(`check/${reply.url}`);
      } catch (err) {
        box.show({ ok: false, message: err.message, fix: err.fix, command: err.command });
        notice(`${err.message} ${err.fix || ""}`.trim(), "stop");
      } finally { use.disabled = false; }
    });
    const wall = est.wall || {};
    const waiting = priceWords(e);
    return h("tr", { dataset: { example: e.id, price: (e.price || {}).state || "" } },
      h("td", {}, h("span", { class: "text" }, e.name), h("span", { class: "sub" }, e.source),
        e.not_offered ? h("span", { class: "sub" }, h("span", { class: "st warn" }, "Not offered"), ` ${e.not_offered}`) : null),
      h("td", { class: "nowrap" }, e.grid ? gridWords(e.grid) : "",
        h("span", { class: "sub" }, e.grid ? `dx ${spacing(e.grid.dx)}, ${stormTime(e.duration_s)}` : "")),
      h("td", {}, schemeName(e.microphysics)),
      h("td", { class: "num" }, est.gpu_need_bytes ? gb(est.gpu_need_bytes) : "",
        e.cells ? h("span", { class: "sub" }, `${num(e.cells / 1e6, 2)} M cells`) : null),
      h("td", { class: "num" }, est.smallest_card_gb ? `${est.smallest_card_gb} GB` : ""),
      h("td", {},
        !e.estimate && waiting ? h("span", { class: "note" }, waiting) :
          e.error && !e.estimate ? h("span", { class: "st warn" }, "Could not price it") :
            fits ? h("span", { class: "text" }, wall.words || "") : h("span", { class: "st warn" }, "Does not fit now"),
        wall.label ? h("span", { class: "sub", title: wall.basis || "" },
          `${wall.label}, from ${wall.basis_label === "measured" ? "measured" : wall.basis_label || "recorded"} step times`) : null,
        e.estimate && waiting ? h("span", { class: "sub" }, waiting) : null,
        e.guided && e.guided.why_not && !e.not_offered ? h("span", { class: "sub" }, e.guided.why_not) : null),
      h("td", { class: "num" }, useWrap));
  });
}

// "From one of CM1's examples": drawn at once, its rows filled as the prices arrive.
function examplesSection() {
  const box = resultBox();
  let used = false;
  const body = h("div", {}, h("p", { class: "note" }, "Pricing the examples on this computer's GPU…"));
  const note = h("p", { class: "note" });
  const section = h("section", { class: "panel section", id: "from-example" },
    h("h2", { class: "cap" }, "From one of CM1's examples"), body, note, box);
  // CM1's own examples, as written. The presets are made with Start from a preset (fitted, with fit.json), so a
  // preset's GM1 settings (a bubble moved for a smaller area) are never lost by importing its namelist alone.
  watchExamples(section, (d, err) => {
    if (err) {
      body.replaceChildren(h("p", { class: "banner stop" }, err.message, h("span", { class: "sub" }, err.fix || "")));
      return;
    }
    body.replaceChildren(table(["Example", "Grid", "Microphysics", { label: "GPU memory", num: true },
      { label: "Smallest card", num: true }, "Wall time on this GPU", ""], exampleRows(d, box, () => { used = true; }),
    { stack: true }));
    note.textContent = d.note || "";
  }, () => used);
  return section;
}

// A panel drawn at once with a line in it, replaced by the real one when it is ready.
function later(id, title, waiting, make) {
  const slot = h("section", { class: "panel", id }, h("h2", { class: "cap" }, title), h("p", { class: "note" }, waiting));
  Promise.resolve().then(make).then((panel) => { if (panel) slot.replaceWith(panel); }).catch((err) => {
    slot.append(h("p", { class: "banner stop" }, err.message || String(err)));
  });
  return slot;
}

// ---- guided

function card(e, onPick) {
  const offered = e.guided && e.guided.offered;
  const est = e.estimate || {};
  const pic = e.picture ?
    h("figure", {}, h("img", { src: e.picture, alt: `${e.plain_name}: ${e.picture_caption || "a picture of this storm"}`,
      loading: "lazy", width: "640", height: "360" }), e.picture_caption ? h("figcaption", {}, e.picture_caption) : null) :
    h("div", { class: "nopic" }, "The picture comes from this example's first run.");
  const waiting = priceWords(e);
  const when = !e.estimate && waiting ? waiting :
    offered ? `${(est.wall && est.wall.words) ? est.wall.words.charAt(0).toUpperCase() + est.wall.words.slice(1) : ""} on your GPU (estimated).` : "";
  const pick = button("Set up this storm", { kind: "primary", disabled: !offered });
  pick.addEventListener("click", () => onPick(e, pick));
  return h("article", { class: `card${offered ? "" : " off"}`, dataset: { example: e.id } },
    pic,
    h("h3", {}, e.plain_name),
    h("p", {}, e.line),
    offered || (!e.estimate && waiting) ? h("p", { class: "when" }, when) :
      h("p", { class: "why" }, (e.guided && e.guided.why_not) || "Not offered on this computer."),
    h("div", { class: "btns" }, pick));
}

async function render(body, args, page) {
  const guided = page.mode === "guided";
  if (guided && homeRoute("guided") === "guided") {
    go("guided");
    return null;
  }
  const holder = h("div", {});
  body.append(holder);
  const session = await api.session();
  page.setData(guided ? "each storm is a CM1 setup that GM1 runs" : `from a CM1 namelist, or from an example · runs folder ${session.root}`);
  if (!guided) {
    // Everything is drawn at once; the presets, the WRF panel and the examples' prices fill in as they arrive.
    // From a WRF or ArWen run sits after the two starts, before CM1's examples.
    holder.replaceChildren(
      h("div", { class: "grid cols-2" }, namelistPanel(session.root),
        later("from-preset", "Start from a preset", "Reading the presets…", presetPanel)),
      later("from-wrf", "From a WRF or ArWen run", "Loading…", wrfPanel),
      examplesSection());
    return null;
  }
  const box = resultBox();
  const onPick = async (e, b) => {
    b.disabled = true;
    try {
      const name = await freeName(e.suggested_name);
      notice(`Setting up ${e.plain_name}…`);
      const reply = await importExample(e, name, box, true);
      if (reply && reply.url) go(runRoute("estimate", reply.run));
    } catch (err) {
      box.show({ ok: false, message: err.message, fix: err.fix, command: err.command }, { always: true });
      notice(`${err.message} ${err.fix || ""}`.trim(), "stop");
    } finally { b.disabled = false; }
  };
  const lead = h("p", { class: "lead" }, "Pick a storm. Each one is set up the way CM1 itself sets it up; GM1 checks it fits ",
    "your GPU before it starts.");
  const gallery = h("div", { class: "gallery" }, h("p", { class: "note" }, "Working out what fits on your GPU…"));
  holder.replaceChildren(lead, gallery, box,
    h("p", { class: "note section" }, "Have your own CM1 namelist? ", toAdvanced("Open it in Advanced"),
      " shows every setting."));
  let picked = false;
  watchExamples(gallery, (d, err) => {
    if (err) {
      gallery.replaceChildren(h("p", { class: "banner stop" }, err.message, h("span", { class: "sub" }, err.fix || "")));
      return;
    }
    const gpu = d.gpu ? `${(d.gpu.name || "your GPU").replace("NVIDIA GeForce ", "")}, ${gb(d.gpu.free_bytes)} free` : null;
    lead.replaceChildren("Pick a storm. Each one is set up the way CM1 itself sets it up; GM1 checks it fits ",
      "your GPU before it starts.", gpu ? h("span", { class: "sub" }, `Your GPU: ${gpu}.`) : null);
    gallery.replaceChildren(...d.examples.map((e) => card(e, (x, b) => { picked = true; return onPick(x, b); })));
  }, () => picked);
  return null;
}

// Guided mode has one gallery, guided/index.js's presets, so New run is an advanced nav item and a
// guided visit goes there; the examples gallery below stays for a build without guided mode.
register("new-run", render, { label: "New run", nav: true, order: 20, mode: "advanced",
  title: (mode) => (mode === "guided" ? "Pick a storm" : "New run") });
