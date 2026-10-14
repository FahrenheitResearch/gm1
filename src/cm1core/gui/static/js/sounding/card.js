// Guided mode's card "Use my own sounding", in the gallery's Explore group. It opens in place on the gallery (no
// new screen), as the WRF card does. Five steps in one panel:
//   1. the sounding: a file the browser reads, pasted text, a path, or one of the balloon soundings GM1 ships
//   2. what GM1 found: the layout, each conversion, each problem with its line and what to do, each warning
//   3. the air: the skew-T and hodograph, the numbers, one hedged sentence on the storm such air usually makes
//   4. how the storm is started, with the start chosen from the sounding and why
//   5. Next, which goes to "How long can it run?" with the kept file and the chosen start
// Every check is one command (POST /api/sounding/check: gpuwm-cm1 sounding convert IN OUT ... --json), and the
// sentences about the file are that command's own, so the page and a terminal say the same thing.
// Next keeps the last check in this tab (sessionStorage, for half an hour), so Back from "How long can it run?"
// opens the card again on the same sounding; closing the card or starting a run forgets it.

import { h, button, num, readout } from "../core.js";
import * as api from "../api.js";
import { go } from "../router.js";
import { commandSlot, showCommand } from "../command.js";
import { ready, t, hint } from "./copy.js";

let cards = 0;
const BACK_KEY = "gm1.sounding.back";
const BACK_MS = 30 * 60 * 1000;

function remembered() {
  try {
    const saved = JSON.parse(sessionStorage.getItem(BACK_KEY) || "null");
    if (saved && saved.request && Date.now() - Number(saved.at || 0) < BACK_MS) return saved;
    sessionStorage.removeItem(BACK_KEY);
  } catch (_) { /* no storage here: the card starts empty */ }
  return null;
}

function remember(request, start) {
  try {
    sessionStorage.setItem(BACK_KEY, JSON.stringify({ request, start, at: Date.now() }));
  } catch (_) { /* too large or no storage: Back shows the card closed */ }
}

function forget() {
  try { sessionStorage.removeItem(BACK_KEY); } catch (_) { /* nothing kept */ }
}

function banner(kind, head, text, fix, line) {
  return h("div", { class: `banner ${kind} snd-finding` },
    h("strong", {}, head), line ? h("span", { class: "snd-line" }, ` ${t("found.line", { line })}`) : null,
    h("p", {}, text),
    fix ? h("p", { class: "sub" }, h("span", { class: "snd-fix" }, `${t("found.fix")} `), fix) : null);
}

function km(m, nd = 1) {
  return m === null || m === undefined ? "–" : num(Number(m) / 1000, nd);
}

// Where the wind (u toward the east, v toward the north) is heading, in words.
function toward(u, v) {
  const names = ["north", "north-east", "east", "south-east", "south", "south-west", "west", "north-west"];
  const angle = (Math.atan2(u, v) * 180) / Math.PI;
  return names[((Math.round(angle / 45) % 8) + 8) % 8];
}

// render(el): the card itself, as the gallery's cards are drawn. The gallery's registerCard hands render a cell
// of the group's grid (div.g-cell); the panel goes after that cell and spans the whole row (.snd-panel).
export async function renderCard(el) {
  await ready();
  const openBtn = h("button", { class: "btn primary", type: "button", "aria-expanded": "false" }, t("card.open"));
  const card = h("article", { class: "g-card snd-card", dataset: { preset: "own-sounding" } },
    h("div", { class: "g-pic none snd-pic" }, h("span", {}, t("card.picture"))),
    h("div", { class: "g-body" },
      h("h2", {}, t("card.title")),
      h("p", { class: "g-line" }, t("card.line")),
      h("div", { class: "btns" }, openBtn)));
  el.append(card);
  let panel = null;
  const fold = (shown) => {
    panel.hidden = !shown;
    openBtn.textContent = shown ? t("card.close") : t("card.open");
    openBtn.setAttribute("aria-expanded", shown ? "true" : "false");
  };
  const open = async (saved) => {
    if (panel) { fold(panel.hidden); return; }
    panel = await openPanel(() => { forget(); fold(false); openBtn.focus(); });
    const anchor = el.classList && el.classList.contains("g-cell") ? el : card;
    // Back from the next screen on a phone: the cell may be folded away under "Show all"; show it.
    if (saved && anchor.hidden) anchor.hidden = false;
    anchor.after(panel);
    // "Show fewer" hides the cell; the panel goes with it.
    if (anchor !== card && typeof MutationObserver === "function") {
      new MutationObserver(() => { if (anchor.hidden && !panel.hidden) fold(false); })
        .observe(anchor, { attributes: true, attributeFilter: ["hidden"] });
    }
    fold(true);
    panel.scrollIntoView({ block: saved ? "start" : "nearest" });
    const first = panel.querySelector("input[type=file]") || panel.querySelector("button, input, select, textarea");
    if (first) first.focus({ preventScroll: true });
    if (saved) panel.restore(saved);
  };
  openBtn.addEventListener("click", () => open(null));
  const saved = remembered();
  if (saved) open(saved);
  return card;
}

async function openPanel(onClose) {
  const id = ++cards;
  const facts = await api.get("/api/sounding/shipped").catch(() => ({ soundings: [], wsl: false, max_bytes: 4194304 }));
  // What the card holds: the last request sent (so a remedy button sends it again with one thing changed), the
  // record that came back, and the start picked.
  const state = { request: null, record: null, start: null };
  let serial = 0;

  const say = h("div", { class: "snd-say", role: "status", "aria-live": "polite" });
  const found = h("section", { class: "snd-step snd-found", hidden: true });
  const air = h("section", { class: "snd-step snd-air", hidden: true });
  const starts = h("section", { class: "snd-step snd-starts", hidden: true });
  const next = h("section", { class: "snd-step snd-next", hidden: true });
  const slot = commandSlot();

  // ---- 1. the sounding
  const file = h("input", { type: "file", id: `snd-file-${id}`, class: "snd-file",
    accept: ".txt,.csv,.tsv,.snd,.dat,.html,.htm,.input_sounding,text/*", "aria-label": t("file.choose") });
  const paste = h("textarea", { class: "input mono snd-paste", rows: "6", spellcheck: "false",
    placeholder: t("file.paste_placeholder"), "aria-label": t("file.paste") });
  const pasteUse = button(t("file.paste_use"), {});
  const shipped = h("select", { class: "input", "aria-label": t("file.shipped") },
    h("option", { value: "" }, t("file.shipped_pick")),
    (facts.soundings || []).map((row) => h("option", { value: row.id }, row.title)));
  const shippedUse = button(t("file.shipped_use"), {});
  const path = h("input", { class: "input mono", placeholder: t("file.path_placeholder"), spellcheck: "false",
    autocomplete: "off", "aria-label": t("file.path") });
  const pathUse = button(t("file.path_use"), {});

  // Nothing of an earlier check stays on screen (or behind Next) once another sounding is asked for.
  function reset() {
    serial += 1;
    state.request = null;
    state.record = null;
    for (const el of [found, air, starts, next]) { el.hidden = true; el.replaceChildren(); }
  }
  function refuse(text, fix) {
    reset();
    say.replaceChildren(banner("stop", t("found.problem"), text, fix || ""));
  }

  async function check(request, startWanted) {
    reset();
    const mine = serial;
    state.request = request;
    say.replaceChildren(h("p", { class: "note" }, t("file.reading")));
    let reply;
    try {
      reply = await api.post("/api/sounding/check", request);
    } catch (err) {
      if (mine !== serial) return;
      say.replaceChildren(banner("stop", t("found.problem"), err.message, err.fix));
      return;
    }
    if (mine !== serial) return;
    slot.set(reply.command);
    say.replaceChildren();
    if (!reply.checked) {
      say.replaceChildren(banner("stop", t("found.problem"), reply.message || "", reply.fix));
      return;
    }
    state.record = reply;
    const offered = ((reply.start || {}).options || []).map((option) => option.id);
    state.start = offered.includes(startWanted) ? startWanted : (reply.start || {}).recommended || "push";
    drawFound(reply);
    if (reply.numbers || reply.preview || reply.verdict) drawAir(reply);
    if (reply.ok && reply.start) { drawStarts(reply); drawNext(reply); }
    found.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }

  file.addEventListener("change", () => {
    const picked = file.files && file.files[0];
    if (!picked) return;
    if (picked.size > (facts.max_bytes || 4194304)) { refuse(t("file.too_large"), t("file.too_large_fix")); return; }
    if (picked.size === 0) { refuse(t("file.empty"), t("file.empty_fix")); return; }
    reset();
    const mine = serial;
    const reader = new FileReader();
    reader.onload = () => { if (mine === serial) check({ text: String(reader.result || ""), name: picked.name }); };
    reader.onerror = () => { if (mine === serial) refuse(t("file.not_read"), ""); };
    reader.readAsText(picked);
  });
  // A sounding given another way: the file field no longer names the one being checked.
  const notTheFile = () => { try { file.value = ""; } catch (_) { /* some browsers keep it */ } };
  pasteUse.addEventListener("click", () => {
    notTheFile();
    if (!paste.value.trim()) { refuse(t("file.none"), ""); return; }
    check({ text: paste.value, name: "pasted-sounding.txt" });
  });
  shippedUse.addEventListener("click", () => {
    notTheFile();
    if (!shipped.value) { refuse(t("file.none"), ""); return; }
    check({ shipped: shipped.value });
  });
  const readPath = () => {
    notTheFile();
    if (!path.value.trim()) { refuse(t("file.none"), ""); return; }
    check({ path: path.value.trim() });
  };
  pathUse.addEventListener("click", readPath);
  path.addEventListener("keydown", (e) => { if (e.key === "Enter") readPath(); });

  // ---- 2. what GM1 found
  function drawFound(r) {
    const kids = [h("h3", { class: "snd-h" }, t("found.title"))];
    if (r.format) {
      kids.push(h("p", { class: "snd-read text" }, t("found.read_as", {
        name: r.name, format: r.format_words, levels: r.levels, top: km(r.top_m),
      })));
    }
    if (r.shipped) kids.push(h("p", { class: "note" }, t("found.shipped", { observation: r.shipped })));
    for (const p of r.problems || []) {
      const offer = p.offer || {};
      const remedies = [];
      if (offer.extend_to_m) {
        const b = button(t("found.extend"), { kind: "primary" });
        b.dataset.remedy = "extend";
        b.addEventListener("click", () => check({ ...state.request, extend: true }));
        remedies.push(b, h("span", { class: "note" }, t("found.extend_note")));
      }
      if (offer.speed_unit) {
        for (const unit of ["kt", "ms"]) {
          const b = button(t(`found.speed_${unit}`), {});
          b.dataset.remedy = `speed-${unit}`;
          b.addEventListener("click", () => check({ ...state.request, speed_unit: unit }));
          remedies.push(b);
        }
      }
      kids.push(banner("stop", t("found.problem"), p.text, p.fix, p.line),
        remedies.length ? h("div", { class: "btns snd-remedies" }, remedies) : null);
    }
    const ext = r.extension || {};
    if (ext.extended) kids.push(banner("warn", t("found.warning"), ext.sentence, ""));
    for (const w of r.warnings || []) kids.push(banner("warn", t("found.warning"), w.text, w.fix, w.line));
    for (const note of r.notes || []) kids.push(h("p", { class: "note" }, note));
    const dropped = r.dropped || [];
    if (dropped.length) {
      // Every line left out, grouped by why (the levels a listing gives under the ground first, as it lists them).
      const groups = new Map();
      for (const row of dropped) {
        const why = String(row.why || "");
        if (!groups.has(why)) groups.set(why, []);
        groups.get(why).push(row.line);
      }
      kids.push(h("details", { class: "snd-details snd-dropped" },
        h("summary", {}, t("found.dropped", { count: dropped.length })),
        h("ul", { class: "g-changes" }, [...groups].map(([why, lines]) => h("li", {},
          t(lines.length === 1 ? "found.dropped_line" : "found.dropped_lines", { lines: lines.join(", "), why }))))));
    }
    if (r.ok) kids.push(h("p", { class: "snd-ok text" }, h("span", { class: "st ok" }), " ", t("found.ok")));
    if (r.converted && (r.conversions || []).length) {
      kids.push(h("details", { class: "snd-details" }, h("summary", {}, t("found.converted")),
        h("ul", { class: "g-changes" }, r.conversions.map((line) => h("li", {}, line)))));
    }
    if (r.text) {
      kids.push(h("details", { class: "snd-details" }, h("summary", {}, t("found.text")),
        h("p", { class: "note" }, t("found.text_note"), hint("found.text_note")),
        h("pre", { class: "snd-text mono", tabindex: "0" }, r.text)));
    }
    if (r.download) {
      kids.push(h("p", { class: "snd-download" },
        h("a", { href: r.download, download: `${r.label || "sounding"}.input_sounding` }, t("found.download")),
        " ", h("span", { class: "note" }, t("found.kept"))));
    }
    found.replaceChildren(...kids.filter(Boolean));
    found.hidden = false;
  }

  // ---- 3. the air
  function drawAir(r) {
    const n = r.numbers || {};
    const value = (v, nd = 0) => (v === null || v === undefined ? "–" : num(Number(v), nd));
    // The picture opens full size in its own tab (it is drawn for a wide screen; on a phone it can be zoomed there).
    const fig = r.preview && r.preview.url
      ? h("figure", { class: "snd-skewt" },
        h("a", { href: r.preview.url, target: "_blank", rel: "noopener" },
          h("img", { src: r.preview.url, alt: t("preview.alt"), width: "1920", height: "1080" })),
        h("figcaption", { class: "note" }, t("preview.open")))
      : r.preview && r.preview.why ? h("p", { class: "note" }, r.preview.why) : null;
    const readouts = r.numbers ? h("div", { class: "readouts" },
      readout([t("numbers.cape"), hint("numbers.cape")], value(n.cape_j_kg), "J/kg"),
      readout([t("numbers.cin"), hint("numbers.cin")], value(n.cin_j_kg), "J/kg"),
      readout([t("numbers.lcl"), hint("numbers.lcl")], km(n.lcl_m), "km"),
      readout([t("numbers.lfc"), hint("numbers.lfc")], km(n.lfc_m), "km"),
      readout([t("numbers.el"), hint("numbers.el")], km(n.el_m), "km"),
      readout(t("numbers.shear6"), value(n.shear_0_6km_m_s), "m/s"),
      readout([t("numbers.srh1"), hint("numbers.srh1")], value(n.srh_0_1km_m2_s2), "m²/s²"),
      readout([t("numbers.srh3"), hint("numbers.srh3")], value(n.srh_0_3km_m2_s2), "m²/s²"))
      : r.numbers_reason ? h("p", { class: "note" }, t("numbers.none", { reason: r.numbers_reason })) : null;
    const said = r.verdict;
    const motion = n.storm_motion_m_s || [];
    const moving = typeof motion[0] === "number" && typeof motion[1] === "number"
      ? h("p", { class: "note snd-motion" }, t("numbers.motion", {
        speed: num(Math.hypot(motion[0], motion[1]), 0), toward: toward(motion[0], motion[1]),
      })) : null;
    air.replaceChildren(...[
      h("h3", { class: "snd-h" }, t("preview.title")),
      fig,
      said ? h("p", { class: "snd-verdict", dataset: { verdict: said.id } }, [said.sentence, ...(said.more || [])].join(" "),
        h("span", { class: "sub" }, said.note)) : null,
      readouts,
      n.el_note ? h("p", { class: "note snd-el" }, n.el_note) : null,
      n.partial_top_m ? h("p", { class: "note snd-partial" }, t("numbers.partial", { top: km(n.partial_top_m) })) : null,
      r.numbers ? h("p", { class: "note" }, t("numbers.basis")) : null,
      moving,
    ].filter(Boolean));
    air.hidden = false;
  }

  // ---- 4. how the storm is started
  function drawStarts(r) {
    const s = r.start;
    const group = h("div", { class: "snd-start-list", role: "radiogroup", "aria-label": t("start.label") });
    for (const option of s.options || []) {
      const input = h("input", { type: "radio", name: `snd-start-${id}`, value: option.id, id: `snd-start-${id}-${option.id}` });
      input.checked = option.id === state.start;
      input.addEventListener("change", () => {
        if (!input.checked) return;
        state.start = option.id;
        paintNext();
        if (remembered()) remember(state.request, state.start);
      });
      group.append(h("label", { class: "snd-start", for: input.id }, input,
        h("span", { class: "snd-start-body" },
          h("strong", {}, option.title),
          option.recommended ? h("span", { class: "st ok snd-rec" }, t("start.recommended")) : null,
          h("span", { class: "snd-start-line" }, option.line))));
    }
    starts.replaceChildren(
      h("h3", { class: "snd-h" }, t("start.title")),
      h("p", { class: "text snd-why" }, s.close_call ? h("span", { class: "st warn" }, t("start.close")) : null,
        s.close_call ? " " : "", s.why),
      group,
      h("p", { class: "note snd-trap" }, s.trap));
    starts.hidden = false;
  }

  // ---- 5. Next
  const goButton = button(t("next.button"), { kind: "primary" });
  goButton.dataset.action = "sounding-next";
  const nextCommand = h("div", {});
  const nextSay = h("div", { role: "status", "aria-live": "polite" });
  function routeFor(r) {
    const base = `guided/fit/${encodeURIComponent(r.preset)}/10m?sounding=${encodeURIComponent(r.sounding)}`;
    return state.start === "strong-bubble" ? `${base}&start=bubble&push=3` : `${base}&start=${state.start}`;
  }
  function paintNext() {
    const r = state.record;
    if (!r || !r.fit_command) { nextCommand.replaceChildren(); return; }
    const flag = state.start === "strong-bubble" ? "--start bubble --custom push=3" : `--start ${state.start}`;
    nextCommand.replaceChildren(showCommand(r.fit_command.replace(/--start \S+/, flag)) || "");
  }
  function drawNext(r) {
    next.replaceChildren(
      h("h3", { class: "snd-h" }, t("next.title")),
      h("p", { class: "note" }, t("next.note")),
      h("div", { class: "btns" }, goButton), nextCommand, nextSay,
      h("div", { class: "snd-about" }, h("h4", {}, t("about.title")), h("p", {}, t("about.text"))));
    next.hidden = false;
    paintNext();
  }
  goButton.addEventListener("click", async () => {
    const r = state.record;
    if (!r || !r.ok || !r.sounding) return;
    const presets = await api.get("/api/presets").catch(() => ({ presets: [] }));
    if (!(presets.presets || []).some((p) => p.id === r.preset)) {
      nextSay.replaceChildren(h("p", { class: "banner warn" }, t("next.waiting")));
      return;
    }
    remember(state.request, state.start);
    go(routeFor(r));
  });

  const close = button(t("card.close"), { small: true });
  close.addEventListener("click", onClose);
  const field = (label, ...kids) => h("div", { class: "snd-way" }, h("div", { class: "snd-way-label" }, label), h("div", { class: "snd-way-body" }, kids));
  const panel = h("section", { class: "panel snd-panel", id: "own-sounding-panel" },
    h("div", { class: "snd-panel-head" }, h("h2", { class: "cap" }, t("card.title")), close),
    h("section", { class: "snd-step" },
      h("h3", { class: "snd-h" }, t("file.title")),
      h("p", { class: "note" }, t("file.formats"), hint("file.formats")),
      field(h("label", { for: file.id }, t("file.choose")), file,
        h("div", { class: "hint" }, t("file.choose_help"))),
      field(t("file.shipped"), h("div", { class: "snd-row" }, shipped, shippedUse),
        h("div", { class: "hint" }, t("file.shipped_help"))),
      field(t("file.paste"), paste, h("div", { class: "btns" }, pasteUse)),
      field(t("file.path"), h("div", { class: "snd-row" }, path, pathUse),
        facts.wsl ? h("div", { class: "hint snd-wsl" }, t("file.wsl")) : null)),
    say, found, air, starts, next, slot);
  // Back from "How long can it run?": the same sounding checked again (the same command), with the start picked.
  panel.restore = (saved) => {
    const request = saved.request || {};
    if (request.path) path.value = request.path;
    if (request.shipped) shipped.value = request.shipped;
    if (typeof request.text === "string" && request.name === "pasted-sounding.txt") paste.value = request.text;
    check(request, saved.start);
  };
  return panel;
}
