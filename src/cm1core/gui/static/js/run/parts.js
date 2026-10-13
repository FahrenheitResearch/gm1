// The pieces the Run and Results screens share, in both modes: the picture viewer with
// its tabs, thumbnails, time slider and Play; the progress block; the run-control buttons, each with the gpuwm-cm1
// command it runs; the time series; the files and their copy lines; the receipt, the disk, the segments and the
// log. Guided mode draws the same pieces with fewer of them shown, in plain words.

import { h, num, clock, stormClock, bytes, button, readout, meter, table, copyline, kv, status as statusChip, parseClock } from "../core.js";
import * as api from "../api.js";
import { notice } from "../router.js";
import { showCommand, liveCommand } from "../command.js";
import { chartGrid } from "../chart.js";

export const FAVOURITES = ["reflectivity", "spin-near-ground", "updraft-5km", "swath"];
// A hurricane run's favourites: its reflectivity, its wind and its pressure. The spin near the ground, the updraft
// at 5 km and the wind swath are made for one thunderstorm and are not drawn for it.
export const HURRICANE_FAVOURITES = ["reflectivity", "wind-lowest", "pressure-lowest"];
// run: the quick-looks' own facts of the run (results' quicklook_run, the live view's run).
export function favouritesOf(run) {
  return run && run.storm_kind === "hurricane" ? HURRICANE_FAVOURITES : FAVOURITES;
}
export const LIVE_STATES = ["running", "waiting"];

// A run with no saved frame that will draw none on its own: "not_started" (imported or ready) or "ended"
// (refused, failed or stopped before its first saved time). null while frames may still come, or when it has some.
export function noFrames(data) {
  const st = (data && data.status) || {};
  // A frame the run wrote counts even before its picture is drawn.
  if (!data || data.view_only || (data.frames || []).length || st.frames) return null;
  if (["imported", "ready"].includes(st.state)) return "not_started";
  if (["failed", "stopped", "finished", "paused"].includes(st.state)) return "ended";
  return null;
}
export const OVER_STATES = ["finished", "stopped", "failed", "paused"];

// Replace an element's children with `kids`, flattening lists and leaving out null, undefined and false
// (replaceChildren would write them as text).
export function put(el, ...kids) {
  el.replaceChildren(...kids.flat(Infinity).filter((kid) => kid !== null && kid !== undefined && kid !== false));
  return el;
}

// ---- words

export function wallWords(seconds) {
  if (seconds === null || seconds === undefined || Number.isNaN(Number(seconds))) return "";
  const s = Math.round(Number(seconds));
  if (s < 90) return `${s} s`;
  if (s < 5400) return `${Math.round(s / 60)} min`;
  return `${Math.floor(s / 3600)} h ${String(Math.round((s % 3600) / 60)).padStart(2, "0")} min`;
}

export function gib(bytesValue) {
  if (bytesValue === null || bytesValue === undefined) return "";
  return `${(bytesValue / 1024 ** 3).toFixed(bytesValue >= 10 * 1024 ** 3 ? 0 : 1)}`;
}

export function gridWords(grid) {
  if (!grid || !grid.nx) return "";
  const dx = grid.dx_m >= 1000 && grid.dx_m % 1000 === 0 ? `${grid.dx_m / 1000} km` : `${num(grid.dx_m)} m`;
  return `${grid.nx} × ${grid.ny} × ${grid.nz}, dx ${dx}`;
}

export function ago(seconds) {
  if (seconds === null || seconds === undefined) return "";
  return `${wallWords(seconds)} ago`;
}

// ---- the header clock (T+), right of the title

export function headerClock(label = "storm time") {
  const el = h("div", { class: "clock", id: "clock" }, stormClock(0), h("small", {}, label));
  const right = document.querySelector(".head .right");
  if (right) right.prepend(el);
  el.set = (seconds, sub) => {
    el.firstChild.textContent = stormClock(seconds || 0);
    if (sub !== undefined) el.lastChild.textContent = sub;
  };
  return el;
}

// ---- one action button, its result and its command (Show command beside every button in advanced mode)

// opts.watch (the inputs body() reads): the line depends on them (Continue's restart and end time, Convert's
// format), so its Show command is command.js's live one, which asks the server for the exact line with a dry run
// and follows the inputs while it is open. Without it the line is known before the press (opts.command).
export function actionButton(id, action, label, opts = {}) {
  const wrap = h("span", { class: `act${opts.advancedOnly ? " advanced-only" : ""}`, dataset: { action } });
  const b = button(label, { kind: opts.kind || "", title: opts.title || null });
  const cmd = h("span", { class: "cmd" });
  const bodyOf = () => (typeof opts.body === "function" ? opts.body() : (opts.body || {}));
  wrap.append(b);
  if (opts.watch) {
    const live = liveCommand(() => ({ path: `${api.runPath(id)}/${action}`, body: bodyOf() }),
      { always: opts.always === true, watch: opts.watch });
    cmd.append(live);
    wrap.setCommand = () => live.refresh();
  } else {
    wrap.setCommand = (line) => {
      cmd.replaceChildren();
      const shown = showCommand(line, { always: opts.always === true });
      if (shown) cmd.append(shown);
    };
    if (opts.command) wrap.setCommand(opts.command);
  }
  b.addEventListener("click", async () => {
    // opts.check: a reason not to send (a field that cannot be read), said instead of sending something else.
    const problem = opts.check ? opts.check() : null;
    if (problem) { notice(problem, "stop"); return; }
    if (opts.confirm && !window.confirm(opts.confirm)) return;
    b.disabled = true;
    try {
      const body = bodyOf();
      const reply = await api.post(`${api.runPath(id)}/${action}`, body);
      wrap.setCommand(reply.command);
      if (reply.ok === false) notice(`${reply.message || "It did not work."} ${reply.fix || ""}`.trim(), "stop");
      else notice(opts.message ? opts.message(reply) : reply.message);
      if (opts.done) opts.done(reply);
    } catch (err) {
      if (err.command) wrap.setCommand(err.command);
      notice(`${err.message} ${err.fix || ""}`.trim(), "stop");
      if (opts.failed) opts.failed(err);
    } finally {
      b.disabled = false;
    }
  });
  wrap.button = b;
  wrap.append(cmd);
  return wrap;
}

// ---- the picture viewer: tabs (or thumbnails), one large picture, and optionally a time slider with Play

// looks: [{id, title, frames: [{time_s, url, thumb_url}]}] or the live view's latest pictures [{id, title, url,
// thumb_url, time_s}]. opts: {order, slider, thumbs, empty, caption(look, frame), note(look, frame), onTime(t),
// limit, tabIds, emptyFor(look), onLook(look)}. note's words go above the picture, where they are seen before it
// (a picture with no storm in it yet says so there). With tabIds only those looks get a tab (the four favourites) and the others
// are shown with select(id) (the More looks row); a look may set playMs (a film plays at its own speed), and a
// frame without a time (a film's in-between picture) is counted, not clocked.
export function viewer(opts = {}) {
  const tabs = h("div", { class: "tabs", role: "tablist" });
  const img = h("img", { alt: "", hidden: true });
  const link = h("a", { target: "_blank", rel: "noopener" }, img);
  const caption = h("figcaption", {});
  const empty = h("p", { class: "muted g-empty" }, opts.empty || "No pictures yet.");
  const figure = h("figure", { class: "g-live" }, link, caption);
  const thumbs = h("div", { class: "thumbs" });
  const slider = h("input", { type: "range", min: "0", max: "0", step: "1", value: "0", "aria-label": "Saved time" });
  const play = button("Play", { small: true });
  const timeEl = h("span", { class: "t" }, "");
  const ticks = h("div", { class: "ticks" });
  const sliderRow = h("div", { class: "slider" }, play, h("div", {}, slider, ticks), timeEl);
  const noteBox = h("div", { class: "viewer-note" });
  const box = h("div", { class: "viewer" }, tabs, noteBox, empty, figure, opts.thumbs ? thumbs : null,
    opts.slider ? sliderRow : null);
  let looks = [];
  let lookId = null;
  let index = -1;
  let playing = null;
  const frameList = () => {
    const look = looks.find((l) => l.id === lookId);
    if (!look) return [];
    if (Array.isArray(look.frames)) return look.frames.filter((f) => f.url);
    return look.url ? [{ time_s: look.time_s, url: look.url, thumb_url: look.thumb_url, storm: look.storm }] : [];
  };
  function paintTabs() {
    tabs.replaceChildren();
    thumbs.replaceChildren();
    for (const look of looks) {
      if (opts.tabIds && !opts.tabIds.includes(look.id)) continue;
      const has = Array.isArray(look.frames) ? look.frames.length > 0 : Boolean(look.url);
      const b = h("button", { type: "button", role: "tab", class: look.id === lookId ? "on" : null,
        "aria-selected": look.id === lookId ? "true" : "false", disabled: has ? null : true,
        title: look.not_drawn ? look.not_drawn.map((n) => n.reason).join("; ") : null, dataset: { look: look.id } },
      look.title || look.id);
      b.addEventListener("click", () => { lookId = look.id; paint(false); });
      tabs.append(b);
      if (opts.thumbs) {
        const frames = Array.isArray(look.frames) ? look.frames : [look];
        const last = frames[frames.length - 1] || {};
        const src = last.thumb_url || last.url;
        const t = h("button", { type: "button", class: look.id === lookId ? "on" : null, disabled: src ? null : true },
          src ? h("img", { src, alt: "", loading: "lazy" }) : h("div", { class: "nopic" }, "not drawn yet"),
          h("span", {}, look.title || look.id));
        t.addEventListener("click", () => { lookId = look.id; paint(false); });
        thumbs.append(t);
      }
    }
  }
  function paintFrame(keep) {
    const list = frameList();
    slider.max = String(Math.max(0, list.length - 1));
    if (!keep || index < 0 || index >= list.length) index = list.length - 1;
    slider.value = String(Math.max(0, index));
    const f = list[index];
    empty.hidden = Boolean(f);
    figure.hidden = !f;
    img.hidden = !f;
    const shown = looks.find((l) => l.id === lookId);
    sliderRow.hidden = Boolean(shown && shown.once) || !f;
    if (!f) {
      const look = looks.find((l) => l.id === lookId);
      empty.textContent = (opts.emptyFor && look && opts.emptyFor(look)) || opts.empty || "No pictures yet.";
      caption.replaceChildren(); noteBox.replaceChildren(); timeEl.textContent = ""; return;
    }
    if (img.getAttribute("src") !== f.url) img.setAttribute("src", f.url);
    link.href = f.url;
    const look = looks.find((l) => l.id === lookId) || {};
    img.alt = `${look.title || lookId} at ${stormClock(f.time_s || 0)}`;
    caption.replaceChildren(...(opts.caption ? [opts.caption(look, f)].flat() :
      [h("b", {}, look.title || lookId), h("span", {}, stormClock(f.time_s || 0))]));
    put(noteBox, opts.note ? opts.note(look, f) : null);
    const counted = f.time_s === null || f.time_s === undefined;
    timeEl.textContent = counted ? `${index + 1} / ${list.length}` : stormClock(f.time_s || 0);
    ticks.replaceChildren(...(list.length > 1 ? [list[0], list[list.length - 1]] : list).map((row, n) =>
      h("span", {}, row.time_s === null || row.time_s === undefined ? String(n ? list.length : 1) :
        stormClock(row.time_s || 0))));
    if (opts.onTime) opts.onTime(f.time_s || 0, f, look);
  }
  let lastLook = null;
  function paint(keep) {
    paintTabs();
    paintFrame(keep);
    if (opts.onLook && lookId !== lastLook) {
      lastLook = lookId;
      opts.onLook(looks.find((l) => l.id === lookId) || null);
    }
  }
  function stopPlaying() {
    if (playing) { clearInterval(playing); playing = null; }
    play.textContent = opts.playLabel || "Play";
  }
  slider.addEventListener("input", () => { index = Number(slider.value); paintFrame(true); });
  play.addEventListener("click", () => {
    if (playing) { stopPlaying(); return; }
    const list = frameList();
    if (list.length < 2) return;
    if (index >= list.length - 1) index = -1;
    play.textContent = opts.pauseLabel || "Pause";
    const look = looks.find((l) => l.id === lookId) || {};
    playing = setInterval(() => {
      if (index >= frameList().length - 1) { stopPlaying(); return; }
      index += 1;
      paintFrame(true);
    }, look.playMs || opts.playMs || 700);
  });
  box.update = (next, keepPlace = true) => {
    // Stay on the frame shown unless it was the newest (then follow the newest). Whether it was the newest is
    // read from the list before this update: the new list already holds the frames that just arrived.
    const atEnd = index < 0 || index >= frameList().length - 1;
    const order = opts.order || [];
    const rank = (id) => { const i = order.indexOf(id); return i < 0 ? order.length : i; };
    looks = [...(next || [])].sort((p, q) => rank(p.id) - rank(q.id));
    if (opts.limit) looks = looks.filter((l) => order.includes(l.id)).slice(0, opts.limit);
    let sameLook = true;
    if (!looks.some((l) => l.id === lookId)) {
      const first = looks.find((l) => (Array.isArray(l.frames) ? l.frames.length : l.url)) || looks[0];
      lookId = first ? first.id : null;
      sameLook = false;
    }
    if (!playing) paint(keepPlace && sameLook && !atEnd);
  };
  box.select = (id) => { stopPlaying(); lookId = id; paint(false); };
  // Another order and other tabs (a hurricane run's favourites, known once its first picture is drawn).
  box.setOrder = (order, tabIds) => { opts.order = order; if (opts.tabIds) opts.tabIds = tabIds || order; };
  // The words shown where a picture would be, when there is none.
  box.setEmpty = (text) => { if (text && text !== opts.empty) { opts.empty = text; paintFrame(true); } };
  box.tabs = tabs;
  box.stop = stopPlaying;
  box.current = () => ({ look: lookId, frame: frameList()[index] || null });
  box.setPlayLabels = (playLabel, pauseLabel) => { opts.playLabel = playLabel; opts.pauseLabel = pauseLabel; stopPlaying(); };
  return box;
}

// The end time Continue offers, as a storm time (T+H:MM:SS): the run's own end for a run stopped or paused before
// it, an hour more for one that reached it (continuing to the same end has nothing left to run).
export function continueClock(st) {
  const end = st.duration_s;
  if (!end) return "";
  const reached = st.state === "finished" || (st.model_time_s != null && st.model_time_s >= end - 1e-6);
  return clock(Math.round(end) + (reached ? 3600 : 0));
}

// The Continue fields' storm time. Blank goes on to the run's own end; anything else must read as T+H:MM:SS, so
// a time typed another way ("90", the old minutes, or "2h") is said, not dropped into a Continue to that end.
export const UNTIL_HELP = "Give the storm time to end at as T+H:MM:SS, for example T+2:30:00.";
export function untilProblem(text) {
  return String(text ?? "").trim() === "" || parseClock(text) ? null : UNTIL_HELP;
}

// A run page for a run that is not there (deleted, renamed, or a mistyped link): say so, and the way back.
export function runNotFound(body, id, err) {
  body.append(h("section", { class: "panel", id: "run-not-found" },
    h("p", { class: "banner warn" }, `There is no run called "${id}".`,
      h("span", { class: "sub" }, "It may have been deleted or renamed, or the link may be mistyped."),
      err && err.message ? h("span", { class: "sub" }, err.message) : null),
    h("div", { class: "btns" }, h("a", { class: "btn primary", href: "#/runs" }, "Back to Runs"))));
  return null;
}

// Whether the run exists: false when the server says it has no such run (a 404) or that the name cannot be a
// run's at all (a 400, "Use a plain name.", as for a mistyped link such as #/run/Supercell%20typo); any other
// trouble is left to the page, which shows it as before.
export const NO_SUCH_RUN_STATUSES = [400, 404];

export async function runExists(id) {
  try {
    await api.get(`${api.runPath(id)}/status`);
    return { ok: true };
  } catch (err) {
    return err && NO_SUCH_RUN_STATUSES.includes(err.status) ? { ok: false, err } : { ok: true };
  }
}

// ---- progress

// A storm time span in words: "2 h", "1 h 30 min", "5 min", "45 s".
export function spanWords(seconds) {
  const s = Math.round(Number(seconds) || 0);
  if (s >= 172800 && s % 3600 === 0) {
    // Two days or more is said in days: "4 days", "3 days 12 h".
    const days = Math.floor(s / 86400), rest = (s % 86400) / 3600;
    return rest ? `${days} days ${rest} h` : `${days} days`;
  }
  if (s >= 3600 && s % 3600 === 0) return `${s / 3600} h`;
  if (s >= 3600) return `${Math.floor(s / 3600)} h ${Math.round((s % 3600) / 60)} min`;
  if (s >= 60 && s % 60 === 0) return `${s / 60} min`;
  return `${s} s`;
}

// The line under a picture that has no storm in it yet (the server's words, gui/storm.py), or null.
export function stormLine(frame) {
  return frame && frame.storm ? h("p", { class: "nostorm", role: "note" }, frame.storm) : null;
}

// When the next picture comes, in words, or "" when none is coming (the run is not going, or the last is drawn).
// pictures: the live view's schedule {interval_s, first_s, next_s, shown_s}; model and total: storm time now and at
// the end; left: wall seconds left (the run's own estimate). The wall time to the next picture is the time left
// shared out over the storm time left. words: {firstStart, firstAt, firstIn, next, nextIn, drawing}, each
// (values) => text, with values {clock, wall, interval}.
export function pictureWords(pictures, { model, total, left, going }, words) {
  if (!pictures || !going || pictures.next_s === null || pictures.next_s === undefined) return "";
  const next = Number(pictures.next_s);
  const at = stormClock(next);
  const interval = spanWords(pictures.interval_s);
  const wall = left !== null && left !== undefined && total > model && next > model
    ? wallWords(Math.max(1, Number(left) * (next - model) / (total - model))) : null;
  if (pictures.shown_s === null || pictures.shown_s === undefined) {
    // Once the model is past the first picture's time the GPU has started: that picture is being drawn.
    if (next <= 0 && Number(model) > next) return (words.firstDrawing || words.drawing)({ clock: at });
    if (next <= 0) return words.firstStart({ clock: at, interval });
    if (next > model && wall) return words.firstIn({ clock: at, wall, interval });
    return words.firstAt({ clock: at, interval });
  }
  if (next <= (model || 0)) return words.drawing({ clock: at });
  return wall ? words.nextIn({ clock: at, wall }) : words.next({ clock: at });
}

// The progress block both Run screens open with: a bar a finger wide, the storm time done of the whole, the share
// done and the time left, and under it when the next picture comes. words: {of(total), percent(n), left(wall),
// leftUnknown} so guided mode says it in its copy and advanced in its own words.
export function progressBlock(words) {
  const fill = h("i", { class: "done" });
  const bar = h("div", { class: "meter progress big", role: "progressbar", "aria-valuemin": "0",
    "aria-valuemax": "100", "aria-valuenow": "0" }, fill);
  const done = h("span", { class: "rp-done" });
  const left = h("span", { class: "rp-left" });
  const sub = h("p", { class: "rp-sub" });
  const box = h("div", { class: "runprogress" }, h("div", { class: "rp-top" }, done, left), bar, sub);
  // model, total: storm time now and at the end (s); percent: the status's share done; leftS: wall seconds left,
  // or null; going: the run is running or waiting; next: the next-picture words ("" for none).
  box.set = ({ model, total, percent, leftS, going, running, next }) => {
    const share = percent !== null && percent !== undefined ? Number(percent)
      : (total ? 100 * (Number(model) || 0) / total : 0);
    const shown = Math.max(0, Math.min(100, share || 0));
    fill.style.left = "0";
    fill.style.width = `${shown.toFixed(1)}%`;
    bar.setAttribute("aria-valuenow", shown.toFixed(0));
    bar.setAttribute("aria-label", words.percent(Math.round(shown)));
    put(done, h("b", {}, stormClock(model || 0)), total ? h("small", {}, ` ${words.of(stormClock(total))}`) : null,
      h("small", { class: "rp-pc" }, ` · ${words.percent(Math.round(shown))}`));
    left.textContent = running ? (leftS !== null && leftS !== undefined ? words.left(wallWords(leftS)) : words.leftUnknown)
      : "";
    left.hidden = !going;
    sub.textContent = next || "";
  };
  return box;
}

export function progressBar() {
  const bar = meter([{ cls: "done", from: 0, to: 0 }]);
  bar.classList.add("progress");
  bar.set = (percent) => { bar.firstChild.style.width = `${Math.max(0, Math.min(100, percent || 0)).toFixed(1)}%`; };
  return bar;
}

// The advanced readouts: model time of end, time left, s per step, model s per wall s, GPU memory, last write.
export function advancedReadouts(live, st, p) {
  const gpu = live.gpu || {};
  const used = gpu.memory_used_bytes;
  const planned = live.planned_gpu_bytes;
  const lw = live.last_write;
  const lwText = lw ? (lw.kind === "frame" ? `frame ${lw.index ?? lw.number}` : `restart ${lw.number}`) : "none yet";
  const lwSub = lw ? [lw.bytes ? bytes(lw.bytes) : null, lw.age_s != null ? ago(lw.age_s) : null,
    stormClock(lw.model_time_s || 0)].filter(Boolean).join(" · ") : "";
  const model = p.model_time_s ?? st.model_time_s;
  const total = p.duration_s ?? st.duration_s;
  const going = LIVE_STATES.includes(st.state);
  const rows = [
    ["Model time", model != null ? stormClock(model) : "", null, total ? `of ${stormClock(total)}` : null],
    ["Time left", st.state === "running" && (p.eta_s ?? st.eta_s) != null ? wallWords(p.eta_s ?? st.eta_s) : "",
      null, going && p.phase && p.phase !== "stepping" ? p.phase.replace(/-/g, " ") : null],
    // The unit is shown only beside a number.
    ["Seconds per step", p.s_per_step_recent != null ? p.s_per_step_recent.toFixed(3) : "",
      p.s_per_step_recent != null ? "s" : null, p.s_per_step_mean != null ? `recent; ${p.s_per_step_mean.toFixed(3)} s mean` : null],
    ["Model s per wall s", p.model_s_per_wall_s != null ? p.model_s_per_wall_s.toFixed(1) : ""],
    ["GPU memory", used != null ? gib(used) : "", used != null ? "GiB" : null,
      used != null ? `in use on the card${gpu.memory_total_bytes ? ` of ${gib(gpu.memory_total_bytes)} GiB` : ""}` +
        (planned ? `; the estimate planned ${gib(planned)} GiB` : "") :
        (planned ? `the estimate planned ${gib(planned)} GiB` : null)],
    ["Last write", lwText, null, lwSub],
  ];
  // A run that is not going shows only the readouts that have a value (no empty tiles).
  return rows.filter((row) => going || row[1] !== "" || row[0] === "Last write").map((row) => readout(...row));
}

// ---- the time series

export function seriesPanel(series, opts = {}) {
  if (!series || !(series.charts || []).length) {
    return h("p", { class: "muted" }, opts.empty || "The charts appear after the first saved frame is drawn.");
  }
  return chartGrid(series, opts);
}

// ---- files and copy lines

export function filesPanel(data, opts = {}) {
  const rows = data.cm1out || [];
  const total = rows.reduce((sum, r) => sum + (r.bytes || 0), 0);
  const body = [];
  body.push(h("p", { class: "note", style: "margin-top:0" }, h("code", {}, data.cm1out_folder || "cm1out"),
    rows.length ? `  ${rows.length} file${rows.length === 1 ? "" : "s"}, ${bytes(total)}` : "  no files yet"));
  if (rows.length) {
    body.push(h("details", { open: opts.open ? true : null }, h("summary", {}, "Files"),
      h("div", { class: "body" }, table(["File", { label: "Size", num: true }], rows.map((r) =>
        h("tr", {}, h("td", { class: "mono" }, h("a", { href: r.url, download: r.name }, r.name)),
          h("td", { class: "num" }, bytes(r.bytes || 0)))), { compact: true }))));
  }
  if ((data.copy || []).length) {
    body.push(h("div", { class: "section" }, h("h3", { class: "cap" }, "Open it"),
      data.copy.map((c) => h("div", {}, h("div", { class: "note" }, c.label), copyline(c.text, c.label)))));
  }
  return body;
}

// ---- receipt, disk, segments, start-up, log, warnings

export function receiptPanel(rows) {
  if (!rows || !rows.length) return h("p", { class: "muted" }, "The receipt is written when a segment ends.");
  return kv(rows.map((r) => [r.label, r.note ? [r.value, h("span", { class: "hint" }, r.note)] : r.value]));
}

export function diskPanel(disk) {
  if (!disk) return null;
  const rows = [
    ["Raw frames", "segments/*/frames", disk.frames_bytes],
    ["Restarts", "segments/*/restarts", disk.restarts_bytes],
    ["cm1out", "cm1out/", disk.cm1out_bytes],
    ["Quick-looks", "quicklook/", disk.quicklook_bytes],
    ["The whole folder", "", disk.total_bytes],
  ];
  return table(["", "", { label: "Size", num: true }], rows.map(([a, b, c]) =>
    h("tr", {}, h("td", {}, a), h("td", { class: "muted mono" }, b), h("td", { class: "num" }, bytes(c || 0)))),
  { compact: true });
}

// Where a continued segment picked up: "from restart 2 of segment 001", or "from a restart" when only its folder
// is known (a run without the supervisor's segment rows).
function resumedWords(from) {
  const number = from.restart ?? from.number;
  if (number === undefined || number === null) return "from a restart";
  return `from restart ${number}` + (from.segment ? ` of segment ${String(from.segment).padStart(3, "0")}` : "");
}

export function segmentsTable(segments) {
  if (!segments || !segments.length) return null;
  return table(["Segment", "Kind", { label: "Model time", num: true }, { label: "Wall", num: true }, "Ended", "Host"],
    segments.map((s) => h("tr", {},
      h("td", { class: "mono" }, s.segment),
      h("td", {}, s.kind || "", s.resumed_from ? h("span", { class: "sub" }, resumedWords(s.resumed_from)) : null),
      h("td", { class: "num" }, s.model_time_s != null ? stormClock(s.model_time_s) : ""),
      h("td", { class: "num" }, s.wall_s != null ? wallWords(s.wall_s) : ""),
      h("td", {}, s.reason || (s.ended_utc ? "ended" : "")),
      h("td", {}, s.host || ""))), { compact: true, stack: true });
}

export function startupTable(startup) {
  if (!startup || !(startup.stages || []).length) return null;
  const rows = startup.stages.map(([name, seconds]) => h("tr", {},
    h("td", {}, String(name).replace(/_/g, " ")), h("td", { class: "num" }, `${Number(seconds).toFixed(1)} s`)));
  if (startup.nvrtc_programs) {
    rows.push(h("tr", {}, h("td", { class: "muted" }, `${startup.nvrtc_programs} NVRTC programs compiled`),
      h("td", { class: "num muted" }, startup.nvrtc_compile_s != null ? `${Number(startup.nvrtc_compile_s).toFixed(1)} s` : "")));
  }
  return table(["Stage", { label: "Time", num: true }], rows, { compact: true });
}

export function logPanel(log) {
  if (!log || !(log.lines || []).length) return h("p", { class: "muted" }, "No log yet.");
  const pre = h("pre", { class: "log" }, log.lines.join("\n"));
  requestAnimationFrame(() => { pre.scrollTop = pre.scrollHeight; });
  return [pre, log.url ? h("p", { class: "note" }, h("a", { href: log.url, target: "_blank", rel: "noopener" },
    "The whole log"), `  ${bytes(log.bytes || 0)}`) : null];
}

export function warningsBlock(warnings) {
  if (!warnings || !warnings.length) return null;
  return h("div", { class: "section" }, warnings.map((w) => h("p", { class: "banner warn" }, w.text,
    w.code ? h("span", { class: "sub hint" }, w.code) : null)));
}

// Where the run was started, for a run this GUI is only attached to.
export function attachedLine(live) {
  const a = live.attached || {};
  if (!a.started_by || a.launched_here) return null;
  const where = a.host && a.host !== a.this_host ? ` on ${a.host}` : "";
  const how = a.local === false ? " This GUI reads its folder and can stop or pause it through its control files." :
    " This GUI reads its folder; Stop and Pause work through its control files.";
  return h("p", { class: "note" }, `Started from ${a.started_by}${where}.${how}`);
}

export function stateChip(st) {
  return statusChip(st.state || "imported", st.label || "");
}

// Collapsible block with a title and a note after it.
export function fold(title, note, body, open = false) {
  return h("details", { open: open ? true : null }, h("summary", {}, title,
    note ? h("span", { class: "muted" }, `  ${note}`) : null), h("div", { class: "body" }, body));
}

// ---- more looks and films

// Fill "{name}" from values in a looks.json ui string.
export function words(data, key, values = {}) {
  let text = ((data && data.looks_ui) || {})[key] || "";
  for (const [name, value] of Object.entries(values)) text = text.replaceAll(`{${name}}`, String(value));
  return text;
}

// The looks after the four favourites, as the viewer takes them: a per-frame look's saved pictures, a look drawn
// once per run (one picture, no slider), or a film's pictures (played at the film's own speed).
export function moreLookRows(data) {
  const byId = Object.fromEntries((data.looks || []).map((l) => [l.id, l]));
  return (data.more_looks || []).map((row) => {
    if (row.kind === "film") {
      const film = row.film || {};
      return { id: row.id, title: row.title, is_new: row.is_new, film: true, frames: film.frames || [],
        playMs: film.fps ? Math.round(1000 / film.fps) : 80 };
    }
    const look = byId[row.id] || {};
    return { ...look, id: row.id, title: row.title, is_new: row.is_new, once: row.kind === "once",
      frames: look.frames || [] };
  });
}

// The film of a look as viewer frames (advanced: the swirl film shows beside the per-frame looks).
export function filmLook(data, id, title) {
  const film = (data.films || {})[id];
  if (!film || !(film.frames || []).length) return null;
  return { id, title, film: true, frames: film.frames, playMs: film.fps ? Math.round(1000 / film.fps) : 80 };
}

// The row under the picture: Save as film (or Make the film), its progress and Stop, the finished film's files,
// and Draw it / Draw the new looks for a run whose pictures lack a look. opts: {always, refresh()}.
export function filmBar(id, data, lookId, opts = {}) {
  const bar = h("div", { class: "film-bar", dataset: { look: lookId || "" } });
  if (!lookId) return bar;
  const more = (data.more_looks || []).find((row) => row.id === lookId) || null;
  const filmable = (data.film_looks || []).includes(lookId);
  const film = (data.films || {})[lookId] || { state: "none" };
  const post = async (path, body, b) => {
    if (b) b.disabled = true;
    try {
      const reply = await api.post(`${api.runPath(id)}/${path}`, body);
      notice(reply.message || "");
      if (opts.refresh) setTimeout(opts.refresh, 800);
    } catch (err) {
      notice(`${err.message} ${err.fix || ""}`.trim(), "stop");
      if (b) b.disabled = false;
    }
  };
  if (more && more.action === "all") {
    if (data.drawing_looks) {
      bar.append(h("p", { class: "note" }, words(data, "drawing")));
    } else {
      const b = button(words(data, "draw_new"), { kind: "primary", small: true });
      b.addEventListener("click", () => post("look", { look: "all" }, b));
      bar.append(h("div", { class: "film-row" }, b,
        h("span", { class: "note" }, words(data, "draw_new_about", { seconds: (data.looks_about_s || {}).all || "" })),
        showCommand((data.look_commands || {}).all, { always: opts.always })));
    }
  } else if (more && more.action === "look") {
    const b = button(words(data, "draw"), { small: true });
    b.addEventListener("click", () => post("look", { look: lookId }, b));
    bar.append(h("div", { class: "film-row" }, b, showCommand((data.look_commands || {})[lookId], { always: opts.always })));
  }
  const available = !more || more.available || more.action === "film";
  if (!filmable || !available) return bar;
  if (film.state === "running") {
    const p = film.progress || {};
    const line = p.total ? words(data, "making", { done: p.done || 0, total: p.total }) : words(data, "making_start");
    const row = h("div", { class: "film-row" }, h("span", { class: "film-progress", role: "status" }, line));
    if (p.total) {
      const pct = Math.max(0, Math.min(100, (100 * (p.done || 0)) / p.total));
      row.append(h("span", { class: "film-meter", "aria-hidden": "true" }, h("span", { style: `width:${pct.toFixed(1)}%` })));
    }
    if (film.running_here) {
      const stop = button(words(data, "stop"), { kind: "stop", small: true });
      stop.addEventListener("click", () => post("film", { look: lookId, stop: true }, stop));
      row.append(stop);
    }
    bar.append(row);
    return bar;
  }
  const label = lookId === "swirl" && !(film.frames || []).length ? words(data, "make_film") : words(data, "save_film");
  const b = button(label, { small: true, kind: lookId === "swirl" && !(film.frames || []).length ? "primary" : "" });
  b.addEventListener("click", () => post("film", { look: lookId }, b));
  const about = (data.looks_about_s || {})[lookId];
  bar.append(h("div", { class: "film-row" }, b,
    about ? h("span", { class: "note" }, words(data, "film_about", { seconds: about })) : null,
    showCommand((data.film_commands || {})[lookId], { always: opts.always })));
  if (film.state === "failed" && film.progress && film.progress.message) {
    bar.append(h("p", { class: "note warn" }, words(data, "failed", { message: film.progress.message })));
  } else if (film.state === "stopped") {
    bar.append(h("p", { class: "note" }, words(data, "stopped", { done: (film.progress || {}).done || 0 })));
  }
  const urls = film.urls || {};
  if (film.state === "done" && Object.keys(urls).length) {
    const links = [];
    if (urls.mp4) links.push(h("a", { href: urls.mp4, download: "" }, words(data, "download_mp4")));
    if (urls.gif) links.push(h("a", { href: urls.gif, download: "" }, words(data, "download_gif")));
    if (urls.apng) links.push(h("a", { href: urls.apng, target: "_blank", rel: "noopener" }, words(data, "download_apng")));
    const parts = [words(data, "ready"), " "];
    links.forEach((a, n) => { if (n) parts.push(" · "); parts.push(a); });
    bar.append(h("p", { class: "note film-files" }, ...parts));
    if (!data.mp4_available && !urls.mp4) bar.append(h("p", { class: "note" }, words(data, "no_mp4")));
  }
  return bar;
}

// The More looks row under the four favourites' tabs (guided): one small button per look.
export function moreLooksRow(data, current, onPick) {
  const rows = data.more_looks || [];
  if (!rows.length) return null;
  const list = h("div", { class: "more-looks-list", role: "list" });
  for (const row of rows) {
    const b = h("button", { type: "button", class: `more-look${row.id === current ? " on" : ""}${row.available ? "" : " off"}`,
      role: "listitem", "aria-pressed": row.id === current ? "true" : "false", dataset: { look: row.id },
      title: row.available ? row.short || null : row.why_not || null },
    row.title);
    b.addEventListener("click", () => onPick(row.id));
    list.append(b);
  }
  return h("div", { class: "more-looks" }, h("span", { class: "more-looks-cap cap" }, words(data, "more_looks")), list);
}
