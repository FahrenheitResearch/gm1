// Screen 8, Run (guided: 7.2 "Watching"). One screen, two views of the same run folder:
//
//   advanced  progress, model time, time left, s/step, model-s per wall-s, GPU memory, the last write and the
//             start-up stages; the latest quick-looks; the time series (with CM1's curve on its stock storm); the
//             log; Stop, Pause at next restart, Force stop (runs on this computer, 60 s after Stop) and Continue,
//             each with its gpuwm-cm1 command.
//   guided    guided mode's header: storm time, time left, one large picture over the four favourite looks that updates
//             with each saved time, and Stop. Everything else is in Advanced.
//
// A run started from a terminal or a batch job reads the same (attach): the page reads the run folder.

import { h, num, clock, parseClock, button, schemeName } from "../core.js";
import * as api from "../api.js";
import { register, go, notice, runRoute } from "../router.js";
import { showCommand } from "../command.js";
import { ready as copyReady, t, looks as favouriteCopy } from "../../guided/copy.js";
import {
  FAVOURITES, LIVE_STATES, OVER_STATES, actionButton, advancedReadouts, attachedLine, fold, headerClock, logPanel,
  pictureWords, progressBlock, put, segmentsTable, seriesPanel, startupTable, stateChip, stormLine, viewer,
  wallWords, warningsBlock, gridWords, continueClock, runExists, runNotFound, untilProblem,
} from "./parts.js";

const POLL_MS = 3000;
const IDLE_POLL_MS = 12000;

// The polled progress record replaces the one the event stream last brought unless that one is newer (both carry
// the runtime's time_utc), so a dropped stream never freezes the clock and a slow poll never steps it back.
function newer(polled, held) {
  if (!polled) return false;
  return !held || !held.time_utc || !polled.time_utc || polled.time_utc >= held.time_utc;
}

// ------------------------------------------------------------------ guided: Watching

async function guidedRun(body, id, page) {
  await copyReady();
  const runUrl = encodeURIComponent(id);
  const clockEl = headerClock(t("watch.storm_time"));
  const progressEl = progressBlock({
    of: (total) => t("watch.progress_of", { total }), percent: (n) => t("watch.percent", { percent: n }),
    left: (wall) => t("watch.left", { wall }), leftUnknown: t("watch.left_unknown"),
  });
  const pictureCopy = {
    firstStart: (v) => t("watch.first_picture_start", v), firstAt: (v) => t("watch.first_picture_at", v),
    firstIn: (v) => t("watch.first_picture_in", v), next: (v) => t("watch.next_picture", v),
    nextIn: (v) => t("watch.next_picture_in", v), drawing: (v) => t("watch.next_picture_drawing", v),
    firstDrawing: (v) => t("watch.first_picture_drawing", v),
  };
  const state = h("p", { class: "g-state" });
  const chip = h("span", {});
  const stop = button(t("watch.stop"), { kind: "stop" });
  const openResults = button(t("watch.open_results"), { kind: "primary" });
  openResults.hidden = true;
  openResults.addEventListener("click", () => go(`guided/results/${runUrl}`));
  // A run imported but not started has nothing to watch: say so, and where it starts.
  const startIt = button(t("watch.start_it"), { kind: "primary" });
  startIt.hidden = true;
  startIt.addEventListener("click", () => go(runRoute("estimate", id)));
  const titles = Object.fromEntries(favouriteCopy().map((l) => [l.id, l.title]));
  const view = viewer({
    order: FAVOURITES, limit: 4, empty: t("watch.first_picture"),
    caption: (look, f) => [h("b", {}, t("watch.picture_at", { look: titles[look.id] || look.title || look.id,
      clock: clock(f.time_s || 0) }))],
    note: (look, f) => stormLine(f),
  });
  view.classList.add("g-viewer");
  body.append(
    h("section", { class: "panel" }, progressEl, state, h("div", { class: "btns" }, chip, stop, openResults, startIt)),
    h("section", { class: "panel section" }, view));

  let status = {};
  let progress = {};
  let pictures = null;
  // The saved times the processor has not drawn yet (a fast GPU outruns the pictures): live.pictures_behind.
  let behind = null;
  let closed = false;
  let painting = false;
  function paint() {
    if (painting) return;
    painting = true;
    requestAnimationFrame(() => { painting = false; paintNow(); });
  }
  function paintNow() {
    const model = progress.model_time_s ?? status.model_time_s ?? 0;
    const total = progress.duration_s ?? status.duration_s;
    const left = status.state === "running" ? (progress.eta_s ?? status.eta_s) : null;
    clockEl.set(model);
    page.setData(`${id} · ${status.label || ""}`);
    // Time left is shown only while the run goes; a run that has ended has none to show. The next picture's time
    // (and, before the first, when the first comes) is said under the bar while the run goes.
    const going = LIVE_STATES.includes(status.state);
    const next = pictureWords(pictures, { model, total, left, going: status.state === "running" }, pictureCopy);
    // Before the first picture its time is said where the picture will be; after, the next one's under the bar.
    const first = Boolean(pictures) && (pictures.shown_s === null || pictures.shown_s === undefined);
    const over = OVER_STATES.includes(status.state);
    // Pictures behind the GPU: while it runs, by more than one saved time; once it has ended, any.
    const drawing = behind && (over ? behind.behind > 0 : behind.behind > 1) ? behind : null;
    const drawWords = !drawing ? "" : t(over ? "watch.drawing" : "watch.behind",
      { drawn: drawing.drawn, saved: drawing.saved, wall: wallWords(drawing.about_s) });
    progressEl.set({ model, total, percent: status.percent, leftS: left, going, running: status.state === "running",
      next: [first ? "" : next, drawWords].filter(Boolean).join(" ") });
    if (first && next) view.setEmpty(next);
    if (OVER_STATES.includes(status.state) && !status.frames) view.setEmpty(t("results.no_frames"));
    chip.replaceChildren(stateChip(status));
    const s = status.state;
    const actions = new Set(status.actions || []);
    stop.hidden = !actions.has("stop");
    openResults.hidden = !OVER_STATES.includes(s);
    // GM1's own GPU queue is taken on every start: "another run is using it" only when one is.
    const queue = status.queue || {};
    const queued = Number(queue.ahead || 0) > 0 || Boolean(queue.holder);
    const words = { waiting: t(queued ? "watch.waiting_gpu" : "watch.starting"),
      finished: t(drawing ? "watch.finished_drawing" : "watch.finished", { wall: drawing ? wallWords(drawing.about_s) : "" }),
      stopped: t("watch.stopped"), paused: t("watch.stopped"), failed: t("watch.failed") };
    let text = words[s] || "";
    if (s === "running" && !status.model_time_s && !progress.model_time_s) text = t("watch.starting");
    const notStarted = ["imported", "ready"].includes(s);
    if (notStarted) { text = t("watch.not_started"); view.setEmpty(t("watch.not_started")); }
    startIt.hidden = !notStarted;
    state.textContent = text;
  }
  stop.addEventListener("click", async () => {
    // A run with no saved point yet cannot be continued after Stop: say so before it is lost.
    const noRestart = status.last_restart === null || status.last_restart === undefined;
    if (noRestart && !status.restartable && !window.confirm(t("watch.stop_confirm"))) return;
    stop.disabled = true;
    try {
      const reply = await api.post(`${api.runPath(id)}/stop`);
      notice(reply.ok ? t("watch.stopping") : `${reply.message} ${reply.fix || ""}`.trim());
    } catch (err) {
      notice(`${err.message} ${err.fix || ""}`.trim());
    } finally { stop.disabled = false; }
  });
  let timer = null;
  async function poll() {
    if (closed) return;
    try {
      const live = await api.get(`${api.runPath(id)}/live`);
      // The whole status, as the server reads it now: a merge would keep keys that no longer apply (a Stop asked
      // of the segment before a Continue).
      if (live.status) status = live.status;
      if (newer(live.progress, progress)) progress = live.progress;
      pictures = live.pictures || null;
      behind = live.pictures_behind || null;
      view.update((live.looks || []).map((l) => ({ ...l, title: titles[l.id] || l.title })));
      paint();
    } catch (_) { /* the run folder may still be writing; the next poll reads it */ }
    if (!closed) timer = setTimeout(poll, OVER_STATES.includes(status.state) && !behind ? IDLE_POLL_MS : POLL_MS);
  }
  const close = api.follow(id, {
    status: (st) => { status = st; paint(); },
    progress: (p) => { progress = p; paint(); },
    frame: () => { clearTimeout(timer); timer = setTimeout(poll, 800); },
    end: () => { clearTimeout(timer); timer = setTimeout(poll, 500); },
  });
  paint();
  poll();
  return () => { closed = true; clearTimeout(timer); close(); view.stop(); };
}

// ------------------------------------------------------------------ advanced: Run

async function advancedRun(body, id, page) {
  const clockEl = headerClock("model time");
  let live = await api.get(`${api.runPath(id)}/live`);
  let status = live.status || {};
  let progress = live.progress || {};
  const commands = live.commands || {};

  // Progress and the buttons
  const stateLine = h("div", { class: "runline" });
  const bar = progressBlock({ of: (total) => `of ${total}`, percent: (n) => `${n}% done`,
    left: (wall) => `about ${wall} left`, leftUnknown: "time left: measuring" });
  const frameCopy = {
    firstStart: (v) => `First frame at ${v.clock} (the starting state), once the GPU has started; then one every ${v.interval}.`,
    firstAt: (v) => `First frame at ${v.clock}.`,
    firstIn: (v) => `First frame at ${v.clock}, in about ${v.wall}.`,
    next: (v) => `Next frame at ${v.clock}.`,
    nextIn: (v) => `Next frame at ${v.clock}, in about ${v.wall}.`,
    drawing: (v) => `Frame at ${v.clock}: its quick-looks are being drawn.`,
    firstDrawing: (v) => `First frame at ${v.clock}: its quick-looks are being drawn.`,
  };
  const readouts = h("div", { class: "readouts" });
  const attached = h("div", {});
  const warnings = h("div", {});
  const until = h("input", { class: "input num", style: "width:110px", placeholder: "T+2:30:00",
    "aria-label": "Continue to (storm time, T+H:MM:SS)" });
  const stopped = { at: null };
  const startB = actionButton(id, "start", "Start", { kind: "primary", command: commands.start, always: true,
    done: (reply) => { if (reply.ok) refresh(); } });
  const stop = actionButton(id, "stop", "Stop", { kind: "stop", command: commands.stop, always: true,
    done: () => { stopped.at = Date.now(); refresh(); } });
  const pause = actionButton(id, "pause", "Pause at next restart", { command: commands.pause, done: () => refresh() });
  const force = actionButton(id, "force-stop", "Force stop", { kind: "stop", command: commands.force_stop,
    confirm: "Force stop kills the run at once. The frame being written is lost; saved frames and restarts are kept.",
    done: () => refresh() });
  const resume = actionButton(id, "resume", "Continue", { kind: "primary", command: commands.resume,
    body: () => (parseClock(until.value) ? { until_s: parseClock(until.value) } : {}), watch: [until],
    check: () => untilProblem(until.value),
    done: (reply) => { if (reply.ok) refresh(); } });
  until.addEventListener("input", () => { until.dataset.touched = "1"; });
  const untilWrap = h("label", { class: "muted untilbox" }, "to storm time ", until);
  const buttons = h("div", { class: "btns runbtns" }, startB, stop, pause, force, resume, untilWrap);
  const stopNote = h("p", { class: "note" });
  const results = h("a", { href: `#/${runRoute("results", id)}` }, "Open Results");
  body.append(h("section", { class: "panel" }, stateLine, bar, readouts, h("div", { class: "section" }, buttons),
    stopNote, attached), warnings);

  // Latest quick-looks
  const view = viewer({ thumbs: true, empty: "The first quick-look appears once the first frame is saved and drawn.",
    note: (look, f) => stormLine(f) });
  const viewCap = h("h2", { class: "cap" }, "Latest quick-looks");
  const looksSection = h("section", { class: "panel section" }, viewCap, view, h("p", { class: "note" },
    "Drawn on the CPU after each frame lands (nice 19, no GPU); the run never waits for them. ", results));
  // A run that has not started, or ended before its first frame, has no pictures or charts: one line says so.
  const framelessNote = h("p", { class: "note section", hidden: true });
  body.append(looksSection, framelessNote);

  // The time series
  const seriesBox = h("div", {});
  const seriesSection = h("section", { class: "panel section" }, h("h2", { class: "cap" }, "Over time",
    h("span", { class: "n" }, "domain extremes at every saved frame")), seriesBox);
  body.append(seriesSection);

  // Start-up, segments, the log, the event stream
  const startupBox = h("div", {});
  const segmentsBox = h("div", {});
  const logBox = h("div", {});
  const events = h("div", { class: "log events mono" });
  const more = h("div", { class: "section" },
    fold("Start-up", "", startupBox), fold("Segments", "", segmentsBox), fold("Log", "", logBox),
    fold("Events", "as they arrive", events));
  body.append(more);
  const commandsLog = h("p", { class: "note" });
  body.append(commandsLog);

  function paintProgress() {
    const st = status;
    const p = progress;
    const model = p.model_time_s ?? st.model_time_s;
    clockEl.set(model || 0, st.duration_s || p.duration_s ? `of ${clock(st.duration_s || p.duration_s)}` : "model time");
    const grid = gridWords(st.grid || (live.admitted || {}).grid);
    page.setData([id, live.case, grid, schemeName(st.microphysics || (live.admitted || {}).microphysics)].filter(Boolean).join(" · "));
    const steps = p.step != null && p.total_steps ? ` · step ${num(p.step)} of ${num(p.total_steps)}` : "";
    const started = (live.admitted || {}).time_utc ? Date.parse(live.admitted.time_utc) : null;
    stateLine.replaceChildren(stateChip(st), h("span", { class: "muted" }, steps,
      started && LIVE_STATES.includes(st.state) ? ` · started ${wallWords((Date.now() - started) / 1000)} ago` : "",
      st.host ? ` · ${st.host}` : ""));
    const total = p.duration_s ?? st.duration_s;
    const leftS = st.state === "running" ? (p.eta_s ?? st.eta_s ?? null) : null;
    const next = pictureWords(live.pictures, { model: model || 0, total, left: leftS, going: st.state === "running" },
      frameCopy);
    const first = Boolean(live.pictures) && (live.pictures.shown_s === null || live.pictures.shown_s === undefined);
    bar.set({ model: model || 0, total, percent: st.percent, leftS, going: LIVE_STATES.includes(st.state),
      running: st.state === "running", next: first ? "" : next });
    if (first && next) view.setEmpty(next);
    readouts.replaceChildren(...advancedReadouts(live, st, p));
    const actions = new Set(st.actions || []);
    startB.hidden = !actions.has("start");
    stop.hidden = !actions.has("stop");
    pause.hidden = !actions.has("pause");
    force.hidden = !(st.force_stop_offered && actions.has("force-stop"));
    resume.hidden = !actions.has("resume");
    untilWrap.hidden = resume.hidden;
    if (!until.dataset.touched) until.value = continueClock({ ...st, duration_s: st.duration_s || (live.admitted || {}).duration_s });
    let words = "";
    if (st.stop_requested_s_ago != null && LIVE_STATES.includes(st.state)) {
      words = `Stop asked ${Math.round(st.stop_requested_s_ago)} s ago: the run finishes the frame being written, ` +
        "then ends." + (st.local ? " If it has not ended 30 s after Stop it is sent SIGTERM; Force stop is offered after 60 s." : "");
    } else if (LIVE_STATES.includes(st.state)) {
      words = "Stop finishes the frame being written, then ends the run. Pause stops right after the next restart " +
        "is written, so Continue picks up there. Closing this page does not stop the run.";
    } else if (actions.has("resume")) {
      words = st.state === "finished" ? "Continue runs it longer from its last restart, to the end time given." :
        "Continue goes on from the last restart, to the end time given.";
    } else if (actions.has("start")) {
      words = "Start runs it on this computer's GPU, with the cm1out and quick-look workers on its CPUs. The Estimate " +
        "page says what it needs first.";
    } else if (st.state === "stopped" && st.final_record === false) {
      words = "The run's process is gone and its log has no final record. The frames saved before are kept.";
    }
    stopNote.textContent = words;
  }

  function paintLive() {
    attached.replaceChildren(attachedLine(live) || "");
    warnings.replaceChildren(warningsBlock(live.warnings) || "");
    if (live.end && live.end.reason && !["completed", "stopped", "paused"].includes(live.end.reason)) {
      warnings.append(h("p", { class: "banner stop" }, `The run ended: ${live.end.reason}`,
        live.end.message ? h("span", { class: "sub" }, live.end.message) : null,
        live.refused ? h("span", { class: "sub" }, `${live.refused.reason || ""} ${live.refused.remedy || ""}`) : null));
    }
    const notStarted = ["imported", "ready"].includes(status.state);
    const frameless = !live.frames_drawn && !status.frames && (notStarted || OVER_STATES.includes(status.state));
    looksSection.hidden = frameless;
    seriesSection.hidden = frameless;
    framelessNote.hidden = !frameless;
    framelessNote.textContent = !frameless ? "" : notStarted
      ? "No quick-looks or charts yet: they appear here once the run has started and saved its first frame."
      : "This run ended before it saved a frame, so it has no quick-looks or charts. The log below says why it ended.";
    view.update(live.looks || []);
    const shown = (live.looks || []).find((l) => l.url);
    put(viewCap, "Latest quick-looks", shown ? h("span", { class: "n" },
      `${live.frames_drawn || 0} frame${live.frames_drawn === 1 ? "" : "s"} drawn, the latest ${clock(shown.time_s || 0)}`) : null);
    seriesBox.replaceChildren(seriesPanel(live.series, { now: status.model_time_s, tmax: status.duration_s, limit: 6 }));
    startupBox.replaceChildren(startupTable(live.startup) || h("p", { class: "muted" }, "Written right before the first step."));
    segmentsBox.replaceChildren(segmentsTable(live.segments) || h("p", { class: "muted" }, "None yet."));
    logBox.replaceChildren(...[logPanel(live.log)].flat().filter(Boolean));
    commandsLog.replaceChildren();
    if (live.commands_log) {
      commandsLog.append("Every command this GUI ran for this run is in ", h("a", { href: live.commands_log,
        target: "_blank", rel: "noopener" }, "commands.log"), ", ready to replay from a terminal. ",
      showCommand(commands.status) || "");
    }
    for (const b of [startB, stop, pause, force, resume]) b.setCommand(commands[b.dataset.action.replace("-", "_")] || null);
  }

  let closed = false;
  let timer = null;
  async function refresh() {
    clearTimeout(timer);
    if (closed) return;
    try {
      live = await api.get(`${api.runPath(id)}/live`);
      // Replaced, not merged: "Stop asked N s ago" and Force stop belong to the segment they were asked of, and a
      // Continue's new segment must not inherit them.
      if (live.status) status = live.status;
      if (newer(live.progress, progress)) progress = live.progress;
      paintLive();
      paintProgress();
    } catch (_) { /* the next poll reads it */ }
    if (!closed) timer = setTimeout(refresh, OVER_STATES.includes(status.state) ? IDLE_POLL_MS : POLL_MS);
  }
  let painting = false;
  const repaint = () => {
    if (painting) return;
    painting = true;
    requestAnimationFrame(() => { painting = false; paintProgress(); });
  };
  const close = api.follow(id, {
    status: (st) => { status = st; repaint(); },
    progress: (p) => { progress = p; repaint(); },
    frame: () => { clearTimeout(timer); timer = setTimeout(refresh, 800); },
    end: () => { clearTimeout(timer); timer = setTimeout(refresh, 500); },
    any: (kind, data) => {
      if (kind === "status") return;
      const text = kind === "progress" ? `step ${data.step ?? ""}${data.heartbeat ? " (heartbeat)" : ""}` :
        kind === "frame" ? `frame ${data.index} at ${clock(data.model_time_s || 0)}` :
        kind === "restart" ? `restart ${data.number} at ${clock(data.model_time_s || 0)}` :
        kind === "end" ? `end: ${data.reason}` : kind === "warning" ? `warning: ${data.code || ""}` : kind;
      events.prepend(h("div", {}, `${data.segment || ""}  ${text}`));
      while (events.childNodes.length > 200) events.lastChild.remove();
    },
  });
  paintLive();
  paintProgress();
  timer = setTimeout(refresh, POLL_MS);
  return () => { closed = true; clearTimeout(timer); close(); view.stop(); };
}

// ------------------------------------------------------------------ the screen

export async function renderRun(body, args, page = {}) {
  const id = decodeURIComponent(args[0] || "");
  const mode = page.mode || api.currentMode();
  const frame = { setData: page.setData || ((text) => { const d = document.getElementById("data"); if (d) d.textContent = text; }) };
  if (!id) {
    body.append(h("p", { class: "banner warn" }, "No run named.", h("span", { class: "sub" }, "Open one from Runs.")));
    return null;
  }
  const known = await runExists(id);
  if (!known.ok) return runNotFound(body, id, known.err);
  return mode === "guided" ? guidedRun(body, id, frame) : advancedRun(body, id, frame);
}

register("run", (body, args, page) => renderRun(body, args, page),
  { title: (mode) => (mode === "guided" ? "Watching" : "Run"), runPage: true });
