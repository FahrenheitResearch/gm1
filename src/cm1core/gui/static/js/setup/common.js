// Helpers the six setup screens share: loading a run's setup, plain-word numbers, CM1 words as grey hints, and
// running one action (a POST whose reply names the gpuwm-cm1 command it ran).

import { h, button, num, paren } from "../core.js";
import * as api from "../api.js";
import { notice, go, runRoute } from "../router.js";
import { showCommand } from "../command.js";

export const MINUS = "−";

export function runId(args) {
  return decodeURIComponent(args[0] || "");
}

export async function loadSetup(id) {
  return api.get(`${api.runPath(id)}/setup`);
}

// CM1's own word, shown small and grey after the plain words.
export function hint(text) {
  return text ? h("span", { class: "hint cm1" }, paren(text)) : null;
}

export function gb(bytes) {
  if (bytes === null || bytes === undefined) return "";
  const v = bytes / 1e9;
  if (v >= 10) return `${v.toFixed(0)} GB`;
  if (v >= 1) return `${v.toFixed(1)} GB`;
  return `${v.toFixed(2)} GB`;
}

export function signed(x, nd = 1) {
  if (x === null || x === undefined || Number.isNaN(x)) return "";
  return num(x, nd);
}

export function durationWords(seconds) {
  if (seconds === null || seconds === undefined) return "";
  if (seconds < 90) return `about ${Math.max(10, Math.round(seconds / 10) * 10)} seconds`;
  const minutes = seconds / 60;
  if (minutes < 90) {
    const m = Math.round(minutes);
    return `about ${m} minute${m === 1 ? "" : "s"}`;
  }
  const hours = (minutes / 60).toFixed(1).replace(/\.0$/, "");
  return `about ${hours} hour${hours === "1" ? "" : "s"}`;
}

export function stormTime(seconds) {
  if (seconds === null || seconds === undefined) return "";
  if (seconds % 3600 === 0) {
    const hrs = seconds / 3600;
    return `${hrs} hour${hrs === 1 ? "" : "s"}`;
  }
  if (seconds >= 3600) return `${(seconds / 3600).toFixed(2).replace(/0+$/, "").replace(/\.$/, "")} hours`;
  return `${Math.round(seconds / 60)} minutes`;
}

export function spacing(metres) {
  if (metres === null || metres === undefined) return "";
  if (metres >= 1000 && metres % 1000 === 0) return `${metres / 1000} km`;
  return `${num(metres, metres % 1 ? 1 : 0)} m`;
}

export function areaWords(v) {
  if (!v || !v.nx || !v.dx) return "";
  const w = (v.nx * v.dx) / 1000;
  const d = ((v.ny || v.nx) * (v.dy || v.dx)) / 1000;
  return `${num(w, w % 1 ? 1 : 0)} × ${num(d, d % 1 ? 1 : 0)} km`;
}

export function gridWords(v) {
  if (!v || !v.nx) return "";
  return `${v.nx} × ${v.ny} × ${v.nz}`;
}

// The estimator's check levels as status classes.
export function levelState(level) {
  return { ok: "ok", info: "note", note: "note", warn: "warn", warning: "warn", fail: "fail", error: "fail",
    stop: "fail", refused: "fail" }[level] || "note";
}

// The data line under a setup screen's title.
export function dataLine(s, extra) {
  const v = s.values || {};
  const bits = [s.id];
  if (v.nx) bits.push(`${gridWords(v)}, dx ${spacing(v.dx)}`);
  if (v.timax) bits.push(stormTime(v.timax));
  const scheme = s.report && s.report.component && s.report.component.scheme;
  if (scheme) bits.push(scheme);
  if (extra) bits.push(extra);
  return bits.join(" · ");
}

// A holder for the result of the last action on a screen: its message and fix, and its command.
export function resultBox() {
  const box = h("div", { class: "result" });
  box.show = (reply, opts = {}) => {
    box.replaceChildren();
    if (!reply) return;
    const ok = reply.ok !== false;
    if (reply.message || reply.fix) {
      box.append(h("p", { class: `banner${ok ? "" : " stop"}` }, reply.message || "",
        reply.fix ? h("span", { class: "sub" }, reply.fix) : null));
    }
    if (opts.diff !== false && reply.diff) box.append(h("pre", { class: "diff" }, reply.diff));
    if (reply.settings_changed && reply.settings_changed.length) {
      box.append(h("pre", { class: "diff" }, reply.settings_changed.map((row) =>
        `settings.toml  ${row.key}: ${row.from === null || row.from === undefined ? "CM1's value" : row.from} → ` +
        `${row.to === null || row.to === undefined ? "CM1's value" : row.to}`).join("\n")));
    }
    const cmd = showCommand(reply.command, { always: opts.always === true });
    if (cmd) box.append(cmd);
  };
  return box;
}

// Run one POST from a button: the button waits, the notice says what happened, the result box shows the command.
export async function act(b, path, body, box, opts = {}) {
  if (b) b.disabled = true;
  try {
    const reply = await api.post(path, body || {});
    if (box) box.show(reply, opts);
    if (reply.ok === false) notice(`${reply.message || "It did not work."} ${reply.fix || ""}`.trim(), "stop");
    else if (opts.notice !== false && reply.message) notice(reply.message);
    return reply;
  } catch (err) {
    if (box) box.show({ ok: false, message: err.message, fix: err.fix, command: err.command }, opts);
    notice(`${err.message} ${err.fix || ""}`.trim(), "stop");
    return null;
  } finally {
    if (b) b.disabled = false;
  }
}

// Estimate again when the estimate is older than the run's namelist (a second, host only).
export async function ensureEstimate(id, s) {
  if (!s.report.present || s.report.exit === 2 || s.has_run) return s;
  if (s.estimate && !s.estimate_stale && s.geometry && !s.geometry_stale) return s;
  try {
    await api.post(`${api.runPath(id)}/estimate`);
    notice(s.estimate ? "Estimated again: the run's namelist changed since the last estimate." :
      "Priced the run for this computer's GPU; nothing ran on it.");
  } catch (err) {
    notice(`${err.message} ${err.fix || ""}`.trim(), "warn");
    return s;
  }
  return loadSetup(id);
}

export function toAdvanced(label = "Show everything") {
  const b = h("button", { type: "button", class: "linkbtn" }, label);
  b.addEventListener("click", async () => {
    try { await api.setMode("advanced"); } catch (err) { notice(err.message, "stop"); }
  });
  return b;
}

export function nextButton(label, page, id, kind = "primary", disabled = false, title = null) {
  return button(label, { kind, disabled, title, onclick: () => go(runRoute(page, id)) });
}

// A plain sentence block for guided mode.
export function lead(...kids) {
  return h("p", { class: "lead" }, ...kids);
}

// A folder that was not made by gpuwm-cm1 import has no setup to show.
export function notImported(body, s) {
  body.append(h("p", { class: "banner warn" },
    "This folder was not made by gpuwm-cm1 import, so it has no namelist or coverage report to show.",
    h("span", { class: "sub" }, "Its run and results open from Runs.")));
  return !s.report.present;
}
