// "Show command": every action in the GUI is one gpuwm-cm1 command. This is the one component that
// shows it: a link that reveals the exact line, copyable. Advanced mode puts one beside every button; guided mode
// shows it on Start and on Results (always = true).
//
//   showCommand(line)            a line known before the press (a run's Stop, Start, Estimate ...)
//   liveCommand(request)         a line that depends on what the page holds (a form's Save): opening it asks the
//                                server for the exact line with {"dry_run": true}, so nothing runs and nothing
//                                changes, and it follows the inputs while it is open
//   actionButton(label, opts)    a button with its Show command beside it

import { h, copyline, button, append } from "./core.js";
import * as api from "./api.js";

export function showCommand(command, opts = {}) {
  if (!command) return null;
  const always = opts.always === true;
  const line = copyline(command, opts.title || "The gpuwm-cm1 command this runs");
  line.hidden = !opts.open;
  const label = opts.label || "Show command";
  const toggle = h("button", { type: "button", "aria-expanded": opts.open ? "true" : "false" },
    opts.open ? "Hide command" : label);
  toggle.addEventListener("click", () => {
    line.hidden = !line.hidden;
    toggle.textContent = line.hidden ? label : "Hide command";
    toggle.setAttribute("aria-expanded", line.hidden ? "false" : "true");
  });
  const box = h("div", { class: `showcmd${always ? "" : " advanced-only"}`, dataset: { command } }, toggle, line);
  return box;
}

// request() returns {path, body} for the POST the button makes, null when there is nothing to do yet, or
// {error} with a line saying what to fix. opts.watch: elements (or a container) whose input changes redraw it.
export function liveCommand(request, opts = {}) {
  const always = opts.always === true;
  const label = opts.label || "Show command";
  const out = h("div", { class: "livecmd" });
  out.hidden = true;
  const toggle = h("button", { type: "button", "aria-expanded": "false" }, label);
  const box = h("div", { class: `showcmd live${always ? "" : " advanced-only"}`, dataset: { command: "" } }, toggle, out);
  let ticket = 0;
  const say = (text) => h("p", { class: "note cmdnote" }, text);
  async function refresh() {
    if (out.hidden) return;
    const mine = ++ticket;
    let req;
    try { req = request(); } catch (err) { req = { error: err.message }; }
    if (!req || req.error) {
      box.dataset.command = "";
      out.replaceChildren(say((req && req.error) || opts.empty || "Change something first; the command shows here."));
      return;
    }
    out.replaceChildren(say("Asking the server for the exact line…"));
    try {
      const reply = await api.dryRun(req.path, req.body || {});
      if (mine !== ticket) return;
      box.dataset.command = reply.command || "";
      out.replaceChildren(copyline(reply.command, opts.title || "The gpuwm-cm1 command this runs"));
    } catch (err) {
      if (mine !== ticket) return;
      const shown = err.command || (err.body && err.body.command_template) || "";
      box.dataset.command = err.command || "";
      out.replaceChildren();
      append(out, [shown ? copyline(shown, "The gpuwm-cm1 command this runs") : null,
        say(`${err.message} ${err.fix || ""}`.trim())]);
    }
  }
  let timer = null;
  const soon = () => { clearTimeout(timer); timer = setTimeout(refresh, 250); };
  for (const el of [].concat(opts.watch || [])) {
    if (!el) continue;
    el.addEventListener("input", soon);
    el.addEventListener("change", soon);
  }
  toggle.addEventListener("click", () => {
    out.hidden = !out.hidden;
    toggle.textContent = out.hidden ? label : "Hide command";
    toggle.setAttribute("aria-expanded", out.hidden ? "false" : "true");
    refresh();
  });
  box.refresh = refresh;
  return box;
}

// A button with its "Show command" under it. opts: {action, kind, small, disabled, title, onclick,
// command (a known line) | request (a live one), watch, always}. The wrapper's .button is the button itself.
export function actionButton(label, opts = {}) {
  const b = button(label, { kind: opts.kind, small: opts.small, disabled: opts.disabled, title: opts.title,
    onclick: opts.onclick });
  if (opts.action) b.dataset.action = opts.action;
  const cmd = opts.command ? showCommand(opts.command, { always: opts.always }) :
    opts.request ? liveCommand(opts.request, { always: opts.always, watch: opts.watch, empty: opts.empty }) : null;
  const wrap = h("span", { class: "action", dataset: { action: opts.action || "" } }, b, cmd);
  wrap.button = b;
  wrap.cmd = cmd;
  return wrap;
}

// A holder whose content is replaced by the command of the last action a screen ran.
export function commandSlot(opts = {}) {
  const slot = h("div", { class: "cmdslot" });
  slot.set = (command) => {
    slot.replaceChildren();
    const shown = showCommand(command, opts);
    if (shown) slot.append(shown);
  };
  return slot;
}
