// Small DOM and number helpers shared by every screen. No framework, no network.

export const MINUS = "−";

// h("div", {class: "x", onclick: fn}, child, "text", [more children])
export function h(tag, attrs, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") el.className = v;
    else if (k === "text") el.textContent = v;
    else if (k === "html") el.innerHTML = v;
    else if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2), v);
    else if (k === "dataset") Object.assign(el.dataset, v);
    else el.setAttribute(k, v === true ? "" : String(v));
  }
  append(el, kids);
  return el;
}

export function append(el, kids) {
  for (const kid of kids.flat(Infinity)) {
    if (kid === null || kid === undefined || kid === false) continue;
    el.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
  }
  return el;
}

export function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// a number with a true minus sign and thousands separators
export function num(x, nd = 0) {
  if (x === null || x === undefined || Number.isNaN(x)) return "";
  const s = Math.abs(x).toLocaleString("en-US", { minimumFractionDigits: nd, maximumFractionDigits: nd });
  return (x < 0 && Number(s.replace(/,/g, "")) !== 0 ? MINUS : "") + s;
}

// Model time since the start, one form everywhere: T+H:MM:SS (T+0:05:00, T+1:25:06).
export function clock(t) {
  const s = Math.max(0, Math.round(Number(t) || 0));
  const two = (n) => String(n).padStart(2, "0");
  return `T+${Math.floor(s / 3600)}:${two(Math.floor((s % 3600) / 60))}:${two(s % 60)}`;
}

// A storm time typed in that one form, back to seconds: "T+2:30:00", "2:30:00" and "2:30" (hours and minutes)
// are the same time. null for anything else, or for T+0:00:00.
export function parseClock(text) {
  const m = /^\s*(?:T\+)?\s*(\d+):([0-5]\d)(?::([0-5]\d))?\s*$/i.exec(String(text ?? ""));
  if (!m) return null;
  const seconds = Number(m[1]) * 3600 + Number(m[2]) * 60 + Number(m[3] || 0);
  return seconds > 0 ? seconds : null;
}

// A stored number as a person reads it: the shortest form within `sig` significant digits, trailing zeros
// dropped (1.4000000000000001 is shown as 1.4). The stored value is not changed.
export function numText(x, sig = 7) {
  if (x === null || x === undefined || x === "") return "";
  const n = Number(x);
  if (!Number.isFinite(n)) return String(x);
  if (n === 0) return "0";
  return String(Number(n.toPrecision(sig)));
}

// A microphysics scheme's id as the pages name it ("morrison-mp10" is "Morrison (10)"); a name is kept.
export const SCHEME_NAMES = { "wsm6-mp6": "WSM6 (6)", "thompson-mp8": "Thompson (8)", "morrison-mp10": "Morrison (10)" };
export function schemeName(id) {
  return id ? SCHEME_NAMES[id] || id : "";
}

// A grey hint in brackets; a hint that already is one bracketed phrase is shown as it is.
export function paren(text) {
  const s = String(text);
  if (s.startsWith("(") && s.endsWith(")")) {
    let depth = 0;
    for (let i = 0; i < s.length; i++) {
      if (s[i] === "(") depth++;
      else if (s[i] === ")") depth--;
      if (depth === 0 && i < s.length - 1) return `(${s})`;
    }
    return s;
  }
  return `(${s})`;
}

export function bytes(n) {
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)} GB`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)} MB`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)} KB`;
  return `${n} B`;
}

export function panel(cap, ...kids) {
  return h("section", { class: "panel" }, cap ? h("h2", { class: "cap" }, cap) : null, kids);
}

export function status(state, label) {
  return h("span", { class: `st ${state}` }, label);
}

// one line of a check list: a status square and plain words
export function checkline(state, text) {
  return h("div", { class: "checkline" }, h("span", { class: `st ${state}` }), h("span", {}, text));
}

export function button(label, opts = {}) {
  const cls = ["btn", opts.kind || "", opts.small ? "small" : ""].join(" ").trim();
  const b = h("button", { class: cls, type: "button", disabled: opts.disabled || null, title: opts.title || null }, label);
  if (opts.onclick) b.addEventListener("click", opts.onclick);
  return b;
}

export function table(headers, rows, opts = {}) {
  const thead = h("thead", {}, h("tr", {}, headers.map((c) =>
    h("th", { class: [c.num ? "num" : "", c.cls || ""].join(" ").trim() || null }, c.label ?? c))));
  const tbody = h("tbody", {}, rows);
  const cls = [opts.compact ? "compact" : "", opts.stack ? "stack" : "", opts.cls || ""].join(" ").trim() || null;
  return h("div", { class: "tablewrap" }, h("table", { class: cls }, thead, tbody));
}

export function kv(pairs) {
  return h("dl", { class: "kv" }, pairs.map(([k, v]) => [h("dt", {}, k), h("dd", {}, v)]));
}

export function readout(k, v, unit, sub) {
  return h("div", { class: "readout" },
    h("div", { class: "k" }, k),
    h("div", { class: "v" }, v, unit ? h("small", {}, unit) : null),
    sub ? h("div", { class: "s" }, sub) : null);
}

// a "copy" line: the text in a code box and a Copy button (clipboard only; nothing leaves the page)
// Copy text to the clipboard: the Clipboard API where the page may use it (https or localhost), else a hidden
// textarea and the browser's copy command (a page opened at another computer's address over http). true if copied.
export async function copyText(text) {
  try {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(String(text));
      return true;
    }
  } catch (_) { /* fall through to the textarea */ }
  const area = document.createElement("textarea");
  area.value = String(text);
  area.setAttribute("readonly", "");
  area.style.position = "fixed";
  area.style.top = "-1000px";
  area.style.opacity = "0";
  document.body.append(area);
  area.select();
  let ok = false;
  try { ok = document.execCommand("copy"); } catch (_) { ok = false; }
  area.remove();
  return ok;
}

// A button that copies what text() returns and says whether it did; when the browser refuses, it selects
// `select` (an element holding the text) so Ctrl+C copies it.
export function copyButton(label, text, opts = {}) {
  const b = button(label, { small: opts.small !== false });
  b.addEventListener("click", async () => {
    const ok = await copyText(typeof text === "function" ? text() : text);
    if (!ok && opts.select) {
      const range = document.createRange();
      range.selectNodeContents(opts.select);
      const sel = window.getSelection();
      sel.removeAllRanges();
      sel.addRange(range);
    }
    b.textContent = ok ? "Copied" : "Selected: press Ctrl+C";
    setTimeout(() => { b.textContent = label; }, ok ? 1200 : 4000);
  });
  return b;
}

export function copyline(text, label) {
  const code = h("code", { title: label || null }, text);
  return h("div", { class: "copyline" }, code, copyButton("Copy", text, { select: code }));
}

export function meter(parts) {
  // parts: [{cls, from, to}] in 0..1 of the bar
  return h("div", { class: "meter" }, parts.map((p) =>
    h("i", { class: p.cls, style: `left:${(100 * p.from).toFixed(2)}%;width:${(100 * (p.to - p.from)).toFixed(2)}%` })));
}
