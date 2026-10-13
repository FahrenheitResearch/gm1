// Guided mode's words. Every string a guided screen shows comes from gui/copy/guided.json and looks.json
// (served at /api/copy), so one lint covers them all. CM1's own names only ever appear in a small grey hint.

import { h } from "../js/core.js";
import * as api from "../js/api.js";

let copy = null;
let loading = null;

export async function ready() {
  if (copy) return copy;
  if (!loading) {
    loading = api.get("/api/copy").then((c) => { copy = c; return c; });
    loading.catch(() => { loading = null; });
  }
  return loading;
}

function fill(text, vars) {
  let out = text;
  for (const [k, v] of Object.entries(vars || {})) out = out.split(`{${k}}`).join(String(v));
  return out;
}

// t("budget.title"), t("gallery.about", {wall, gpu})
export function t(key, vars) {
  const row = copy && copy.guided.strings[key];
  return fill(row ? row.text : key, vars);
}

// The CM1 name for a string, if it has one (drawn as a hint).
export function hint(key, value) {
  const row = copy && (copy.guided.strings[key] || copy.guided.terms[key]);
  if (!row || !row.hint) return null;
  return h("span", { class: "hint" }, value === undefined || value === null ? row.hint : `${row.hint} = ${value}`);
}

// A plain term and, after it, its CM1 name as a grey hint: [text, hint element].
export function term(key, value) {
  const row = (copy && copy.guided.terms[key]) || { text: key };
  return [row.text, hint(key, value)];
}

export function looks() {
  return (copy && copy.looks.looks) || [];
}

// ---- numbers in words (estimates get two significant figures and "about")

function sig2(x) {
  return Number(Number(x).toPrecision(2)).toLocaleString("en-US");
}

export function wallWords(seconds) {
  if (seconds === null || seconds === undefined || Number.isNaN(seconds)) return "";
  if (seconds < 100) return `${sig2(seconds)} s`;
  if (seconds < 6000) return `${sig2(seconds / 60)} min`;
  return `${sig2(seconds / 3600)} h`;
}

export function spanWords(seconds) {
  const s = Math.round(seconds);
  if (s >= 172800 && s % 3600 === 0) {
    // Two days or more is said in days: "4 days", "3 days 12 h".
    const days = Math.floor(s / 86400), rest = (s % 86400) / 3600;
    return rest ? `${days} days ${rest} h` : `${days} days`;
  }
  if (s % 3600 === 0) return `${s / 3600} h`;
  if (s >= 3600) return `${Math.floor(s / 3600)} h ${Math.round((s % 3600) / 60)} min`;
  if (s % 60 === 0) return `${s / 60} min`;
  return `${s} s`;
}

// A size as the fit's own sentences say it: one decimal under 100 GB, whole GB above (15.8 GB, 212 GB).
export function gb(bytes) {
  if (bytes === null || bytes === undefined) return "";
  const value = bytes / 1e9;
  return value < 100 ? `${value.toFixed(1)} GB` : `${Math.round(value)} GB`;
}

export function metres(m) {
  // Any spacing is said as it is: 67 m, 78.2 m, 1.25 km.
  if (m >= 1000) return `${Number((m / 1000).toFixed(3))} km`;
  return `${Number(Number(m).toFixed(1))} m`;
}

export function shortGpu(name) {
  return String(name || "").replace(/^nvidia\s+(geforce\s+)?/i, "").trim();
}
