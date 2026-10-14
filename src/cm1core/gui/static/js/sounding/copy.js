// The Use my own sounding card's words: every string it shows comes from gui/copy/sounding.json (served at
// /api/sounding/copy), so the copy lint covers them all. CM1's own names appear only in a small grey hint.

import { h, paren } from "../core.js";
import * as api from "../api.js";

let copy = null;
let loading = null;

export async function ready() {
  if (copy) return copy;
  if (!loading) {
    loading = api.get("/api/sounding/copy").then((c) => { copy = c; return c; });
    loading.catch(() => { loading = null; });
  }
  return loading;
}

export function t(key, vars) {
  const row = copy && copy.strings[key];
  let text = row ? row.text : key;
  for (const [k, v] of Object.entries(vars || {})) text = text.split(`{${k}}`).join(String(v));
  return text;
}

export function hint(key) {
  const row = copy && copy.strings[key];
  return row && row.hint ? h("span", { class: "hint cm1" }, paren(row.hint)) : null;
}
