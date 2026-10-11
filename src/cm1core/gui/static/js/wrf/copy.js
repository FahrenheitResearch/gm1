// The column route's words: every string the card and the panel show comes from gui/copy/wrf.json (served at
// /api/wrf/copy), so the copy lint covers them all. CM1 and WRF names appear only in a small grey hint.

import { h, paren } from "../core.js";
import * as api from "../api.js";

let copy = null;
let loading = null;

export async function ready() {
  if (copy) return copy;
  if (!loading) {
    loading = api.get("/api/wrf/copy").then((c) => { copy = c; return c; });
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
