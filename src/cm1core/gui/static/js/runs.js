// Screen 1, Runs: every run folder under the runs folder, and the GPU strip.
//
// Runs started from a terminal or a batch job are listed as well (attach): the list is every folder
// with a run.json under the runs folder, and plain gpuwm-cm1 cartesian output folders, view-only. A run on
// another host reads Stale when it has written nothing for 60 s, and "Stopped without a final record" once its
// batch job is no longer listed. Guided mode shows the same list in fewer columns.

import { h, num, clock, bytes, button, table, meter, copyline, status as statusChip } from "./core.js";
import * as api from "./api.js";
import { register, go, homeRoute, openRunRoute } from "./router.js";
import { wallWords } from "./run/parts.js";

const EVERY_MS = 3000;

// A folder by its last part ("…/gm1-runs"), for guided mode's header line.
function shortPath(path) {
  if (!path) return "";
  const parts = String(path).split(/[\\/]+/).filter(Boolean);
  return parts.length > 1 ? `…/${parts[parts.length - 1]}` : String(path);
}

function gpuStrip(g) {
  const total = g.total_bytes || 1;
  const inUse = g.in_use_bytes ?? g.used_bytes ?? 0;
  const mine = g.gm1_bytes || 0;
  const other = Math.max(0, inUse - mine);
  const gib = (v) => (v / 1024 ** 3).toFixed(1);
  const runs = (g.gm1_runs || []).map((r) =>
    `${r.name}${r.used_bytes ? ` ${gib(r.used_bytes)} GiB` : ""}${r.state === "waiting" ? " (waiting)" : ""}`);
  return h("section", { class: "panel gpu-strip", dataset: { gpu: String(g.index) } },
    h("span", { class: "cap", style: "margin:0" }, `GPU ${g.index}`),
    h("span", { class: "text" }, g.name),
    h("span", { class: "muted advanced-only" }, [g.driver ? `driver ${g.driver}` : null, g.sm].filter(Boolean).join(" · ")),
    meter([{ cls: "other", from: 0, to: other / total }, { cls: "need", from: other / total, to: inUse / total }]),
    h("span", {}, `${gib(inUse)} of ${gib(total)} GiB in use`),
    runs.length ? h("span", { class: "muted" }, `GM1: ${runs.join(", ")}`) :
      h("span", { class: "muted" }, "no GM1 run on it"),
    g.note ? h("span", { class: "muted gpu-note" }, g.note) : null);
}

async function render(body, args, page) {
  const strips = h("div", { class: "strips" });
  const holder = h("div", { class: "section" });
  const foot = h("div", { class: "section" });
  body.append(strips, holder, foot);
  const newRun = button("New run", { kind: "primary", onclick: () => go(homeRoute() === "guided" ? "guided" : "new-run") });
  const right = document.querySelector(".head .right");
  if (right) right.prepend(h("div", { class: "btns" }, newRun));

  let session = {};
  try { session = await api.session(); } catch (_) { /* the list below says what went wrong */ }
  // How the list is made is advanced-mode detail; guided shows the runs alone.
  foot.append(
    h("p", { class: "note advanced-only" }, "Runs started from a terminal or a batch job show here as well: the list is every folder ",
      "with a run.json under ", h("code", {}, session.root || "the runs folder"), ", and plain gpuwm-cm1 cartesian ",
      "output folders, which open view-only. A run on another computer reads Stale when it has written nothing for ",
      "60 s, and Stopped without a final record once its batch job is no longer listed."),
    h("div", { class: "advanced-only" }, h("p", { class: "note" }, "The same list from a terminal:"),
      copyline(`gpuwm-cm1 status --root ${session.root || "RUNS"}`, "status")));

  async function drawGpus() {
    try {
      const g = await api.get("/api/gpus");
      if (!g.gpus.length) {
        strips.replaceChildren(h("p", { class: "banner warn" }, g.message || "No NVIDIA GPU found on this computer.",
          h("span", { class: "sub" }, "Runs started on a cluster still show here; System says what is missing.")));
        return;
      }
      strips.replaceChildren(...g.gpus.map(gpuStrip));
    } catch (_) { /* the next tick tries again */ }
  }

  async function draw() {
    const d = await api.get("/api/runs");
    const mode = api.currentMode();
    // Guided names the runs folder by its last part; advanced gives the whole path.
    const where = mode === "guided" ? shortPath(d.root) : d.root;
    page.setData([where, `${d.runs.length} ${d.runs.length === 1 ? "run" : "runs"}`,
      mode === "guided" ? "" : d.host || session.host || ""].filter(Boolean).join(" · "));
    if (!d.runs.length) {
      holder.replaceChildren(h("p", { class: "text" }, "No runs in this folder yet."),
        h("div", { class: "btns" }, button(mode === "guided" ? "Pick a storm" : "New run", {
          kind: "primary", onclick: () => go(homeRoute() === "guided" ? "guided" : "new-run") })));
      return;
    }
    const rows = d.runs.map((r) => {
      const st = r.status || {};
      const grid = r.grid ? `${r.grid.nx} × ${r.grid.ny} × ${r.grid.nz}` : "";
      const dx = r.grid && r.grid.dx_m ? `dx ${num(r.grid.dx_m)} m` : "";
      const model = st.model_time_s != null ? clock(st.model_time_s) : "";
      const fallback = r.view_only ? `run/${r.url}` : ["imported", "ready"].includes(st.state) ?
        (mode === "guided" ? `estimate/${r.url}` : `check/${r.url}`) : `run/${r.url}`;
      const target = openRunRoute(r, mode, fallback);
      const sub = [];
      if (r.view_only) sub.push("view only");
      if (r.segments > 1) sub.push(`${r.segments} segments`);
      if (st.state === "stale" && st.last_record_age_s != null) sub.push(`last word ${wallWords(st.last_record_age_s)} ago`);
      const job = r.scheduler && r.scheduler.job_id ? `${r.scheduler.kind || "job"} ${r.scheduler.job_id}` : "";
      const tr = h("tr", { class: "link", tabindex: "0", dataset: { run: r.id, state: st.state || "" },
        onclick: () => go(target) },
      h("td", { class: "runname" }, h("a", { href: `#/${target}` }, r.name)),
      h("td", { class: "wide-only" }, r.case || ""),
      h("td", { class: "nowrap wide-only advanced-only" }, grid, dx ? h("span", { class: "sub" }, dx) : null),
      h("td", {}, statusChip(st.state || "imported", st.label || ""), sub.length ? h("span", { class: "sub" }, sub.join(" · ")) : null),
      h("td", { class: "num" }, model, st.duration_s ? h("span", { class: "sub" }, `of ${clock(st.duration_s)}`) : null),
      h("td", { class: "num wide-only advanced-only" }, r.wall_s != null ? wallWords(r.wall_s) : ""),
      h("td", { class: "num wide-only advanced-only" }, r.disk_bytes != null ? bytes(r.disk_bytes) : ""),
      h("td", { class: "wide-only advanced-only" }, r.host || "", job ? h("span", { class: "sub" }, job) : null));
      tr.addEventListener("keydown", (ev) => { if (ev.key === "Enter") go(target); });
      return tr;
    });
    holder.replaceChildren(table(["Name", { label: "Case", cls: "wide-only" },
      { label: "Grid", cls: "wide-only advanced-only" }, "Status",
      { label: mode === "guided" ? "Storm time" : "Model time", num: true },
      { label: "Wall", num: true, cls: "wide-only advanced-only" },
      { label: "Disk", num: true, cls: "wide-only advanced-only" },
      { label: "Host", cls: "wide-only advanced-only" }], rows, { stack: true, cls: "runs" }));
  }

  let timer = null;
  let ticks = 0;
  let closed = false;
  const tick = async () => {
    if (closed) return;
    try { await draw(); } catch (err) { holder.replaceChildren(h("p", { class: "banner stop" }, err.message)); }
    if (ticks % 2 === 0) await drawGpus();
    ticks += 1;
    if (!closed) timer = setTimeout(tick, EVERY_MS);
  };
  await tick();
  return () => { closed = true; clearTimeout(timer); };
}

register("runs", render, { label: "Runs", nav: true, title: "Runs", order: 10 });
