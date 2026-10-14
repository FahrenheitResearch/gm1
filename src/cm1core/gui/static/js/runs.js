// Screen 1, Runs: every run folder under the runs folder, and the GPU strip.
//
// Runs started from a terminal or a batch job are listed as well (attach): the list is every folder
// with a run.json under the runs folder, and plain gpuwm-cm1 cartesian output folders, view-only. A run on
// another host reads Stale when it has written nothing for 60 s, and "Stopped without a final record" once its
// batch job is no longer listed. Guided mode shows the same list in fewer columns.
//
// Every row says how much disk the run uses, and the line over the list says the total. Delete removes a run's
// folder; "Keep pictures only" removes its big files (the cm1out files, the saved frames and restarts) and keeps
// the pictures. Both ask first, naming the run and its size, and neither is offered while the run is going.
// Under Windows (WSL2) the runs folder is also named as File Explorer sees it, with a button that opens it.
//
// Open CM1 output: CM1's own netCDF output (a folder of cm1out files, or one file) anywhere on this computer,
// opened for pictures. The files stay where they are; a folder in the runs list names them, opens view-only and
// shows the looks, films and slices drawn from them (POST /api/cm1-output, gpuwm-cm1 quicklook FILES).

import { h, num, stormClock, bytes, button, table, meter, copyline, status as statusChip } from "./core.js";
import * as api from "./api.js";
import { register, go, notice, homeRoute, openRunRoute } from "./router.js";
import { wallWords } from "./run/parts.js";
import { folderLine } from "./folder.js";

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

// The words the two delete buttons ask with: the run by name, and its size.
export function deleteWords(name, size, keepPictures) {
  return keepPictures
    ? `Delete the big files of ${name} and keep its pictures?\n\nIt uses ${size} now; nearly all of that is freed. ` +
      "Its pictures and its results page stay. Its data files go, so it can no longer be run longer, and no new " +
      "pictures or slices can be drawn from it. This cannot be undone."
    : `Delete ${name}?\n\nIt uses ${size}. Its whole folder is removed: the pictures, the files and its settings. ` +
      "This cannot be undone.";
}

async function render(body, args, page) {
  const strips = h("div", { class: "strips" });
  const where = h("div", { class: "section runs-where" });
  const holder = h("div", { class: "section" });
  const foot = h("div", { class: "section" });
  body.append(strips, where, holder, foot);
  const newRun = button("New run", { kind: "primary", onclick: () => go(homeRoute() === "guided" ? "guided" : "new-run") });
  const openForm = cm1OutputForm();
  const openOutput = button("Open CM1 output", { title: "pictures of CM1's own netCDF output, with nothing to run",
    onclick: () => { openForm.hidden = !openForm.hidden; if (!openForm.hidden) openForm.querySelector("input").focus(); } });
  const right = document.querySelector(".head .right");
  if (right) right.prepend(h("div", { class: "btns" }, openOutput, newRun));
  body.insertBefore(openForm, where);

  let session = {};
  try { session = await api.session(); } catch (_) { /* the list below says what went wrong */ }
  // How the list is made is advanced-mode detail; guided shows the runs alone.
  foot.append(
    h("p", { class: "note advanced-only" }, "Runs started from a terminal or a batch job show here as well: the list is every folder ",
      "with a run.json under ", h("code", {}, session.root || "the runs folder"), ", plain gpuwm-cm1 cartesian ",
      "output folders, which open view-only, and folders of CM1 netCDF output, which open for pictures only. ",
      "A run on another computer reads Stale when it has written nothing for ",
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

  let busy = false;
  // Delete a run, or its big files: ask first (the run by name, and its size), then say what was freed.
  async function remove(r, keepPictures, picturesOnly) {
    if (busy) return;
    if (picturesOnly) {
      // CM1 output opened for pictures: only its pictures go, never CM1's files
      if (!window.confirm(`Remove ${r.name} from the list?\n\nIts pictures (${bytes(r.disk_bytes || 0)}) are ` +
        `deleted. CM1's files in ${(r.cm1_output || {}).source || "their folder"} are not touched.`)) return;
    } else if (!window.confirm(deleteWords(r.name, bytes(r.disk_bytes || 0), keepPictures))) return;
    busy = true;
    try {
      const reply = await api.post(`${api.runPath(r.id)}/delete`, keepPictures ? { keep_pictures: true } : {});
      notice(`${reply.message} ${bytes(reply.freed_bytes || 0)} freed.${reply.note ? ` ${reply.note}` : ""}`);
    } catch (err) {
      notice(`${err.message} ${err.fix || ""}`.trim(), "warn");
    } finally { busy = false; }
    try { await draw(); } catch (_) { /* the next tick draws it */ }
  }

  let shownWhere = null;
  async function draw() {
    const d = await api.get("/api/runs");
    const mode = api.currentMode();
    // Guided names the runs folder by its last part; advanced gives the whole path.
    const short = mode === "guided" ? shortPath(d.root) : d.root;
    page.setData([short, `${d.runs.length} ${d.runs.length === 1 ? "run" : "runs"}`,
      d.runs.length ? `${bytes(d.total_bytes || 0)} on disk` : "",
      mode === "guided" ? "" : d.host || session.host || ""].filter(Boolean).join(" · "));
    // Where the runs are, as this computer's file manager names it; drawn again only when it changes, so a
    // "Copied" on its button is not wiped by the next tick.
    const key = [d.root, d.root_windows || "", d.can_open_folder ? "1" : "", mode].join("|");
    if (key !== shownWhere) {
      shownWhere = key;
      where.replaceChildren(folderLine({ label: "Your runs are in", path: d.root, windows: d.root_windows,
        canOpen: d.can_open_folder }));
    }
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
      const model = st.model_time_s != null ? stormClock(st.model_time_s, st.duration_s || st.model_time_s) : "";
      const fallback = r.cm1_output ? `results/${r.url}` : r.view_only ? `run/${r.url}` : ["imported", "ready"].includes(st.state) ?
        (mode === "guided" ? `estimate/${r.url}` : `check/${r.url}`) : `run/${r.url}`;
      const target = openRunRoute(r, mode, fallback);
      const sub = [];
      if (r.cm1_output) sub.push("pictures only");
      else if (r.view_only) sub.push("view only");
      if (r.segments > 1) sub.push(`${r.segments} segments`);
      if (st.state === "stale" && st.last_record_age_s != null) sub.push(`last word ${wallWords(st.last_record_age_s)} ago`);
      const job = r.scheduler && r.scheduler.job_id ? `${r.scheduler.kind || "job"} ${r.scheduler.job_id}` : "";
      const tr = h("tr", { class: "link", tabindex: "0", dataset: { run: r.id, state: st.state || "" },
        onclick: () => go(target) },
      h("td", { class: "runname" }, h("a", { href: `#/${target}` }, r.name)),
      h("td", { class: "wide-only" }, r.case || ""),
      h("td", { class: "nowrap wide-only advanced-only" }, grid, dx ? h("span", { class: "sub" }, dx) : null),
      h("td", {}, statusChip(st.state || "imported", st.label || ""), sub.length ? h("span", { class: "sub" }, sub.join(" · ")) : null),
      h("td", { class: "num" }, model, st.duration_s ? h("span", { class: "sub" }, `of ${stormClock(st.duration_s, st.duration_s)}`) : null),
      h("td", { class: "num wide-only advanced-only" }, r.wall_s != null ? wallWords(r.wall_s) : ""),
      h("td", { class: "num rundisk" }, r.disk_bytes != null ? bytes(r.disk_bytes) : "",
        r.trimmed ? h("span", { class: "sub" }, "pictures only") : null),
      h("td", { class: "wide-only advanced-only" }, r.host || "", job ? h("span", { class: "sub" }, job) : null),
      h("td", { class: "runacts" }, r.cm1_output && r.cm1_output.opened ? deleteButtons(r, st, true) :
        r.view_only ? "" : deleteButtons(r, st)));
      tr.addEventListener("keydown", (ev) => { if (ev.key === "Enter" && ev.target === tr) go(target); });
      return tr;
    });
    holder.replaceChildren(table(["Name", { label: "Case", cls: "wide-only" },
      { label: "Grid", cls: "wide-only advanced-only" }, "Status",
      { label: mode === "guided" ? "Storm time" : "Model time", num: true },
      { label: "Wall", num: true, cls: "wide-only advanced-only" },
      { label: mode === "guided" ? "Size" : "Disk", num: true },
      { label: "Host", cls: "wide-only advanced-only" },
      { label: "", cls: "runacts" }], rows, { stack: true, cls: "runs" }));
  }

  // A run that is going is stopped first; a run on another computer that went quiet may still be going.
  // CM1 output opened for pictures has one button: it removes the pictures, never CM1's files.
  function deleteButtons(r, st, picturesOnly) {
    const going = ["running", "waiting", "stale"].includes(st.state);
    const why = going ? "Stop it first: a storm that is running is not deleted." : null;
    const trim = button("Keep pictures only", { small: true, disabled: going, title: why ||
      "Delete the big files, keep the pictures" });
    trim.dataset.action = "trim";
    trim.hidden = Boolean(r.trimmed) || ["imported", "ready"].includes(st.state) || Boolean(picturesOnly);
    const del = button("Delete", { small: true, kind: "stop", disabled: going, title: why || "Delete this run's folder" });
    del.dataset.action = "delete";
    if (picturesOnly) del.title = "Remove these pictures from the list; CM1's files are not touched";
    for (const [b, keep] of [[trim, true], [del, false]]) {
      b.addEventListener("click", (ev) => { ev.stopPropagation(); ev.preventDefault(); remove(r, keep, picturesOnly); });
      b.addEventListener("keydown", (ev) => ev.stopPropagation());
    }
    return h("div", { class: "btns" }, trim, del);
  }

  // Open CM1 output: the path of CM1's netCDF output (a folder, or one file) and a name for it in the list.
  function cm1OutputForm() {
    const path = h("input", { class: "input", type: "text", placeholder: "/path/to/the/cm1/run (or …/cm1out.nc)",
      "aria-label": "Folder or file of CM1 output", spellcheck: "false", style: "width:100%" });
    const name = h("input", { class: "input", type: "text", placeholder: "a name for the list",
      "aria-label": "Name", spellcheck: "false", style: "width:260px" });
    // clear of the button above it, at any width (a refusal is a block under it, never over it)
    const reply = h("div", { class: "note", style: "margin-top:12px" });
    path.addEventListener("input", () => {
      if (name.dataset.typed) return;
      const parts = path.value.trim().replace(/[\\/]+$/, "").split(/[\\/]+/).filter(Boolean);
      let last = parts.length ? parts[parts.length - 1] : "";
      if (/\.nc$/i.test(last)) last = parts.length > 1 ? parts[parts.length - 2] : last.replace(/\.nc$/i, "");
      name.value = last.replace(/[^A-Za-z0-9._-]+/g, "-").replace(/^[^A-Za-z0-9]+/, "").slice(0, 64);
    });
    name.addEventListener("input", () => { name.dataset.typed = "1"; });
    const open = button("Open for pictures", { kind: "primary" });
    open.addEventListener("click", async () => {
      open.disabled = true;
      put(reply, "Checking the files…");
      try {
        const r = await api.post("/api/cm1-output", { path: path.value.trim(), name: name.value.trim() });
        put(reply, r.message, h("code", { class: "sub" }, r.command || ""));
        notice(r.message);
        go(`results/${r.url}`);
      } catch (err) {
        put(reply, h("div", { class: "banner stop" }, err.message, err.fix ? h("span", { class: "sub" }, err.fix) : null));
      } finally { open.disabled = false; }
    });
    const form = h("section", { class: "panel section cm1-output-form", hidden: true },
      h("h2", { class: "cap" }, "Open CM1 output", h("span", { class: "n" }, "pictures only")),
      h("p", { class: "text" }, "CM1's own netCDF output (cm1out.nc, or one cm1out_NNNNNN.nc per saved time), or a ",
        "GM1 run's cm1out folder: GM1 draws its looks, films and slices on the computer's processor. Nothing is run, ",
        "and the files stay where they are."),
      h("label", { class: "muted" }, "Folder or file", path),
      h("div", { class: "btns section" }, h("label", { class: "muted" }, "Name ", name), open),
      reply,
      h("p", { class: "note" }, "With the run's namelist.input beside the files, a hurricane run is drawn with the ",
        "hurricane looks; files on stretched levels are drawn at their own heights."));
    return form;
  }

  function put(el, ...kids) { el.replaceChildren(...kids.filter((k) => k !== null && k !== undefined && k !== "")); }

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
