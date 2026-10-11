// The domain map of `gpuwm-cm1 chain inspect --map-png`, where a drag draws the box GM1 runs in. The reply says
// where the domain sits in the picture (domain_px, in the picture's own pixels) and how many cells it has, so a
// drag is turned into whole cells: i0 from the west side and j0 from the south side, both from 0, then the size.
// The three cells along each side that no box may use are shaded in the picture itself.

import { h } from "../core.js";

function cellAt(map, img, event) {
  const box = map.domain_px;
  const rect = img.getBoundingClientRect();
  if (!rect.width || !rect.height) return null;
  const px = ((event.clientX - rect.left) * map.width) / rect.width;
  const py = ((event.clientY - rect.top) * map.height) / rect.height;
  const fx = (px - box.left) / box.width;
  const fy = (box.top + box.height - py) / box.height;
  const nx = map.cells.west_east;
  const ny = map.cells.south_north;
  return [Math.min(nx - 1, Math.max(0, Math.floor(fx * nx))), Math.min(ny - 1, Math.max(0, Math.floor(fy * ny)))];
}

// boxMap(onBox): a figure whose .show(map) draws a map reply, .setBox([i0, j0, ni, nj]) outlines a box, and whose
// drags call onBox([i0, j0, ni, nj]).
export function boxMap(onBox, words) {
  const img = h("img", { alt: words.alt, width: "960", height: "540", draggable: "false" });
  const outline = h("div", { class: "chain-box", "aria-hidden": "true" });
  outline.hidden = true;
  const stage = h("div", { class: "chain-stage" }, img, outline);
  const caption = h("figcaption", { class: "note" });
  const fig = h("figure", { class: "chain-map" }, stage, caption);
  let map = null;
  let box = null;
  let anchor = null;

  const place = () => {
    if (!map || !box) { outline.hidden = true; return; }
    const d = map.domain_px;
    const nx = map.cells.west_east;
    const ny = map.cells.south_north;
    const [i0, j0, ni, nj] = box;
    const left = (d.left + (i0 / nx) * d.width) / map.width;
    const top = (d.top + ((ny - j0 - nj) / ny) * d.height) / map.height;
    outline.style.left = `${(100 * left).toFixed(3)}%`;
    outline.style.top = `${(100 * top).toFixed(3)}%`;
    outline.style.width = `${(100 * (ni / nx) * d.width / map.width).toFixed(3)}%`;
    outline.style.height = `${(100 * (nj / ny) * d.height / map.height).toFixed(3)}%`;
    outline.hidden = false;
  };
  const spanOf = (a, b) => [Math.min(a[0], b[0]), Math.min(a[1], b[1]),
    Math.abs(a[0] - b[0]) + 1, Math.abs(a[1] - b[1]) + 1];

  stage.addEventListener("pointerdown", (event) => {
    if (!map) return;
    const at = cellAt(map, img, event);
    if (!at) return;
    anchor = at;
    box = spanOf(at, at);
    place();
    try { stage.setPointerCapture(event.pointerId); } catch (_) { /* a synthetic event */ }
    event.preventDefault();
  });
  stage.addEventListener("pointermove", (event) => {
    if (!map || !anchor) return;
    const at = cellAt(map, img, event);
    if (!at) return;
    box = spanOf(anchor, at);
    place();
  });
  const finish = (event) => {
    if (!anchor) return;
    const at = map ? cellAt(map, img, event) : null;
    if (at) box = spanOf(anchor, at);
    anchor = null;
    place();
    if (box) onBox(box.slice());
  };
  stage.addEventListener("pointerup", finish);
  stage.addEventListener("pointercancel", () => { anchor = null; });

  fig.show = (reply, legend) => {
    if (!reply || !reply.url) return;
    map = reply;
    img.src = reply.url;
    caption.textContent = legend || "";
    fig.dataset.nx = String(reply.cells.west_east);
    fig.dataset.ny = String(reply.cells.south_north);
    place();
  };
  fig.setBox = (value) => { box = value ? value.slice() : null; place(); };
  fig.img = img;
  return fig;
}
