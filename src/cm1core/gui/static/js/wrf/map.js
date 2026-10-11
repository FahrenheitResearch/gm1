// The domain map: the picture `gpuwm-cm1 sounding times --map-png` drew, where a click picks a grid column. The
// reply says where the grid sits in the picture (grid_box_px, in the picture's own pixels), so a click anywhere
// inside it is one column: i from the west edge, j from the south edge, both from 1.

import { h } from "../core.js";
import { t } from "./copy.js";

export function gridPoint(map, img, event) {
  const box = map.grid_box_px;
  const [W, H] = map.size;
  const rect = img.getBoundingClientRect();
  if (!rect.width || !rect.height) return null;
  const px = ((event.clientX - rect.left) * W) / rect.width;
  const py = ((event.clientY - rect.top) * H) / rect.height;
  const fx = (px - box[0]) / (box[2] - box[0]);
  const fy = (box[3] - py) / (box[3] - box[1]);
  if (fx < 0 || fx >= 1 || fy < 0 || fy >= 1) return null;
  return [Math.floor(fx * map.nx) + 1, Math.floor(fy * map.ny) + 1];
}

// mapView(onPick): a figure whose .show(map) draws a map reply and whose clicks call onPick([i, j]).
export function mapView(onPick) {
  const img = h("img", { alt: t("map.label"), width: "960", height: "540", draggable: "false" });
  const empty = h("div", { class: "wrf-map-empty" }, t("card.picture"));
  const caption = h("figcaption", { class: "note" }, t("map.help"));
  const fig = h("figure", { class: "wrf-map" }, empty, caption);
  let current = null;
  img.addEventListener("click", (event) => {
    if (!current) return;
    const ij = gridPoint(current, img, event);
    if (ij) onPick(ij);
  });
  fig.show = (map) => {
    if (!map || !map.url) return;
    current = map;
    img.src = map.url;
    if (!img.isConnected) fig.replaceChildren(img, caption);
    fig.dataset.nx = String(map.nx);
    fig.dataset.ny = String(map.ny);
  };
  fig.img = img;
  return fig;
}
