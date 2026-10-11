// Guided mode's card "Use a sounding from my WRF or ArWen run", in the gallery's Explore group. It opens
// in place on the gallery (no new screen): the file, the time, the map, the air above the chosen spot, the plain
// note on what this is, and Next, which keeps the column for the storm and goes to "How long can it run?".

import { h, button } from "../core.js";
import * as api from "../api.js";
import { go } from "../router.js";
import { actionButton } from "../command.js";
import { ready, t } from "./copy.js";
import { columnFlow } from "./flow.js";

// render(el, ctx): the card itself, as the gallery's cards are drawn (a picture area, a title, one line, a button).
// The gallery's registerCard hands render a cell of the group's grid (div.g-cell); the panel goes after that cell, so it
// is a grid item of its own and spans the whole row (.wrf-panel), and it folds away with the cell.
export async function renderCard(el) {
  await ready();
  const body = h("div", { class: "wrf-open-body" });
  const openBtn = h("button", { class: "btn primary", type: "button", "aria-expanded": "false" }, t("card.open"));
  const card = h("article", { class: "g-card wrf-card", dataset: { preset: "wrf-column" } },
    h("div", { class: "g-pic none wrf-pic" }, h("span", {}, t("card.picture"))),
    h("div", { class: "g-body" },
      h("h2", {}, t("card.title")),
      h("p", { class: "g-line" }, t("card.line")),
      h("div", { class: "btns" }, openBtn)));
  el.append(card);
  let panel = null;
  openBtn.addEventListener("click", async () => {
    if (panel) {
      const shown = panel.hidden;
      panel.hidden = !shown;
      openBtn.textContent = shown ? t("card.close") : t("card.open");
      openBtn.setAttribute("aria-expanded", shown ? "true" : "false");
      return;
    }
    panel = await openPanel();
    // In place: the panel spans the gallery's whole row, right after this card's cell.
    const anchor = el.classList && el.classList.contains("g-cell") ? el : card;
    anchor.after(panel);
    // "Show fewer" hides the cell; the panel goes with it (and the button says Choose a file again).
    if (anchor !== card && typeof MutationObserver === "function") {
      new MutationObserver(() => {
        if (anchor.hidden && !panel.hidden) {
          panel.hidden = true;
          openBtn.textContent = t("card.open");
          openBtn.setAttribute("aria-expanded", "false");
        }
      }).observe(anchor, { attributes: true, attributeFilter: ["hidden"] });
    }
    openBtn.textContent = t("card.close");
    openBtn.setAttribute("aria-expanded", "true");
    panel.scrollIntoView({ block: "nearest" });
    // Keyboard users land in the panel's first field, the file.
    const first = panel.querySelector("input");
    if (first) first.focus({ preventScroll: true });
  });
  return card;
}

async function openPanel() {
  const status = h("div", { class: "wrf-next-say", role: "status", "aria-live": "polite" });
  let flow = null;
  const request = () => {
    const s = flow && flow.state;
    if (!s || !s.column || !s.point) return { error: t("error.pick_point") };
    return { path: "/api/wrf/use", body: { path: s.path, ...s.when, ...s.point, target: "guided", budget: "1h" } };
  };
  const next = actionButton(t("use.next"), { action: "wrf-use", kind: "primary", request, disabled: true });
  flow = await columnFlow({ guided: true, onColumn: (state) => { next.button.disabled = !state; } });
  next.button.addEventListener("click", async () => {
    const req = request();
    if (req.error) return;
    next.button.disabled = true;
    status.replaceChildren(h("p", { class: "note" }, t("use.keeping")));
    try {
      const reply = await api.post(req.path, req.body);
      if (!reply.ok) {
        status.replaceChildren(h("p", { class: "banner stop" }, reply.message || "", h("span", { class: "sub" }, reply.fix || "")));
        return;
      }
      const presets = await api.get("/api/presets").catch(() => ({ presets: [] }));
      if (!(presets.presets || []).some((p) => p.id === reply.preset)) {
        status.replaceChildren(h("p", { class: "banner warn" }, t("use.waiting")));
        return;
      }
      go(reply.route);
    } catch (err) {
      status.replaceChildren(h("p", { class: "banner stop" }, err.message, h("span", { class: "sub" }, err.fix || "")));
    } finally { next.button.disabled = false; }
  });
  const close = button(t("card.close"), { small: true });
  const panel = h("section", { class: "panel wrf-panel", id: "wrf-card-panel" },
    h("div", { class: "wrf-panel-head" }, h("h2", { class: "cap" }, t("card.title")), close),
    flow, h("div", { class: "btns wrf-next" }, next), status);
  close.addEventListener("click", () => {
    panel.hidden = true;
    const opener = document.querySelector(".wrf-card .btns .btn");
    if (opener) { opener.textContent = t("card.open"); opener.setAttribute("aria-expanded", "false"); opener.focus(); }
  });
  return panel;
}
