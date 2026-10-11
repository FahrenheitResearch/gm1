// The "Make your own environment" card: two plain sliders, how unstable the air is and how much the
// wind changes with height, a sentence on the kind of storm such air usually makes, and the fit's "About N min".
// Each setting is fitted as `gpuwm-cm1 preset fit your-environment --instability I --shear S`; the sentence and
// numbers come back with the fit (CM1's getcape on the model's own levels). The picture is the nearest setting
// that was run for real, and says which.

import { h } from "../js/core.js";
import { go } from "../js/router.js";
import { t } from "./copy.js";
import { fitCard, showFit } from "./gallery.js";

const STEPS = 20; // slider steps: settings 0, 0.05, ... 1

// What each setting states (cm1import/analytic_sounding.py: CAPE_RANGE_J_KG, SHEAR_RANGE_M_S).
function capeTarget(i) { return 300 + 2700 * i; }
function shearTarget(s) { return 2 + 38 * s; }

function nearest(pictures, i, s) {
  let best = null;
  for (const pic of pictures || []) {
    if (typeof pic.instability !== "number" || typeof pic.shear !== "number") continue;
    const d = (pic.instability - i) ** 2 + (pic.shear - s) ** 2;
    if (!best || d < best.d) best = { d, pic };
  }
  return best ? best.pic : null;
}

function slider(id, label, low, high, value) {
  const input = h("input", { type: "range", id, min: "0", max: String(STEPS), step: "1", value: String(Math.round(value * STEPS)) });
  const shown = h("span", { class: "g-env-value", "aria-live": "polite" });
  const row = h("div", { class: "g-env-slider" },
    h("label", { for: id }, label),
    input,
    h("div", { class: "g-env-ends", "aria-hidden": "true" }, h("span", {}, low), h("span", {}, high)),
    shown);
  return { row, input, shown };
}

export function environmentCard(p, alive) {
  const env = (p.sounding && p.sounding.environment) || {};
  let inst = typeof env.instability === "number" ? env.instability : 0.5;
  let shear = typeof env.shear === "number" ? env.shear : 0.5;
  const img = h("img", { alt: p.name, loading: "lazy" });
  const caption = h("figcaption", {});
  const fig = h("figure", { class: "g-pic" }, img, caption);
  const unstable = slider(`g-env-i`, t("env.instability"), t("env.instability_low"), t("env.instability_high"), inst);
  const windy = slider(`g-env-s`, t("env.shear"), t("env.shear_low"), t("env.shear_high"), shear);
  const sentence = h("p", { class: "g-env-sentence", "aria-live": "polite" });
  const numbers = h("p", { class: "g-facts" });
  const fitLine = h("p", { class: "g-fit" }, t("gallery.checking"));
  const why = h("p", { class: "g-why" });
  const choose = h("button", { class: "btn primary", type: "button", disabled: true }, t("gallery.choose"));
  const customize = p.customize ? h("button", { class: "btn g-cz-link", type: "button", disabled: true }, t("gallery.customize")) : null;
  const card = h("article", { class: "g-card g-env", dataset: { preset: p.id } },
    fig,
    h("div", { class: "g-body" },
      h("h2", {}, p.name),
      h("p", { class: "g-line" }, p.line),
      unstable.row, windy.row, sentence, numbers, fitLine, why,
      h("div", { class: "btns" }, choose, customize)));

  let budget = p.budget;
  let serial = 0;
  let timer = null;

  function paintTargets() {
    unstable.shown.textContent = t("env.cape", { cape: Math.round(capeTarget(inst) / 10) * 10 });
    windy.shown.textContent = t("env.shear_value", { shear: Math.round(shearTarget(shear)) });
    const pic = nearest(p.pictures, inst, shear);
    if (pic) {
      img.src = `/api/presets/${encodeURIComponent(pic.picture)}/picture`;
      caption.textContent = t("env.picture_nearest", {
        instability: pic.instability, shear: pic.shear, caption: pic.caption || "",
      });
    } else if (p.picture && p.picture.url) {
      img.src = p.picture.url;
      caption.textContent = p.picture.caption || "";
    } else {
      fig.replaceChildren(h("div", { class: "g-pic none" }, h("span", {}, t("gallery.picture_none"))));
    }
  }

  async function fitNow() {
    const mine = ++serial;
    choose.disabled = true;
    fitLine.textContent = t("gallery.checking");
    why.textContent = "";
    const environment = { instability: inst, shear, hodograph: "curved" };
    let reply;
    try {
      reply = await fitCard(p, { environment });
    } catch (err) {
      if (!alive() || mine !== serial) return;
      card.classList.add("off");
      fitLine.textContent = t("gallery.not_offered");
      why.textContent = `${err.message} ${err.fix || ""}`.trim();
      return;
    }
    if (!alive() || mine !== serial || !reply) return;
    const made = ((reply.fit || {}).sounding || {}).environment || {};
    sentence.textContent = made.sentence || "";
    const n = made.numbers;
    numbers.textContent = n && typeof n.cape_j_kg === "number"
      ? t("env.numbers", { cape: Math.round(n.cape_j_kg), shear: Math.round(n.shear_0_6km_m_s), srh: Math.round(n.srh_0_3km_m2_s2) })
      : "";
    const got = showFit(card, fitLine, why, choose, reply);
    if (got) budget = got;
    if (customize) customize.disabled = choose.disabled;
  }

  function changed() {
    inst = Number(unstable.input.value) / STEPS;
    shear = Number(windy.input.value) / STEPS;
    paintTargets();
    clearTimeout(timer);
    timer = setTimeout(fitNow, 350);
  }
  unstable.input.addEventListener("input", changed);
  windy.input.addEventListener("input", changed);
  choose.addEventListener("click", () => {
    const query = `instability=${inst}&shear=${shear}&hodograph=curved`;
    go(`guided/fit/${encodeURIComponent(p.id)}/${budget}?${query}`);
  });
  if (customize) {
    customize.addEventListener("click", () => {
      const query = `instability=${inst}&shear=${shear}&hodograph=curved&customize=1`;
      go(`guided/fit/${encodeURIComponent(p.id)}/${budget}?${query}`);
    });
  }
  paintTargets();
  card.fitNow = () => {
    if (card.fitted) return;
    card.fitted = true;
    fitNow();
  };
  return card;
}
