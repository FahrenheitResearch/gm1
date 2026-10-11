// Advanced mode, New run: "From a WRF or ArWen run". The same steps as guided mode's card, then a
// namelist (CM1's supercell example, or your own) and a name; Import and check runs
//   gpuwm-cm1 import NAMELIST --wrf FILE [--wrf-time T] (--wrf-at=LAT,LON | --wrf-ij I,J) --out RUNDIR --json
// and opens the imported run's Check screen.

import { h, button } from "../core.js";
import * as api from "../api.js";
import { go, notice } from "../router.js";
import { actionButton } from "../command.js";
import { resultBox } from "../setup/common.js";
import { ready, t, hint } from "./copy.js";
import { columnFlow } from "./flow.js";

const NAME_RE = /^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/;

export async function wrfPanel() {
  await ready();
  const box = resultBox();
  let flow = null;
  const own = h("input", { class: "input mono", placeholder: "~/cm1/run/namelist.input", "aria-label": t("tab.namelist_own"),
    spellcheck: "false" });
  own.hidden = true;
  const choice = h("select", { class: "input", "aria-label": t("tab.namelist") },
    h("option", { value: "example" }, t("tab.namelist_default")),
    h("option", { value: "own" }, t("tab.namelist_own")));
  choice.addEventListener("change", () => { own.hidden = choice.value !== "own"; });
  const name = h("input", { class: "input", placeholder: "my-wrf-storm", "aria-label": t("tab.name"), spellcheck: "false" });
  // The name follows the chosen spot until the user types one of their own.
  let autoName = "";
  name.addEventListener("input", () => { autoName = ""; });
  const follow = h("input", { type: "checkbox", checked: true, "aria-label": t("tab.follow") });
  const request = () => {
    const s = flow && flow.state;
    if (!s || !s.column || !s.point) return { error: t("error.pick_point") };
    const runName = name.value.trim();
    if (!runName || !NAME_RE.test(runName)) return { error: "Give the run a name: letters, digits, '.', '_' or '-'." };
    const namelist = choice.value === "own" ? (own.value.trim() ? { path: own.value.trim() } : null) : { example: "cm1-supercell" };
    if (!namelist) return { error: "Give your namelist's path." };
    return { path: "/api/wrf/use", body: { path: s.path, ...s.when, ...s.point, target: "import", name: runName, namelist,
      frame: follow.checked ? "follow" : "namelist" } };
  };
  const go1 = actionButton(t("tab.import"), { action: "wrf-use", kind: "primary", request,
    watch: [name, own, choice, follow] });
  go1.button.disabled = true;
  flow = await columnFlow({ guided: false, onColumn: (state) => {
    go1.button.disabled = !state;
    if (state && (!name.value || name.value === autoName)) {
      const p = state.column.point || {};
      autoName = `wrf-${p.i}-${p.j}-${String((state.column.source || {}).times || "").slice(0, 13).replace(/[^0-9]/g, "")}`;
      name.value = autoName;
    }
    if (go1.cmd && go1.cmd.refresh) go1.cmd.refresh();
  } });
  go1.button.addEventListener("click", async () => {
    const req = request();
    if (req.error) { notice(req.error, "stop"); return; }
    go1.button.disabled = true;
    notice(t("tab.importing"));
    try {
      const reply = await api.post(req.path, req.body);
      box.show(reply);
      if (reply.url) {
        notice(reply.message + (reply.fix ? ` ${reply.fix}` : ""), reply.ok ? "" : "warn");
        setTimeout(() => go(`check/${reply.url}`), 300);
      } else {
        notice(`${reply.message} ${reply.fix || ""}`.trim(), "stop");
      }
    } catch (err) {
      box.show({ ok: false, message: err.message, fix: err.fix, command: err.command });
      notice(`${err.message} ${err.fix || ""}`.trim(), "stop");
    } finally { go1.button.disabled = false; }
  });
  return h("section", { class: "panel section wrf-tab", id: "from-wrf" },
    h("h2", { class: "cap" }, t("tab.title")),
    h("p", { class: "note" }, t("tab.lead"), hint("tab.lead")),
    flow,
    h("div", { class: "wrf-import" },
      h("div", { class: "field" }, h("label", {}, t("tab.namelist"), hint("tab.namelist_help")),
        h("div", {}, choice, own, h("div", { class: "hint" }, t("tab.namelist_help")))),
      h("div", { class: "field" }, h("label", {}, t("tab.name")), name),
      h("label", { class: "wrf-check" }, follow, " ", t("tab.follow"), hint("tab.follow"),
        h("span", { class: "sub" }, t("tab.follow_help"))),
      h("div", { class: "btns" }, go1),
      h("p", { class: "note" }, t("tab.frame_note"))),
    box);
}
