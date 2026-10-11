// Guided screen 3: Watching. It is the Run screen (js/run/run.js) drawn with mode = "guided": guided mode's
// header (storm time, time left), one large picture over the four favourite looks that updates with each saved
// time, and Stop. Pause, Force stop, the rates and the log are in Advanced; the components are the same.

import { renderRun } from "../js/run/run.js";

export function watch(body, args, page = {}) {
  return renderRun(body, args, { ...page, mode: "guided" });
}
