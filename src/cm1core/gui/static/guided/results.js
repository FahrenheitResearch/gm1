// Guided screen 4: Results. It is the Results screen (js/run/results.js) drawn with mode = "guided":
// the four favourite looks in order (reflectivity, spin near the ground, updraft at 5 km, the wind swath), a time
// slider and Play, "What am I looking at" beside the picture, then Open the files, Run it longer and Show
// everything. The components are the Advanced screen's.

import { renderResults } from "../js/run/results.js";

export function results(body, args, page = {}) {
  return renderResults(body, args, { ...page, mode: "guided" });
}
