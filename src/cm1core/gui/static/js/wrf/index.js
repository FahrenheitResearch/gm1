// The column route: guided mode's gallery card and advanced mode's New run panel. The card joins
// the gallery through its registerCard (group Explore) when the gallery has it; the panel is drawn by New run.

import * as gallery from "../../guided/gallery.js";
import { renderCard } from "./card.js";

export { wrfPanel } from "./tab.js";

if (typeof gallery.registerCard === "function") {
  gallery.registerCard({ group: "explore", order: 20, id: "wrf-column", render: (el) => renderCard(el) });
}
