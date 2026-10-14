// Use my own sounding: guided mode's gallery card for a sounding file of the user's own. The card joins the
// gallery through its registerCard (group Explore, after Make your own environment) when the gallery has it.

import * as gallery from "../../guided/gallery.js";
import { renderCard } from "./card.js";

if (typeof gallery.registerCard === "function") {
  gallery.registerCard({ group: "explore", order: 19.5, id: "own-sounding", render: (el) => renderCard(el) });
}
