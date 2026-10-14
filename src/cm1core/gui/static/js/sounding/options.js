// The Customize choices of a storm on the user's own sounding, as they are with the start the card chose (the push
// of rising air has a strength in m/s and no bubble size): gpuwm-cm1 preset options PRESET --start START --json.
// The budget screen asks through here; null when they cannot be read, and the preset's own choices are kept.

import * as api from "../api.js";

export async function startOptions(preset, start) {
  try {
    const reply = await api.post("/api/sounding/options", { preset, start }, { quiet: true });
    return reply && reply.customize ? reply.customize : null;
  } catch (_) {
    return null;
  }
}
