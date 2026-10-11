// Where a folder is, in the words this computer's file manager uses, and a button that opens it there.
//
// Under Windows GM1 runs inside WSL2, so the server knows a run by its Linux path (~/gm1-runs/RUN) while File
// Explorer knows it as \\wsl.localhost\DISTRO\home\NAME\gm1-runs\RUN. The server gives both (root_windows, folder_windows);
// this draws the Windows name first, ready to copy, with the Linux path under it. On Linux there is one path.
// "Open folder" asks the server to show the folder: File Explorer under WSL2, the desktop's file manager on Linux
// (POST /api/open-folder). It is drawn only when the server says this computer can (can_open_folder).

import { h, button, copyline } from "./core.js";
import * as api from "./api.js";
import { notice } from "./router.js";

// folderLine({label, path, windows, canOpen, run}): run is the run's id, or nothing for the runs folder.
export function folderLine({ label, path, windows, canOpen, run }) {
  const open = button("Open folder", { small: true });
  open.dataset.action = "open-folder";
  open.hidden = !canOpen;
  open.addEventListener("click", async () => {
    open.disabled = true;
    try {
      const reply = await api.post("/api/open-folder", run ? { run } : {}, { quiet: true });
      notice(reply.message || "Opened the folder.");
    } catch (err) {
      notice(`${err.message} ${err.fix || ""}`.trim(), "warn");
    } finally { open.disabled = false; }
  });
  const box = h("div", { class: "folderline", dataset: { path: path || "", windows: windows || "" } },
    h("div", { class: "folder-top" }, h("span", { class: "note" }, windows ? `${label} (in Windows):` : `${label}:`), open),
    copyline(windows || path || "", "folder"));
  if (windows) box.append(h("p", { class: "note haspath folder-linux" }, "In Linux (inside WSL) it is ", h("code", {}, path)));
  return box;
}
