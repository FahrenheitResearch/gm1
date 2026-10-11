"""The guided/advanced mode, stored per OS user.

``~/.config/gpuwm-cm1/gui.json`` (``$XDG_CONFIG_HOME`` is honoured) holds
``{"mode": "guided" | "advanced"}``, plus one flag recording that the switch
to advanced on a first namelist import has happened.  The mode is a view: it
changes what the pages draw, never what runs, and nothing here writes inside
a run directory.

- No stored mode and no runs under ``--root``: guided (a first run).
- No stored mode and runs present: advanced (someone who has run GM1 from a
  terminal already).
- Importing a namelist in guided mode switches to advanced once, and says so.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import threading
from typing import Any

MODES = ("guided", "advanced")
PREFS_NAME = "gui.json"
IMPORT_SWITCH_NOTICE = (
    "Switched to Advanced because you imported a CM1 namelist: it shows every setting. "
    "Guided is one click away at the top."
)


def config_dir() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"), ".config")
    return Path(base) / "gpuwm-cm1"


def require_mode(value: Any) -> str:
    if value not in MODES:
        raise ValueError(f"mode must be one of {', '.join(MODES)}, not {value!r}")
    return str(value)


class Prefs:
    """Read and write ``gui.json``; safe across the server's threads."""

    def __init__(self, directory: Path | None = None) -> None:
        self.directory = Path(directory) if directory is not None else config_dir()
        self.path = self.directory / PREFS_NAME
        self._lock = threading.Lock()

    def load(self) -> dict[str, Any]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def _save(self, data: dict[str, Any]) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        handle, temporary = tempfile.mkstemp(prefix=".gui-", suffix=".json", dir=self.directory)
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                json.dump(data, stream, indent=2, sort_keys=True)
                stream.write("\n")
            os.replace(temporary, self.path)
        except BaseException:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise

    def mode(self, runs_exist: bool) -> tuple[str, str]:
        """(mode, where it came from): stored, first-launch or runs-present."""

        stored = self.load().get("mode")
        if stored in MODES:
            return str(stored), "stored"
        return ("advanced", "runs-present") if runs_exist else ("guided", "first-launch")

    def set_mode(self, mode: str) -> str:
        mode = require_mode(mode)
        with self._lock:
            data = self.load()
            data["mode"] = mode
            self._save(data)
        return mode

    def switch_on_import(self, shown: str) -> str | None:
        """After a namelist import: advanced, once.  Returns the notice or None.

        ``shown`` is the mode the user was looking at when they imported.
        """

        with self._lock:
            data = self.load()
            if data.get("import_switch_done"):
                return None
            data["import_switch_done"] = True
            data["mode"] = "advanced"
            self._save(data)
        return IMPORT_SWITCH_NOTICE if shown == "guided" else None


__all__ = ["IMPORT_SWITCH_NOTICE", "MODES", "Prefs", "config_dir", "require_mode"]
