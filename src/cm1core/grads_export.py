"""Make a published GM1 history frame readable by CM1's own ecosystem.

CM1's ``output_format=1`` is a GrADS direct-access binary32 ``.dat`` plus a
``.ctl`` descriptor (``src/writeout.F:8761-8995`` and ``:9265-9528``).  Nothing
in the meteorological ecosystem opens GM1's private ``CM1FHS1`` envelope;
everything in it opens this.

The conversion itself is Rust (``rust/cm1-state/src/bin/gm1_history_grads_export.rs``)
because the project binds file formats and data paths to Rust.  This module is
the door: it locates the sealed executable, checks its ABI, and hands it the
frame and the grid spacing the descriptor's coordinates need.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
from typing import Any

GRADS_EXPORT_TOOL = "gm1-history-grads-export"
GRADS_EXPORT_ABI = "gpuwm-cm1-history-grads-export/v1"


def require_grads_exporter(path: Any = None) -> Path:
    """Admit exactly one executable that speaks this export ABI."""

    if path is None:
        from .rust_tools import verify_native_tools

        executable = Path(verify_native_tools()[GRADS_EXPORT_TOOL])
    else:
        executable = Path(path).resolve(strict=True)
    completed = subprocess.run(
        [os.fspath(executable), "--abi"], check=False, capture_output=True, text=True
    )
    abi = completed.stdout.strip()
    if completed.returncode != 0 or abi != GRADS_EXPORT_ABI:
        raise RuntimeError(
            f"{executable} reports ABI {abi!r}, not {GRADS_EXPORT_ABI!r}; it does "
            "not speak this frame's field table"
        )
    return executable


def export_history_frame_as_grads(
    frame_directory: Any,
    output_directory: Any,
    *,
    dx_m: float,
    dy_m: float,
    dz_m: float,
    prefix: str = "cm1out",
    executable: Any = None,
) -> dict[str, Any]:
    """Write CM1 ``output_format=1`` records and descriptors beside a frame.

    ``dx_m``/``dy_m``/``dz_m`` are required, not inferred: the descriptor's
    ``xdef``/``ydef``/``zdef`` are physical coordinates, and a descriptor
    written with a guessed spacing puts every field on a grid the run never
    used while still opening cleanly in every reader.
    """

    for name, value in (("dx_m", dx_m), ("dy_m", dy_m), ("dz_m", dz_m)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f"{name} must be a positive grid spacing in metres")
    tool = require_grads_exporter(executable)
    frame = Path(frame_directory)
    completed = subprocess.run(
        [
            os.fspath(tool),
            "--frame", os.fspath(frame),
            "--out", os.fspath(Path(output_directory)),
            "--prefix", prefix,
            "--dx", repr(float(dx_m)),
            "--dy", repr(float(dy_m)),
            "--dz", repr(float(dz_m)),
        ],
        check=False, capture_output=True, text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"GrADS export failed with exit {completed.returncode}: {detail}")
    return json.loads(completed.stdout)


__all__ = [
    "GRADS_EXPORT_ABI",
    "GRADS_EXPORT_TOOL",
    "export_history_frame_as_grads",
    "require_grads_exporter",
]
