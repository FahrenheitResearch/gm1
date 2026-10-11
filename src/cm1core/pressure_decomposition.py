"""CM1 ``pdcomp.F`` pressure decomposition, published under CM1's names.

Authority
---------
``output_pdcomp`` (``run/namelist.input:277``, ``src/param.F:727-729``,
``README.namelist:1449-1454``) calls ``src/pdcomp.F:10`` ``pidcomp`` at every
history time (``src/cm1.F:3483-3503``).  Its four Poisson inversions become
the history variables ``pipb``, ``pipdl``, ``pipdn`` and, with ``icor=1``,
``pipc`` (``src/writeout.F:6252-6340``, nondimensional pi-prime at scalar
points), and their vertical gradient forces ``pgradb``, ``pgraddl``,
``pgraddn``, ``pgradc`` (``src/writeout.F:8478-8580``, m/s/s at w levels).

The arithmetic lives in :mod:`cm1core.cuda_pdcomp_open`, including why an
open lateral boundary needs no solver of its own in CM1 (``param.F:5178-5205``
doubles the transform and ``poiss.F:80-114`` reflects the forcing into it).
This module maps a run onto that selection and writes two frames per history
time: ``pdcomp-<frame>/`` (the ``pip*`` rectangles) and ``pdcomp-w-<frame>/``
(the ``pgrad*`` w-level rectangles, ``nz+1`` levels).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .cuda_pdcomp_open import (
    PDCOMP_SCALAR_NAMES,
    PDCOMP_W_NAMES,
    SOURCE_ANCHORS,
    PdcompSelection,
    pdcomp_selection_defect,
)

PDCOMP_SCHEMA = "gpuwm-cm1.pressure-decomposition/v2"

#: What a reader is told, verbatim, wherever the decomposition appears.
PDCOMP_UNVALIDATED = (
    "UNVALIDATED end to end against a CM1 run. The forcings, fluxes, divx, "
    "thv0 and init3d.F coefficients are CM1's own statements bit for bit "
    "(gfortran-built pristine pdcomp.F capture), and the mirrored open-boundary "
    "inversion agrees with CM1's poiss.F/singleton.F to FFT round-off on "
    "bounded uniform and stretch_z decks. pipdn inherits this lane's "
    "advu/advv/advw kernels, so its CM1 fidelity is that operator's; no storm "
    "run has been differenced against CM1's pipb/pipdl/pipdn/pipc output."
)

__all__ = [
    "PDCOMP_SCALAR_NAMES", "PDCOMP_SCHEMA", "PDCOMP_UNVALIDATED", "PDCOMP_W_NAMES",
    "SOURCE_ANCHORS", "pdcomp_selection_for", "publish_pdcomp_frames",
]


def pdcomp_selection_for(config: Any) -> PdcompSelection:
    """This run's ``pidcomp`` selection, from the bound runtime configuration."""

    g = config.grid
    p = config.profile
    return PdcompSelection(
        ni=int(g.nx), nj=int(g.ny), nk=int(g.nz),
        dx=float(g.dx_m), dy=float(g.dy_m), dz=float(g.dz_m), dt=float(config.dt_s),
        wbc=int(config.wbc), ebc=int(config.ebc), sbc=int(config.sbc), nbc=int(config.nbc),
        icor=int(config.icor),
        imoist=1 if int(p.nqv) >= 1 else 0, iice=int(p.iice),
        nqv=max(int(p.nqv), 1), nql1=int(p.nql1), nql2=int(p.nql2),
        nqs1=int(p.nqs1), nqs2=int(p.nqs2),
    )


def publish_pdcomp_frames(runtime: Any, root: Path, index: int) -> list[dict[str, Any]]:
    """Invert this history time's forcings and write CM1's eight names."""

    from .high_frequency_output import HighFrequencyWindow, write_high_frequency_window

    selection = pdcomp_selection_for(runtime.config)
    defect = pdcomp_selection_defect(selection)
    if defect is not None:
        raise ValueError(f"output_pdcomp refused: {defect}")
    fields, record = runtime._pdcomp_fields()
    g = runtime.config.grid
    common = {
        "model_time_s": runtime.clock.model_seconds,
        "step": runtime.clock.completed_steps,
    }
    extra = {"schema": PDCOMP_SCHEMA, "pdcomp": record, "unvalidated": PDCOMP_UNVALIDATED}
    return [
        write_high_frequency_window(
            Path(root) / f"pdcomp-{index:05d}",
            {name: fields[name] for name in selection.names},
            HighFrequencyWindow(0, g.nx, 0, g.ny, 0, g.nz),
            order=list(selection.names), nx=g.nx, ny=g.ny, nz=g.nz,
            extra={"kind": "pdcomp", "grid_output": "s",
                   "units": "nondimensional", **extra},
            **common,
        ),
        write_high_frequency_window(
            Path(root) / f"pdcomp-w-{index:05d}",
            {name: fields[name] for name in selection.w_names},
            HighFrequencyWindow(0, g.nx, 0, g.ny, 0, g.nz + 1),
            order=list(selection.w_names), nx=g.nx, ny=g.ny, nz=g.nz + 1,
            extra={"kind": "pdcomp-w", "grid_output": "w", "units": "m/s/s", **extra},
            **common,
        ),
    ]
