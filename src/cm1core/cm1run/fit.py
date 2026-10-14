"""Fit a preset to this GPU and a time budget: the guided mode's auto-fit.

A preset is a CM1 r21.1 namelist plus
a few facts about how it may be made smaller (:class:`PresetSpec`).  The fit
prices candidates with :func:`cm1core.cm1run.estimate.price` and takes the
first that fits, in this order:

1. the preset at its base size, if GPU need <= 0.95 x free GPU memory, locked
   host memory <= 50 % of RAM, disk <= 50 % of free space and the central wall
   estimate <= the budget;
2. a smaller horizontal area (dx, dt, nz and the physics fixed), in steps of 8
   cells on both sides together -- a preset with one repeating direction
   shrinks that one first -- down to the preset's minimum and to where the
   storm-track check has the storm within 10 km of an edge before the end;
3. the next coarser rung of the preset's ladder (dt scaled with dx, nsound
   re-derived by the importer, as CM1 does), then step 2 again;
4. a shorter storm time, in 15-minute steps down to the preset's minimum,
   each time repeating 1-3;
5. throughout, the output interval is raised (15 min, then each longer one
   that splits the storm time into whole frames, ...) until the disk need is
   at most half the free space;
6. nothing fits: the preset is not offered, with the smallest GPU memory and
   the shortest time it would need, in plain words.

A setting the user chose in guided Customize (``--custom``) is *pinned* (``PresetSpec.pinned``) and skipped by
these steps: ``grid`` (no coarser rung), ``area`` (one size), ``time`` (one storm time), ``pictures`` (no wider
output interval).  The not-offered reason then says which choices hold it back.

With a choice of the user's own the fit does not turn down what can run.  When steps 1-5 find nothing, they are
gone through once more with the area allowed down to the smallest a storm can start in
(``PresetSpec.hard_min_width_km``: 16 km, or the warm bubble's own width) and a storm that comes close to the
edge, or reaches it, taken with a warning that says when (``FitResult.warnings``) instead of being turned down.
What is still not offered is what cannot run: more GPU memory, RAM or disk than is free even at that smallest
area, more time than the budget (the ``any`` budget has no limit), a setting the import refuses, and a storm on a
sounding file that leaves its area (the run's own start refuses that one).

Guards: only nx, ny, dx, dy, dtl, timax and tapfrq change (physics, boundaries, sounding and initiation never
do); the fitted namelist is re-read by the real parser and must import with no refusal; every change is a
sentence in ``fit.json``.  A domain larger than the largest GM1 has run is offered, with a warning.
"""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
import time
import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import rates as _rates
from .estimate import (
    _PRICING_BUDGET_MIB,
    Deck,
    Estimate,
    EstimateError,
    FIT_EDGE_CLEARANCE_M,
    FITS_FRACTION_OF_FREE,
    duration_words,
    price,
)
from ..cm1import.translate import TranslationError
from .gpuinfo import GpuInfo, HostInfo

FIT_SCHEMA = "gpuwm-cm1.fit/v1"

#: The budgets the guided screen offers, in seconds.  ``any`` is "as long as it takes": no limit on the time, for
#: a run that needs more than overnight's 10 hours.
BUDGETS: Mapping[str, float] = {"10m": 600.0, "1h": 3600.0, "overnight": 36000.0, "any": math.inf}
BUDGET_WORDS: Mapping[str, str] = {
    "10m": "about 10 minutes",
    "1h": "about 1 hour",
    "overnight": "overnight (up to 10 hours)",
    "any": "as long as it takes",
}

def _budget_words(key: str, seconds: float) -> str:
    """A budget in words: its button's, or the time itself for a budget given in seconds."""

    if key in BUDGET_WORDS:
        return BUDGET_WORDS[key]
    return duration_words(seconds) if math.isfinite(seconds) else BUDGET_WORDS["any"]


#: Shares of RAM and free disk a fitted run may take (section 7.4, step 1).
HOST_FRACTION = 0.5
DISK_FRACTION = 0.5
#: Horizontal shrink step, cells per side.
SHRINK_STEP = 8
#: Storm-time shortening step (section 7.4, step 4).
SHORTEN_STEP_S = 900.0

#: The most levels of the runs the measured speeds come from: past it an offered run carries a note about its time.
MEASURED_LEVELS_MAX = 80

#: The namelist words a fit may change; everything else is the preset's.
FIT_KEYS = ("nx", "ny", "dx", "dy", "dtl", "timax", "tapfrq")

#: Words a fit must never change: physics, boundaries, sounding, initiation.
FROZEN_KEYS = (
    "nz", "dz", "ptype", "imoist", "ihail", "sgsmodel", "tconfig", "bbc", "sfcmodel", "isfcflx",
    "wbc", "ebc", "sbc", "nbc", "tbc", "bcturbs", "isnd", "iwnd", "iinit", "irandp", "imove",
    "umove", "vmove", "icor", "fcor", "lspgrad", "eqtset", "stretch_z", "ztop", "zd", "irdamp",
    "hrdamp", "psolver", "kdiv", "ramp_sgs", "ramp_time",
)


class FitError(ValueError):
    """A preset that cannot be fitted for a reason other than size (a refusal)."""


# ------------------------------------------------------------- presets


@dataclass(frozen=True)
class PresetSpec:
    """One guided preset: a CM1 namelist and how the fit may change it.

    ``min_width_km`` maps a grid spacing (m) to the smallest area side at it;
    the entry ``0`` is the default for every spacing without its own entry.
    ``shrink`` is ``"both"`` (square shrink) or ``"repeating-first"`` (shrink
    the periodic direction first, as the squall line does).
    """

    id: str
    name: str
    namelist_text: str
    ladder_m: tuple[float, ...] = ()
    min_width_km: Mapping[float, float] = field(default_factory=lambda: {0.0: 64.0})
    min_along_repeating_km: float | None = None
    min_duration_s: float = 3600.0
    shrink: str = "both"
    sounding_text: str | None = None
    offered: bool = True
    not_offered_reason: str | None = None
    line: str = ""
    #: GM1-only settings the preset carries into every candidate (for example
    #: ``initialization.bubble_center_x_m``/``_y_m`` in the iorigin frame, an
    #: offset from the centre under iorigin = 2, so it survives a shrink).
    gm1_settings: Mapping[str, Any] = field(default_factory=dict)
    #: Settings the user chose (guided Customize, ``--custom``) that the fit never changes: ``grid`` (no coarser
    #: rung), ``area`` (one size), ``time`` (one storm time), ``pictures`` (no wider output interval).
    pinned: frozenset[str] = frozenset()
    #: ``initialization.random_seed`` for random bumps from GM1's own numbers (``--custom bumps=seed:N``).
    random_seed: int | None = None
    #: The smallest area side a storm can start in (km), below ``min_width_km``: with a choice of the user's own
    #: the fit goes down to it, with a warning, before it says a run does not fit.
    hard_min_width_km: float | None = None
    #: One-line notes the user's choices carry into every fit (an unusual grid spacing).
    notes: tuple[str, ...] = ()
    #: The area starts at the most one restart file holds at this spacing, not at the preset's own (Customize).
    start_capped: bool = False

    def smallest_width_m(self, dx_m: float) -> float:
        """The smallest area side the fit may go down to at ``dx_m``: the preset's own, or with a choice of the
        user's own the smallest a storm can start in when that is less."""

        own = self.min_width_m(dx_m)
        if self.pinned and self.hard_min_width_km is not None:
            # Customize's own floor: no more than the preset's, unless the warm bubble itself is wider (a bubble
            # placed off centre needs room on both sides of the centre, and no smaller area can be priced).
            return float(self.hard_min_width_km) * 1000.0
        return own

    def min_width_m(self, dx_m: float) -> float:
        table = dict(self.min_width_km)
        for spacing in sorted((k for k in table if k > 0.0)):
            if dx_m <= spacing + 1e-6:
                return table[spacing] * 1000.0
        return table.get(0.0, 0.0) * 1000.0

    @classmethod
    def from_mapping(cls, spec: Mapping[str, Any], namelist_text: str, *, sounding_text: str | None = None) -> "PresetSpec":
        """``preset.json`` as a preset folder stores it beside ``namelist.input``."""

        widths = spec.get("min_width_km", 64.0)
        if isinstance(widths, Mapping):
            table = {0.0 if key in ("default", "0") else float(key): float(value) for key, value in widths.items()}
        else:
            table = {0.0: float(widths)}
        return cls(
            id=str(spec["id"]),
            name=str(spec.get("name", spec["id"])),
            namelist_text=namelist_text,
            ladder_m=tuple(float(value) for value in spec.get("ladder_m", ())),
            min_width_km=table,
            min_along_repeating_km=(
                None if spec.get("min_along_repeating_km") is None else float(spec["min_along_repeating_km"])
            ),
            min_duration_s=float(spec.get("min_duration_s", 3600.0)),
            shrink=str(spec.get("shrink", "both")),
            sounding_text=sounding_text,
            offered=bool(spec.get("offered", True)),
            not_offered_reason=spec.get("not_offered_reason"),
            line=str(spec.get("line", "")),
            gm1_settings=dict(spec.get("gm1_settings") or {}),
            pinned=frozenset(str(knob) for knob in spec.get("pinned") or ()),
            random_seed=None if spec.get("random_seed") is None else int(spec["random_seed"]),
            hard_min_width_km=(
                None if spec.get("hard_min_width_km") is None else float(spec["hard_min_width_km"])
            ),
            notes=tuple(str(note) for note in spec.get("custom_notes") or ()),
            start_capped=bool(spec.get("start_capped")),
        )

    @classmethod
    def from_directory(cls, path: str | os.PathLike[str]) -> "PresetSpec":
        """``<id>/namelist.input`` and ``<id>/preset.json`` (and ``input_sounding``)."""

        path = Path(path)
        spec = json.loads((path / "preset.json").read_text(encoding="utf-8"))
        sounding = path / "input_sounding"
        return cls.from_mapping(
            spec, (path / "namelist.input").read_text(encoding="ascii"),
            sounding_text=sounding.read_text(encoding="ascii") if sounding.is_file() else None,
        )


# ------------------------------------------------------- namelist editing


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return ".true." if value else ".false."
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        text = repr(float(value))
        return text if ("." in text or "e" in text) else text + ".0"
    return str(value)


def set_namelist_values(text: str, values: Mapping[str, Any]) -> str:
    """Replace ``key = value`` assignments in CM1 namelist text, keeping layout.

    Keys are matched case-insensitively outside comments.  A key the text
    does not assign is added to the end of its CM1 r21.1 group.  The result is
    re-read by the native parser before anything uses it (:func:`fit_preset`).
    """

    pending = {key.lower(): value for key, value in values.items()}
    lines = text.splitlines(keepends=True)
    out = []
    for line in lines:
        body, bang, comment = line.partition("!")
        for key in list(pending):
            pattern = re.compile(rf"(?i)(?<![\w%])({re.escape(key)}\s*=\s*)([^,/\n]*?)(\s*(?:,|/|\n|$))")
            match = pattern.search(body)
            if match:
                replacement = match.group(1) + _format_value(pending[key]) + match.group(3)
                body = body[: match.start()] + replacement + body[match.end():]
                del pending[key]
        out.append(body + bang + comment)
    text = "".join(out)
    if pending:
        from ..cm1import import defaults as _defaults

        inventory = _defaults.load()
        for key, value in pending.items():
            groups = inventory.groups_of(key)
            if not groups:
                raise FitError(f"{key} is not a CM1 r21.1 namelist key")
            group = groups[0]
            match = re.search(rf"(?im)^\s*&{re.escape(group)}\b.*?^\s*/", text, re.S)
            if not match:
                raise FitError(f"the namelist has no &{group} group to add {key} to")
            end = match.end()
            slash = text.rfind("/", match.start(), end)
            text = text[:slash] + f" {key} = {_format_value(value)},\n " + text[slash:]
    return text


# ------------------------------------------------------------- candidates


@dataclass(frozen=True)
class Candidate:
    nx: int
    ny: int
    dx_m: float
    dt_s: float
    duration_s: float
    output_interval_s: float

    def values(self) -> dict[str, Any]:
        return {
            "nx": self.nx, "ny": self.ny, "dx": float(self.dx_m), "dy": float(self.dx_m),
            "dtl": float(self.dt_s), "timax": float(self.duration_s), "tapfrq": float(self.output_interval_s),
        }


@dataclass
class _Base:
    """The preset read once: its effective CM1 settings and translation inputs."""

    imported: Any
    effective: dict[str, Any]
    nx: int
    ny: int
    dx_m: float
    dt_s: float
    duration_s: float
    output_interval_s: float
    periodic_x: bool
    periodic_y: bool
    #: The thinnest level: dz, or dz_bot on a stretched column (param.F:7819-8137).
    dz_min_m: float = 0.0


#: CM1's own supercell deck steps 6 s over 500 m levels (run/config_files/supercell/namelist.input:16-17), so
#: its strongest updrafts (~60 m/s) cross 0.7 of a level per step.  A coarser rung scales the step with dx but
#: keeps the preset's levels, so the step is also held to this ratio: past it the vertical Courant number of a
#: mature storm reaches CM1's own stop (1.5, misclibs.F:2714) and the run ends there, as CM1's would.
CM1_DT_PER_DZ_S_PER_M = 6.0 / 500.0


def rung_dt_s(base: "_Base", dx_m: float) -> float:
    """The time step at a rung: scaled with dx (as CM1 advises), held to CM1's own step per level height."""

    if dx_m == base.dx_m:
        # The preset's own step, as it is: base.dt_s * dx / dx can come out one unit in the last place short (6.0 at
        # 1000.0001 m gives 5.999999999999999), and the run's clock check then refuses every candidate.
        return base.dt_s
    dt = base.dt_s * (dx_m / base.dx_m)
    if base.dz_min_m <= 0.0:
        return dt
    cap = CM1_DT_PER_DZ_S_PER_M * base.dz_min_m
    if dt <= cap:
        return dt
    # A whole multiple of the preset's own step, so it still divides the preset's output times.
    return max(base.dt_s, base.dt_s * math.floor(cap / base.dt_s + 1e-9))


def _dt_at(base: "_Base", dx_m: float, duration_s: float) -> float | None:
    """The time step a spacing of the user's own takes with the preset's levels and output times, as Customize gives
    it (``customize.snapped_time_step``), or None when none lands."""

    from .customize import CustomError, snapped_time_step

    effective = base.effective
    rstfrq = float(effective.get("rstfrq", 0.0) or 0.0)
    try:
        return snapped_time_step(
            dx_m, float(effective.get("dz", 0.0) or 0.0) or base.dz_min_m,
            [duration_s, base.output_interval_s, *([rstfrq] if rstfrq > 0.0 else [])],
            moving=int(effective.get("imove", 0) or 0) == 1,
        )
    except CustomError:
        return None


def _read_preset(preset: PresetSpec, workdir: Path) -> _Base:
    from ..cm1import import import_namelist

    path = workdir / f"{preset.id}.namelist.input"
    path.write_text(preset.namelist_text, encoding="ascii")
    imported = import_namelist(path)
    if imported.translation is None:
        raise FitError(
            f"preset {preset.id} does not import: "
            + (imported.component.reason if imported.component.component is None else "; ".join(imported.cm1_stops))
        )
    effective = dict(imported.effective)
    return _Base(
        imported=imported, effective=effective,
        nx=int(effective["nx"]), ny=int(effective["ny"]), dx_m=float(effective["dx"]),
        dt_s=float(effective["dtl"]), duration_s=float(effective["timax"]),
        output_interval_s=float(effective["tapfrq"]),
        periodic_x=int(effective["wbc"]) == 1 and int(effective["ebc"]) == 1,
        periodic_y=int(effective["sbc"]) == 1 and int(effective["nbc"]) == 1,
        dz_min_m=(
            float(effective.get("dz", 0.0)) if int(effective.get("stretch_z", 0) or 0) == 0
            else min(float(effective.get("dz", 0.0)), float(effective.get("dz_bot", 0.0) or effective.get("dz", 0.0)))
        ),
    )


def _restart_at_end(effective: Mapping[str, Any]) -> dict[str, float]:
    """The run's one restart at the end, for a namelist whose negative rstfrq writes none.

    CM1 writes no restart for rstfrq < 0 (README.namelist:103-104); the imported run keeps one at the end
    (``cm1import.coverage``'s ``restart_at_end``, which the run's own deck carries), so a fit prices that one
    restart and not the translator's ``|rstfrq|`` cadence.  Before this, the 250 m famous cards priced two 5.2 GB restarts
    (and about 51.9 GB of disk) where their runs, made the same way, wrote one.
    """

    rstfrq, timax = effective.get("rstfrq"), effective.get("timax")
    if (isinstance(rstfrq, (int, float)) and not isinstance(rstfrq, bool) and rstfrq < 0
            and isinstance(timax, (int, float)) and not isinstance(timax, bool) and timax > 0):
        return {"run.restart_interval_s": float(timax)}
    return {}


def _random_numbers(effective: Mapping[str, Any], settings: Mapping[str, Any]) -> dict[str, Any]:
    """The random-number setting the import gives a run with random bumps (``coverage.runtime_keys``).

    irandp = 1 (and iinit 7 and 8's noise) with no seed given draws CM1's own random numbers
    (``random_generator = "cm1"``), so a candidate is priced as the run will be made instead of refused for having
    no recorded seed."""

    random = int(effective.get("irandp", 0) or 0) == 1 or int(effective.get("iinit", 1) or 1) in (7, 8)
    if random and "initialization.random_seed" not in settings and "initialization.random_generator" not in settings:
        return {"initialization.random_generator": "cm1"}
    return {}


def _deck_for(base: _Base, candidate: Candidate, gm1_only: Mapping[str, Any], sounding: bytes | None) -> Deck:
    """The candidate as a priced deck, through the one translator (no subprocess)."""

    from ..cm1import.translate import PRODUCT_GM1_ONLY, render_toml, translate

    effective = dict(base.effective)
    effective.update(candidate.values())
    imported = base.imported
    runtime = dict(PRODUCT_GM1_ONLY)
    runtime.update(_restart_at_end(effective))
    runtime.update(gm1_only)
    runtime.update(_random_numbers(effective, runtime))
    from ..cm1import import defaults as _defaults

    param9 = _defaults.load().fields["param9"]
    translation = translate(
        effective,
        component=imported.component.component,
        gm1_only=runtime,
        output={key: effective[key] for key in param9},
        declared=imported.stated | frozenset(FIT_KEYS),
    )
    raw = tomllib.loads(render_toml(translation, ("# fit candidate",)))
    return Deck(
        raw=raw, source=f"fit candidate {candidate.nx}x{candidate.ny} at {candidate.dx_m:g} m",
        kind="mapping", name="fit candidate", sounding_bytes=sounding,
    )


def _output_intervals(default_s: float, duration_s: float, dt_s: float | None = None) -> list[float]:
    """The preset's interval (15 min, or 5 for the quick look), then each longer one the runtime admits, up to
    one frame at the end.

    The runtime admits an interval only when it is a whole number of time steps and splits the storm time into
    whole frames (the history cadence rule), so doubling is not enough: 5 min over 2 h would reach 80 min, and
    30 min does not split 1 h 45 min.  5 min over 2 h gives 5, 10, 15, 20, 30, 40, 60 and 120 min.
    """

    def whole(value: float) -> bool:
        return abs(value - round(value)) < 1e-6

    values = [default_s]
    if default_s <= 0.0 or duration_s <= default_s:
        return values
    for multiple in range(2, int(math.floor(duration_s / default_s + 1e-9)) + 1):
        interval = default_s * multiple
        if whole(duration_s / interval) and (not dt_s or whole(interval / dt_s)):
            values.append(interval)
    if values[-1] < duration_s:
        values.append(duration_s)  # one frame at the end: always a whole number of steps
    return values


@dataclass(frozen=True)
class _Tried:
    candidate: Candidate
    estimate: Estimate | None
    ok: bool
    why: tuple[str, ...]
    settings: Mapping[str, Any] = field(default_factory=dict)


# ------------------------------------------------------------------- fit


@dataclass(frozen=True)
class FitResult:
    """What the guided screen shows for one preset, budget and GPU."""

    preset: PresetSpec
    budget: str
    budget_s: float
    status: str  # "fits" | "not-offered"
    reason: str
    namelist_text: str | None
    gm1_settings: Mapping[str, Any]
    changes: tuple[Mapping[str, Any], ...]
    estimate: Estimate | None
    summary: str
    gpu: GpuInfo | None
    host: HostInfo | None
    candidates_priced: int
    elapsed_s: float
    needs: Mapping[str, Any] = field(default_factory=dict)
    #: What an offered run should be read with (``{"kind", "sentence", ...}``): the storm reaches the edge, a
    #: domain larger than any GM1 has run, a spacing outside what GM1 ships.  A warning never stops a run.
    warnings: tuple[Mapping[str, Any], ...] = ()

    @property
    def offered(self) -> bool:
        return self.status == "fits"

    @property
    def sentences(self) -> list[str]:
        return [change["sentence"] for change in self.changes]

    def record(self) -> dict[str, Any]:
        estimate = self.estimate
        return {
            "schema": FIT_SCHEMA,
            "preset": self.preset.id,
            "preset_name": self.preset.name,
            "budget": self.budget,
            # No limit ("as long as it takes") is null: JSON has no infinity.
            "budget_s": self.budget_s if math.isfinite(self.budget_s) else None,
            "budget_words": _budget_words(self.budget, self.budget_s),
            "status": self.status,
            "reason": self.reason,
            "summary": self.summary,
            "changes": [dict(change) for change in self.changes],
            "sentences": self.sentences,
            "warnings": [str(warning["sentence"]) for warning in self.warnings],
            "warning_records": [dict(warning) for warning in self.warnings],
            "gm1_settings": dict(self.gm1_settings),
            "needs": dict(self.needs),
            "gpu": None if self.gpu is None else self.gpu.record(),
            "host": None if self.host is None else self.host.record(),
            "estimate": None if estimate is None else {
                "cells": estimate.cells,
                "grid": dict(estimate.grid),
                "gpu_memory_bytes": estimate.gpu_need_bytes,
                "host_locked_bytes": estimate.host_locked_bytes,
                "disk_bytes": estimate.disk_bytes,
                "wall_s": estimate.wall_s,
                "wall_low_s": estimate.wall.low,
                "wall_high_s": estimate.wall.high,
                "wall_label": estimate.wall.label,
                "wall_rate_label": estimate.wall.basis_label,
                "min_edge_clearance_km": estimate.min_edge_clearance_km,
                "storm_track_status": estimate.storm.get("status"),
                "dt_s": estimate.dt_s,
                "steps": estimate.steps,
                "largest_square_side_cells": _largest_side(estimate, self.gpu),
            },
            "rule": "cm1core.cm1run.fit",
            "candidates_priced": self.candidates_priced,
            "elapsed_s": round(self.elapsed_s, 3),
        }


#: The cells one restart file holds: 16 GiB at the 280 bytes a cell of a storm's carried state (the estimate's
#: restart-size check prices a restart the same way).
_RESTART_CELLS = 17179869184 / 280.0
#: The estimate's checks that one saved file (a frame, a restart) is within the 16 GiB its format holds.
_FILE_CHECKS = ("frame-size", "restart-size")


def _limits(estimate: Estimate, budget_s: float, *, min_edge_m: float, largest_cells: int) -> tuple[str, ...]:
    """Which of the fit's conditions this candidate breaks (empty: it fits)."""

    broken = []
    if estimate.gpu is not None and estimate.gpu_need.value > FITS_FRACTION_OF_FREE * estimate.gpu.free_bytes:
        broken.append("gpu")
    if estimate.host_ram_bytes and estimate.host_locked.value > HOST_FRACTION * estimate.host_ram_bytes:
        broken.append("host")
    if estimate.disk_free_bytes is not None and estimate.disk.value > DISK_FRACTION * estimate.disk_free_bytes:
        broken.append("disk")
    if estimate.wall.value > budget_s:
        broken.append("time")
    # A domain larger than the largest GM1 has run (``largest_cells``) is not a limit: it runs where it fits, and
    # the fit says so in a warning (_warnings).
    # One saved frame and one restart each hold 16 GiB (the estimate's frame-size and restart-size checks): a
    # limit a smaller area lifts, like memory, so the fit shrinks the area for it instead of giving up.
    if any(check.level == "stop" and check.id in _FILE_CHECKS for check in estimate.checks):
        broken.append("file")
    if any(check.level == "stop" and check.id not in ("gpu-memory", "disk", "storm-track", *_FILE_CHECKS)
           for check in estimate.checks):
        broken.append("refused")
    storm = estimate.storm
    if storm.get("status") == "evaluated" and not storm.get("meets_edge_rule", True):
        broken.append("edge")
    # The run's own start refuses a storm that loses the room it needs on a sounding file (the storm-track check's
    # "door"): no warning takes that one, whatever the user chose.
    if storm.get("door") == "refused":
        broken.append("track")
    return tuple(broken)


def _door_refuses_track(estimate: Estimate, deck: Deck) -> bool:
    """Whether the run's start would refuse this candidate's storm track where the estimate does not follow it.

    The estimate's storm-track check follows CM1's warm bubble only (iinit = 1); the run's start
    (``cartesian_door.require_run_geometry``, which the import runs too) also follows the push of rising air
    (iinit = 12, the famous days) from its centre, and refuses a storm on a sounding file that loses the room it needs
    before the end.  Without this the fit offered small areas the import then refused."""

    if estimate.storm.get("status") != "not-evaluated-not-a-warm-bubble":
        return False
    from .. import cartesian_door as door
    from ..errors import CM1Refusal
    from .estimate import _model_config

    try:
        config = _model_config(deck.raw)
        selection, _ = door.require_initializer_selection(config)
        if selection.isnd not in door.RUN_GEOMETRY_ADMITTED_ISND:
            return False
        door.require_run_geometry(
            config, door.require_grid_extent(config), selection=selection,
            moving_frame=door.require_moving_frame(config), center_xy_m=door.bubble_center_m(config),
            sounding_label="input_sounding", sounding_bytes=deck.sounding_bytes or b"",
        )
    except CM1Refusal as refusal:
        return str(refusal.detail.code).startswith("CM1-CARTESIAN-RUN-GEOMETRY")
    except (TypeError, ValueError, KeyError, AttributeError):
        return False
    return False


def fit_preset(
    preset: PresetSpec,
    gpu: GpuInfo | None,
    budget: str | float,
    *,
    host: HostInfo,
    cache: Sequence[_rates.MeasuredPoint] | None = None,
    min_edge_clearance_m: float = FIT_EDGE_CLEARANCE_M,
    verify: bool = True,
) -> FitResult:
    """Fit ``preset`` to ``gpu``'s free memory, ``host`` and ``budget`` (7.4).

    ``budget`` is ``"10m"``, ``"1h"``, ``"overnight"`` or seconds.  ``host``
    carries the RAM and the free disk under the runs folder.  ``verify``
    re-reads the fitted namelist with the native parser and re-prices it (the
    guard every fitted namelist must pass); tests of the search alone may skip
    it.
    """

    started = time.perf_counter()
    budget_key = str(budget) if isinstance(budget, str) else "custom"
    budget_s = BUDGETS[budget] if isinstance(budget, str) else float(budget)
    if not preset.offered:
        return FitResult(
            preset=preset, budget=budget_key, budget_s=budget_s, status="not-offered",
            reason=preset.not_offered_reason or "Not offered yet.", namelist_text=None, gm1_settings={},
            changes=(), estimate=None, summary="", gpu=gpu, host=host, candidates_priced=0,
            elapsed_s=time.perf_counter() - started,
        )
    record = _rates.load_record()
    largest = int(record["largest_domain_cells"]["value"])
    cache = _rates.read_cache() if cache is None else list(cache)
    sounding = preset.sounding_text.encode("ascii") if preset.sounding_text else None
    priced = 0
    tried: list[_Tried] = []
    # What the estimate said of each candidate it could not price (the not-offered reason quotes the first).
    refusals: list[str] = []

    with tempfile.TemporaryDirectory(prefix="gpuwm-cm1-fit-") as scratch:
        base = _read_preset(preset, Path(scratch))
        # A setting the user chose is pinned: the fit changes only the others.
        pinned = preset.pinned
        rungs = [base.dx_m] + ([] if "grid" in pinned else [value for value in preset.ladder_m if value > base.dx_m])
        durations = [base.duration_s]
        while "time" not in pinned and durations[-1] - SHORTEN_STEP_S >= preset.min_duration_s - 1e-6:
            durations.append(durations[-1] - SHORTEN_STEP_S)
        width_m, depth_m = base.nx * base.dx_m, base.ny * base.dx_m

        seen: dict[Candidate, _Tried] = {}

        def evaluate(candidate: Candidate) -> _Tried:
            if candidate not in seen:
                seen[candidate] = evaluate_once(candidate)
            return seen[candidate]

        def evaluate_once(candidate: Candidate) -> _Tried:
            nonlocal priced
            # The closure tiling is the estimate's own call (estimate.price's
            # recommendation): the whole domain, or tile 128 where the whole
            # domain does not fit or tile 128 costs no step time.  The
            # registry budget is set from the chosen candidate's own peak once
            # it is chosen; while pricing, it must not be what refuses.
            settings: dict[str, Any] = dict(preset.gm1_settings)
            if preset.random_seed is not None:
                settings["initialization.random_seed"] = int(preset.random_seed)
            settings["runtime.memory_budget_mib"] = _PRICING_BUDGET_MIB
            try:
                deck = _deck_for(base, candidate, settings, sounding)
                estimate = price(deck, gpu, host=host, cache=cache)
                advice = estimate.recommendation
                option = advice.get("tile_128_option") or {}
                # Neither tiling fits: price the smaller one, so a card that
                # is not offered is told the least memory the preset needs.
                smaller_and_neither_fits = (
                    estimate.gpu_fits is False and option
                    and option["gpu_memory_bytes"] < estimate.gpu_need.value
                )
                if advice.get("tile_nx") or smaller_and_neither_fits:
                    tile = advice if advice.get("tile_nx") else option
                    settings["runtime.tile_nx"] = int(tile["tile_nx"])
                    settings["runtime.tile_ny"] = int(tile["tile_ny"])
                    deck = _deck_for(base, candidate, settings, sounding)
                    estimate = price(deck, gpu, host=host, cache=cache)
                    priced += 1
            except (EstimateError, TranslationError) as error:
                priced += 1
                refusals.append(str(error))
                return _Tried(candidate, None, False, (f"refused: {error}",), settings)
            priced += 1
            # Raise the output interval until the disk fits (step 5), unless the user chose it.
            if estimate.disk_free_bytes is not None and "pictures" not in preset.pinned:
                intervals = _output_intervals(candidate.output_interval_s, candidate.duration_s, candidate.dt_s)
                for interval in intervals[1:]:
                    if estimate.disk.value <= DISK_FRACTION * estimate.disk_free_bytes:
                        break
                    wider = replace(candidate, output_interval_s=interval)
                    try:
                        deck = _deck_for(base, wider, settings, sounding)
                        trial = price(deck, gpu, host=host, cache=cache)
                    except (EstimateError, TranslationError):
                        priced += 1
                        continue  # an interval the runtime refuses is not a choice; try the next
                    priced += 1
                    candidate, estimate = wider, trial
            broken = _limits(estimate, budget_s, min_edge_m=min_edge_clearance_m, largest_cells=largest)
            if "track" not in broken and _door_refuses_track(estimate, deck):
                broken = (*broken, "track")
            result = _Tried(candidate, estimate, not broken, broken, settings)
            tried.append(result)
            return result

        def sizes_for(dx_m: float) -> list[tuple[int, int]]:
            """Base size at this spacing, then shrinking by 8 cells to the floor."""

            nx = int(round(width_m / dx_m))
            ny = int(round(depth_m / dx_m))
            floor = preset.smallest_width_m(dx_m)
            along = None if preset.min_along_repeating_km is None else preset.min_along_repeating_km * 1000.0
            out = [(nx, ny)]
            if "area" in preset.pinned:
                return out  # the area the user chose, and no other
            if preset.shrink == "repeating-first" and (base.periodic_x or base.periodic_y):
                # Shrink the repeating direction first, down to its own floor.
                repeat_floor = along if along is not None else floor
                while True:
                    cx, cy = out[-1]
                    if base.periodic_y and (cy - SHRINK_STEP) * dx_m >= repeat_floor - 1e-6:
                        out.append((cx, cy - SHRINK_STEP))
                    elif base.periodic_x and not base.periodic_y and (cx - SHRINK_STEP) * dx_m >= repeat_floor - 1e-6:
                        out.append((cx - SHRINK_STEP, cy))
                    else:
                        break
                while True:
                    cx, cy = out[-1]
                    if base.periodic_y and (cx - SHRINK_STEP) * dx_m >= floor - 1e-6:
                        out.append((cx - SHRINK_STEP, cy))
                    elif base.periodic_x and not base.periodic_y and (cy - SHRINK_STEP) * dx_m >= floor - 1e-6:
                        out.append((cx, cy - SHRINK_STEP))
                    else:
                        break
                return out
            while True:
                cx, cy = out[-1]
                if min(cx - SHRINK_STEP, cy - SHRINK_STEP) * dx_m < floor - 1e-6:
                    break
                out.append((cx - SHRINK_STEP, cy - SHRINK_STEP))
            return out

        def strict(item: _Tried) -> bool:
            return item.ok

        def with_warnings(item: _Tried) -> bool:
            """Fits, but for a storm that comes close to the edge or reaches it, which the run itself allows."""

            if item.estimate is None or set(item.why) - {"edge"}:
                return False
            return item.estimate.storm.get("door") != "refused"

        # With a choice of the user's own, what can run is offered: the steps once as they are, then once more
        # taking a storm near the edge with a warning (the candidates are priced once, and kept).
        chosen: _Tried | None = None
        for accept in ([strict, with_warnings] if preset.pinned else [strict]):
            for duration in durations:
                for dx_m in rungs:
                    dt = rung_dt_s(base, dx_m)
                    sizes = sizes_for(dx_m)
                    # Resource needs fall as the area shrinks and the edge
                    # clearance falls with it, so the first fitting size in the
                    # fit's order is the largest one whose resources fit, if
                    # that one still meets the floor and the edge rule.  Find it
                    # by bisection over the ordered sizes.
                    def cand(i: int) -> Candidate:
                        nx, ny = sizes[i]
                        return Candidate(nx, ny, dx_m, dt, duration, base.output_interval_s)

                    first = evaluate(cand(0))
                    if accept(first):
                        chosen = first
                        break
                    resource_limits = {"gpu", "host", "disk", "time", "size", "file"}
                    if not (set(first.why) & resource_limits):
                        # Not a size problem (edge rule or a refusal at base size):
                        # smaller areas only make the edge worse.
                        continue
                    low, high = 0, len(sizes) - 1
                    last = evaluate(cand(high)) if high > 0 else first
                    if last.estimate is None and first.estimate is not None and high > 0:
                        # The smallest areas can be ones the run refuses (a warm bubble placed off centre no
                        # longer lies inside them): the search then ends at the smallest area that can be priced.
                        priced_at, refused_at = 0, high
                        while refused_at - priced_at > 1:
                            middle = (priced_at + refused_at) // 2
                            if evaluate(cand(middle)).estimate is None:
                                refused_at = middle
                            else:
                                priced_at = middle
                        high = priced_at
                        last = evaluate(cand(high))
                        if high == 0:
                            continue  # only the whole area can be priced, and it does not fit
                    if set(last.why) & resource_limits:
                        continue  # even the smallest area does not fit on resources
                    # invariant: sizes[low] breaks a resource limit, sizes[high] does not
                    best = last
                    while high - low > 1:
                        middle = (low + high) // 2
                        trial = evaluate(cand(middle))
                        if set(trial.why) & resource_limits:
                            low = middle
                        else:
                            high, best = middle, trial
                    if accept(best):
                        chosen = best
                        break
                if chosen is not None:
                    break
            if chosen is not None:
                break

        if chosen is None:
            def price_grid(dx_m: float, width_m: float, depth_m: float, duration_s: float,
                           more_cells: int = 0) -> _Tried | None:
                """Another spacing over an area (rounded up to whole cells, and ``more_cells`` more a side), priced
                as the fit prices its own candidates: a suggestion in the not-offered reason is checked this way
                before it is made."""

                # At the spacing in use the step is the run's own (a preset can ship a shorter one than the rule gives).
                dt = base.dt_s if dx_m == base.dx_m else _dt_at(base, dx_m, duration_s)
                if dt is None:
                    return None
                nx = max(8, int(math.ceil(width_m / dx_m - 1e-9)) + more_cells)
                ny = max(8, int(math.ceil(depth_m / dx_m - 1e-9)) + more_cells)
                return evaluate(Candidate(nx, ny, dx_m, dt, duration_s, base.output_interval_s))

            return _not_offered(
                preset, budget_key, budget_s, gpu, host, list(tried), priced, started, base,
                refusals=refusals, price_grid=price_grid,
            )
        candidate = chosen.candidate
        settings = dict(chosen.settings)
        estimate = chosen.estimate
        assert estimate is not None
        wanted_mib = int(estimate.recommendation.get("memory_budget_mib", 8192))
        if gpu is not None:
            wanted_mib = min(wanted_mib, int(FITS_FRACTION_OF_FREE * gpu.free_bytes / (1024 * 1024)))
        settings["runtime.memory_budget_mib"] = max(1024, wanted_mib)
        # Only the words whose value changed are rewritten, so a preset that
        # fits as it is keeps its namelist byte for byte.
        changed = {
            key: value for key, value in candidate.values().items()
            if not math.isclose(float(base.effective[key]), float(value), rel_tol=0.0, abs_tol=1e-9)
        }
        text = set_namelist_values(preset.namelist_text, changed) if changed else preset.namelist_text
        if verify:
            estimate = _verify(preset, text, base, settings, gpu, host, cache, Path(scratch))
            priced += 1
            broken = _limits(estimate, budget_s, min_edge_m=min_edge_clearance_m, largest_cells=largest)
            # A storm near the edge was taken with a warning (with_warnings); anything else must still hold.
            if set(broken) - (set() if chosen.ok else {"edge"}):
                raise FitError(
                    f"the fitted namelist of {preset.id} re-imports but no longer fits ({', '.join(broken)})"
                )
        changes = _changes(base, candidate, tried, budget_key, budget_s, gpu, host,
                           start_capped=preset.start_capped)
        summary = _summary(candidate, estimate, gpu)
        warnings = _warnings(preset, base, candidate, estimate, largest, min_edge_clearance_m)
        return FitResult(
            preset=preset, budget=budget_key, budget_s=budget_s, status="fits", reason="",
            namelist_text=text, gm1_settings=settings, changes=tuple(changes), estimate=estimate,
            summary=summary, gpu=gpu, host=host, candidates_priced=priced,
            elapsed_s=time.perf_counter() - started,
            needs={
                "gpu_memory_bytes": estimate.gpu_need_bytes,
                "disk_bytes": estimate.disk_bytes,
                "wall_s": estimate.wall_s,
            },
            warnings=tuple(warnings),
        )


def _verify(
    preset: PresetSpec, text: str, base: _Base, settings: Mapping[str, Any], gpu: GpuInfo | None,
    host: HostInfo, cache: Sequence[_rates.MeasuredPoint], scratch: Path,
) -> Estimate:
    """The guard: re-read the fitted namelist with the real parser, and re-price it."""

    from ..cm1import import import_namelist

    path = scratch / "fitted.namelist.input"
    path.write_text(text, encoding="ascii")
    effective = import_namelist(path, translate_now=False).effective
    imported = import_namelist(path, gm1_only={**_restart_at_end(effective), **_random_numbers(effective, settings),
                                               **settings})
    if imported.translation is None or imported.unknown or imported.cm1_stops:
        raise FitError(f"the fitted namelist of {preset.id} does not import cleanly")
    for key in FROZEN_KEYS:
        if key in base.effective and imported.effective.get(key) != base.effective.get(key):
            raise FitError(f"the fit changed {key}, which it must never change")
    raw = tomllib.loads(imported.toml())
    sounding = preset.sounding_text.encode("ascii") if preset.sounding_text else None
    deck = Deck(raw=raw, source=str(path), kind="namelist", name=f"{preset.id} (fitted)", sounding_bytes=sounding,
                namelist_import=imported)
    return price(deck, gpu, host=host, cache=cache)


def _largest_side(estimate: Estimate, gpu: GpuInfo | None) -> int | None:
    """The cells along one side of the largest square area, at this run's levels, that fits the GPU's free memory
    (``estimate.largest_square_domain``, the estimate's own "largest box"): the finest grid a card can run a storm
    at is its smallest area over this."""

    if gpu is None or not estimate.cells:
        return None
    from .estimate import largest_square_domain

    per_cell = estimate.priced_peak_bytes / estimate.cells
    box = largest_square_domain(
        free_bytes=gpu.free_bytes, nz=int(estimate.grid["nz"]), dx_m=float(estimate.grid["dx_m"]),
        component=estimate.component, priced_per_cell=min(per_cell, 420.0) if per_cell else 420.0,
    )
    return int(box["n"]) or None


def _share_that_fits(estimate: Estimate, gpu: GpuInfo | None) -> float | None:
    """For a priced candidate too large for the GPU: the share of its cells that would fit the free memory, by the
    memory model's own sum (a fixed part, and the rest in step with the cells)."""

    if gpu is None:
        return None
    model = _rates.load_record()["memory_model"]
    fixed = float(model["fixed_bytes"]) + float(model.get("component_extra_bytes", {}).get(estimate.component, 0.0))
    usable = FITS_FRACTION_OF_FREE * gpu.free_bytes - fixed
    grows = estimate.gpu_need.value - fixed
    if usable <= 0.0 or grows <= 0.0 or usable >= grows:
        return None
    return usable / grows


def _area_words(candidate: Candidate) -> str:
    """The area simulated, exactly: "59.0 x 59.0 km" for 880 cells of 67 m, "120 km" for a whole number."""

    width, depth = candidate.nx * candidate.dx_m / 1000.0, candidate.ny * candidate.dx_m / 1000.0

    def km(value: float) -> str:
        return f"{value:.0f}" if abs(value - round(value)) < 0.05 else f"{value:.1f}"

    return f"{km(width)} km" if candidate.nx == candidate.ny else f"{km(width)} × {km(depth)} km"


def _warnings(
    preset: PresetSpec, base: _Base, candidate: Candidate, estimate: Estimate, largest_cells: int,
    min_edge_m: float,
) -> list[dict[str, Any]]:
    """What an offered run should be read with: each a sentence, none of them a reason not to run."""

    out: list[dict[str, Any]] = [{"kind": "spacing", "sentence": note} for note in preset.notes]
    storm = estimate.storm
    side_m = min(candidate.nx, candidate.ny) * candidate.dx_m
    area = _area_words(candidate)
    own_m = preset.min_width_m(candidate.dx_m)
    # Two signs of an area too small for the storm: smaller than the area the preset was set up for, or short of
    # the room the storm-track check gives the storm.  Either one gets the same note, whatever the track check
    # predicts for a view that follows the storm: on an RTX 4090 a 67 m storm left a 34 km area within 45 minutes
    # although the check, with the view moving as it advised, had it staying inside.
    small = side_m < own_m - 1e-6
    edge = storm.get("status") == "evaluated" and not storm.get("meets_edge_rule", True)
    if small or edge:
        clearance = storm.get("minimum_edge_clearance_km")
        exit_s = storm.get("exit_time_s") if edge else None
        needs = storm.get("admission_clearance_km")
        required = storm.get("required_domain_km") or ()
        if small:
            size = f"smaller than the {_km(own_m)} this storm was set up for"
        elif required:
            size = f"smaller than the {max(float(value) for value in required):.0f} km the storm-track check gives this storm"
        else:
            size = "smaller than the room the storm-track check gives this storm"
        sentence = (f"This {area} area is {size}: the storm may leave an area this small; pictures near the edges "
                    "then show edge effects.")
        if edge and not storm.get("stays_inside", True) and exit_s is not None:
            # The storm-track check's own exit time: when the predicted storm is closer to an open side than the
            # room it needs (its "admission clearance").  A prediction, said as one.
            room = f"closer than {needs:.0f} km to the edge" if needs is not None else "too close to the edge"
            sentence += (f" The storm-track check puts it {room} from the start." if exit_s < 30.0 else
                         f" The storm-track check puts it {room} from about T+{exit_s / 60.0:.0f} min.")
        elif edge and clearance is not None:
            sentence += (f" The storm-track check puts it within {clearance:.0f} km of the edge (GM1 aims for "
                         f"{min_edge_m / 1000.0:.0f} km).")
        record: dict[str, Any] = {"kind": "edge", "exit_time_s": exit_s, "clearance_km": clearance, "lead": sentence,
                                  "set_up_for_km": round(own_m / 1000.0, 3)}
        pair = storm.get("recommended_pair") or {}
        frame, bubble = pair.get("frame_m_s"), pair.get("bubble_km")
        follows = (
            edge and int(base.effective.get("imove", 0) or 0) == 1 and int(base.effective.get("iinit", 0) or 0) in (1, 12)
            and frame and bubble and not pair.get("uses_it")
        )
        if follows:
            record["recommended_pair"] = {"frame_m_s": list(frame), "bubble_km": list(bubble),
                                          "stays_inside": pair.get("stays_inside"),
                                          "minimum_edge_clearance_km": pair.get("minimum_edge_clearance_km")}
            long_km = max(candidate.nx, candidate.ny) * candidate.dx_m / 1000.0
            if "area" not in preset.pinned:
                # Choosing the area makes Customize apply the storm-track check's frame and bubble for it.
                record["follow_area_km"] = round(long_km, 3)
                sentence += (f" Keep this area as your own choice (--custom area={round(long_km, 2):g}) and GM1 moves the view "
                             f"as the storm-track check predicts the storm moves ({float(frame[0]):g} m/s east, "
                             f"{float(frame[1]):g} m/s north); the prediction is not exact.")
        out.append({**record, "sentence": sentence})
    levels = int(estimate.grid.get("nz", 0) or 0)
    if levels > MEASURED_LEVELS_MAX:
        # The speeds the time comes from were measured with 40 to 80 levels.  One run with 200 levels (a 150 m grid,
        # levels every 100 m, on an RTX 4090 another run was sharing) took about twice its estimate.
        out.append({"kind": "levels", "sentence": (
            f"The time is estimated from runs with at most {MEASURED_LEVELS_MAX} levels. With {levels} levels expect "
            "it to take longer: the one run measured with 200 levels took about twice its estimate.")})
    if estimate.cells > largest_cells:
        out.append({"kind": "size", "sentence": (
            f"{estimate.cells / 1e6:.0f} million cells is more than the largest area GM1 has run "
            f"({largest_cells / 1e6:.0f} million), so this size is untried.")})
    return out


def _gb(value: float) -> str:
    return f"{value / 1e9:.1f} GB"


def _spacing(dx_m: float) -> str:
    """A grid spacing as people say it: 250 m, 500 m, 1 km."""

    return f"{dx_m / 1000.0:g} km" if dx_m >= 1000.0 else f"{dx_m:g} m"


def _gb_round(value: float) -> str:
    """A large size as people say it: 4,700 GB, 470 GB, 47 GB, 4.7 GB."""

    gb = value / 1e9
    if gb >= 1000.0:
        return f"{round(gb, -2):,.0f} GB"
    if gb >= 100.0:
        return f"{round(gb, -1):,.0f} GB"
    return f"{gb:.0f} GB" if gb >= 10.0 else f"{gb:.1f} GB"


def _km(value_m: float) -> str:
    return f"{value_m / 1000.0:.0f} km"


#: The fit's limits in plain words, for a preset no candidate fits (step 6).
LIMIT_WORDS: Mapping[str, str] = {
    "gpu": "GPU memory", "host": "RAM", "disk": "disk space", "time": "time", "size": "room within the largest area GM1 has run",
    "file": "room in one saved file (a frame or a restart holds 16 GiB)",
}


def _because(reasons: set[str], budget_key: str, budget_s: float, gpu: GpuInfo | None, host: HostInfo) -> str:
    words = []
    if "gpu" in reasons and gpu is not None:
        words.append(f"fits in your GPU's {_gb(gpu.free_bytes)} of free memory")
    if "time" in reasons:
        words.append(_finish_words(budget_key, budget_s))
    if "host" in reasons and host.ram_total_bytes:
        words.append(f"needs no more than half of your {_gb(host.ram_total_bytes)} of RAM")
    if "disk" in reasons and host.disk_free_bytes:
        words.append(f"fits in half of your {_gb(host.disk_free_bytes)} of free disk")
    if "size" in reasons:
        words.append("stays within the largest area GM1 has run")
    if "file" in reasons:
        words.append("fits in one restart file (16 GiB)")
    if not words:
        return ""
    return " so it " + " and ".join(words)


def _changes(
    base: _Base, candidate: Candidate, tried: Sequence[_Tried], budget_key: str, budget_s: float,
    gpu: GpuInfo | None, host: HostInfo, *, start_capped: bool = False,
) -> list[dict[str, Any]]:
    """Each change the fit made, as a sentence with its reason."""

    base_reasons: set[str] = set()
    for item in tried:
        if (item.candidate.nx, item.candidate.ny, item.candidate.dx_m, item.candidate.duration_s) == (
            base.nx, base.ny, base.dx_m, base.duration_s
        ):
            base_reasons |= set(item.why)

    def reasons_of(pick: Any, **same: Any) -> str:
        """Why one step was needed: the limits its alternative broke (the base's, when none was tried)."""

        rows = [item for item in tried if all(getattr(item.candidate, key) == value for key, value in same.items())]
        chosen = pick(rows, key=lambda item: item.candidate.nx * item.candidate.ny) if rows else None
        return _because(set(chosen.why) if chosen is not None else base_reasons, budget_key, budget_s, gpu, host)

    # A coarser grid was needed because even the smallest area at the preset's spacing broke a limit; a smaller
    # area because the whole area at the chosen spacing did; a shorter storm because every area and spacing at
    # the preset's storm time did.
    because_coarser = reasons_of(min, dx_m=base.dx_m, duration_s=base.duration_s)
    # A smaller area: what the next larger area tried at this spacing broke is what binds (the whole area can break
    # more, memory as well as time, where only the time held the chosen size back).
    larger = [item for item in tried
              if item.candidate.dx_m == candidate.dx_m and item.candidate.duration_s == candidate.duration_s
              and item.candidate.nx * item.candidate.ny > candidate.nx * candidate.ny and item.estimate is not None]
    next_larger = min(larger, key=lambda item: item.candidate.nx * item.candidate.ny) if larger else None
    because_smaller = (
        _because(set(next_larger.why), budget_key, budget_s, gpu, host) if next_larger is not None else ""
    ) or reasons_of(max, dx_m=candidate.dx_m, duration_s=candidate.duration_s)
    because_shorter = reasons_of(min, duration_s=base.duration_s)
    changes: list[dict[str, Any]] = []
    width_before = (base.nx * base.dx_m, base.ny * base.dx_m)
    width_after = (candidate.nx * candidate.dx_m, candidate.ny * candidate.dx_m)
    # When the preset's own spacing did not fit the GPU, say what its full-size version needs, so a card on a
    # smaller GPU tells the user what a larger one would run (the estimate's own price of the base candidate).
    # Free memory, as the fit tests it: a run is offered only where its need is at most FITS_FRACTION_OF_FREE of
    # what is free, so the free memory it needs is the need over that fraction (as the not-offered reason says;
    # the sentence once said the need itself, 5 % short of what a card must have free).
    full_size_need = ""
    for item in tried:
        c = item.candidate
        if (c.nx, c.ny, c.dx_m, c.duration_s) == (base.nx, base.ny, base.dx_m, base.duration_s) and item.estimate:
            if "gpu" in item.why and item.estimate.gpu_need is not None:
                full_size_need = (
                    f"; its full {_spacing(base.dx_m)} version needs about "
                    f"{_gb(item.estimate.gpu_need.value / FITS_FRACTION_OF_FREE)} of free GPU memory"
                )
            break
    if candidate.dx_m != base.dx_m:
        keys: dict[str, list[Any]] = {
            "dx": [base.dx_m, candidate.dx_m], "dy": [base.dx_m, candidate.dx_m],
            "dtl": [base.dt_s, candidate.dt_s],
        }
        # The same area at a coarser spacing has fewer cells across it; the
        # area sentence below names nx/ny only when the area itself changed.
        area_changed = (
            abs(width_after[0] - width_before[0]) > 1.0 or abs(width_after[1] - width_before[1]) > 1.0
        )
        if (candidate.nx, candidate.ny) != (base.nx, base.ny) and not area_changed:
            keys.update(nx=[base.nx, candidate.nx], ny=[base.ny, candidate.ny])
        changes.append({
            "keys": keys,
            "sentence": (
                f"Used {_spacing(candidate.dx_m)} grid spacing instead of {_spacing(base.dx_m)} (and a "
                f"{candidate.dt_s:g} s time step instead of {base.dt_s:g} s"
                + (
                    f", no longer than CM1's own 6 s per 500 m of level height with these "
                    f"{_spacing(base.dz_min_m)} levels"
                    if candidate.dt_s < base.dt_s * candidate.dx_m / base.dx_m - 1e-9 else ""
                )
                + f"){because_coarser}"
                + full_size_need
                + "."
            ),
        })
    if abs(width_after[0] - width_before[0]) > 1.0 or abs(width_after[1] - width_before[1]) > 1.0:
        square = abs(width_after[0] - width_after[1]) < 1.0 and abs(width_before[0] - width_before[1]) < 1.0
        if square and _km(width_after[0]) == _km(width_before[0]):
            # A change of a few cells reads the same in whole km: said to a tenth, so it is not "40 km instead of 40".
            area = (f"Made the area {width_after[0] / 1000.0:.1f} km wide instead of "
                    f"{width_before[0] / 1000.0:.1f} km")
        elif square:
            area = f"Made the area {_km(width_after[0])} wide instead of {_km(width_before[0])}"
        else:
            area = (
                f"Made the area {_km(width_after[0])} × {_km(width_after[1])} instead of "
                f"{_km(width_before[0])} × {_km(width_before[1])}"
            )
        if start_capped:
            # The area a fine grid starts from is not the preset's but the most one restart file holds (Customize).
            area += f" (the most one restart file holds at {_spacing(base.dx_m)})"
        changes.append({
            "keys": {"nx": [base.nx, candidate.nx], "ny": [base.ny, candidate.ny]},
            "sentence": area + because_smaller + ".",
        })
    if candidate.duration_s != base.duration_s:
        changes.append({
            "keys": {"timax": [base.duration_s, candidate.duration_s]},
            "sentence": (
                f"Shortened the storm time to {duration_words(candidate.duration_s)} instead of "
                f"{duration_words(base.duration_s)}{because_shorter}."
            ),
        })
    if candidate.output_interval_s != base.output_interval_s:
        changes.append({
            "keys": {"tapfrq": [base.output_interval_s, candidate.output_interval_s]},
            "sentence": (
                f"Saves a picture every {duration_words(candidate.output_interval_s)} instead of every "
                f"{duration_words(base.output_interval_s)} so the files fit in half of your free disk."
            ),
        })
    return changes


def article(words: str) -> str:
    """"A" or "An" before a number as it is said aloud: an 8 km grid, an 82.9 m grid, an 11 km area, a 100 m grid."""

    whole = re.match(r"\d*", words.strip()).group(0)
    vowel = whole.startswith("8") or (whole[:2] in ("11", "18") and len(whole) % 3 == 2)
    return "An" if vowel else "A"


def _summary(candidate: Candidate, estimate: Estimate, gpu: GpuInfo | None) -> str:
    width = candidate.nx * candidate.dx_m / 1000.0
    depth = candidate.ny * candidate.dx_m / 1000.0
    size = f"{width:.0f} × {depth:.0f}" if abs(width - depth) > 0.5 else f"{width:.0f} × {width:.0f}"
    area = f"{article(size)} {size} km area"            # "An 80 × 80 km area", "A 120 × 120 km area"
    odd_spacing = abs(candidate.dx_m / 50.0 - round(candidate.dx_m / 50.0)) > 1e-6
    if odd_spacing or abs(width - round(width)) > 0.05 or abs(depth - round(depth)) > 0.05:
        # A spacing of the user's own that is not a round number (67 m, 82.9 m), or an area that is not a whole
        # number of km, is said exactly, with its cells, whether or not the area comes out whole.
        size = f"{width:.1f} × {depth:.1f}"
        area = f"{article(size)} {size} km area ({candidate.nx} × {candidate.ny} cells)"
    card = "an RTX 3080 (no GPU found here)" if gpu is None else f"your {_short_name(gpu.name)}"
    wall = duration_words(estimate.wall_s)
    return (
        f"{area} at {_spacing(candidate.dx_m)}, {duration_words(candidate.duration_s)} of storm time. "
        f"About {wall} on {card} (estimated). "
        f"Needs {_gb(estimate.gpu_need.value)} of GPU memory and about {_gb(estimate.disk.value)} of disk."
        # With no GPU found there is no free memory to hold the need to: say so, so the number is not read as checked.
        + (" No GPU was found, so whether that memory is free was not checked." if gpu is None else "")
    )


def _short_name(name: str) -> str:
    return re.sub(r"(?i)^nvidia\s+(geforce\s+)?", "", name).strip()


def _gb_versus(need: float, have: float) -> tuple[str, str]:
    """A need and what is free, rounded so that different amounts never read the same ("8.31 GB" against
    "8.27 GB", not "8.3 GB" against "8.3 GB")."""

    for digits in (1, 2, 3):
        a, b = f"{need / 1e9:.{digits}f} GB", f"{have / 1e9:.{digits}f} GB"
        if a != b:
            return a, b
    return a, b


#: What holds an area back, in the words of "the largest area that fits ...".
_BINDS: Mapping[str, str] = {
    "gpu": "your GPU's free memory", "host": "half of your RAM", "disk": "half of your free disk",
    "file": "one restart file (16 GiB)",
}


def _storm_need(left: _Tried, estimates: Sequence[_Tried], price_grid: Any) -> _Tried | None:
    """The smallest area the storm stays inside to the end, at the spacing and storm time of ``left`` (an area it
    leaves), priced; None when no area tried kept it in.

    The fit's search looked at a few areas only, so the gap between the largest area the storm leaves and the
    smallest it stays in is halved down to 8 cells: the area is then the storm's need, not where the search
    stopped."""

    c = left.candidate
    same = [item for item in estimates if item.candidate.dx_m == c.dx_m and item.candidate.duration_s == c.duration_s]
    kept = [item for item in same if "track" not in item.why]
    need = min(kept, key=lambda item: item.candidate.nx * item.candidate.ny) if kept else None
    if need is None or price_grid is None:
        return need
    gone = [item.candidate.nx for item in same if "track" in item.why and item.candidate.nx < need.candidate.nx]
    low, high, n0 = max(gone, default=need.candidate.nx), need.candidate.nx, need.candidate
    while high - low > 8:
        middle = (low + high) // 2
        trial = price_grid(c.dx_m, middle * c.dx_m, middle * c.dx_m * n0.ny / n0.nx, c.duration_s)
        if trial is None or trial.estimate is None or "track" in trial.why:
            low = middle
        else:
            high, need = middle, trial
    return need


def _track_words(
    leaves: Sequence[_Tried], estimates: Sequence[_Tried], preset: PresetSpec, budget_key: str, budget_s: float,
    checked: Any, price_grid: Any = None,
) -> tuple[str, float | None]:
    """Why no area fits: the largest that fits this computer and the time is one the storm leaves before the end (the
    run's own start refuses it).  Says the area the storm needs when one was tried, and a coarser grid that fits it,
    checked by pricing (``checked``).  Returns the sentence and that grid spacing (or None)."""

    widest = max(leaves, key=lambda item: item.candidate.nx * item.candidate.ny)
    c = widest.candidate
    same = [item for item in estimates if item.candidate.dx_m == c.dx_m and item.candidate.duration_s == c.duration_s]
    larger = [item for item in same if item.candidate.nx * item.candidate.ny > c.nx * c.ny]
    broken = {why for item in larger for why in item.why}
    words = [f"fits {_BINDS[why]}" for why in sorted(broken & set(_BINDS))]
    if "time" in broken:
        words.append(_finish_words(budget_key, budget_s))
    need = _storm_need(widest, estimates, price_grid)
    if words:
        text = (f"At {_spacing(c.dx_m)} the largest area that {' and '.join(words)} is {_area_words(c)}, and the "
                "storm leaves it before the end, which the run's own start refuses")
    else:
        text = (f"At {_spacing(c.dx_m)} the storm leaves even a {_area_words(c)} area before the end, which the run's "
                "own start refuses")
    if need is not None:
        text += f"; it needs about {_area_words(need.candidate)} to stay inside"
    text += "."
    grid: float | None = None
    if need is not None and "grid" in preset.pinned:
        n = need.candidate
        # The same cells over the larger area: a coarser grid in step with its side, checked by pricing.
        guess = c.dx_m * math.sqrt((n.nx * n.ny) / max(1, c.nx * c.ny))
        grid = checked(guess, n.nx * n.dx_m, n.ny * n.dx_m, n.duration_s)
        if grid is not None:
            text += f" A {_spacing(grid)} grid over that area fits."
    if "time" in preset.pinned:
        text += " A shorter storm time keeps the storm inside longer."
    return text, grid


def _finish_words(budget_key: str, budget_s: float) -> str:
    """A budget as what a run must do: "finishes in about 1 hour", "finishes overnight (within 10 hours)"."""

    if budget_key == "overnight":
        return "finishes overnight (within 10 hours)"
    if budget_key == "any" or not math.isfinite(budget_s):
        return "finishes"
    return f"finishes in {_budget_words(budget_key, budget_s)}"


def _not_offered(
    preset: PresetSpec, budget_key: str, budget_s: float, gpu: GpuInfo | None, host: HostInfo,
    tried: Sequence[_Tried], priced: int, started: float, base: _Base, *,
    refusals: Sequence[str] = (), price_grid: Any = None,
) -> FitResult:
    """Step 6: the smallest GPU and the shortest time any candidate needed, in plain words.

    ``price_grid(dx_m, width_m, depth_m, duration_s)`` prices another spacing over an area, so a spacing the reason
    suggests is one the fit itself would take, not only what a sum gives."""

    estimates = [item for item in tried if item.estimate is not None]
    finest_fit: float | None = None
    suggest: dict[str, float] = {}
    resources = {"gpu", "host", "disk", "time", "file"}

    def runs(item: _Tried | None) -> bool:
        return item is not None and item.estimate is not None and not ((resources | {"refused", "track"}) & set(item.why))

    def checked(dx_m: float, width_m: float, depth_m: float, duration_s: float,
                apart_from: frozenset[str] = frozenset(), more_cells: int = 0) -> float | None:
        """The finest whole-metre spacing near ``dx_m`` that the fit prices as fitting ``width_m`` x ``depth_m``:
        coarser a step at a time until one fits, then finer while the next still does, so the spacing named does
        not depend on where the search started.  ``apart_from``: limits that are not this suggestion's to lift
        (the time, when the reason is memory)."""

        if price_grid is None:
            return None

        def fits(value: float) -> bool:
            item = price_grid(value, width_m, depth_m, duration_s, more_cells)
            return item is not None and runs(replace(item, why=tuple(set(item.why) - apart_from)))

        value = float(math.ceil(dx_m))
        for _ in range(12):
            if fits(value):
                break
            value = value + 1.0 if value < 200.0 else float(math.ceil(value * 1.02))
        else:
            return None
        for _ in range(8):
            if value - 1.0 < 1.0 or not fits(value - 1.0):
                break
            value -= 1.0
        return value

    if not estimates:
        # What the estimate itself said, when it said something: "could not be priced" alone hides the cause.
        said = refusals[0] if refusals else ""
        reason = "It could not be priced on this machine" + (f": {said.rstrip('.')}." if said else ".")
        needs: dict[str, Any] = {"kind": "unpriced"}
    else:
        # A pinned area is the user's own: no "smallest size" then, only the size they chose.
        at = "at the area you chose" if "area" in preset.pinned else "even at its smallest size"
        if "area" not in preset.pinned and any("track" in item.why for item in estimates):
            # Smaller areas were priced too, but the storm leaves them and the run's own start refuses that.
            at = "even at the smallest area the storm stays inside"
        # With a choice of the user's own a storm near the edge is a warning, so the small areas the fit went down
        # to count here: the reason is what stops the smallest of them, not what stops the whole area.
        # A storm the run's own start refuses ("track") is never offered, so the small areas it is refused in do not
        # count either: the reason is then that the areas that fit are too small for the storm (below).
        barred = {"refused", "size", "track"} if preset.pinned else {"edge", "refused", "size", "track"}
        allowed = [item for item in estimates if not (barred & set(item.why))] or estimates
        smallest = min(allowed, key=lambda item: item.estimate.gpu_need.value)  # type: ignore[union-attr]
        fastest = min(allowed, key=lambda item: item.estimate.wall.value)  # type: ignore[union-attr]
        s_est = smallest.estimate
        f_est = fastest.estimate
        assert s_est is not None and f_est is not None
        width = smallest.candidate.nx * smallest.candidate.dx_m
        depth = smallest.candidate.ny * smallest.candidate.dx_m
        parts = []
        smallest_host = min(allowed, key=lambda item: item.estimate.host_locked.value)  # type: ignore[union-attr]
        smallest_disk = min(allowed, key=lambda item: item.estimate.disk.value)  # type: ignore[union-attr]
        # Areas that fit this computer and the time, where the storm leaves before the end and the run's start
        # refuses it: no budget and no card changes that, only a larger area (at a coarser grid) or a shorter storm.
        fits_but_leaves = [item for item in estimates if "track" in item.why and not (resources & set(item.why))]
        gone = [item for item in estimates if "track" in item.why]
        if gone and not fits_but_leaves:
            # Every area the storm leaves is also too much for this computer or the time: the reason is then what
            # stops the smallest area the storm stays inside, which the search may not have tried itself.
            need = _storm_need(max(gone, key=lambda item: item.candidate.nx * item.candidate.ny), estimates,
                               price_grid)
            if need is not None and need not in allowed:
                allowed = [need, *allowed]
            smallest = min(allowed, key=lambda item: item.estimate.gpu_need.value)  # type: ignore[union-attr]
            fastest = min(allowed, key=lambda item: item.estimate.wall.value)  # type: ignore[union-attr]
            s_est, f_est = smallest.estimate, fastest.estimate
            assert s_est is not None and f_est is not None
            width = smallest.candidate.nx * smallest.candidate.dx_m
            depth = smallest.candidate.ny * smallest.candidate.dx_m
            smallest_host = min(allowed, key=lambda item: item.estimate.host_locked.value)  # type: ignore[union-attr]
            smallest_disk = min(allowed, key=lambda item: item.estimate.disk.value)  # type: ignore[union-attr]
        if fits_but_leaves:
            kind = "limits"
            text, grid_hint = _track_words(fits_but_leaves, estimates, preset, budget_key, budget_s, checked,
                                           price_grid)
            parts.append(text)
            if grid_hint is not None:
                suggest["grid"] = grid_hint
        elif gpu is not None and all("gpu" in item.why for item in allowed):
            kind = "gpu"
            parts.append(
                f"Needs {_gb_versus(s_est.gpu_need.value / FITS_FRACTION_OF_FREE, gpu.free_bytes)[0]} of free GPU "
                f"memory {at} ({_area_words(smallest.candidate)}); yours has "
                f"{_gb_versus(s_est.gpu_need.value / FITS_FRACTION_OF_FREE, gpu.free_bytes)[1]}."
            )
            share = _share_that_fits(s_est, gpu)
            if "grid" in preset.pinned and share:
                # What would fit at the smallest area: the sum's own finest spacing (3 % coarser than it gives),
                # never finer than one restart file holds over that area, then priced as the fit prices a run and
                # made coarser a metre at a time until it does fit, so the spacing named is one the fit takes.
                dx_m = smallest.candidate.dx_m
                finest = math.ceil(1.03 * dx_m / math.sqrt(share))
                finest = max(finest, math.ceil(1.01 * width / math.sqrt(_RESTART_CELLS / int(s_est.grid["nz"]))))
                # Priced over the smallest area the fit would go down to at that spacing: the floor, and up to 7
                # cells more a side (the fit moves in steps of 8 cells from the area it starts at).
                # What is priced is the memory: the time, the disk and the RAM have reasons of their own.
                others = frozenset({"time", "disk", "host"})
                if "area" in preset.pinned or any("track" in item.why for item in estimates):
                    # The area is the user's own, or the smallest the storm stays inside: the spacing is priced
                    # over that area (Customize keeps it to within half a step of 8 cells at another spacing).
                    least, over = width, "that area"
                    sure = checked(finest, width, depth, smallest.candidate.duration_s, others, 4)
                else:
                    least = preset.smallest_width_m(float(finest))
                    over = f"its smallest area ({_km(least)})"
                    sure = checked(finest, least, least * depth / width, smallest.candidate.duration_s,
                                   others, 7)
                value = sure if sure is not None else float(finest)
                parts.append(
                    f"The finest grid that fits {over} here is "
                    f"{'' if sure is not None else 'about '}{_spacing(value)}."
                )
                finest_fit = float(value)
                suggest["grid"] = float(value)
        elif all("file" in item.why for item in allowed) and "grid" in preset.pinned:
            # Not a matter of this computer: one restart file holds 16 GiB, whatever the card.
            kind = "limits"
            smallest_cells = min(allowed, key=lambda item: item.estimate.cells)  # type: ignore[union-attr]
            cells = smallest_cells.estimate.cells  # type: ignore[union-attr]
            dx_m = smallest_cells.candidate.dx_m
            finest_fit = float(math.ceil(1.01 * dx_m * math.sqrt(cells / _RESTART_CELLS)))
            suggest["grid"] = finest_fit
            parts.append(
                f"Even its smallest area ({_km(smallest_cells.candidate.nx * dx_m)}) is {cells / 1e6:.0f} million "
                f"cells at {_spacing(dx_m)}, and one restart file holds about {_RESTART_CELLS / 1e6:.0f} million "
                f"(16 GiB); the finest grid that fits here is about {finest_fit:.0f} m."
            )
        elif all("time" in item.why for item in allowed):
            kind = "time"
            # The area the time is for: a longer time lets the fit take a larger one, so the two numbers differ.
            where = ("at the area you chose" if "area" in preset.pinned
                     else f"even at the smallest area the storm stays inside ({_area_words(fastest.candidate)})"
                     if any("track" in item.why for item in estimates)
                     else f"even at its smallest area ({_area_words(fastest.candidate)})")
            parts.append(
                f"Takes about {duration_words(f_est.wall_s)} on your GPU {where}; "
                f"pick a longer time than {_budget_words(budget_key, budget_s)}."
            )
        elif host.ram_total_bytes and all("host" in item.why for item in allowed):
            kind = "host"
            locked = smallest_host.estimate.host_locked.value  # type: ignore[union-attr]
            parts.append(
                f"Needs about {_gb_round(locked / HOST_FRACTION)} of RAM {at} "
                f"({_km(smallest_host.candidate.nx * smallest_host.candidate.dx_m)}), since GM1 uses at most half of "
                f"it; this computer has {_gb_round(host.ram_total_bytes)}."
            )
        elif host.disk_free_bytes and all("disk" in item.why for item in allowed):
            kind = "disk"
            written = smallest_disk.estimate.disk.value  # type: ignore[union-attr]
            fewest = "" if "pictures" in preset.pinned else " and with the fewest saved pictures"
            parts.append(
                f"Needs about {_gb_round(written / DISK_FRACTION)} of free disk {at}{fewest} (GM1 uses at most half "
                f"of what is free); the runs folder has {_gb_round(host.disk_free_bytes)} free."
            )
        elif any("edge" in item.why for item in estimates) and not (
            {"gpu", "host", "disk", "time", "size", "file"} & {reason for item in allowed for reason in item.why}
        ):
            kind = "edge"
            edges = [
                item.estimate.min_edge_clearance_km for item in estimates
                if "edge" in item.why and item.estimate is not None and item.estimate.min_edge_clearance_km is not None
            ]
            closest = max(edges) if edges else None
            parts.append(
                "The storm would come too close to the edge of every area it fits in"
                + (f" (at best {closest:.1f} km; it needs {FIT_EDGE_CLEARANCE_M / 1000:.0f} km)" if closest is not None else "")
                + "."
            )
        else:
            kind = "limits"
            reasons = sorted({reason for item in allowed for reason in item.why} - {"edge", "refused"})
            short = [LIMIT_WORDS[reason] for reason in reasons if reason in LIMIT_WORDS]
            parts.append(f"It does not fit this computer {at.replace(' size', '')}"
                         + (": too little " + " and ".join(short) if short else "") + ".")
        reason = " ".join(parts)
        pinned = [knob for knob in ("grid", "area", "time", "pictures") if knob in preset.pinned]
        if pinned:
            # The user's own choices hold the fit back: say which, so they can hand them back to GM1.
            words = {"grid": "grid spacing", "area": "area", "time": "storm time", "pictures": "picture interval"}
            said = [words[knob] for knob in pinned]
            said = said[0] if len(said) == 1 else ", ".join(said[:-1]) + " and " + said[-1]
            reason = (reason[:-1] + f", or let GM1 choose the {said}."
                      if reason.endswith(".") and "pick a longer time" in reason
                      else reason + f" Or let GM1 choose the {said}.")
        needs = {
            "smallest_gpu_memory_bytes": s_est.gpu_need.value,
            "smallest_gpu_free_bytes_needed": s_est.gpu_need.value / FITS_FRACTION_OF_FREE,
            "smallest_width_km": width / 1000.0,
            "shortest_wall_s": f_est.wall_s,
            "pinned": pinned,
            "kind": kind,
            "finest_grid_m": finest_fit,
            "largest_square_side_cells": _largest_side(s_est, gpu),
            # A setting that would be offered, for a one-tap button ({"grid": 23.0}); empty when none was found.
            "suggest": dict(suggest),
        }
    return FitResult(
        preset=preset, budget=budget_key, budget_s=budget_s, status="not-offered", reason=reason,
        namelist_text=None, gm1_settings={}, changes=(), estimate=None, summary="", gpu=gpu, host=host,
        candidates_priced=priced, elapsed_s=time.perf_counter() - started, needs=needs,
    )


__all__ = [
    "BUDGETS",
    "BUDGET_WORDS",
    "FIT_KEYS",
    "FIT_SCHEMA",
    "FROZEN_KEYS",
    "FitError",
    "FitResult",
    "PresetSpec",
    "fit_preset",
    "set_namelist_values",
]
