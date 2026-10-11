"""Compare a CM1 run against a gpuwm-cm1 run, field by field and time by time.

Bit identity is not the expectation.  CM1 r21.1 built here is gfortran with no
``-DDP``, so its prognostic arrays are IEEE binary32 -- the same width as
GM1's -- but the two codes evaluate the same mathematics in a different order,
on different hardware, with different reductions.  What must hold is that the
disagreement starts at rounding and *does not grow*: the useful output is the
divergence curve, not a pass/fail on one frame.

For each field at each history time this reports

  * ``max_abs``   -- the largest absolute difference and where it is;
  * ``max_rel``   -- that difference divided by the field's own RMS magnitude
                     in the CM1 frame, which is a stated, reproducible
                     denominator that does not explode near zero crossings;
  * ``rmse`` and ``nrmse`` -- the whole-field error and its normalised form;
  * ``max_abs_over_peak`` -- the same largest difference against the field's
                     own peak magnitude in that CM1 frame.  On a field that is
                     zero over most of the domain (a warm bubble's thpert, a
                     quiescent w) the RMS denominator is dominated by
                     background and makes a small error look enormous; this
                     column is the one to read for such fields;
  * ``first_exceedance`` -- the first model time at which ``nrmse`` crosses the
                     declared tolerance for that field.

Staggered fields are compared on their own grids: CM1's ``output_format=1``
history writes u, v and w at native stagger, and GM1's carriers are the same
interior extents, so no interpolation is introduced between the two codes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field as _field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .grads import CM1History
from .gm1_frames import GM1History

#: Comparison field -> the gpuwm-cm1 carrier that holds it.  A tuple value
#: selects one species out of a species-major carrier.
#:
#: ``th3d`` is the trap.  Its field-table meaning string says "current
#: potential temperature" and its CM1 name suggests ``th``, but the carrier
#: holds the *perturbation*: on the isnd=5 base state its values run
#: -1.2e-3 .. 0.96 K while CM1's ``th`` runs 300 .. 489 K.  It is CM1's
#: ``thpert``, so that is what it is compared against.  Pairing it with ``th``
#: reports an nrmse of 1.0 at every frame, which looks like a catastrophic
#: dycore failure and is only a name.
FIELD_MAP: Mapping[str, str | tuple[str, int]] = {
    "th3d": "th3d",
    "prs": "prs",
    "rho": "rho",
    "u3d": "u3d",
    "v3d": "v3d",
    "w3d": "w3d",
    "closure.tkea": "closure.tkea",
    "q3d:qv": ("q3d", 0),
    "q3d:qc": ("q3d", 1),
    "q3d:qr": ("q3d", 2),
    "q3d:qi": ("q3d", 3),
    "q3d:qs": ("q3d", 4),
    "q3d:qg": ("q3d", 5),
    # CM1 &param9 products.  These carry CM1's own writeout.F name_output
    # strings on both sides, so the pair is the identity; what the comparison
    # establishes is that GM1's calcuh/calcvort/solve3-swath transcriptions
    # give CM1's answer on the same state.
    "uh": "uh",
    "xvort": "xvort",
    "yvort": "yvort",
    "zvort": "zvort",
    "pv": "pv",
    "sws": "sws",
    "svs": "svs",
    "sps": "sps",
    "srs": "srs",
    "sgs": "sgs",
    "sus": "sus",
    "shs": "shs",
    "rain": "rain",
    "prate": "prate",
    "cape": "cape",
    "cin": "cin",
    "lcl": "lcl",
    "lfc": "lfc",
    "th": "th",
    "pwat": "pwat",
    "lwp": "lwp",
}

#: CM1 name for each comparable field.
CM1_NAME: Mapping[str, str] = {
    "th3d": "thpert", "prs": "prs", "rho": "rho", "u3d": "u", "v3d": "v",
    "w3d": "w", "closure.tkea": "tke", "q3d:qv": "qv", "q3d:qc": "qc",
    "q3d:qr": "qr", "q3d:qi": "qi", "q3d:qs": "qs", "q3d:qg": "qg",
    **{
        name: name for name in (
            "uh", "xvort", "yvort", "zvort", "pv", "sws", "svs", "sps", "srs",
            "sgs", "sus", "shs", "rain", "prate", "cape", "cin", "lcl", "lfc",
            "th", "pwat", "lwp",
        )
    },
}

#: Default normalised-RMSE tolerances.  These are *stated*, not derived: they
#: are the level at which a difference stops being rounding and starts being a
#: different answer, for a comparison of a few hundred steps.  Raise them with
#: ``--tolerance FIELD=VALUE`` and say so in the receipt rather than silently.
DEFAULT_TOLERANCE: Mapping[str, float] = {
    "th3d": 1e-5,
    "prs": 1e-6,
    "rho": 1e-6,
    "u3d": 1e-4,
    "v3d": 1e-4,
    "w3d": 1e-3,
    "closure.tkea": 1e-2,
    "q3d:qv": 1e-5,
    "q3d:qc": 1e-2,
    "q3d:qr": 1e-2,
    "q3d:qi": 1e-2,
    "q3d:qs": 1e-2,
    "q3d:qg": 1e-2,
    # A &param9 diagnostic is a pure function of the completed state, so its
    # difference is the state's difference put through the operator.  Each
    # limit below is stated with the reason it is where it is; where a field
    # is essentially zero in a given case its nrmse is the ratio of two small
    # numbers and the absolute difference is what the reader should look at,
    # which is why ``max|d|`` and ``cm1_peak`` are both in every row.
    "uh": 1e-2,
    "xvort": 1e-2,
    "yvort": 1e-2,
    # zvort and pv are tiny wherever a case has no organised rotation, so
    # their normalised statistics are dominated by the normalisation.  The
    # operators themselves are pinned exactly by the analytic solid-body and
    # linear-theta tests in tests/test_cm1_output_diagnostics.py.
    "zvort": 5e-2,
    "pv": 3e-1,
    # A run-long extremum carries the worst single step of the whole run, so
    # it is larger than the instantaneous state difference it is built from.
    "sws": 1e-2,
    "svs": 1e-2,
    # sps is min(prs - prs0) and therefore carries the two initializers' own
    # base-state pressure difference as a CONSTANT offset: CM1 and gpuwm-cm1
    # form prs0 by the same expression (base.F:146, p00*pi0**(cp/rd)) from
    # base states that differ at f32 rounding, and the low-level pressure
    # perturbation is a small number to carry that offset on.
    "sps": 3e-1,
    "srs": 1e-2,
    "sgs": 1e-2,
    # sus is max w at 5 km AGL, which is near zero until a case has an updraft
    # that deep.
    "sus": 1e-1,
    "shs": 1e-2,
    "rain": 1e-2,
    "prate": 1e-2,
    "cape": 1e-3,
    "cin": 1e-2,
    # LCL and LFC are heights found by interpolation between two model levels,
    # so a small state difference can move them by part of a layer.
    "lcl": 1e-3,
    "lfc": 1e-3,
    # th is th0 + tha: the perturbation's own difference normalised by a ~300 K
    # absolute field, so roughly th3d's nrmse divided by 450.
    "th": 1e-5,
    "pwat": 1e-3,
    "lwp": 1e-2,
}


@dataclass(frozen=True)
class FieldDifference:
    field: str
    cm1_field: str
    model_time_s: float
    shape: tuple[int, ...]
    max_abs: float
    max_abs_at: tuple[int, ...]
    cm1_at_max: float
    gm1_at_max: float
    max_rel: float
    rmse: float
    nrmse: float
    cm1_rms: float
    cm1_peak: float
    max_abs_over_peak: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "cm1_field": self.cm1_field,
            "model_time_s": self.model_time_s,
            "shape": list(self.shape),
            "max_abs": self.max_abs,
            "max_abs_at_kji": list(self.max_abs_at),
            "cm1_at_max": self.cm1_at_max,
            "gm1_at_max": self.gm1_at_max,
            "max_rel": self.max_rel,
            "rmse": self.rmse,
            "nrmse": self.nrmse,
            "cm1_rms": self.cm1_rms,
            "cm1_peak": self.cm1_peak,
            "max_abs_over_peak": self.max_abs_over_peak,
        }


@dataclass
class Comparison:
    case: str
    tolerance: Mapping[str, float]
    interior_cells: int = 0
    rows: list[FieldDifference] = _field(default_factory=list)
    skipped: list[dict[str, Any]] = _field(default_factory=list)

    def curve(self, name: str) -> list[FieldDifference]:
        return [r for r in self.rows if r.field == name]

    def first_exceedance(self, name: str) -> FieldDifference | None:
        limit = self.tolerance.get(name, math.inf)
        for row in self.curve(name):
            if row.nrmse > limit:
                return row
        return None

    def fields(self) -> list[str]:
        seen: list[str] = []
        for row in self.rows:
            if row.field not in seen:
                seen.append(row.field)
        return seen

    def growing(self, name: str) -> bool:
        """True when the last frame's nrmse exceeds the first non-zero one."""

        curve = [r.nrmse for r in self.curve(name)]
        nonzero = [v for v in curve if v > 0.0]
        return bool(nonzero) and curve[-1] > nonzero[0] * 10.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "gm1.differential-comparison/v1",
            "case": self.case,
            "interior_cells_trimmed": self.interior_cells,
            "tolerance_nrmse": dict(self.tolerance),
            "fields": self.fields(),
            "skipped": self.skipped,
            "first_exceedance": {
                name: (row.as_dict() if (row := self.first_exceedance(name)) else None)
                for name in self.fields()
            },
            "divergence_curve": [r.as_dict() for r in self.rows],
        }

    def report(self) -> str:
        lines = [f"differential comparison: {self.case}", ""]
        if self.interior_cells:
            lines.append(
                f"interior only: {self.interior_cells} cells trimmed from each lateral edge, "
                "so a lateral-boundary difference is excluded from these statistics"
            )
            lines.append("")
        for name in self.fields():
            limit = self.tolerance.get(name, math.inf)
            lines.append(f"{name}  (CM1 {CM1_NAME.get(name, name)})  tolerance nrmse <= {limit:g}")
            lines.append(
                f"  {'t (s)':>10} {'max|d|':>12} {'at (k,j,i)':>16} {'rmse':>12} "
                f"{'nrmse':>12} {'|d|/peak':>11}"
            )
            for row in self.curve(name):
                mark = "  EXCEEDS" if row.nrmse > limit else ""
                # A 2-D surface product has a (j,i) index, not a (k,j,i) one.
                at = "(" + ",".join(str(value) for value in row.max_abs_at[-3:]) + ")"
                lines.append(
                    f"  {row.model_time_s:10.3f} {row.max_abs:12.5e} {at:>16} "
                    f"{row.rmse:12.5e} {row.nrmse:12.5e} {row.max_abs_over_peak:11.4e}{mark}"
                )
            first = self.first_exceedance(name)
            if first is not None:
                lines.append(
                    f"  FIRST EXCEEDANCE at t = {first.model_time_s:g} s: nrmse "
                    f"{first.nrmse:.5e} > {limit:g}, max|d| {first.max_abs:.5e} at "
                    f"{first.max_abs_at} (CM1 {first.cm1_at_max:.8g}, GM1 {first.gm1_at_max:.8g})"
                )
            lines.append("")
        for row in self.skipped:
            lines.append(f"skipped {row['field']}: {row['reason']}")
        return "\n".join(lines)


def _as_array(words, shape: Sequence[int]) -> np.ndarray:
    return np.frombuffer(memoryview(words), dtype="<f4").reshape(tuple(shape))


def compare(
    cm1: CM1History,
    gm1: GM1History,
    *,
    case: str = "unnamed",
    tolerance: Mapping[str, float] | None = None,
    fields: Sequence[str] | None = None,
    time_atol_s: float = 1e-3,
    interior_cells: int = 0,
) -> Comparison:
    limits = dict(DEFAULT_TOLERANCE)
    if tolerance:
        limits.update(tolerance)
    if interior_cells < 0:
        raise ValueError("interior_cells trims whole cells from each lateral edge; it cannot be negative")
    result = Comparison(case=case, tolerance=limits, interior_cells=interior_cells)

    wanted = list(fields) if fields else list(FIELD_MAP)
    available = set(gm1.field_names())
    cm1_available = set(cm1.field_names())

    pairs: list[tuple[str, str, str, int | None]] = []
    for name in wanted:
        source = FIELD_MAP[name]
        carrier, species = (source, None) if isinstance(source, str) else source
        cm1_name = CM1_NAME[name]
        if carrier not in available:
            result.skipped.append({"field": name, "reason": f"gpuwm-cm1 frame carries no {carrier!r}"})
            continue
        if cm1_name not in cm1_available:
            result.skipped.append({"field": name, "reason": f"CM1 history carries no {cm1_name!r}; enable its output selector"})
            continue
        pairs.append((name, carrier, cm1_name, species))

    for index, frame in enumerate(gm1.frames):
        time = frame.model_time_s
        cm1_index = _match_time(cm1, time, time_atol_s)
        if cm1_index is None:
            result.skipped.append({
                "field": "*",
                "reason": f"gpuwm-cm1 frame {index + 1} at t={time} s has no CM1 history frame within {time_atol_s} s",
            })
            continue
        for name, carrier, cm1_name, species in pairs:
            gm1_words, gm1_shape = frame.read(carrier)
            a = _as_array(gm1_words, gm1_shape)
            if species is not None:
                a = a[species]
            cm1_words, cm1_shape = cm1.read(cm1_name, cm1_index)
            b = _as_array(cm1_words, cm1_shape)
            # CM1 writes a 2-D surface field with a level count of 0, and the
            # GrADS reader returns it as (1, ny, nx).  gpuwm-cm1 publishes the
            # same plane as (ny, nx).  That is one descriptor convention, not
            # two grids, so it is squeezed rather than reported as a mismatch.
            if b.ndim == a.ndim + 1 and b.shape[0] == 1 and b.shape[1:] == a.shape:
                b = b[0]
            if a.shape != b.shape:
                result.skipped.append({
                    "field": name,
                    "reason": f"shape mismatch: gpuwm-cm1 {a.shape} vs CM1 {b.shape}; the two runs are not on the same grid",
                })
                continue
            if interior_cells:
                trim = interior_cells
                if min(a.shape[-1], a.shape[-2]) <= 2 * trim:
                    result.skipped.append({
                        "field": name,
                        "reason": f"interior trim of {trim} cells leaves nothing of a {a.shape} field",
                    })
                    continue
                a = a[..., trim:-trim, trim:-trim]
                b = b[..., trim:-trim, trim:-trim]
            result.rows.append(_difference(name, cm1_name, time, a, b))
    return result


def _match_time(cm1: CM1History, time: float, atol: float) -> int | None:
    for index, value in enumerate(cm1.model_time_s):
        if abs(value - time) <= atol:
            return index + 1
    return None


def _difference(name: str, cm1_name: str, time: float, a: np.ndarray, b: np.ndarray) -> FieldDifference:
    wide_a = a.astype(np.float64)
    wide_b = b.astype(np.float64)
    delta = np.abs(wide_a - wide_b)
    flat = int(np.argmax(delta))
    at = np.unravel_index(flat, delta.shape)
    rms = float(np.sqrt(np.mean(wide_b * wide_b)))
    rmse = float(np.sqrt(np.mean(delta * delta)))
    max_abs = float(delta.flat[flat])
    peak = float(np.max(np.abs(wide_b)))
    return FieldDifference(
        field=name,
        cm1_field=cm1_name,
        model_time_s=time,
        shape=tuple(int(v) for v in a.shape),
        max_abs=max_abs,
        max_abs_at=tuple(int(v) for v in at),
        cm1_at_max=float(wide_b[at]),
        gm1_at_max=float(wide_a[at]),
        max_rel=(max_abs / rms) if rms > 0.0 else float("inf") if max_abs > 0.0 else 0.0,
        rmse=rmse,
        nrmse=(rmse / rms) if rms > 0.0 else float("inf") if rmse > 0.0 else 0.0,
        cm1_rms=rms,
        cm1_peak=peak,
        max_abs_over_peak=(max_abs / peak) if peak > 0.0 else float("inf") if max_abs > 0.0 else 0.0,
    )
