"""Native pictures of a run's restart checkpoint: ``gpuwm-cm1 render RUNDIR --product P``.

A run folder holds two kinds of saved state.  The history frames (``segments/NNN/frames``) are what the quick-looks
and the cm1out writer read.  The restart checkpoints (``segments/NNN/restarts/restart-NNNNNN/restart.cm1fcp``) are
the complete state at a large step, written so a run can be continued; the native renderer (``cm1-render``) reads
that envelope.  This module lets a user point the renderer at a run folder and pick a product by name: it finds the
restart, reads the carriers each product needs, derives the CM1 diagnostics a restart does not carry (reflectivity,
vertical vorticity, updraft helicity) with the same code the quick-looks use, and hands the planes to the Rust
renderer, which draws them in its own style with a footer that says what the picture is.

What a picture says about itself, always:

* it is a restart checkpoint at T+..., not a history frame (restarts are usually written at the end of a segment,
  so they are not every saved time);
* the values in it (lowest and highest) and the colour scale it is drawn on;
* how each field was made: read from the restart, or derived by the picture tool by CM1's formula, and from what;
* the height of the level shown, the domain and its grid, and whether the domain moves.

What is read and checked: the envelope's 8-byte magic and JSON header; its contract hash against the one the
restart's own ``restart-publication.json`` recorded when the restart was published; the SHA-256 of every array
read against the header.  The padded carriers (u, v, w with their halo) are cut to the model's own cells.  A field
the restart does not hold, and that cannot be derived from it, is refused by name.

Nothing here opens a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
from typing import Any, Mapping, Sequence

import numpy as np

from .render_products import (ENVIRONMENT_PRODUCTS, LEVEL_PRODUCTS, PRODUCTS, PRODUCT_WORDS,
                              REFLECTIVITY_PRODUCTS, cannot_draw)

RESTART_FILE = "restart.cm1fcp"
PUBLICATION = "restart-publication.json"
MAGIC = b"CM1FCP1\0"
MAX_HEADER_BYTES = 16 * 1024 * 1024
RESTART_DIR = re.compile(r"^restart-(\d{6})$")
SEGMENT_DIR = re.compile(r"^\d{3,}$")

# CM1 r21.1 constants.f / param.F, as the quick-looks and the cm1out writer use them.
P00 = 1.0e5
RD = 287.04
RV = 461.5
CP = 1005.7
REPS = RV / RD
REFLECTIVITY_HEIGHT_M = 1000.0


class RestartRenderError(ValueError):
    """A request this route cannot honour; the fields are the refusal's."""

    def __init__(self, code: str, reason: str, remedy: str, option: str | None = None, value: Any = None):
        super().__init__(reason)
        self.code, self.reason, self.remedy, self.option, self.value = code, reason, remedy, option, value


def _refuse(code: str, reason: str, remedy: str, option: str | None = None, value: Any = None) -> "None":
    raise RestartRenderError(code, reason, remedy, option, value)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


# ----------------------------------------------------------------------------------------------- finding a restart
@dataclass(frozen=True)
class RestartChoice:
    number: int
    segment: int | None
    directory: Path
    model_time_s: float | None
    step: int | None

    @property
    def envelope(self) -> Path:
        return self.directory / RESTART_FILE

    def as_dict(self) -> dict[str, Any]:
        return {"number": self.number, "segment": self.segment, "model_time_s": self.model_time_s,
                "step": self.step, "path": os.fspath(self.envelope)}


def _restart_row(directory: Path, segment: int | None) -> RestartChoice | None:
    match = RESTART_DIR.match(directory.name)
    if match is None or not (directory / RESTART_FILE).is_file():
        return None
    receipt = _read_json(directory / PUBLICATION)
    if not isinstance(receipt, dict):
        return None
    time_s = receipt.get("model_time_s")
    step = receipt.get("step")
    return RestartChoice(int(match.group(1)), segment, directory,
                         float(time_s) if isinstance(time_s, (int, float)) else None,
                         int(step) if isinstance(step, int) else None)


def run_root(target: Path) -> Path | None:
    """The run folder ``target`` belongs to (one with ``segments/``), when there is one."""

    for candidate in (target, *target.parents):
        if (candidate / "segments").is_dir():
            return candidate
        if candidate.parent == candidate:
            break
    return None


def restarts_of(root: Path) -> list[RestartChoice]:
    """Every published restart of a run folder, oldest segment first (staging folders are never listed)."""

    found: list[RestartChoice] = []
    segments = root / "segments"
    if segments.is_dir():
        for entry in sorted(segments.iterdir(), key=lambda p: p.name):
            if entry.name.startswith(".") or not SEGMENT_DIR.match(entry.name) or not entry.is_dir():
                continue
            restarts = entry / "restarts"
            if not restarts.is_dir():
                continue
            for directory in sorted(restarts.iterdir(), key=lambda p: p.name):
                if directory.name.startswith(".") or not directory.is_dir():
                    continue
                row = _restart_row(directory, int(entry.name))
                if row is not None:
                    found.append(row)
    restarts = root / "restarts"
    if restarts.is_dir():
        for directory in sorted(restarts.iterdir(), key=lambda p: p.name):
            if not directory.name.startswith(".") and directory.is_dir():
                row = _restart_row(directory, None)
                if row is not None:
                    found.append(row)
    return found


def _lineage(root: Path) -> list[RestartChoice] | None:
    """The restarts along the run's own lineage (``gpuwm-cm1 run`` folders), as Continue picks them."""

    if not (root / "run.json").is_file():
        return None
    try:
        from .cm1run.rundir import RunDirectory

        rows = RunDirectory(root).lineage_restarts()
    except Exception:  # noqa: BLE001 - an unreadable run.json falls back to the plain listing
        return None
    out = []
    for row in rows:
        choice = _restart_row(Path(row.path), row.segment)
        if choice is not None:
            out.append(choice)
    return out


def choose(target: Path, pick: str | None) -> tuple[Path | None, RestartChoice, list[RestartChoice]]:
    """The restart a request names: ``target`` a run folder, a restart folder or a restart.cm1fcp file; ``pick``
    None or "latest" (the newest along the run's lineage), a restart number, or ``SEGMENT:NUMBER``."""

    target = Path(target)
    if target.is_file():
        if target.name != RESTART_FILE and target.suffix.lower() != ".cm1fcp":
            home = run_root(target.parent)
            _refuse("CM1-RENDER-RUN", f"{target.name} is not a restart checkpoint (restart.cm1fcp)",
                    (f"give its run folder, {home}" if home is not None else
                     "give a run folder, a restart folder or its restart.cm1fcp"), "history", str(target))
        target = target.parent
    if not target.is_dir():
        _refuse("CM1-RENDER-RUN", f"{target} is not a folder", "give a run folder", "history", str(target))
    if RESTART_DIR.match(target.name) and (target / RESTART_FILE).is_file():
        segment = target.parent.parent.name if target.parent.name == "restarts" else ""
        row = _restart_row(target, int(segment) if SEGMENT_DIR.match(segment) else None)
        if row is None:
            _refuse("CM1-RENDER-RUN", f"{target} has no {PUBLICATION}, so it is not a published restart",
                    "give a restart that was published (renamed into place with its receipt)", "history",
                    str(target))
        if pick not in (None, "latest"):
            _refuse("CM1-RENDER-RUN", "--restart picks a restart in a run folder, and this is one restart already",
                    "give the run folder with --restart, or this restart without it", "restart", pick)
        return run_root(target), row, [row]
    root = target
    rows = restarts_of(root)
    if not rows:
        frames = any((root / "segments").glob("*/frames/frame-*")) if (root / "segments").is_dir() else False
        _refuse("CM1-RENDER-NO-RESTART",
                f"{root.name} has no published restart checkpoint" + (
                    ": a run writes its restart at the end of a segment (and every restart interval), and this "
                    "one has none yet" if frames else ""),
                "the quick-looks draw every saved frame (gpuwm-cm1 quicklook RUNDIR); the native renderer draws "
                "a restart, once the run has written one", "history", str(root))
    lineage = _lineage(root) or rows
    if pick in (None, "latest"):
        return root, lineage[-1], rows
    text = str(pick).strip()
    segment: int | None = None
    if ":" in text:
        head, text = text.split(":", 1)
        if not head.strip().isdigit():
            _refuse("CM1-RENDER-RUN", f"--restart {pick!r} is not SEGMENT:NUMBER", "for example --restart 2:3",
                    "restart", pick)
        segment = int(head)
    if not text.strip().isdigit():
        _refuse("CM1-RENDER-RUN", f"--restart {pick!r} is not a restart number",
                "pass latest, a restart number from --list, or SEGMENT:NUMBER", "restart", pick)
    number = int(text)
    if segment is not None:
        hits = [r for r in rows if r.number == number and r.segment == segment]
    else:
        hits = [r for r in lineage if r.number == number] or [r for r in rows if r.number == number]
    if not hits:
        listed = ", ".join(f"{r.number} (segment {r.segment})" for r in rows)
        _refuse("CM1-RENDER-RUN", f"this run has no restart {pick}; it has {listed}",
                "pass one of those, or latest", "restart", pick)
    return root, hits[-1], rows


# ----------------------------------------------------------------------------------------------- reading a restart
#: Plain words for the moisture species a restart carries (a number concentration takes its species' word).
SPECIES_WORDS = {"qv": "water vapour", "qc": "cloud water", "qr": "rain", "qi": "cloud ice", "qs": "snow",
                 "qg": "graupel or hail", "qh": "hail"}


@dataclass(frozen=True)
class Carrier:
    """What the quick-look frame reader needs to know of one carrier."""

    name: str
    meaning: str
    units: str
    shape: tuple[int, ...]


def _f64_bits(text: Any) -> float | None:
    try:
        return struct.unpack(">d", bytes.fromhex(str(text)))[0]
    except (ValueError, TypeError):
        return None


class RestartFrame:
    """One published restart checkpoint, read as the quick-looks read a history frame.

    It offers the frame names the quick-look code asks for (``u3d``, ``w3d``, ``th3d``, ``q3d``, ``nc3d``, ``prs``,
    ``rho``, the swaths ...), each cut from the restart's own carriers, plus ``th`` (the full potential temperature)
    derived from the restart's pressure and density by CM1's equation of state, because a restart carries no base
    state.  Raw restart names (``fields.th3d``, ``tke.tke3d`` ...) are offered too.
    """

    def __init__(self, choice: RestartChoice) -> None:
        self.choice = choice
        self.directory = choice.directory
        self.path = choice.envelope
        self.receipt = _read_json(self.directory / PUBLICATION)
        if not isinstance(self.receipt, dict):
            _refuse("CM1-RENDER-RUN", f"{self.directory} has no readable {PUBLICATION}",
                    "give a published restart", "history", str(self.directory))
        self.header, self.payload_offset = self._header()
        self._arrays = {row["name"]: row for row in self.header.get("arrays") or [] if isinstance(row, dict)}
        self._specs = {}
        for spec in (self.header.get("contract") or {}).get("array_specs") or []:
            if isinstance(spec, dict) and isinstance(spec.get("name"), str):
                self._specs[spec["name"]] = spec
        self._cache: dict[str, np.ndarray] = {}
        self.checked: list[str] = []
        self.publication = {"config": dict(self.receipt.get("config") or {}),
                            "dt_s": self.receipt.get("dt_s"),
                            "swaths": self.receipt.get("swaths") or {}}
        geometry = (self.receipt.get("source_identity") or {}).get("geometry") or {}
        grid = self.publication["config"].get("grid") or geometry
        if not grid:
            _refuse("CM1-RENDER-RUN", f"{self.directory.name} records no grid", "give a restart GM1 published",
                    "history", str(self.directory))
        self.grid = dict(grid)
        self.nx, self.ny, self.nz = int(grid["nx"]), int(grid["ny"]), int(grid["nz"])
        self.carriers = self._names()

    # -- the envelope ----------------------------------------------------------------------------------------------
    def _header(self) -> tuple[dict[str, Any], int]:
        with self.path.open("rb") as stream:
            prefix = stream.read(16)
            if len(prefix) < 16 or prefix[:8] != MAGIC:
                _refuse("CM1-HISTORY-ENVELOPE", f"{self.path} does not start with the CM1FCP1 magic",
                        "give a restart checkpoint GM1 wrote", "history", str(self.path))
            length = int.from_bytes(prefix[8:16], "little")
            if length > MAX_HEADER_BYTES:
                _refuse("CM1-HISTORY-ENVELOPE", f"{self.path.name} declares a {length}-byte header",
                        "give a restart checkpoint GM1 wrote", "history", str(self.path))
            raw = stream.read(length)
        try:
            header = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as error:
            _refuse("CM1-HISTORY-ENVELOPE", f"{self.path.name}'s header is not JSON ({error})",
                    "give a restart checkpoint GM1 wrote", "history", str(self.path))
        if not isinstance(header, dict):
            _refuse("CM1-HISTORY-ENVELOPE", f"{self.path.name}'s header is not a JSON object",
                    "give a restart checkpoint GM1 wrote", "history", str(self.path))
        recorded = (self.receipt.get("forecast_envelope") or {}).get("contract_sha256")
        if not isinstance(recorded, str) or recorded != header.get("contract_sha256"):
            _refuse("CM1-RENDER-CONTRACT-MISMATCH",
                    f"the restart's receipt records contract {recorded}, and the envelope's header declares "
                    f"{header.get('contract_sha256')}: the file is not the restart the receipt describes",
                    "render a restart whose receipt and envelope belong together", "history", str(self.path))
        size = self.path.stat().st_size
        offset = 16 + length
        if int(header.get("payload_nbytes") or -1) != size - offset:
            _refuse("CM1-HISTORY-ENVELOPE",
                    f"{self.path.name} holds {size - offset} payload bytes and its header declares "
                    f"{header.get('payload_nbytes')}", "give a complete restart checkpoint", "history",
                    str(self.path))
        return header, offset

    def raw(self, name: str) -> np.ndarray:
        """A restart carrier, its bytes checked against the header's SHA-256, in its declared shape."""

        if name in self._cache:
            return self._cache[name]
        row = self._arrays.get(name)
        spec = self._specs.get(name)
        if row is None or spec is None:
            _refuse("CM1-RENDER-FIELD", f"this restart holds no {name}", "pass --list-fields for what it holds",
                    "field", name)
        shape = tuple(int(v) for v in spec["shape"])
        count = int(np.prod(shape))
        nbytes = int(row["nbytes"])
        if nbytes != 4 * count:
            _refuse("CM1-HISTORY-ENVELOPE", f"{name} declares {nbytes} bytes for shape {shape}",
                    "give a restart checkpoint GM1 wrote", "field", name)
        with self.path.open("rb") as stream:
            stream.seek(self.payload_offset + int(row["offset"]))
            data = stream.read(nbytes)
        if len(data) != nbytes or hashlib.sha256(data).hexdigest() != row.get("sha256"):
            _refuse("CM1-HISTORY-ENVELOPE", f"{name}'s bytes do not match the SHA-256 in the restart's header",
                    "the file is damaged; continue or render from another restart", "field", name)
        array = np.frombuffer(data, dtype="<f4").reshape(shape)
        self.checked.append(name)
        self._cache[name] = array
        return array

    def holds(self, name: str) -> bool:
        return name in self._arrays and name in self._specs

    def spec(self, name: str) -> Mapping[str, Any]:
        return self._specs.get(name) or {}

    # -- the frame names --------------------------------------------------------------------------------------------
    def _species(self) -> tuple[list[str], list[str]]:
        """(mass rows, number rows) of fields.q3d, from its meaning ('current qv,qc,... then nci,...')."""

        meaning = str(self.spec("fields.q3d").get("meaning") or "")
        lists = [word.strip(",") for word in meaning.split() if "," in word]
        if len(lists) >= 1:
            mass = [w for w in lists[0].split(",") if w]
            number = [w for w in lists[1].split(",") if w] if len(lists) > 1 else []
            return mass, number
        return [], []

    def _names(self) -> dict[str, Carrier]:
        nx, ny, nz = self.nx, self.ny, self.nz
        out: dict[str, Carrier] = {}

        def add(name: str, source: str, shape: tuple[int, ...], meaning: str | None = None) -> None:
            spec = self.spec(source)
            out[name] = Carrier(name, meaning or str(spec.get("meaning") or name), str(spec.get("units") or ""),
                                shape)

        for name, stag in (("u3d", (nz, ny, nx + 1)), ("v3d", (nz, ny + 1, nx)), ("w3d", (nz + 1, ny, nx))):
            if self.holds(f"padded.{name}"):
                add(name, f"padded.{name}", stag, {"u3d": "west-east velocity (domain-relative)",
                                                   "v3d": "south-north velocity (domain-relative)",
                                                   "w3d": "vertical velocity"}[name])
        # CM1's th3d is the potential temperature minus the base state (the restart's own words say only
        # "current potential temperature"); the full one is derived below as th.
        for name in ("th3d", "pp3d", "prs", "rho", "rf"):
            if self.holds(f"fields.{name}"):
                add(name, f"fields.{name}", tuple(self.spec(f"fields.{name}")["shape"]),
                    "potential temperature perturbation (minus the base state)" if name == "th3d" else None)
        if self._wind_halo() is not None:
            out["zvort"] = Carrier("zvort", "vertical vorticity dv/dx - du/dy (CM1's calcvort), derived from the "
                                   "restart's winds and their boundary halo", "1/s", (nz, ny, nx))
        if self.holds("tke.tke3d"):
            add("tke", "tke.tke3d", tuple(self.spec("tke.tke3d")["shape"]))
        mass, number = self._species()
        if self.holds("fields.q3d") and mass:
            q3d = tuple(self.spec("fields.q3d")["shape"])
            if q3d[0] == len(mass) + len(number):
                out["q3d"] = Carrier("q3d", "current " + ",".join(mass) + " mixing ratios", "kg/kg",
                                     (len(mass), *q3d[1:]))
                if number:
                    out["nc3d"] = Carrier("nc3d", "current " + ",".join(number) + " number concentrations", "1/kg",
                                          (len(number), *q3d[1:]))
                for row, species in enumerate(mass + number):
                    kind = "mixing ratio" if row < len(mass) else "number concentration"
                    words = SPECIES_WORDS.get(species.lstrip("n").replace("c", "q", 1) if row >= len(mass) else species)
                    meaning = f"{words} {kind}" if words else f"{species} {kind}"
                    out[species] = Carrier(species, meaning, "kg/kg" if row < len(mass) else "1/kg", q3d[1:])
        if {"fields.prs", "fields.rho"} <= set(self._arrays) and "qv" in out:
            out["th"] = Carrier("th", "potential temperature, from the restart's pressure and density (CM1's "
                                "equation of state)", "K", (nz, ny, nx))
        for name in ("sws", "svs", "sps", "sus", "shs", "srs", "sgs", "sws2", "svs2", "sps2", "sus2", "shs2",
                     "srs2", "sgs2"):
            if self.holds(f"swath.{name}"):
                add(name, f"swath.{name}", tuple(self.spec(f"swath.{name}")["shape"]))
        # Every raw name too, for --field.
        for name, spec in self._specs.items():
            if name not in out and len(spec.get("shape") or ()) in (2, 3):
                out[name] = Carrier(name, str(spec.get("meaning") or name), str(spec.get("units") or ""),
                                    tuple(int(v) for v in spec["shape"]))
        return out

    def has(self, name: str) -> bool:
        return name in self.carriers

    def _crop(self, padded: np.ndarray, want: tuple[int, ...], name: str) -> np.ndarray:
        offsets = []
        for have, need in zip(padded.shape, want):
            extra = have - need
            if extra < 0 or extra % 2:
                _refuse("CM1-RENDER-FIELD", f"{name}: the restart's padded carrier {padded.shape} does not hold the "
                        f"model's {want} cells with an even halo", "give a restart GM1 wrote", "field", name)
            offsets.append(extra // 2)
        index = tuple(slice(o, o + n) for o, n in zip(offsets, want))
        return padded[index]

    def array(self, name: str) -> np.ndarray:
        if name in self._cache:
            return self._cache[name]
        if name not in self.carriers:
            _refuse("CM1-RENDER-FIELD", f"this restart holds no {name}", "pass --list-fields for what it holds",
                    "field", name)
        carrier = self.carriers[name]
        if name in ("u3d", "v3d", "w3d"):
            value = self._crop(self.raw(f"padded.{name}"), carrier.shape, name)
        elif name in ("th3d", "pp3d", "prs", "rho", "rf"):
            value = self.raw(f"fields.{name}")
        elif name == "tke":
            value = self.raw("tke.tke3d")
        elif name in ("q3d", "nc3d"):
            q = self.raw("fields.q3d")
            rows = len(self._species()[0])
            value = q[:rows] if name == "q3d" else q[rows:]
        elif name == "th":
            value = self._theta()
        elif name == "zvort":
            value = self._zvort()
        elif name.startswith(("s", )) and self.holds(f"swath.{name}"):
            value = self.raw(f"swath.{name}")
        else:
            mass, number = self._species()
            if name in mass + number:
                value = self.raw("fields.q3d")[(mass + number).index(name)]
            else:
                value = self.raw(name)
        self._cache[name] = value
        return value

    def _theta(self) -> np.ndarray:
        """Full potential temperature from CM1's equation of state on the restart's own words:
        T = prs / (rd rho (1 + max(qv, 0) reps)) and theta = T / (prs/p00)**(rd/cp).  CM1 recomputes rho from th,
        pi and qv at the end of each step (solve3.F), so this returns that step's temperature to rounding."""

        prs = self.raw("fields.prs").astype(np.float64)
        rho = self.raw("fields.rho").astype(np.float64)
        qv = np.maximum(self.array("qv").astype(np.float64), 0.0)
        temperature = prs / (RD * rho * (1.0 + qv * REPS))
        return (temperature / np.power(prs / P00, RD / CP)).astype(np.float32)

    def _wind_halo(self) -> tuple[tuple[int, int, int], tuple[int, int, int]] | None:
        """(u halo, v halo): the cells each padded wind carries around the model's own, per axis, when both carry
        at least one sideways; None otherwise."""

        if not (self.holds("padded.u3d") and self.holds("padded.v3d")):
            return None
        nx, ny, nz = self.nx, self.ny, self.nz
        halos = []
        for name, want in (("padded.u3d", (nz, ny, nx + 1)), ("padded.v3d", (nz, ny + 1, nx))):
            shape = tuple(int(v) for v in self.spec(name).get("shape") or ())
            if len(shape) != 3 or any(have < need or (have - need) % 2 for have, need in zip(shape, want)):
                return None
            halos.append(tuple((have - need) // 2 for have, need in zip(shape, want)))
        if min(halos[0][1], halos[0][2], halos[1][1], halos[1][2]) < 1:
            return None
        return halos[0], halos[1]  # type: ignore[return-value]

    def _zvort(self) -> np.ndarray:
        """Vertical vorticity at scalar points: CM1 r21.1 calcvort (misclibs.F:2037-2049, Cartesian, no terrain),
        operation for operation in float32.

        Every corner of the domain, its edge included, is ``(v(i)-v(i-1))*rdx - (u(j)-u(j-1))*rdy``; at the edge
        the difference reaches into the restart's halo, which CM1's lateral boundary conditions filled (the edge
        value copied outward for an open boundary, the far side's for a periodic one), as CM1's own loop does.  A
        cell is the quarter-sum of its four corners in CM1's order.  (GM1's history frames repeat the inner
        neighbour in the outermost ring instead; inside it the two are the same numbers.)"""

        halo = self._wind_halo()
        if halo is None:
            _refuse("CM1-RENDER-FIELD", "this restart's winds carry no halo to difference at the domain's edge",
                    "give a restart GM1 wrote", "field", "zvort")
        (ku, ju, iu), (kv, jv, iv) = halo  # type: ignore[misc]
        nx, ny, nz = self.nx, self.ny, self.nz
        pu, pv = self.raw("padded.u3d"), self.raw("padded.v3d")
        rdx = np.float32(1.0 / float(self.grid["dx_m"]))
        rdy = np.float32(1.0 / float(self.grid["dy_m"]))
        quarter = np.float32(0.25)
        out = np.empty((nz, ny, nx), np.float32)
        for k in range(nz):
            u = pu[ku + k, ju - 1:ju + ny + 1, iu:iu + nx + 1]      # rows j = 0..ny+1 (CM1), faces i = 1..nx+1
            v = pv[kv + k, jv:jv + ny + 1, iv - 1:iv + nx + 1]      # faces j = 1..ny+1, columns i = 0..nx+1
            tem = (v[:, 1:] - v[:, :-1]) * rdx - (u[1:, :] - u[:-1, :]) * rdy   # the ny+1 by nx+1 corners
            out[k] = quarter * (((tem[:-1, :-1] + tem[:-1, 1:]) + tem[1:, :-1]) + tem[1:, 1:])
        return out

    # -- what QuickFrame reads -------------------------------------------------------------------------------------
    @property
    def model_time_s(self) -> float:
        value = self.receipt.get("model_time_s")
        return float(value) if isinstance(value, (int, float)) else float("nan")

    @property
    def moving_frame(self) -> dict[str, Any]:
        config = self.publication["config"]
        scalars = self.receipt.get("scheduler_scalars") or {}
        return {"umove": float(config.get("u_move_m_s") or 0.0), "vmove": float(config.get("v_move_m_s") or 0.0),
                "domainlocx": _f64_bits(scalars.get("domainlocx")) or 0.0,
                "domainlocy": _f64_bits(scalars.get("domainlocy")) or 0.0}

    def close(self) -> None:
        self._cache.clear()


def geometry(frame: RestartFrame) -> Any:
    """The grid, with zf rebuilt by CM1's own level construction from the restart's vertical record.

    A restart carries no zf (README.namelist:1512-1516: the grid is rebuilt, not carried).  The level record's
    SHA-256 is checked against the rebuilt one, so a stretched grid is drawn at the heights the run used."""

    from .cm1out.settings import Geometry

    grid = frame.grid
    nz = frame.nz
    dz = np.float32(grid["dz_m"])
    vertical = grid.get("vertical") if isinstance(grid.get("vertical"), dict) else None
    zf = None
    note = "uniform levels"
    if vertical and int(vertical.get("stretch_z") or 0) != 0:
        from .cm1_vertical_grid import build_vertical_levels

        try:
            levels = build_vertical_levels(
                stretch_z=int(vertical["stretch_z"]), nk=int(vertical.get("nk") or nz),
                dz_m=float(vertical.get("dz_reference_m") or grid["dz_m"]),
                ztop_m=float(vertical.get("ztop_m") or grid.get("ztop_m")),
                str_bot_m=float(vertical.get("str_bot_m") or 0.0), str_top_m=float(vertical.get("str_top_m") or 0.0),
                dz_bot_m=float(vertical.get("dz_bot_m") or 0.0), dz_top_m=float(vertical.get("dz_top_m") or 0.0),
                levels_m=tuple(vertical.get("levels_m") or ()))
        except Exception as error:  # noqa: BLE001 - said plainly below
            _refuse("CM1-RENDER-GRID", f"the restart's stretched levels cannot be rebuilt: {error}",
                    "render a restart written by this version of GM1", "history", str(frame.path))
        if vertical.get("sha256") and levels.sha256 != vertical["sha256"]:
            _refuse("CM1-RENDER-GRID", "the rebuilt stretched levels are not the ones the run recorded "
                    f"(SHA-256 {levels.sha256} against {vertical['sha256']})",
                    "render with the GM1 version that made the run", "history", str(frame.path))
        zf = np.asarray(levels.zf, dtype=np.float32).reshape(-1)
        if zf.shape[0] != nz + 1:
            # CM1-indexed with ghosts: the faces 1..nz+1.
            zf = zf[1:nz + 2] if zf.shape[0] >= nz + 3 else zf[:nz + 1]
        note = "stretched levels"
    if zf is None:
        zf = (np.arange(nz + 1, dtype=np.float32) * dz).astype(np.float32)
    zh = (np.float32(0.5) * (zf[1:] + zf[:-1])).astype(np.float32)
    mh = (dz / (zf[1:] - zf[:-1])).astype(np.float32)
    rmh = (np.float32(1.0) / mh).astype(np.float32)
    dt = float(frame.receipt.get("dt_s") or 0.0)
    out = Geometry(frame.nx, frame.ny, nz, float(grid["dx_m"]), float(grid["dy_m"]), float(dz), zf, zh, mh, rmh, dt)
    return out, note


# ----------------------------------------------------------------------------------------------- words
def _clean(text: str) -> str:
    """The renderer's font is upper-case ASCII: say minus, times and degrees in letters it has."""

    # the long dash by its code point: the release scan refuses it as a character and as an escape
    swaps = {"\u2212": "-", "\u2013": "-", chr(0x2014): "-", "\u00d7": "x", "\u00b2": "2", "\u00b3": "3",
             "\u207b\u00b9": "-1", "\u2032": "'", "\u00b0": " deg", "\u00b7": "|", "\u2009": " "}
    for old, new in swaps.items():
        text = text.replace(old, new)
    return text.encode("ascii", "replace").decode("ascii")


def fmt_value(value: float) -> str:
    """A value for a footer: 3 significant digits, no exponent for everyday sizes."""

    if value == 0 or not math.isfinite(value):
        return "0" if value == 0 else str(value)
    if float(value).is_integer() and abs(value) < 1e9:
        return f"{value:.0f}"
    magnitude = abs(value)
    if magnitude >= 100:
        return f"{value:.0f}"
    if magnitude >= 1:
        return f"{value:.3g}" if magnitude < 10 else f"{value:.1f}"
    digits = max(2, -int(math.floor(math.log10(magnitude))) + 2)
    return f"{value:.{digits}f}".rstrip("0").rstrip(".") if digits < 9 else f"{value:.2e}"


def fmt_height(z_m: float) -> str:
    if z_m >= 1000.0:
        text = f"{z_m / 1000.0:.2f}".rstrip("0").rstrip(".")
        return f"{text} km"
    return f"{z_m:.0f} m"


def fmt_grid(dx_m: float) -> str:
    return f"{dx_m / 1000.0:g} km" if dx_m >= 1000.0 else f"{dx_m:g} m"


def clock(model_time_s: float, hurricane: bool) -> str:
    total = max(0, int(round(model_time_s)))
    if hurricane:
        whole = total // 3600
        shown = str(whole) if total % 3600 == 0 else f"{total / 3600:.1f}"
        days = whole // 24
        return f"T+{shown} h ({days} d {whole % 24} h)" if days else f"T+{shown} h"
    return f"T+{total // 3600}:{total % 3600 // 60:02d}:{total % 60:02d}"


def spin_range(dx_m: float) -> float:
    """The quick-looks' spin range for a grid spacing (quicklook.products.spin_range): 0.01 1/s at 1 km, scaled
    as 1/dx and rounded to 1-2-5."""

    from .quicklook.products import spin_range as quicklook_spin_range

    return quicklook_spin_range(dx_m)


#: The scalar fields the picture tool makes from a restart rather than reads from it.
DERIVED_FIELDS = ("th", "zvort")

#: How the picture tool makes vertical vorticity from a restart, said under every picture that shows it.
SPIN_SOURCE = ("derived by the picture tool from the restart's winds, at the domain's edge from their boundary halo "
               "as CM1 does (a restart carries no vorticity)")


def hurricane_w_range() -> float:
    """A hurricane's updraft scale: the quick-looks' (quicklook.products.HURRICANE_W_RANGE, 10 m/s), on which a
    grown eyewall's 4-10 m/s fills the ramp instead of its darkest third."""

    from .quicklook.products import HURRICANE_W_RANGE

    return float(HURRICANE_W_RANGE)


def uh_range(dx_m: float, hurricane: bool) -> float | None:
    """The updraft-helicity scale's top, or None for the renderer's own 0-300 m2/s2 with colour breaks at the
    published 25 and 100 thresholds.  On a thunderstorm grid finer than 3 km a supercell's helicity runs to the
    high hundreds or thousands (GM1's 1 km run: 696, its swath 903; a 500 m run's swath 2863), which the 300 scale
    paints white; there the quick-looks' range is used (quicklook.products.UH_RANGE, 1000 m2/s2)."""

    if hurricane or dx_m >= 3000.0:
        return None
    from .quicklook.products import UH_RANGE

    return float(UH_RANGE)


def uh_scale_words(top: float | None) -> str:
    if top is None:
        return "0 to 300 m2/s2, fixed; colours change at the published 25 and 100 m2/s2 thresholds"
    return f"0 to {top:g} m2/s2, fixed (the quick-looks' range for a grid finer than 3 km)"


def plain_units(units: str) -> str:
    """A unit as the pictures write it: ``m s-1`` is m/s, ``kg kg-1`` kg/kg, ``s-1`` 1/s, ``m2 s-2`` m2/s2."""

    text = str(units or "").strip()
    if text in ("#/kg",):
        return "1/kg"
    if text == "1":
        return "nondimensional"
    tokens = text.split()
    pattern = re.compile(r"^([A-Za-z]+)(-?\d+)?$")
    if not tokens or not all(pattern.match(t) for t in tokens):
        return text
    above, below = [], []
    for token in tokens:
        name, power = pattern.match(token).groups()  # type: ignore[union-attr]
        exponent = int(power) if power else 1
        if exponent < 0:
            below.append(name + (str(-exponent) if exponent != -1 else ""))
        else:
            above.append(name + (str(exponent) if exponent != 1 else ""))
    if not below:
        return " ".join(above)
    return (" ".join(above) or "1") + "/" + "/".join(below)


def hurricane_spin(dx_m: float) -> float:
    """A hurricane's spin range: twice the thunderstorm range for the grid (0.002 1/s at 8 km, 0.004 at 4 km).
    A hurricane's core spins at a few 1e-3 1/s on these grids (GM1's 8 km run: up to 0.0063 at 81 h) and its rain
    bands at a few 1e-4; this range draws the bands and saturates only the core, where the thunderstorm range
    of 0.001 at 8 km would saturate the whole eyewall."""

    return 2.0 * spin_range(dx_m)


# ----------------------------------------------------------------------------------------------- planes
@dataclass
class Picture:
    """Everything the renderer is handed for one product."""

    product: str
    planes: list[np.ndarray]
    names: list[str]
    units: str
    title: str | None = None
    panel_title: str | None = None
    classification: str | None = None
    bounds: tuple[float, float] | None = None
    w_max: float | None = None
    zeta_max: float | None = None
    scale_words: str = ""
    method: list[str] = field(default_factory=list)
    level_words: str = ""
    extent_m: tuple[float, float] = (0.0, 0.0)
    record: dict[str, Any] = field(default_factory=dict)
    vertical_speed: bool = False
    #: The three-panel picture's vorticity panel title (it is the one panel drawn on a chosen level).
    spin_title: str | None = None
    #: The unit word a single-panel legend prints, when the palette's own is not the field's (the scalar product).
    legend_units: str | None = None


@dataclass
class Request:
    product: str
    field: str | None = None
    level: int | None = None
    height_m: float | None = None
    #: A hurricane is drawn over the inner area the quick-looks show (the eyewall and the inner rain bands)
    #: unless the whole area is asked for.
    whole: bool = False


class Planes:
    """The fields of one restart the products share, each made once."""

    def __init__(self, frame: RestartFrame, hurricane: bool) -> None:
        from .quicklook.frame import QuickFrame

        self.frame = frame
        self.geometry, self.levels_note = geometry(frame)
        self.qf = QuickFrame(frame, self.geometry, key=frame.directory.name)  # type: ignore[arg-type]
        self.hurricane = hurricane
        self._reflectivity = None

    @property
    def dx(self) -> float:
        return self.qf.dx

    def level_for(self, request: Request, *, default_height: float | None = None,
                  faces: bool = False) -> tuple[int, str, float]:
        """The level a request names, how to say it and its height.  ``--level K`` counts as CM1 and
        ``gpuwm-cm1 quicklook --slice --level`` do: 1 is the lowest."""

        z = self.qf.zf if faces else self.qf.zh
        count = len(z)
        kind = "w level" if faces else "level"
        if request.level is not None:
            if not 1 <= request.level <= count:
                _refuse("CM1-RENDER-REQUEST", f"--level {request.level} is outside the {count} levels",
                        f"pass --level between 1 (the lowest) and {count}, or --height M", "level", request.level)
            k = int(request.level) - 1
        elif request.height_m is not None:
            top = float(z[-1])
            if float(request.height_m) > top:
                _refuse("CM1-RENDER-REQUEST", f"--height {float(request.height_m):g} m is above this run's highest "
                        f"{kind}, {fmt_height(top)} above ground ({top:g} m)",
                        f"pass a height up to {top:g} m, or --level {count} for the highest", "height",
                        request.height_m)
            k = self.qf.level_near(float(request.height_m), faces=faces)
        elif default_height is not None:
            k = self.qf.level_near(default_height, faces=faces)
        else:
            k = 0
        lowest = " (the lowest)" if k == 0 else ""
        return k,f"{kind} {k + 1} of {count}{lowest}, {fmt_height(float(z[k]))} above ground", float(z[k])

    def reflectivity(self) -> Any:
        if self._reflectivity is None:
            from .quicklook import reflectivity as R

            self._reflectivity = R.volume(self.qf)
        return self._reflectivity

    def dbz_volume(self, product: str = "reflectivity") -> tuple[np.ndarray, str]:
        from .quicklook import reflectivity as R

        component = str(self.qf.config.get("microphysics") or "")
        if component in R.OWN_REFLECTIVITY:
            scheme = R.PLAIN_SCHEME.get(component, component)
            _refuse("CM1-RENDER-NO-REFLECTIVITY",
                    f"{scheme} computes its reflectivity inside the model and writes it to the history frames "
                    "only; a restart carries none, and GM1 has no CPU copy of that formula",
                    "the quick-looks draw this run's reflectivity from its frames; from the restart, draw "
                    "updraft, vorticity, cold-pool or a swath", "product", product)
        refl = self.reflectivity()
        if not refl.available:
            _refuse("CM1-RENDER-NO-REFLECTIVITY", (refl.reason or "this run has no reflectivity").rstrip("."),
                    "draw updraft, vorticity, cold-pool or a swath", "product", product)
        operator = refl.operator
        if "calc_refl10cm" in operator:
            operator = ("CM1's calc_refl10cm (Morrison, 10 cm) on the CPU, from the restart's rain, snow and "
                        f"{self.dense()} (not a radar), with the temperature from its pressure and density")
        return refl.dbz, operator

    def dense(self) -> str:
        category = (self.frame.publication["config"].get("dense_ice") or {}).get("category")
        return category if category in ("hail", "graupel") else "graupel"

    def w_colmax(self) -> np.ndarray:
        return self.qf.colmax("w3d")

    def zeta(self, k: int) -> np.ndarray:
        return self.qf.zeta_level(k)[0]


def _extent(nx: int, ny: int, dx: float, dy: float) -> tuple[float, float]:
    return nx * dx, ny * dy


def build(planes: Planes, request: Request) -> Picture:
    """The planes, scales and words of one product, over the area the picture shows."""

    picture = _build(planes, request)
    if planes.hurricane and not request.whole:
        _inner(planes, picture)
    return picture


def inner_window(n: int, d: float, width_m: float) -> slice:
    """The cells within ``width_m`` around the middle of ``n`` cells of ``d`` metres (all of them if fewer)."""

    cells = int(round(width_m / d))
    if cells >= n:
        return slice(0, n)
    start = (n - cells) // 2
    return slice(start, start + cells)


def _inner(planes: Planes, picture: Picture) -> None:
    """Cut a hurricane picture to the inner area around the middle of the domain, as the quick-looks show it."""

    from .quicklook.products import HURRICANE_INNER

    qf = planes.qf
    ny, nx = picture.planes[0].shape
    if (ny, nx) != (qf.ny, qf.nx):
        return
    sj = inner_window(ny, qf.dy, HURRICANE_INNER)
    si = inner_window(nx, qf.dx, HURRICANE_INNER)
    if (sj.stop - sj.start, si.stop - si.start) == (ny, nx):
        return
    picture.planes = [np.ascontiguousarray(p[sj, si]) for p in picture.planes]
    width = (si.stop - si.start) * qf.dx
    height = (sj.stop - sj.start) * qf.dy
    picture.extent_m = (width, height)
    picture.method.append(f"the inner {width / 1000:g} x {height / 1000:g} km around the middle of the "
                          f"{qf.extent[0] / 1000:g} km area, as the quick-looks show a hurricane "
                          "(--whole draws all of it)")
    picture.record["inner_area_km"] = [width / 1000.0, height / 1000.0]


def _build(planes: Planes, request: Request) -> Picture:
    frame, qf, product = planes.frame, planes.qf, request.product
    dx, dy = qf.dx, qf.dy
    extent = _extent(qf.nx, qf.ny, dx, dy)
    sr = spin_range(dx)
    hurricane = planes.hurricane

    if product == "reflectivity":
        dbz, operator = planes.dbz_volume(product)
        plane = dbz.max(axis=0)
        return Picture(product, [plane], ["dbz_column_max"], "dBZ", panel_title="COLUMN MAXIMUM, ALL LEVELS",
                       scale_words="NWS reflectivity colours, 5 dBZ classes",
                       method=[f"Reflectivity by {operator}", "column maximum over every level (CM1 calccref)"],
                       level_words="column maximum", extent_m=extent, record={"operator": operator})
    if product == "reflectivity-level":
        dbz, operator = planes.dbz_volume(product)
        if request.level is not None:
            k, words, z = planes.level_for(request)
            plane = dbz[k]
            how = f"{words}, the level's own value"
            title = f"GM1 REFLECTIVITY AT {fmt_height(z).upper()}"
        else:
            from .quicklook import reflectivity as R

            z = float(request.height_m if request.height_m is not None else REFLECTIVITY_HEIGHT_M)
            if not float(qf.zh[0]) <= z <= float(qf.zh[-1]):
                _refuse("CM1-RENDER-REQUEST", f"--height {z:g} is outside the scalar levels "
                        f"({float(qf.zh[0]):g} to {float(qf.zh[-1]):g} m)", "pass a height between them", "height", z)
            plane, (lo, hi) = R.at_height(dbz, qf.zh, z)
            how = f"{fmt_height(z)} above ground, linear in Z between the {fmt_height(lo)} and {fmt_height(hi)} levels"
            title = f"GM1 REFLECTIVITY AT {fmt_height(z).upper()}"
        return Picture(product, [plane], ["dbz_level"], "dBZ", title=title,
                       panel_title=how.split(",")[0].upper(),
                       scale_words="NWS reflectivity colours, 5 dBZ classes",
                       method=[f"Reflectivity by {operator}", how], level_words=how, extent_m=extent,
                       record={"operator": operator})
    if product == "updraft":
        plane = planes.w_colmax()
        top = hurricane_w_range()
        bounds = (0.0, top) if hurricane else None
        return Picture(product, [plane], ["w_column_max"], "m/s", panel_title="W, COLUMN MAXIMUM OVER ALL LEVELS",
                       bounds=bounds,
                       scale_words=(f"0 to {top:g} m/s, fixed for a hurricane (the quick-looks' range; its eyewall "
                                    "rises at 4 to 10 m/s)" if hurricane else "0 to 60 m/s, fixed"),
                       method=["w read from the restart (its halo cut off), column maximum over every w level"],
                       level_words="column maximum", extent_m=extent, vertical_speed=True)
    if product == "vorticity":
        k, words, height = planes.level_for(request)
        plane = planes.zeta(k)
        z = hurricane_spin(dx) if hurricane else sr
        return Picture(product, [plane], [f"zeta_k{k + 1}"], "1/s",
                       title=f"GM1 VERTICAL VORTICITY AT {fmt_height(height).upper()}",
                       panel_title=f"ZETA, LEVEL {k + 1}", bounds=(-z, z),
                       scale_words=(f"-{fmt_value(z)} to {fmt_value(z)} 1/s, fixed for a hurricane at dx {fmt_grid(dx)}"
                                    if hurricane else
                                    f"-{fmt_value(z)} to {fmt_value(z)} 1/s, fixed for dx {fmt_grid(dx)} "
                                    "(the quick-looks' spin range)"),
                       method=[f"vertical vorticity dv/dx - du/dy at the cell corners, averaged to the centres "
                               f"(CM1's calcvort), on {words}", SPIN_SOURCE],
                       level_words=words, extent_m=extent)
    if product == "cold-pool":
        k, words, height = planes.level_for(request)
        plane = qf.theta_prime(k)
        return Picture(product, [plane], [f"thpert_k{k + 1}"], "K",
                       title=f"GM1 THETA PERTURBATION AT {fmt_height(height).upper()}",
                       panel_title=f"THETA MINUS THE BASE STATE, LEVEL {k + 1}",
                       scale_words="-12 to 4 K, fixed",
                       method=[f"the restart's th3d (potential temperature minus the base state) on {words}"],
                       level_words=words, extent_m=extent)
    if product == "scalar":
        name = request.field
        if not name:
            _refuse("CM1-RENDER-REQUEST", "the scalar product draws one field, and none was named",
                    "pass --field NAME (gpuwm-cm1 render RUNDIR --list-fields lists them)", "field", None)
        if not frame.has(name):
            _refuse("CM1-RENDER-FIELD", f"this restart holds no field {name!r}",
                    "gpuwm-cm1 render RUNDIR --list-fields lists the fields a restart holds", "field", name)
        carrier = frame.carriers[name]
        values = frame.array(name)
        shape = values.shape
        height = None
        if len(shape) == 2:
            plane, words = np.asarray(values, np.float32), "one value per column"
        elif len(shape) == 3:
            faces = shape[0] == qf.nz + 1
            if shape[0] not in (qf.nz, qf.nz + 1) or shape[1] not in (qf.ny, qf.ny + 1) \
                    or shape[2] not in (qf.nx, qf.nx + 1):
                _refuse("CM1-RENDER-FIELD", f"{name} has shape {shape}, not the model's own cells",
                        "pick a field on the model grid (u3d, v3d, w3d, th3d, prs, qv ...)", "field", name)
            k, words, height = planes.level_for(request, faces=faces)
            plane = np.asarray(values[k], np.float32)
        else:
            _refuse("CM1-RENDER-FIELD", f"{name} is not a horizontal or 3-D field", "pick another field", "field",
                    name)
        units = plain_units(carrier.units) or "no unit"
        if name in DERIVED_FIELDS:
            source = "made by the picture tool from the restart"
        else:
            source = "read from the restart" + (" (its halo cut off)" if name in ("u3d", "v3d", "w3d") else "")
        return Picture(product, [plane], [re.sub(r"[^A-Za-z0-9_]", "_", name)], units,
                       title=f"GM1 {name.upper()}" + (f" AT {fmt_height(height).upper()}" if height is not None
                                                       else ""),
                       panel_title=_clean(f"{name}, {carrier.meaning}, {units}")[:70].upper(),
                       scale_words="the picture's own lowest to highest value",
                       method=[_clean(f"{name}: {carrier.meaning} ({units}), {source}, {words}")],
                       level_words=words, extent_m=_extent(plane.shape[1], plane.shape[0], dx, dy),
                       record={"field": name}, legend_units=_clean(units).upper()[:24])
    if product in ("supercell", "vortex-composite", "storm-fusion"):
        dbz, operator = planes.dbz_volume(product)
        k, words, height = planes.level_for(request)
        trio = [dbz.max(axis=0), planes.w_colmax(), planes.zeta(k)]
        # The spin scale is the quick-looks' range for this grid spacing (a finer grid resolves stronger spin);
        # the updraft scales are each product's own, smaller for a hurricane, whose eyewall rises at 4-10 m/s
        # (the three panels take the quick-looks' hurricane range, as the updraft picture does).
        if product == "supercell":
            w_max = hurricane_w_range() if hurricane else 60.0
        elif product == "vortex-composite":
            w_max = 10.0 if hurricane else 30.0
        else:
            w_max = 4.0 if hurricane else 8.0
        z_max = hurricane_spin(dx) if hurricane else sr
        picture = Picture(product, trio, ["dbz_column_max", "w_column_max", f"zeta_k{k + 1}"], "",
                          w_max=w_max, zeta_max=z_max,
                          scale_words=(f"updraft 0 to {fmt_value(w_max)} m/s, spin -{fmt_value(z_max)} to "
                                       f"{fmt_value(z_max)} 1/s at dx {fmt_grid(dx)}, fixed"),
                          method=[f"Reflectivity by {operator}, column maximum",
                                  "w: column maximum over every w level, read from the restart",
                                  f"spin: dv/dx - du/dy (CM1's calcvort) on {words}, {SPIN_SOURCE}"],
                          level_words=f"spin on {words}", extent_m=extent, vertical_speed=True,
                          record={"operator": operator},
                          spin_title=(f"VERTICAL VORTICITY AT {fmt_height(height).upper()}"
                                      if product == "supercell" else None))
        return picture
    if product == "updraft-helicity":
        if float(qf.zf[-1]) < 5000.0:
            _refuse("CM1-RENDER-REQUEST", "this model top is below 5 km, so there is no 2-5 km layer",
                    "draw another product", "product", product)
        plane = qf.uh()[0]
        top = uh_range(dx, hurricane)
        return Picture(product, [plane], ["uh_2_5km"], "m2/s2", panel_title="2-5 KM UPDRAFT HELICITY",
                       bounds=(0.0, top) if top is not None else None, scale_words=uh_scale_words(top),
                       method=["CM1's calcuh sum (misclibs.F:1513-1528) over 2-5 km, on the restart's w and the spin "
                               "derived from its winds and their boundary halo by the picture tool",
                               "a rotating-updraft measure, counted only where w and the spin are both positive"],
                       level_words="2 to 5 km above ground", extent_m=extent)
    if product.startswith("swath-"):
        return _swath(planes, request, extent, sr)
    _refuse("CM1-RENDER-REQUEST", f"unknown render product {product!r}; a restart draws {', '.join(PRODUCTS)}",
            "pick one of those", "product", product)
    raise AssertionError  # unreachable


SHORT = {"swath-wind": "STRONGEST WIND", "swath-vorticity": "STRONGEST SPIN",
         "swath-pressure": "DEEPEST PRESSURE DROP", "swath-helicity": "STRONGEST UPDRAFT HELICITY"}
SWATHS = {
    "swath-wind": ("sws", "m/s", "strongest wind speed at the lowest model level"),
    "swath-vorticity": ("svs", "1/s", "strongest vertical vorticity at the lowest model level"),
    "swath-pressure": ("sps", "Pa", "lowest pressure perturbation at the lowest model level"),
    "swath-helicity": ("shs", "m2/s2", "strongest 2-5 km updraft helicity"),
}


def _swath(planes: Planes, request: Request, extent: tuple[float, float], sr: float) -> Picture:
    frame, qf, product = planes.frame, planes.qf, request.product
    name, units, what = SWATHS[product]
    record = frame.publication.get("swaths") or {}
    layout = record.get("layout") or {}
    canvas = f"{name}2"
    if not frame.has(name):
        why = ("GM1 keeps no vorticity or helicity swath on stretched levels" if planes.qf.stretched
               else "this run kept no such swath")
        _refuse("CM1-RENDER-NO-SWATH", f"this restart holds no {name} swath: {why}",
                "draw swath-wind or swath-pressure, or the quick-looks", "product", product)
    from .tornado_diagnostics import SWATH_INITIAL_VALUES

    moving = qf.domain_moves
    if moving and frame.has(canvas):
        plane = np.array(frame.array(canvas), dtype=np.float32)
        ny, nx = plane.shape
        where = (f"on the ground: a ground-fixed canvas of {nx * qf.dx / 1000:g} x {ny * qf.dy / 1000:g} km "
                 "that the moving domain crossed; ground it never reached is drawn as 0")
        extent = (nx * qf.dx, ny * qf.dy)
    else:
        plane = np.array(frame.array(name), dtype=np.float32)
        where = ("in the domain's own cells, which move with it over the ground (no ground-fixed canvas was kept)"
                 if moving else "on the ground (the domain is fixed to the ground)")
    # A cell no step has reached still holds CM1's initial value (init3d.F:191-201: 200000 Pa for sps, -1000
    # for svs); it is drawn as 0, no swath, and counted out of the values the picture states.
    unvisited = plane == np.float32(SWATH_INITIAL_VALUES[name])
    if SWATH_INITIAL_VALUES[name] != 0.0 and bool(unvisited.any()):
        plane[unvisited] = 0.0
    first = (record.get("first_step") or {}).get(name)
    since = "since the run's start" if first in (1, None) else f"since step {first}"
    if product == "swath-wind":
        # The quick-looks' wind ranges: 40 m/s for a thunderstorm (SWATH_WIND_RANGE), on which an ordinary
        # supercell's 20-30 m/s gusts fill the ramp, and 80 m/s for a hurricane (HURRICANE_WIND_RANGE).
        from .quicklook.products import HURRICANE_WIND_RANGE, SWATH_WIND_RANGE

        top = float(HURRICANE_WIND_RANGE if planes.hurricane else SWATH_WIND_RANGE)
        bounds = (0.0, top)
        scale = (f"0 to {top:g} m/s ground-relative, fixed" + (" for a hurricane" if planes.hurricane else "")
                 + " (the quick-looks' wind range)")
    elif product == "swath-vorticity":
        top = 2.0 * sr
        bounds, scale = (0.0, top), f"0 to {fmt_value(top)} 1/s, fixed for dx {fmt_grid(qf.dx)}"
    elif product == "swath-pressure":
        # A thunderstorm's lowest-level pressure drop deepens as the grid resolves smaller vortices: a few hPa
        # at 1 km, tens of hPa at 100 m; a hurricane's is 50-70 hPa at any grid.
        deep = 8000.0 if planes.hurricane else min(8000.0, 50000.0 * sr)
        bounds = (-deep, 0.0)
        scale = (f"-{deep:g} to 0 Pa, fixed" +
                 (" for a hurricane" if planes.hurricane else f" for a thunderstorm at dx {fmt_grid(qf.dx)}"))
    else:
        top = uh_range(qf.dx, planes.hurricane)
        bounds, scale = ((0.0, top) if top is not None else None), uh_scale_words(top)
    z0 = fmt_height(float(qf.zh[0]))
    level = f"the lowest model level, {z0} above ground" if product != "swath-helicity" else "2 to 5 km above ground"
    speed = " ground-relative" if product == "swath-wind" else ""
    return Picture(product, [plane], [f"{name}_swath"], units,
                   panel_title=f"{SHORT[product]}{speed.upper()}, {since.upper()}", bounds=bounds,
                   scale_words=scale,
                   method=[f"CM1's own running {what} ({name}, solve3.F:855-1136), kept by the model every step "
                           f"{since}, read from the restart", where],
                   level_words=level, extent_m=extent,
                   record={"swath": name, "canvas": canvas if (moving and frame.has(canvas)) else name,
                           "ground_canvas_shape": layout.get("ground_canvas_shape")})


# ----------------------------------------------------------------------------------------------- the renderer
def write_grads(directory: Path, planes: Sequence[np.ndarray], names: Sequence[str]) -> Path:
    """The planes as one GrADS record (CM1's own format, little-endian float32), for ``cm1-render --format grads``."""

    ny, nx = planes[0].shape
    for plane in planes:
        if plane.shape != (ny, nx):
            raise ValueError("every plane of one picture must have one shape")
    data = directory / "planes.dat"
    with data.open("wb") as stream:
        for plane in planes:
            stream.write(np.ascontiguousarray(plane, dtype="<f4").tobytes())
    rows = "\n".join(f"{name} 0 99 {name}" for name in names)
    control = directory / "planes.ctl"
    control.write_text(
        f"dset ^{data.name}\ntitle gm1 native picture planes\nundef -9.99e33\n"
        f"xdef {nx} linear 1 1\nydef {ny} linear 1 1\nzdef 1 linear 1 1\n"
        f"tdef 1 linear 00:00Z01jan2000 1hr\nvars {len(names)}\n{rows}\nendvars\n",
        encoding="ascii",
    )
    return control


def finite_range(plane: np.ndarray) -> tuple[float, float] | None:
    finite = plane[np.isfinite(plane)]
    if finite.size == 0:
        return None
    return float(finite.min()), float(finite.max())


# ----------------------------------------------------------------------------------------------- the door
@dataclass
class RunWords:
    name: str
    facts_line: str
    hurricane: bool


def run_words(root: Path | None, frame: RestartFrame, dx: float, dy: float) -> RunWords:
    """The run's name, the quick-look footer's facts (simulated, GM1 version, microphysics, dx) and whether it is
    a hurricane run, from the run folder's own records; a restart outside a run folder says less."""

    config = frame.publication["config"]
    name = root.name if root is not None else frame.directory.name
    if root is None:
        from .quicklook.facts import scheme_words

        return RunWords(name, f"simulated | {scheme_words(config)} | dx {fmt_grid(dx)}", False)
    from .cm1out import frames as F
    from .cm1out import settings as T
    from .quicklook.facts import run_facts

    layout = F.discover(root)
    settings = None
    try:
        settings = T.resolve(layout)
    except Exception:  # noqa: BLE001 - the namelist reader is a native tool; the deck says the same words
        deck = root / "deck.toml"
        if deck.is_file():
            try:
                settings = T.from_deck(deck)
            except Exception:  # noqa: BLE001 - a run without either still draws, with fewer words
                settings = None
    facts = run_facts(layout, config, dx, dy, settings=settings)
    line = f"simulated | GM1 {facts.version} | {facts.scheme_line} | dx {fmt_grid(dx)}"
    return RunWords(facts.name, line, facts.storm_kind == "hurricane")


def _plain_range(picture: Picture) -> tuple[str, list[Any]]:
    """The values in the picture, said plainly, and as numbers for the receipt."""

    units = picture.units
    if len(picture.planes) == 3:
        r = [finite_range(p) for p in picture.planes]
        parts = []
        if r[0]:
            parts.append(f"reflectivity up to {r[0][1]:.0f} dBZ")
        if r[1]:
            parts.append(f"updraft up to {fmt_value(r[1][1])} m/s")
        if r[2]:
            parts.append(f"spin {fmt_value(r[2][0])} to {fmt_value(r[2][1])} 1/s")
        return "Values in this picture: " + ", ".join(parts), [list(x) if x else None for x in r]
    r = finite_range(picture.planes[0])
    if r is None:
        return "Values in this picture: none finite", [None]
    lo, hi = r
    if picture.product in ("reflectivity", "reflectivity-level"):
        text = (f"Values in this picture: up to {hi:.0f} dBZ" if hi >= 5.0
                else "Values in this picture: nothing at or above 5 dBZ")
    else:
        frame = " ground-relative" if picture.product == "swath-wind" else ""
        text = f"Values in this picture: {fmt_value(lo)} to {fmt_value(hi)} {units}{frame}"
    return text, [[lo, hi]]


def words_for(picture: Picture, choice: RestartChoice, run: RunWords, planes: Planes) -> dict[str, Any]:
    """Every line the renderer draws for this picture, checked by the quick-looks' honesty lint."""

    qf = planes.qf
    when = clock(qf.t, run.hurricane)
    caveat = f"Restart checkpoint at {when}, not a history frame"
    value_line, values = _plain_range(picture)
    scale = f"Colour scale: {picture.scale_words}." if picture.scale_words else ""
    lx, ly = qf.extent
    levels = (f"{qf.nz} {planes.levels_note}, lowest at {fmt_height(float(qf.zh[0]))}")
    moving = (f"the domain moves at {fmt_value(qf.umove)}/{fmt_value(qf.vmove)} m/s ground-relative"
              if qf.domain_moves else "the domain is fixed to the ground")
    segment = f"segment {choice.segment:03d}, " if choice.segment is not None else ""
    footers = [
        f"{value_line}. {scale}".strip(),
        "; ".join(picture.method) + ".",
        f"Domain {lx / 1000:g} x {ly / 1000:g} km, dx {fmt_grid(qf.dx)}, {levels}; {moving}.",
        f"{run.facts_line} | {segment}restart {choice.number} | drawn from the restart by gpuwm-cm1 render",
    ]
    three = len(picture.planes) == 3
    note = None if three else caveat
    classification = caveat.upper() if picture.product in ("vortex-composite", "storm-fusion") else None
    if picture.product == "supercell":
        note = caveat
    texts = [_clean(t) for t in footers]
    vertical = [texts[0], texts[1]] if picture.vertical_speed else []
    from .quicklook.style import lint

    problems = lint([*texts, caveat], vertical=vertical)
    if problems:
        _refuse("CM1-RENDER-WORDS", "the picture's words break the picture rules: " + "; ".join(problems),
                "this is a defect in gpuwm-cm1 render; report it with the command", "product", picture.product)
    return {"note": _clean(note) if note else None, "classification": _clean(classification) if classification else None,
            "footers": texts, "time_words": when.upper(), "label": f"{run.name} | {segment}restart {choice.number}",
            "values": values, "caveat": caveat}


def _output_path(arguments: Any, root: Path | None, choice: RestartChoice, request: Request) -> tuple[Path, bool, str | None]:
    """(where the PNG goes, whether an existing one may be replaced, its path under quicklook/ for the GUI)."""

    if arguments.output is not None:
        requested = Path(arguments.output).expanduser()
        try:
            parent = requested.parent.resolve(strict=True)
        except OSError as error:
            _refuse("CM1-RENDER-OUTPUT", f"the output folder cannot be found: {error}",
                    "create the folder and pass a new .png path in it", "output", str(arguments.output))
        output = parent / requested.name
        if output.suffix.lower() != ".png" or output.exists():
            _refuse("CM1-RENDER-OUTPUT", "--output must be a new .png file; an existing picture is never overwritten",
                    "choose a .png path that does not exist yet, or leave --output out to write into the run's "
                    "quicklook/native/ folder", "output", str(output))
        _writable(parent, "pass --output PICTURE.png in a folder you can write")
        return output, False, None
    if root is None:
        _refuse("CM1-RENDER-OUTPUT", "this restart is not inside a run folder, so there is no quicklook/ folder "
                "to put the picture in", "pass --output PICTURE.png", "output", None)
    parts = [request.product]
    if request.field:
        parts.append(re.sub(r"[^A-Za-z0-9_.-]", "_", request.field))
    if request.level is not None:
        parts.append(f"level{request.level}")
    elif request.height_m is not None:
        parts.append(f"{request.height_m:g}m")
    if request.whole:
        parts.append("whole")
    segment = f"s{choice.segment:03d}-" if choice.segment is not None else ""
    name = "-".join(parts) + f"-{segment}r{choice.number:06d}.png"
    folder = root / "quicklook" / "native"
    # Checked before anything is drawn, and made only when the picture is ready.
    _writable(folder, "pass --output PICTURE.png in a folder you can write")
    return folder / name, True, f"native/{name}"


def _writable(folder: Path, remedy: str) -> None:
    """Refuse, before anything is drawn, a folder the picture cannot be written into (or made in)."""

    existing = folder
    while not existing.exists() and existing.parent != existing:
        existing = existing.parent
    if not existing.is_dir():
        _refuse("CM1-RENDER-OUTPUT", f"cannot write {folder}: {existing} is not a folder", remedy, "output",
                str(folder))
    if not os.access(existing, os.W_OK | os.X_OK):
        what = "" if existing == folder else f" (it would be made in {existing})"
        _refuse("CM1-RENDER-OUTPUT", f"cannot write {folder}{what}: permission denied", remedy, "output",
                str(folder))


def _make_folder(folder: Path) -> None:
    try:
        folder.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        _refuse("CM1-RENDER-OUTPUT", f"cannot write {folder}: {error.strerror or error}",
                "pass --output PICTURE.png in a folder you can write", "output", str(folder))


def list_restarts(root: Path | None, rows: Sequence[RestartChoice], default: RestartChoice, as_json: bool,
                  hurricane: bool = False) -> int:
    if as_json:
        print(json.dumps({"schema": "gpuwm-cm1.render-restarts/v1", "run": os.fspath(root) if root else None,
                          "default": default.as_dict(), "restarts": [r.as_dict() for r in rows]},
                         indent=2, sort_keys=True))
        return 0
    print(f"Restart checkpoints of {root.name if root else default.directory}:")
    print("  restart  segment  storm time            step  file")
    for row in rows:
        mark = "  (the default)" if row == default else ""
        segment = f"{row.segment:03d}" if row.segment is not None else "-"
        when = clock(row.model_time_s, hurricane) if row.model_time_s is not None else "?"
        try:
            path = row.envelope.relative_to(root) if root else row.envelope
        except ValueError:
            path = row.envelope
        print(f"  {row.number:<7}  {segment:<7}  {when:<20}  {row.step if row.step is not None else '?':>5}  "
              f"{path}{mark}")
    print("Pick one with --restart NUMBER (or SEGMENT:NUMBER).")
    return 0


def list_fields(frame: RestartFrame, as_json: bool) -> int:
    rows = [{"name": c.name, "shape": list(c.shape), "units": c.units, "meaning": c.meaning}
            for c in frame.carriers.values()]
    if as_json:
        print(json.dumps({"schema": "gpuwm-cm1.render-fields/v1", "restart": os.fspath(frame.path), "fields": rows},
                         indent=2, sort_keys=True))
        return 0
    print(f"Fields of {frame.path} (for --product scalar --field NAME):")
    for row in rows:
        shape = "x".join(str(v) for v in row["shape"])
        print(f"  {row['name']:<28} {shape:<14} {row['units']:<14} {row['meaning']}")
    return 0


def _run_names_near(target: Path, limit: int = 4000) -> list[str]:
    """Run folders beside a path that does not exist whose names are close to it (a mistyped run name)."""

    import difflib
    import itertools

    parent = target.parent
    try:
        if not parent.is_dir():
            return []
        names = [entry.name for entry in itertools.islice(parent.iterdir(), limit)
                 if entry.is_dir() and (entry / "segments").is_dir()]
    except OSError:
        return []
    return difflib.get_close_matches(target.name, names, n=3, cutoff=0.6)


def door(arguments: Any) -> int:
    """``gpuwm-cm1 render RUNDIR|RESTART [--restart PICK] --product P ...``"""

    from .errors import refuse

    try:
        return _door(arguments)
    except RestartRenderError as error:
        refuse(code=error.code, gate="rust-io-render", reason=error.reason, remedy=error.remedy,
               option=error.option, value=error.value)
    return 2


def _door(arguments: Any) -> int:
    import subprocess
    import tempfile

    target = Path(arguments.history).expanduser()
    if not target.exists():
        near = _run_names_near(target)
        _refuse("CM1-HISTORY-PATH", f"{target} does not exist",
                "give a run folder or a restart.cm1fcp" + (f"; did you mean {', '.join(near)}?" if near else ""),
                "history", str(target))
    target = target.resolve()
    root, choice, rows = choose(target, getattr(arguments, "restart", None))
    as_json = bool(getattr(arguments, "json", False))
    if getattr(arguments, "list", False):
        hurricane = False
        try:
            frame0 = RestartFrame(choice)
            hurricane = run_words(root, frame0, float(frame0.grid["dx_m"]), float(frame0.grid["dy_m"])).hurricane
        except Exception:  # noqa: BLE001 - the list does not need the words
            pass
        return list_restarts(root, rows, choice, as_json, hurricane)
    frame = RestartFrame(choice)
    if getattr(arguments, "list_fields", False):
        return list_fields(frame, as_json)
    product = arguments.product
    if product is None:
        _refuse("CM1-RENDER-REQUEST", "no --product was named",
                f"pass --product, one of {', '.join(PRODUCTS)}", "product", None)
    if product in ENVIRONMENT_PRODUCTS:
        _refuse("CM1-RENDER-REQUEST", f"{product} is an environment plane CM1 writes to its history output "
                "(output_cape and its neighbours); a restart carries none",
                "the quick-looks' environment look draws CAPE and CIN from the sounding", "product", product)
    if product not in PRODUCTS:
        _refuse("CM1-RENDER-REQUEST", f"unknown render product {product!r}; from a run folder: {', '.join(PRODUCTS)}",
                "pick one of those", "product", product)
    fields = list(getattr(arguments, "field", []) or [])
    if product != "scalar" and fields:
        _refuse("CM1-RENDER-REQUEST", f"{product} takes its fields from the restart itself; --field is for the "
                "scalar product", "leave --field out, or use --product scalar --field NAME", "field", fields)
    if len(fields) > 1:
        _refuse("CM1-RENDER-REQUEST", "the scalar product draws one field", "pass one --field", "field", fields)
    level = getattr(arguments, "level", None)
    height = getattr(arguments, "height", None)
    if level is not None and height is not None:
        _refuse("CM1-RENDER-REQUEST", "--level and --height both name a level", "pass one of them", "level", level)
    if (level is not None or height is not None) and product not in LEVEL_PRODUCTS:
        _refuse("CM1-RENDER-REQUEST", f"{product} is not drawn on one level, so --level and --height mean nothing "
                "for it", f"leave them out; the products drawn on one level are {', '.join(LEVEL_PRODUCTS)}",
                "level", level if level is not None else height)
    if height is not None and (not math.isfinite(height) or height < 0):
        _refuse("CM1-RENDER-REQUEST", "--height must be a height above ground in metres", "pass --height 1000",
                "height", height)
    request = Request(product, fields[0] if fields else None, level, height,
                      whole=bool(getattr(arguments, "whole", False)))
    dx, dy = float(frame.grid["dx_m"]), float(frame.grid["dy_m"])
    run = run_words(root, frame, dx, dy)
    if request.whole and not run.hurricane:
        _refuse("CM1-RENDER-REQUEST", "--whole is for a hurricane run, which is drawn over its inner 500 km unless "
                "it is given; this run is drawn over its whole domain already", "leave --whole out", "whole", True)
    output, replace, relative = _output_path(arguments, root, choice, request)
    planes = Planes(frame, run.hurricane)
    picture = build(planes, request)
    words = words_for(picture, choice, run, planes)
    _make_folder(output.parent)

    from .rust_tools import NativeToolError, verify_native_tools

    try:
        executable = verify_native_tools()["cm1-render"]
    except (OSError, NativeToolError) as error:
        _refuse("CM1-NATIVE-TOOLS-INVALID", f"the installed native renderer failed manifest verification: {error}",
                "reinstall a platform wheel whose native-tools.json and executable hashes pass")
    ny, nx = picture.planes[0].shape
    lx, ly = picture.extent_m
    with tempfile.TemporaryDirectory(prefix="gpuwm-cm1-render-planes-") as scratch:
        control = write_grads(Path(scratch), picture.planes, picture.names)
        temporary = output.with_name(f".{output.name}.{os.getpid()}.partial.png")
        command = [str(executable), "--input", str(control), "--format", "grads", "--output", str(temporary),
                   "--product", product]
        for name in picture.names:
            command += ["--field", name]
        command += ["--nx", str(nx), "--ny", str(ny), "--nz", "1", "--level", "0",
                    "--label", _clean(words["label"]), "--valid-seconds", repr(float(planes.qf.t)),
                    "--time-words", _clean(words["time_words"])]
        for flag, value in (("--title", picture.title), ("--panel-title", picture.panel_title),
                            ("--classification", words["classification"]), ("--note", words["note"])):
            if value:
                command += [flag, _clean(value)]
        for line in words["footers"]:
            command += ["--footer", line]
        if picture.bounds is not None:
            command += ["--bounds", f"{picture.bounds[0]!r}:{picture.bounds[1]!r}"]
        if picture.w_max is not None:
            command += ["--w-max", repr(picture.w_max)]
        if picture.zeta_max is not None:
            command += ["--zeta-max", repr(picture.zeta_max)]
        if picture.spin_title:
            command += ["--spin-title", _clean(picture.spin_title)]
        if picture.legend_units:
            command += ["--units", _clean(picture.legend_units)]
        if lx > 0 and ly > 0 and abs(lx / ly - 1.0) > 1e-9:
            command += ["--aspect", f"{lx!r}:{ly!r}"]
        try:
            completed = subprocess.run(command, text=True, capture_output=True, check=False)
            if completed.returncode != 0:
                _refuse("CM1-RUST-RENDER-FAILED", completed.stderr.strip() or f"cm1-render exited {completed.returncode}",
                        "report this with the command; the restart itself is not changed")
            payload = temporary.read_bytes()
            if not payload.startswith(b"\x89PNG\r\n\x1a\n"):
                _refuse("CM1-RUST-RENDER-FAILED", "the native renderer did not write a PNG",
                        "reinstall the wheel and try again")
            if replace:
                os.replace(temporary, output)
            else:
                try:
                    os.link(temporary, output)
                except OSError as error:
                    _refuse("CM1-RENDER-OUTPUT", f"the picture could not be put in place without overwriting: {error}",
                            "choose a new output path", "output", str(output))
        finally:
            temporary.unlink(missing_ok=True)
    receipt = {
        "schema": "gpuwm-cm1.render/v2",
        "status": "rendered",
        "model": "GM1",
        "source": "restart",
        "product": product,
        "product_words": PRODUCT_WORDS[product],
        "output": os.fspath(output),
        "file": relative,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "restart": choice.as_dict(),
        "run": os.fspath(root) if root else None,
        "contract_sha256": frame.header.get("contract_sha256"),
        "contract_sha256_source": "restart-publication.json",
        "arrays_checked": sorted(set(frame.checked)),
        "model_time_s": planes.qf.t,
        "clock": clock(planes.qf.t, run.hurricane),
        "units": picture.units,
        "value_range": words["values"],
        "colour_scale": picture.scale_words,
        "level": picture.level_words,
        "method": picture.method,
        "caveat": words["caveat"],
        "footer": words["footers"],
        "grid": {"nx": planes.qf.nx, "ny": planes.qf.ny, "nz": planes.qf.nz, "dx_m": planes.qf.dx,
                 "dy_m": planes.qf.dy, "levels": planes.levels_note},
        "native_abi": "gpuwm-cm1-render/v1",
        "math_authority": "CM1 r21.1-derived mathematics",
        "visualization": "bilinear cell-center reconstruction with 2x2 supersampling; visualization-only; "
                         "source arrays unchanged",
        **{key: value for key, value in picture.record.items() if key not in ("operator",)},
    }
    if picture.record.get("operator"):
        receipt["reflectivity_operator"] = picture.record["operator"]
    planes.frame.close()
    if as_json:
        print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False, default=float))
    else:
        print(f"drew {PRODUCT_WORDS[product].lower()} to {output}")
        print(f"  {words['footers'][0]}")
        print(f"  {words['caveat']} (restart {choice.number}"
              + (f", segment {choice.segment:03d}" if choice.segment is not None else "") + ")")
    return 0


__all__ = [
    "ENVIRONMENT_PRODUCTS", "LEVEL_PRODUCTS", "PRODUCTS", "PRODUCT_WORDS", "Picture", "Planes", "Request",
    "REFLECTIVITY_PRODUCTS", "RestartChoice", "RestartFrame", "RestartRenderError", "build", "cannot_draw",
    "choose", "clock", "finite_range", "fmt_value", "geometry", "hurricane_spin", "hurricane_w_range", "inner_window", "plain_units", "restarts_of",
    "run_root", "spin_range", "uh_range", "write_grads",
]
