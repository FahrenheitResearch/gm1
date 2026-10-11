"""Read CM1 r21.1's ``output_format=1`` history.

CM1's ``output_format=1`` is its own direct-access binary32: for each history
time it writes, per output file, every declared variable's levels back to
back, each level ``nx*ny`` little-endian IEEE f32 words with x fastest.  There
is no container, no record marker and no library: the ``.ctl`` descriptor CM1
writes beside it names the variables, their level counts and their order.

Reading it here therefore reads raw words, which is what the comparator wants
and what keeps NetCDF out of Python.
"""

from __future__ import annotations

import array
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

_DSET = re.compile(r"^\s*dset\s+(.+?)\s*$", re.IGNORECASE)
_DIM = re.compile(r"^\s*([xyz])def\s+(\d+)\s+linear\s+(\S+)\s+(\S+)", re.IGNORECASE)
# CM1 writes ``zdef N levels`` followed by the N heights whenever the vertical
# coordinate is not uniform (``writeout.F``'s GrADS descriptor takes the same
# branch as ``param.F:7814``'s selector).  The heights may be on the same line
# or on the lines that follow, and they are in km, as the ``.ctl`` title says.
_LEVELS = re.compile(r"^\s*([xyz])def\s+(\d+)\s+levels\b(.*)$", re.IGNORECASE)
_NUMBER = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[EeDd][-+]?\d+)?")
_TDEF = re.compile(r"^\s*tdef\s+(\d+)", re.IGNORECASE)
_VARS = re.compile(r"^\s*vars\s+(\d+)", re.IGNORECASE)
_UNDEF = re.compile(r"^\s*undef\s+(\S+)", re.IGNORECASE)


@dataclass(frozen=True)
class Variable:
    name: str
    levels: int  # 0 in the ctl means a single 2D surface field
    description: str

    @property
    def nz(self) -> int:
        return max(self.levels, 1)


@dataclass(frozen=True)
class GradsDataset:
    """One CM1 ``.ctl`` and the ``.dat`` files it describes."""

    ctl: Path
    template: str
    nx: int
    ny: int
    nz: int
    x0: float
    dx: float
    y0: float
    dy: float
    z0: float
    dz: float
    times: int
    undef: float
    variables: tuple[Variable, ...]
    #: The descriptor's own level heights, in the ``.ctl``'s units (km).  A
    #: uniform ``zdef ... linear`` carries none and ``z0``/``dz`` describe it
    #: completely; a stretched one carries all ``nz`` of them and ``z0``/``dz``
    #: are then only the first height and the first spacing.
    levels: tuple[float, ...] = ()

    def variable(self, name: str) -> Variable:
        for var in self.variables:
            if var.name == name:
                return var
        raise KeyError(f"{self.ctl.name} has no variable {name!r}; it has {[v.name for v in self.variables]}")

    def offset_words(self, name: str) -> int:
        words = 0
        for var in self.variables:
            if var.name == name:
                return words
            words += var.nz * self.nx * self.ny
        raise KeyError(name)

    def frame_words(self) -> int:
        return sum(v.nz * self.nx * self.ny for v in self.variables)

    def path_for(self, index: int) -> Path:
        """``index`` is 1-based, matching CM1's ``cm1out_000001_s.dat``."""

        if "%y4" in self.template:
            return self.ctl.parent / self.template.replace("%y4", f"{index:04d}")
        return self.ctl.parent / self.template

    def read(self, name: str, index: int) -> tuple[array.array, tuple[int, int, int]]:
        """Return ``(words, (nz, ny, nx))`` for one variable at one time.

        ``words`` is a ``float32`` :class:`array.array` in the file's own
        slow-to-fast order: level, then y, then x.
        """

        var = self.variable(name)
        path = self.path_for(index)
        plane = self.nx * self.ny
        start = self.offset_words(name) * 4
        if "%y4" not in self.template:
            start += (index - 1) * self.frame_words() * 4
        count = var.nz * plane
        buffer = array.array("f")
        with path.open("rb") as handle:
            handle.seek(start)
            raw = handle.read(count * 4)
        if len(raw) != count * 4:
            raise ValueError(f"{path} is short: wanted {count * 4} bytes at {start}, got {len(raw)}")
        buffer.frombytes(raw)
        return buffer, (var.nz, self.ny, self.nx)


def read_ctl(path: Path) -> GradsDataset:
    template = ""
    dims: dict[str, tuple[int, float, float]] = {}
    levels: dict[str, tuple[float, ...]] = {}
    times = 0
    undef = -999999.9
    variables: list[Variable] = []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        index += 1
        match = _DSET.match(line)
        if match:
            template = match.group(1).lstrip("^")
            continue
        match = _DIM.match(line)
        if match:
            dims[match.group(1).lower()] = (int(match.group(2)), float(match.group(3)), float(match.group(4)))
            continue
        match = _LEVELS.match(line)
        if match:
            axis, wanted = match.group(1).lower(), int(match.group(2))
            found = [float(text.replace("D", "E").replace("d", "e"))
                     for text in _NUMBER.findall(match.group(3))]
            while len(found) < wanted and index < len(lines):
                found.extend(
                    float(text.replace("D", "E").replace("d", "e"))
                    for text in _NUMBER.findall(lines[index])
                )
                index += 1
            if len(found) != wanted:
                raise ValueError(
                    f"{path.name} declares {wanted} {axis} levels and carries {len(found)}"
                )
            levels[axis] = tuple(found)
            continue
        match = _TDEF.match(line)
        if match:
            times = int(match.group(1))
            continue
        match = _UNDEF.match(line)
        if match:
            undef = float(match.group(1))
            continue
        match = _VARS.match(line)
        if match:
            for _ in range(int(match.group(1))):
                parts = lines[index].split(None, 3)
                index += 1
                variables.append(Variable(parts[0], int(parts[1]), parts[3].strip() if len(parts) > 3 else ""))
            continue
    nx, x0, dx = dims["x"]
    ny, y0, dy = dims["y"]
    if "z" in dims:
        nz, z0, dz = dims["z"]
        heights: tuple[float, ...] = ()
    else:
        heights = levels["z"]
        nz = len(heights)
        z0 = heights[0]
        # One spacing cannot describe a stretched column; this is the first
        # one, kept only so a caller that wants a nominal figure has one.
        dz = heights[1] - heights[0] if nz > 1 else 0.0
    return GradsDataset(
        ctl=path, template=template, nx=nx, ny=ny, nz=nz,
        x0=x0, dx=dx, y0=y0, dy=dy, z0=z0, dz=dz,
        times=times, undef=undef, variables=tuple(variables),
        levels=heights,
    )


@dataclass(frozen=True)
class CM1History:
    """Every ``.ctl`` a CM1 run wrote, plus its per-frame model times."""

    directory: Path
    datasets: dict[str, GradsDataset]
    model_time_s: tuple[float, ...]
    step: tuple[int, ...]
    dt_s: tuple[float, ...]

    def find(self, name: str) -> tuple[str, GradsDataset]:
        for key, dataset in self.datasets.items():
            if any(v.name == name for v in dataset.variables):
                return key, dataset
        raise KeyError(f"no CM1 output file carries {name!r}; carried: {sorted(self.field_names())}")

    def field_names(self) -> Iterator[str]:
        for dataset in self.datasets.values():
            for var in dataset.variables:
                yield var.name

    def read(self, name: str, index: int) -> tuple[array.array, tuple[int, int, int]]:
        return self.find(name)[1].read(name, index)

    @property
    def frames(self) -> int:
        return len(self.model_time_s)


def open_history(directory: Path, prefix: str = "cm1out") -> CM1History:
    datasets: dict[str, GradsDataset] = {}
    for ctl in sorted(directory.glob(f"{prefix}_*.ctl")):
        stem = ctl.stem[len(prefix) + 1:]
        # Not history: the statistics, the metadata, and the parcels (cm1out_parcel.ctl,
        # parcel.F:2144-2211), whose variables share names with the history's (dbz, th, u ...).
        if stem in ("stats", "metadata", "parcel"):
            continue
        datasets[stem] = read_ctl(ctl)
    meta = directory / f"{prefix}_metadata.dat"
    times: list[float] = []
    steps: list[int] = []
    dts: list[float] = []
    if meta.exists():
        names = _metadata_variables(directory / f"{prefix}_metadata.ctl")
        words = array.array("f")
        words.frombytes(meta.read_bytes())
        stride = len(names)
        columns = {name: index for index, name in enumerate(names)}
        for base in range(0, len(words) - stride + 1, stride):
            times.append(float(words[base + columns["mtime"]]))
            steps.append(int(words[base + columns["nstep"]]))
            dts.append(float(words[base + columns["dt"]]))
    return CM1History(directory, datasets, tuple(times), tuple(steps), tuple(dts))


#: ``writeout.F:9536-9700`` writes one record per variable per frame, and which
#: variables it writes depends on the run: ``adt`` only for ``adapt_dt=1``,
#: ``umove``/``vmove``/``domainlocx``/``domainlocy`` only for ``imove>=1``,
#: four more for radiation.  The stride is therefore not a constant, and
#: ``writeout.F:9721-9739`` writes the list that says what it is.  Reading it
#: is not optional: a fixed four-word stride silently reads ``umove`` as a
#: model time on every translating-frame run.
_METADATA_FALLBACK = ("mtime", "nstep", "dt", "nwrite")


def _metadata_variables(ctl: Path) -> tuple[str, ...]:
    if not ctl.is_file():
        return _METADATA_FALLBACK
    names: list[str] = []
    reading = False
    for line in ctl.read_text(encoding="utf-8", errors="replace").splitlines():
        token = line.strip()
        if not token:
            continue
        head = token.split()[0].lower()
        if head == "vars":
            reading = True
            continue
        if head == "endvars":
            break
        if reading:
            names.append(token.split()[0])
    if not names or not {"mtime", "nstep", "dt"} <= set(names):
        raise ValueError(
            f"{ctl} does not describe CM1's metadata record: it must list at "
            f"least mtime, nstep and dt; it lists {names}"
        )
    return tuple(names)
