"""Run the real CM1 r21.1 binary on a differential case.

The authority tree is read-only.  This module stages a *copy* of its ``run``
directory (the executable, the lookup tables the schemes read, and nothing
else), writes the case's namelist into the copy, runs it there, and hands back
the history it wrote.  Nothing is ever written under the authority root.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from . import namelist as _namelist
from .grads import CM1History, open_history
from .spec import DifferentialCase

#: Everything CM1's schemes open by name at runtime.  Large tables are
#: symlinked rather than copied so a run directory costs kilobytes.
LOOKUP_FILES = (
    "LANDUSE.TBL", "RRTMG_LW_DATA", "RRTMG_SW_DATA", "freezeH2O.dat",
    "ishmael-gamma-tab.bin", "ishmael-qi-qc.bin", "ishmael-qi-qr.bin",
    "lsnudge_0001.dat", "p3_lookupTable_1.dat-v5.4_2momI",
    "p3_lookupTable_1.dat-v5.4_3momI", "p3_lookupTable_2.dat-v5.3",
    "qr_acr_qgV2.dat", "qr_acr_qsV2.dat",
)

#: Where to look for the built CM1 r21.1 oracle, in order.  A worktree that
#: carries its own ``work/native-reference`` wins; a sibling ``mirror``
#: checkout (the shared layout where worktrees sit beside the main clone) is the
#: fallback so a worktree without one still has an oracle.
#: ``$CM1_AUTHORITY_ROOT`` (or ``--authority``) overrides both.
_REPO_ROOT = Path(__file__).resolve().parents[3]
AUTHORITY_SEARCH_PATH: tuple[Path, ...] = (
    _REPO_ROOT / "work" / "native-reference" / "cm1r21.1",
    _REPO_ROOT.parent / "mirror" / "work" / "native-reference" / "cm1r21.1",
)
DEFAULT_AUTHORITY = AUTHORITY_SEARCH_PATH[0]


class CM1RunError(RuntimeError):
    pass


@dataclass(frozen=True)
class CM1Run:
    directory: Path
    namelist: Path
    log: Path
    returncode: int
    wall_s: float
    history: CM1History

    @property
    def terminated_normally(self) -> bool:
        return "Program terminated normally" in self.log.read_text(errors="replace")


def authority_run_dir(authority: Path | None = None) -> Path:
    if authority is not None:
        candidates: tuple[Path, ...] = (authority,)
    elif os.environ.get("CM1_AUTHORITY_ROOT"):
        candidates = (Path(os.environ["CM1_AUTHORITY_ROOT"]),)
    else:
        candidates = AUTHORITY_SEARCH_PATH
    for root in candidates:
        run = root / "run"
        if (run / "cm1.exe").is_file():
            return run
    searched = ", ".join(str(c / "run" / "cm1.exe") for c in candidates)
    raise CM1RunError(
        "the differential harness needs the built CM1 r21.1 oracle and found no cm1.exe at: "
        f"{searched}. Set CM1_AUTHORITY_ROOT or pass --authority."
    )


#: CM1's shipped decks are namelist files; reading one needs the source tree,
#: not a build, so the pristine extraction serves when no cm1.exe exists.
PRISTINE_RUN_DIR = _REPO_ROOT / "upstream" / "cm1-21.1" / "pristine" / "run"


def base_deck(name: str = "supercell", authority: Path | None = None) -> Path:
    try:
        run = authority_run_dir(authority)
    except CM1RunError:
        if authority is not None or not (PRISTINE_RUN_DIR / "config_files").is_dir():
            raise
        run = PRISTINE_RUN_DIR
    deck = run / "config_files" / name / "namelist.input"
    if not deck.is_file():
        available = sorted(p.name for p in (run / "config_files").iterdir() if p.is_dir())
        raise CM1RunError(f"CM1 ships no deck named {name!r}; it ships {available}")
    return deck


def stage(directory: Path, authority: Path | None = None) -> Path:
    """Copy the authority run directory's executable and tables into ``directory``."""

    run = authority_run_dir(authority)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "cm1.exe"
    if not target.exists():
        shutil.copy2(run / "cm1.exe", target)
        target.chmod(0o755)
    for name in LOOKUP_FILES:
        source = run / name
        link = directory / name
        if source.exists() and not link.exists():
            try:
                link.symlink_to(source)
            except OSError:
                shutil.copy2(source, link)
    return directory


def _write_declared_levels(
    case: DifferentialCase, deck: Mapping[str, Mapping[str, object]], directory: Path
) -> None:
    """CM1's ``input_grid_z`` for ``stretch_z=3`` and ``4``.

    ``param.F:8013`` and ``:8083`` open a file, not a namelist group, so the
    level list cannot travel in ``[cm1]`` and is declared once on the GM1 side
    as ``grid.levels_m`` -- the key map already carries it as GM1-only for
    exactly this reason.  Writing it here is what keeps the two sides on one
    list instead of two that have to be kept in step by hand.

    ``param.F:8021`` reads *scalar* levels for ``stretch_z=3`` and
    ``param.F:8086`` reads *w* levels for ``stretch_z=4``, so the expected
    count differs by one and is checked rather than assumed: a short file makes
    CM1 stop at ``param.F``'s 9012 read error with no indication of which end
    was wrong.
    """

    effective: dict[str, object] = {}
    for fields in deck.values():
        effective.update(fields)
    stretch_z = int(effective.get("stretch_z", 0) or 0)
    levels = case.gm1_only.get("grid.levels_m")
    if stretch_z not in (3, 4):
        if levels:
            raise CM1RunError(
                f"[gm1] grid.levels_m is declared but stretch_z={stretch_z} reads no "
                "level file (param.F:8005,8072), so CM1 would ignore it while "
                "gpuwm-cm1 refused it"
            )
        return
    if not levels:
        raise CM1RunError(
            f"stretch_z={stretch_z} reads its levels from input_grid_z "
            "(param.F:8013,8083); declare them once as [gm1] \"grid.levels_m\" "
            "so both sides run the same column"
        )
    wanted = int(effective["nz"]) + (1 if stretch_z == 4 else 0)
    if len(levels) != wanted:
        raise CM1RunError(
            f"stretch_z={stretch_z} reads {wanted} "
            f"{'w' if stretch_z == 4 else 'scalar'} levels (param.F:"
            f"{'8086' if stretch_z == 4 else '8021'}); the case declares {len(levels)}"
        )
    (directory / "input_grid_z").write_text(
        "".join(f"{float(value):.8f}\n" for value in levels), encoding="utf-8"
    )


def run_case(
    case: DifferentialCase,
    directory: Path,
    *,
    base: str | Path = "supercell",
    authority: Path | None = None,
    timeout_s: float = 3600.0,
    sounding: Path | None = None,
    extra_overrides: Mapping[str, object] | None = None,
) -> CM1Run:
    deck_path = base if isinstance(base, Path) else base_deck(base, authority)
    deck = case.namelist_deck(deck_path)
    # ``physics.component`` is GM1's word; CM1 needs the imoist/ptype it means.
    _namelist.apply(deck, dict(case.translate(deck_path).namelist_overrides))
    if extra_overrides:
        unknown = _namelist.apply(deck, dict(extra_overrides))
        if unknown:
            raise CM1RunError(f"extra overrides name keys the deck has no group for: {sorted(unknown)}")
    stage(directory, authority)
    namelist_path = directory / "namelist.input"
    _namelist.write(deck, namelist_path)
    _write_declared_levels(case, deck, directory)
    if sounding is not None:
        shutil.copy2(sounding, directory / "input_sounding")
    log_path = directory / "cm1.log"
    started = time.monotonic()
    with log_path.open("wb") as log:
        completed = subprocess.run(
            ["./cm1.exe"], cwd=directory, stdout=log, stderr=subprocess.STDOUT,
            timeout=timeout_s, check=False,
        )
    wall = time.monotonic() - started
    history = open_history(directory)
    run = CM1Run(directory, namelist_path, log_path, completed.returncode, wall, history)
    if completed.returncode != 0 or not run.terminated_normally:
        tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-25:])
        raise CM1RunError(f"CM1 exited {completed.returncode} in {directory}:\n{tail}")
    return run
