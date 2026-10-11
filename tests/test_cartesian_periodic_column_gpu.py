"""A horizontally uniform column on periodic sides, integrated on a real device.

A periodic direction has no edge.  CM1 r21.1 advances the boundary faces of a
periodic side with every other face (adv.F ``advu`` ``i1``/``i2``, sound.F:386-406
with ``ibw = 0``) and reads across the side at the wrapped index, so a state
that is the same in every column along a periodic direction stays the same
word for word along it, whatever the other direction does.  GM1 broke that
twice on periodic sides:

* advu/advv never advanced faces 1 and ni+1 and ran the open-side inflow edits
  on every side (fixed by the lateral-family ``obc`` word of
  ``cuda_momentum_advection5``);
* the uniform (``stretch_z = 0``) lane's acoustic stage updated the periodic
  boundary faces after the column pass had already formed sound.F's
  horizontal divergence from them, so the first and last columns saw the
  previous small step's boundary face.  A 12x12x115 free-slip column at 100 m
  stopped on the final-RK EOS guard within its first seconds for that reason.

Each case runs the real Cartesian runtime for a few large steps from the
analytic Weisman-Klemp column (``isnd = 5``, no bubble) and checks every
prognostic carrier along each periodic direction word for word, and that the
two copies of a periodic boundary face are one word.  Both vertical families
run: the uniform family and the flat stretched (profile) family.

A second check moves CM1's warm bubble by whole cells on a doubly periodic
domain and compares the closure's step-1 held tendencies, which must move with
it word for word.  They do not yet at the edge columns (the SGS closure's
lateral treatment is the open one on every family); that check is a strict
expected failure until the closure takes the family.

Opt-in (``CM1_RUN_GPU_TESTS=1``): it compiles and runs the whole lane.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("CM1_RUN_GPU_TESTS") != "1",
    reason="bounded GPU gate is opt-in",
)

ROOT = Path(__file__).resolve().parents[1]
CRATE = ROOT / "rust/cm1-state"
DECK = ROOT / "evidence/oracle/periodic-momentum-faces/uniform-periodic-column-600.gpuwm-cm1.toml"
NG = 3
STEPS = 6
PROGNOSTIC = ("u3d", "v3d", "w3d", "th3d", "pp3d")

#: CM1's (wbc, ebc, sbc, nbc): doubly periodic, and CM1's squall-line family
#: (open x, periodic y) with its transpose.
FAMILIES = ((1, 1, 1, 1), (2, 2, 1, 1), (1, 1, 2, 2))

#: param.F:7819-7953 with dz_bot = dz_top = dz: nk1 = 2000/500 = 4,
#: nk3 = (20000-8000)/500 = 24, nk2 = 40-28 = 12 and 12*500 = 8000-2000, so
#: the profile family runs on the uniform family's own 500 m column.
PROFILE_COLUMN = (
    "stretch_z = 1\ndz_bot_m = 500.0\ndz_top_m = 500.0\n"
    "str_bot_m = 2000.0\nstr_top_m = 8000.0\nztop_m = 20000.0\n"
)


@pytest.fixture(scope="module")
def idealized_library() -> Path:
    """The Rust base-state initializer, as the stretched GPU integration test finds it."""

    supplied = os.environ.get("GM1_CM1_STATE_LIBRARY")
    if supplied:
        return Path(supplied).resolve(strict=True)
    target = ROOT / "build/rust-native-target"
    for name in ("libcm1_state.so", "cm1_state.dll", "libcm1_state.dylib"):
        candidate = target / "release" / name
        if candidate.is_file():
            return candidate
    subprocess.run(
        ["cargo", "build", "--release", "--lib", "--target-dir", str(target)],
        cwd=CRATE,
        check=True,
    )
    found = [
        target / "release" / name
        for name in ("libcm1_state.so", "cm1_state.dll", "libcm1_state.dylib")
        if (target / "release" / name).is_file()
    ]
    assert len(found) == 1
    return found[0]


def _deck(tmp_path: Path, sides: tuple[int, int, int, int], *, profile: bool) -> Path:
    replacements = {
        "duration_s": "duration_s = 6.0",
        "history_interval_s": "history_interval_s = 6.0",
        "restart_interval_s": "restart_interval_s = 6.0",
        "memory_budget_mib": "memory_budget_mib = 2048",
        # Diagnostics this check does not read; calcvort and the PV product
        # refuse a stretched column (CM1-OUTPUT-STRETCHED-METRIC).
        "output_dbz": "output_dbz = false",
        "output_pv": "output_pv = 0",
        "output_uh": "output_uh = 0",
        "output_vort": "output_vort = 0",
        "west": f"west = {sides[0]}",
        "east": f"east = {sides[1]}",
        "south": f"south = {sides[2]}",
        "north": f"north = {sides[3]}",
    }
    pending = dict(replacements)
    out = []
    for line in DECK.read_text().splitlines():
        key = line.split("=", 1)[0].strip() if "=" in line and not line.startswith("#") else None
        if key is not None and key in pending:
            out.append(pending.pop(key))
            continue
        out.append(line)
        if profile and line.strip() == "dz_m = 500.0":
            out.append(PROFILE_COLUMN.rstrip("\n"))
    assert not pending, f"deck clauses never matched: {sorted(pending)}"
    path = tmp_path / ("profile.toml" if profile else "uniform.toml")
    path.write_text("\n".join(out) + "\n")
    return path


def _integrate(deck: Path, library: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], Any]:
    cupy = pytest.importorskip("cupy")
    from cm1core.cartesian_door import (
        bubble_center_m,
        build_runtime_config,
        horizontal_origin_m,
        require_initializer_selection,
        require_moving_frame,
        selected_capacity_profile,
    )
    from cm1core.config import load_config
    from cm1core.cuda_cartesian_runtime import CudaCartesianRuntime
    from cm1core.gm1_idealized_residency import IdealizedSource

    config = load_config(deck)
    runtime_config = build_runtime_config(config)
    grid = runtime_config.grid
    selection, _ = require_initializer_selection(config)
    frame = require_moving_frame(config)
    stream = cupy.cuda.Stream(non_blocking=True)

    def read(runtime) -> dict[str, np.ndarray]:
        return {
            name: np.ascontiguousarray(cupy.asnumpy(runtime.dynamics.fields[name]))
            for name in PROGNOSTIC
        }

    # The door's own frame arguments: iorigin=2 and the declared bubble centre.
    with IdealizedSource(
        library, b"", grid, selection=selection, moving_frame=frame,
        origin_xy_m=horizontal_origin_m(config), bubble_center_xy_m=bubble_center_m(config),
    ) as source:
        with stream:
            runtime = CudaCartesianRuntime(
                source, runtime_config, cupy=cupy, stream=stream,
                capacity_profile=selected_capacity_profile(config),
            )
            stretched = bool(runtime.dynamics.stretched)
            initial = read(runtime)
            for _ in range(STEPS):
                runtime.step()
            runtime.synchronize()
            final = read(runtime)
    return initial, final, (grid, stretched)


def _physical(a: np.ndarray, ni: int, nj: int, nk: int) -> tuple[np.ndarray, str]:
    """The carrier's words ``(k, j, i)`` and its stagger (``u``, ``v`` or ``s``).

    The runtime's canonical fields are the physical extents: u on faces
    ``0..ni``, v on rows ``0..nj``, w on levels ``1..nk+1``.
    """

    shapes = {(nk, nj, ni + 1): "u", (nk, nj + 1, ni): "v", (nk + 1, nj, ni): "s", (nk, nj, ni): "s"}
    assert a.shape in shapes, a.shape
    return np.ascontiguousarray(a, dtype=np.float32), shapes[a.shape]


def _differing_along(a: np.ndarray, stagger: str, axis: str) -> int:
    """Words that differ from column 0 (x) or row 0 (y), boundary-face copy included."""

    bits = a.view(np.uint32)
    if axis == "x":
        unique = bits[..., :-1] if stagger == "u" else bits
        return int((unique != unique[..., :1]).sum())
    unique = bits[:, :-1, :] if stagger == "v" else bits
    return int((unique != unique[:, :1, :]).sum())


def _face_copies_differ(a: np.ndarray, stagger: str, axis: str) -> int:
    bits = a.view(np.uint32)
    if axis == "x" and stagger == "u":
        return int((bits[..., 0] != bits[..., -1]).sum())
    if axis == "y" and stagger == "v":
        return int((bits[:, 0, :] != bits[:, -1, :]).sum())
    return 0


@pytest.mark.parametrize("sides", FAMILIES, ids=str)
@pytest.mark.parametrize("profile", [False, True], ids=["uniform-family", "profile-family"])
def test_a_uniform_column_stays_one_word_along_every_periodic_direction(
    tmp_path: Path, idealized_library: Path, sides, profile
) -> None:
    initial, final, (grid, stretched) = _integrate(_deck(tmp_path, sides, profile=profile), idealized_library)
    assert stretched is profile
    ni, nj, nk = grid.nx, grid.ny, grid.nz
    axes = [axis for axis, side in (("x", sides[0]), ("y", sides[2])) if side == 1]
    moved = 0
    for name in PROGNOSTIC:
        start, _ = _physical(initial[name], ni, nj, nk)
        end, stagger = _physical(final[name], ni, nj, nk)
        assert np.all(np.isfinite(end)), name
        moved += int((start.view(np.uint32) != end.view(np.uint32)).sum())
        for axis in axes:
            # Precondition: the analytic column starts the same in every column.
            assert _differing_along(start, stagger, axis) == 0, (name, axis, "initial")
            assert _differing_along(end, stagger, axis) == 0, (name, axis)
            assert _face_copies_differ(end, stagger, axis) == 0, (name, axis, "boundary face copies")
    # The run moved the state, so the invariance is a property of the step.
    assert moved > 0



# ------------------------------------ a bubble beside the periodic boundary


def _history(deck: Path, library: Path, *, steps: int) -> list[dict[str, np.ndarray]]:
    """u and v after each of ``steps`` large steps."""

    cupy = pytest.importorskip("cupy")
    from cm1core.cartesian_door import (
        bubble_center_m,
        build_runtime_config,
        horizontal_origin_m,
        require_initializer_selection,
        require_moving_frame,
        selected_capacity_profile,
    )
    from cm1core.config import load_config
    from cm1core.cuda_cartesian_runtime import CudaCartesianRuntime
    from cm1core.gm1_idealized_residency import IdealizedSource

    config = load_config(deck)
    runtime_config = build_runtime_config(config)
    selection, _ = require_initializer_selection(config)
    frame = require_moving_frame(config)
    stream = cupy.cuda.Stream(non_blocking=True)
    out = []
    with IdealizedSource(
        library, b"", runtime_config.grid, selection=selection, moving_frame=frame,
        origin_xy_m=horizontal_origin_m(config), bubble_center_xy_m=bubble_center_m(config),
    ) as source:
        with stream:
            runtime = CudaCartesianRuntime(
                source, runtime_config, cupy=cupy, stream=stream,
                capacity_profile=selected_capacity_profile(config),
            )
            for _ in range(steps):
                runtime.step()
                runtime.synchronize()
                out.append({n: cupy.asnumpy(runtime.dynamics.fields[n]) for n in ("u3d", "v3d")})
    return out


@pytest.mark.parametrize("profile", [False, True], ids=["uniform-family", "profile-family"])
def test_the_two_copies_of_a_periodic_face_stay_one_word_beside_a_bubble(
    tmp_path: Path, idealized_library: Path, profile
) -> None:
    """A periodic side's face 1 and face ni+1 are one face: CM1 advances both alike.

    The warm bubble sits beside the north-east corner of a doubly periodic
    domain, so its pressure field crosses both periodic boundaries.  The flat
    (profile) acoustic stage used to update those faces with the open side's
    ssopenbc radiation step instead of sound.F's pressure gradient
    (param.F:9182-9219 masks a periodic face in): the copies were 7e-3 m/s apart
    after 20 s and 0.28 m/s after 180 s.
    """

    frames = _history(_bubble_deck(tmp_path, "edge.toml", (5500.0, 5500.0), profile=profile),
                      idealized_library, steps=STEPS)
    assert float(np.abs(frames[-1]["u3d"][..., :-1] - frames[0]["u3d"][..., :-1]).max()) > 0.0
    for step, fields in enumerate(frames, start=1):
        u, v = fields["u3d"], fields["v3d"]
        assert int((u[..., 0].view(np.uint32) != u[..., -1].view(np.uint32)).sum()) == 0, ("u", step)
        assert int((v[:, 0, :].view(np.uint32) != v[:, -1, :].view(np.uint32)).sum()) == 0, ("v", step)


# ------------------------------------------- a shifted bubble, the closure


def _bubble_deck(tmp_path: Path, name: str, centre: tuple[float, float], *, profile: bool) -> Path:
    """The doubly periodic column with CM1's warm bubble (iinit=1) at ``centre`` (iorigin=2)."""

    base = _deck(tmp_path, (1, 1, 1, 1), profile=profile).read_text().splitlines()
    out = []
    for line in base:
        if line.strip() == "iinit = 0":
            out += ["iinit = 1", f"bubble_center_x_m = {centre[0]}", f"bubble_center_y_m = {centre[1]}"]
            continue
        out.append(line)
    assert "iinit = 1" in out
    path = tmp_path / name
    path.write_text("\n".join(out) + "\n")
    return path


HELD = ("uten", "vten", "wten", "thten1")


def _first_held_tendencies(deck: Path, library: Path) -> dict[str, np.ndarray]:
    """The closure's held tendencies of step 1, read before any domain reduction runs."""

    cupy = pytest.importorskip("cupy")
    from cm1core.cartesian_door import (
        bubble_center_m,
        build_runtime_config,
        horizontal_origin_m,
        require_initializer_selection,
        require_moving_frame,
        selected_capacity_profile,
    )
    from cm1core.config import load_config
    from cm1core.cuda_cartesian_runtime import CudaCartesianRuntime
    from cm1core.gm1_idealized_residency import IdealizedSource

    config = load_config(deck)
    runtime_config = build_runtime_config(config)
    selection, _ = require_initializer_selection(config)
    frame = require_moving_frame(config)
    stream = cupy.cuda.Stream(non_blocking=True)
    captured: dict[str, np.ndarray] = {}
    with IdealizedSource(
        library, b"", runtime_config.grid, selection=selection, moving_frame=frame,
        origin_xy_m=horizontal_origin_m(config), bubble_center_xy_m=bubble_center_m(config),
    ) as source:
        with stream:
            runtime = CudaCartesianRuntime(
                source, runtime_config, cupy=cupy, stream=stream,
                capacity_profile=selected_capacity_profile(config),
            )
            execute = runtime.execute

            def capture(operation, *, step):
                execute(operation, step=step)
                if step == 1 and not captured and "held" in operation.phase.value:
                    # The closure may run on streams of its own: wait for the device.
                    cupy.cuda.runtime.deviceSynchronize()
                    captured.update({n: cupy.asnumpy(runtime.dynamics.held[n]) for n in HELD})

            runtime.execute = capture
            runtime.step()
            runtime.synchronize()
    assert set(captured) == set(HELD)
    return captured


#: Open, measured on the RTX 3080 at 3a3609e4: 14 held-wten words (and the
#: held-thten1 words beside them) differ, all in the edge column or row next to
#: the moved bubble, on both vertical families.  The SGS closure applies CM1's
#: open-side lateral treatment on every family -- turbtend.F:67-102 copies the
#: boundary flux (dumx(ni+1) = dumx(ni)) only when the side is open (wbc=2),
#: and turb.F:4133-4400/4792-5280 treat rigid and open stresses by family --
#: so on a periodic side the edge column's turbulent flux divergence is
#: dropped.  Strict: fixing the closure's lateral family makes this pass.
CLOSURE_LATERAL_FAMILY_OPEN = pytest.mark.xfail(
    strict=True,
    reason="the SGS closure runs the open-side edge treatment on every lateral family",
)


@CLOSURE_LATERAL_FAMILY_OPEN
@pytest.mark.parametrize("profile", [False, True], ids=["uniform-family", "profile-family"])
def test_the_closure_tendencies_move_with_a_bubble_on_a_periodic_domain(
    tmp_path: Path, idealized_library: Path, profile
) -> None:
    """A periodic domain has no preferred column: moving the bubble moves the closure's words.

    The two initial states are exact shifts of each other (5 columns east, 3
    rows south); the step-1 held tendencies are compared before the dry-mass
    reduction, the first domain-wide sum, runs.
    """

    east, south = 5, 3
    a = _first_held_tendencies(_bubble_deck(tmp_path, "a.toml", (0.0, 0.0), profile=profile), idealized_library)
    b = _first_held_tendencies(
        _bubble_deck(tmp_path, "b.toml", (1000.0 * east, -1000.0 * south), profile=profile), idealized_library
    )
    assert np.count_nonzero(a["wten"]) and np.count_nonzero(a["thten1"])
    for name in HELD:
        # The unique words: u faces 0..ni-1, v rows 0..nj-1 (face ni is face 0).
        ni, nj = a["thten1"].shape[-1], a["thten1"].shape[-2]
        first, second = a[name][..., :nj, :ni], b[name][..., :nj, :ni]
        moved = np.roll(np.roll(first, east, axis=-1), -south, axis=-2)
        assert int((moved.view(np.uint32) != second.view(np.uint32)).sum()) == 0, name
