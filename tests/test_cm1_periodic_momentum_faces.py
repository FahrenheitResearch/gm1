"""advu/advv/advw on every serial lateral family, word for word.

CM1 r21.1 advects u on faces ``i = i1..i2`` and v on rows ``j = j1..j2``
(adv.F:968-978, 1913-1923), where ``param.F:6678-6681`` makes ``ibw`` (and
``ibe``/``ibs``/``ibn``) zero exactly on a periodic side, so a periodic
domain advances its boundary faces 1 and ni+1.  The inflow edits of adv.F
(the boundary-face flux copy and the divx-piece subtraction) run only where
``doitw``..``doitn`` are set, which is an open side (value 2).  Before the
lateral-family fix the kernels advanced faces 2..ni and applied the inflow
edits on all four sides whatever the family, which is CM1's all-open schedule.

Three authorities, from strongest to most portable:

* CM1's own compiled ``advu``/``advv``/``advw``
  (:mod:`tools.oracle.cm1_flat_stretched_native`, MODE_MOMENTUM_FAMILY): the
  driver fills the wind halos with CM1's ``bcu``/``bcv``/``bcw`` for the
  family and forms the mass fluxes and ``divx`` in the halo as solve2.F does.
  Skips unless the native reference build and g++ are present.
* The module's statement-ordered CPU reference, held to the kernel TEXT through
  the host emulator (g++ only).
* Translation invariance of the CPU reference on periodic domains, which the
  pre-fix schedule breaks at the boundary faces (NumPy only).

A periodic state has ``u(1) = u(ni+1)`` and ``v(1) = v(nj+1)``: both faces are
advanced by the same words from identical inputs, in CM1 and here, so every
fixture keeps that invariant; the halo words CM1 forms from its periodic wind
copies are then the wrapped physical words the kernels read.
"""

from __future__ import annotations

import shutil
from typing import Any, Mapping

import numpy as np
import pytest

from cm1core.cuda_momentum_advection5 import (
    CUDA_MOMENTUM_ADVECTION5_SOURCE,
    MOMENTUM_ALL_OPEN_SIDES,
    cpu_terrain_open_momentum_advection5_inplace,
    momentum_advection5_counts,
    momentum_open_boundary_code,
)
from cm1core.cuda_psolver3_terrain_open import terrain_open_sound_shapes
from cm1core.cuda_terrain_open_boundary import terrain_static_shapes

F32 = np.float32
NG = 3
GXX = shutil.which("g++") is not None
requires_gxx = pytest.mark.skipif(not GXX, reason="the host kernel emulator needs g++")

#: CM1's (wbc, ebc, sbc, nbc).  Doubly periodic; CM1's squall-line family
#: (open x, periodic y) and its transpose; rigid walls beside periodic and
#: open sides; and the all-open family every earlier receipt ran.
FAMILIES = (
    (1, 1, 1, 1),
    (2, 2, 1, 1),
    (1, 1, 2, 2),
    (3, 4, 1, 1),
    (1, 1, 4, 3),
    (3, 3, 2, 2),
    (2, 2, 2, 2),
)


def _bits(value: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(value, dtype=F32).view(np.uint32)


def _differing(reference: np.ndarray, candidate: np.ndarray) -> int:
    return int((_bits(reference) != _bits(candidate)).sum())


def _keep_periodic_faces(arrays: Mapping[str, np.ndarray], sides, ni: int, nj: int) -> None:
    """``u(ni+1) = u(1)`` and ``v(nj+1) = v(1)`` on a periodic direction."""

    if sides[0] == 1:
        arrays["u3d"][..., NG + ni] = arrays["u3d"][..., NG]
    if sides[2] == 1:
        arrays["v3d"][:, NG + nj, :] = arrays["v3d"][:, NG, :]


def _periodic_halo(arrays: Mapping[str, np.ndarray], sides, ni: int, nj: int) -> None:
    """bc.F's periodic ``bcu``/``bcv``/``bcw`` copies, x first then y."""

    u, v, w = arrays["u3d"], arrays["v3d"], arrays["w3d"]
    if sides[0] == 1:
        for i in range(1, NG + 1):
            u[..., NG - i] = u[..., NG + ni - i]
            u[..., NG + ni + i] = u[..., NG + i]
            for s in (v, w):
                s[..., NG - i] = s[..., NG + ni - i]
                s[..., NG + ni - 1 + i] = s[..., NG + i - 1]
    if sides[2] == 1:
        for j in range(1, NG + 1):
            v[:, NG - j] = v[:, NG + nj - j]
            v[:, NG + nj + j] = v[:, NG + j]
            for s in (u, w):
                s[:, NG - j] = s[:, NG + nj - j]
                s[:, NG + nj - 1 + j] = s[:, NG + j - 1]


# ------------------------------------------------------------ CPU reference


def _uniform_state(seed: int, ni: int, nj: int, nk: int, sides) -> tuple[dict, dict]:
    """A horizontally homogeneous flat column with random winds."""

    generator = np.random.default_rng(seed)
    shapes = terrain_open_sound_shapes(ni, nj, nk)
    arrays = {
        name: np.zeros(shapes[name], dtype=F32)
        for name in ("u3d", "uten", "rru", "v3d", "vten", "rrv", "w3d", "wten", "rrw", "uh", "vh")
    }
    arrays["uh"].fill(F32(1.0))
    arrays["vh"].fill(F32(1.0))
    for name, scale in (("u3d", 12.0), ("v3d", 12.0), ("w3d", 6.0)):
        arrays[name][:] = generator.uniform(-scale, scale, arrays[name].shape).astype(F32)
    _keep_periodic_faces(arrays, sides, ni, nj)
    _periodic_halo(arrays, sides, ni, nj)
    metrics = {
        name: np.ones(shape, dtype=F32)
        for name, shape in terrain_static_shapes(ni=ni, nj=nj, nk=nk).items()
    }
    metrics["dzdx"].fill(F32(0.0))
    metrics["dzdy"].fill(F32(0.0))
    for k in range(nk):
        density = F32(1.12 - 0.018 * k)
        metrics["rho0_3d"][k].fill(density)
        metrics["rr0"][k].fill(F32(1.0) / density)
    for k in range(nk + 1):
        metrics["rf0"][k].fill(F32(1.13 - 0.017 * k))
        metrics["rrf0"][k].fill(F32(1.0) / metrics["rf0"][k, 0, 0])
        metrics["c2"][k].fill(F32(0.5))
        metrics["c1"][k].fill(F32(0.5))
    return arrays, metrics


def _cpu(arrays: dict, metrics: dict, sides, ni: int, nj: int, nk: int) -> tuple[dict, np.ndarray]:
    out = {name: value.copy() for name, value in arrays.items()}
    divx = np.zeros((nk, nj, ni), dtype=F32)
    cpu_terrain_open_momentum_advection5_inplace(
        out, metrics, divx, ni=ni, nj=nj, nk=nk, nrk=2, dx=100.0, dy=100.0, dz=200.0,
        lateral_sides=sides,
    )
    return out, divx


def _roll(field: np.ndarray, kind: str, sides, ni: int, nj: int, *, dx: int, dy: int) -> np.ndarray:
    """A padded u/v/w carrier translated by ``(dx, dy)`` cells, halos refilled.

    Only the unique physical words move (faces ``0..ni-1`` of u, rows
    ``0..nj-1`` of v); the duplicate boundary face and the halo are rebuilt
    from them as a periodic state has them.
    """

    out = field.copy()
    # A row moves with its whole x extent (the halo of an open x side
    # included), a column with its whole y extent.
    rows = slice(NG, NG + nj)
    cols = slice(NG, NG + ni)
    if dy:
        out[:, rows, :] = np.roll(out[:, rows, :], dy, axis=1)
    if dx:
        out[:, :, cols] = np.roll(out[:, :, cols], dx, axis=2)
    carriers = {"u3d": out if kind == "u" else np.zeros((1, nj + 2 * NG, ni + 1 + 2 * NG), F32),
                "v3d": out if kind == "v" else np.zeros((1, nj + 1 + 2 * NG, ni + 2 * NG), F32),
                "w3d": out if kind == "w" else np.zeros((1, nj + 2 * NG, ni + 2 * NG), F32)}
    _keep_periodic_faces(carriers, sides, ni, nj)
    _periodic_halo(carriers, sides, ni, nj)
    return out


@pytest.mark.parametrize("sides", [(1, 1, 1, 1), (2, 2, 1, 1), (1, 1, 2, 2)], ids=str)
def test_cpu_reference_is_translation_invariant_along_periodic_directions(sides) -> None:
    """A periodic direction has no edge: shifting the state shifts every tendency.

    The pre-fix schedule (the all-open word on the same state) leaves the
    boundary faces unadvanced and applies the inflow edits at the edges, so
    it is not invariant; the fixed one is, word for word, and its two copies
    of the boundary face (faces 1 and ni+1) receive the same word.
    """

    ni, nj, nk = 9, 8, 8
    dx, dy = (3 if sides[0] == 1 else 0), (2 if sides[2] == 1 else 0)
    arrays, metrics = _uniform_state(11, ni, nj, nk, sides)
    moved = dict(arrays)
    for name, kind in (("u3d", "u"), ("v3d", "v"), ("w3d", "w")):
        moved[name] = _roll(arrays[name], kind, sides, ni, nj, dx=dx, dy=dy)
    cells = {
        "uten": ("u", (slice(1, nk + 1), slice(NG, NG + nj), slice(NG, NG + ni + 1))),
        "vten": ("v", (slice(1, nk + 1), slice(NG, NG + nj + 1), slice(NG, NG + ni))),
        "wten": ("w", (slice(2, nk + 1), slice(NG, NG + nj), slice(NG, NG + ni))),
    }
    for family, invariant in ((sides, True), (MOMENTUM_ALL_OPEN_SIDES, False)):
        base, _ = _cpu(arrays, metrics, family, ni, nj, nk)
        shifted, _ = _cpu(moved, metrics, family, ni, nj, nk)
        differing = sum(
            _differing(_roll(base[name], kind, sides, ni, nj, dx=dx, dy=dy)[box], shifted[name][box])
            for name, (kind, box) in cells.items()
        )
        if invariant:
            assert differing == 0, family
            if sides[0] == 1:
                assert _differing(base["uten"][..., NG], base["uten"][..., NG + ni]) == 0
                assert np.count_nonzero(base["uten"][1:nk + 1, NG:NG + nj, NG])
            if sides[2] == 1:
                assert _differing(base["vten"][:, NG], base["vten"][:, NG + nj]) == 0
                assert np.count_nonzero(base["vten"][1:nk + 1, NG, NG:NG + ni])
        else:
            assert differing > 0, family


def test_family_word_and_counts_follow_param_f() -> None:
    assert momentum_open_boundary_code((2, 2, 2, 2)) == 0x2222
    assert momentum_open_boundary_code((1, 1, 2, 2)) == 0x2211
    assert momentum_advection5_counts(10, 7, 4, (2, 2, 2, 2)) == (4 * 7 * 9, 4 * 6 * 10, 3 * 7 * 10)
    assert momentum_advection5_counts(10, 7, 4, (1, 1, 1, 1)) == (4 * 7 * 11, 4 * 8 * 10, 3 * 7 * 10)
    assert momentum_advection5_counts(10, 7, 4, (3, 4, 1, 1)) == (4 * 7 * 9, 4 * 8 * 10, 3 * 7 * 10)
    for bad in ((1, 2, 2, 2), (2, 2, 3, 1), (0, 2, 2, 2), (2, 2, 2), (True, 2, 2, 2)):
        with pytest.raises(ValueError):
            momentum_open_boundary_code(bad)


# ------------------------------------------------------- kernel text, host


def _emulate(source: str, arrays: dict, metrics: dict, sides, ni: int, nj: int, nk: int,
             *, flat_column: Mapping[str, Any] | None = None) -> tuple[dict, np.ndarray]:
    """The six momentum kernels of ``source`` on host copies of the carriers."""

    from tools.oracle.cuda_kernel_host_emulator import HostKernelLibrary, launch_grid

    library = HostKernelLibrary.compile(source)
    out = {name: value.copy() for name, value in arrays.items()}
    divx = np.zeros((nk, nj, ni), dtype=F32)
    values: dict[str, Any] = {
        **metrics, **(flat_column or {}),
        "u": out["u3d"], "v": out["v3d"], "w": out["w3d"],
        "rru": out["rru"], "rrv": out["rrv"], "rrw": out["rrw"], "divx": divx,
        "uten": out["uten"], "vten": out["vten"], "wten": out["wten"],
        "uh": out["uh"], "vh": out["vh"], "rho0": metrics["rho0_3d"], "c2a": metrics["c2"],
        "ni": ni, "nj": nj, "nk": nk, "ng": NG, "dx": 100.0, "dy": 100.0, "dz": 200.0,
        "smeps": 1.0e-20, "obc": momentum_open_boundary_code(sides),
    }
    u_words, v_words, w_words = momentum_advection5_counts(ni, nj, nk, sides)
    for name, words in (
        ("cm1_mom5_horizontal_mass_v1", nk * nj * (ni + 1) + nk * (nj + 1) * ni),
        ("cm1_mom5_vertical_mass_v1", (nk + 1) * nj * ni),
        ("cm1_mom5_divergence_v1", nk * nj * ni),
        ("cm1_mom5_advu_v1", u_words),
        ("cm1_mom5_advv_v1", v_words),
        ("cm1_mom5_advw_v1", w_words),
    ):
        kernel = library.kernel(name)
        arguments = []
        for parameter in kernel.parameters:
            value = values[parameter.name]
            if parameter.pointer:
                arguments.append(value)
            elif parameter.base == "float":
                arguments.append(np.float32(value))
            else:
                arguments.append(np.uint32(value))
        grid, block = launch_grid(words, 128)
        kernel(grid, block, tuple(arguments))
    return out, divx


@requires_gxx
@pytest.mark.parametrize("sides", FAMILIES, ids=str)
def test_kernel_text_is_the_cpu_reference_on_every_family(sides) -> None:
    ni, nj, nk = 9, 8, 8
    arrays, metrics = _uniform_state(23, ni, nj, nk, sides)
    for name in ("uten", "vten", "wten"):
        arrays[name][:] = np.random.default_rng(5).uniform(-1, 1, arrays[name].shape).astype(F32)
    expected, expected_divx = _cpu(arrays, metrics, sides, ni, nj, nk)
    observed, observed_divx = _emulate(CUDA_MOMENTUM_ADVECTION5_SOURCE, arrays, metrics, sides, ni, nj, nk)
    assert _differing(expected_divx, observed_divx) == 0
    for name in ("rru", "rrv", "rrw", "uten", "vten", "wten"):
        assert _differing(expected[name], observed[name]) == 0, name
    # Non-vacuous: every tendency moved, and on a periodic side the boundary
    # face moved too.
    for name in ("uten", "vten", "wten"):
        assert _differing(arrays[name], observed[name]) > 0, name
    moved_u = _differing(arrays["uten"][1:nk + 1, NG:NG + nj, NG], observed["uten"][1:nk + 1, NG:NG + nj, NG])
    moved_v = _differing(arrays["vten"][1:nk + 1, NG, NG:NG + ni], observed["vten"][1:nk + 1, NG, NG:NG + ni])
    assert (moved_u > 0) == (sides[0] == 1)
    assert (moved_v > 0) == (sides[2] == 1)


@requires_gxx
def test_the_all_open_word_is_the_pre_fix_text_word_for_word() -> None:
    """The family every earlier receipt ran keeps every word it had."""

    from tools.oracle.momentum_lateral_family_spelling import (
        PRE_LATERAL_FAMILY_SHA256,
        pre_lateral_family_spelling,
    )
    import hashlib

    before = pre_lateral_family_spelling(CUDA_MOMENTUM_ADVECTION5_SOURCE)
    assert hashlib.sha256(before.encode()).hexdigest() == PRE_LATERAL_FAMILY_SHA256
    ni, nj, nk = 9, 8, 8
    arrays, metrics = _uniform_state(31, ni, nj, nk, MOMENTUM_ALL_OPEN_SIDES)
    generator = np.random.default_rng(7)
    for name in ("uten", "vten", "wten"):
        arrays[name][:] = generator.uniform(-1, 1, arrays[name].shape).astype(F32)
    live, live_divx = _emulate(CUDA_MOMENTUM_ADVECTION5_SOURCE, arrays, metrics, MOMENTUM_ALL_OPEN_SIDES, ni, nj, nk)
    old, old_divx = _emulate(before, arrays, metrics, MOMENTUM_ALL_OPEN_SIDES, ni, nj, nk)
    assert _differing(live_divx, old_divx) == 0
    for name in ("rru", "rrv", "rrw", "uten", "vten", "wten"):
        assert _differing(live[name], old[name]) == 0, name
        assert np.count_nonzero(live[name]), name


# ------------------------------------------------------ CM1's own objects


def _inflow_edit_cells(sides, ni: int, nj: int, nk: int) -> dict[str, np.ndarray]:
    """The cells an open side's inflow edit can reach, per tendency box."""

    u = np.zeros((nk, nj, ni + 1), bool)
    v = np.zeros((nk, nj + 1, ni), bool)
    w = np.zeros((nk - 1, nj, ni), bool)
    if sides[2] == 2:
        u[:, 0] = True
        w[:, 0] = True
    if sides[3] == 2:
        u[:, nj - 1] = True
        w[:, nj - 1] = True
    if sides[0] == 2:
        v[:, :, 0] = True
        w[:, :, 0] = True
    if sides[1] == 2:
        v[:, :, ni - 1] = True
        w[:, :, ni - 1] = True
    return {"uten": u, "vten": v, "wten": w}


def _native_ready() -> bool:
    if not GXX:
        return False
    from tools.oracle import cm1_flat_stretched_native as native

    return native.available()


requires_native = pytest.mark.skipif(
    not _native_ready(),
    reason="requires the built native-reference CM1 tree and g++: set CM1_AUTHORITY_ROOT",
)


@pytest.fixture(scope="module")
def native_driver(tmp_path_factory: pytest.TempPathFactory):
    from tools.oracle import cm1_flat_stretched_native as native

    work = tmp_path_factory.mktemp("cm1-periodic-momentum-native")
    return native.build_driver(work), work


def _column_state(seed: int, ni: int, nj: int, nk: int, *, stretched: bool):
    """Winds, tendencies and base-state columns in CM1's padded layout."""

    from cm1core.cm1_vertical_grid import build_vertical_levels
    from tools.oracle import cm1_flat_stretched_native as native

    layout = native.shapes(ni, nj, nk)
    generator = np.random.default_rng(seed)

    def uniform(shape, low, high):
        return generator.uniform(low, high, shape).astype(F32)

    if stretched:
        levels = build_vertical_levels(
            stretch_z=4, nk=nk, dz_m=200.0, ztop_m=2400.0,
            levels_m=(0.0, 30.0, 80.0, 170.0, 320.0, 560.0, 950.0, 1550.0, 2400.0),
        )
        mh = np.ascontiguousarray(levels.mh, dtype=F32)
        mf = np.ascontiguousarray(levels.mf, dtype=F32)
    else:
        mh = np.ones((nk + 2,), F32)
        mf = np.ones((nk + 3,), F32)
    rho0 = uniform((nk + 2,), 0.7, 1.3)
    rf0 = uniform((nk + 2,), 0.7, 1.3)
    c2 = uniform((nk + 2,), 0.3, 0.7)
    columns = {
        "rho0": rho0, "rr0": (F32(1.0) / rho0).astype(F32),
        "rf0": rf0, "rrf0": (F32(1.0) / rf0).astype(F32),
        "c2": c2, "c1": (F32(1.0) - c2).astype(F32),
    }
    winds = {
        "u3d": uniform(layout["U"], -12, 12), "v3d": uniform(layout["V"], -12, 12),
        "w3d": uniform(layout["W"], -6, 6),
        "uten": uniform(layout["U"], -1, 1), "vten": uniform(layout["V"], -1, 1),
        "wten": uniform(layout["W"], -1, 1),
    }
    return layout, columns, winds, mh, mf


@requires_native
@pytest.mark.parametrize("stretched", [True, False], ids=["flat-stretched", "uniform"])
@pytest.mark.parametrize("sides", FAMILIES, ids=str)
def test_advu_advv_advw_are_cm1_words_on_every_family(native_driver, sides, stretched) -> None:
    """CM1's compiled advu/advv/advw (and solve2's fluxes) on the family, 0 words."""

    from cm1core.cuda_gm1_child_stretched_momentum import (
        CM1_R21_SMEPS_F32,
        GM1_FLAT_STRETCHED_MOMENTUM_SOURCE,
    )
    from tools.oracle import cm1_flat_stretched_native as native

    ni, nj, nk = 11, 9, 8
    layout, columns, winds, mh, mf = _column_state(71, ni, nj, nk, stretched=stretched)
    _keep_periodic_faces(winds, sides, ni, nj)
    arrays = {
        **winds,
        **{name: native.column_field(column, layout["S"]) for name, column in columns.items()},
        "mh": native.column_field(mh, layout["S"]), "mf": native.column_field(mf, layout["W"]),
        "rho": np.random.default_rng(3).uniform(0.7, 1.3, layout["S"]).astype(F32),
    }
    cm1 = native.run(*native_driver, native.Call(
        native.MODE_MOMENTUM_FAMILY, ni, nj, nk, nrk=1, iargs=tuple(sides),
        dt=1.0, dx=125.0, dy=125.0, dz=200.0), arrays)

    def compact(column, first, layers):
        block = np.asarray(column, dtype=F32)[first:first + layers, None, None]
        return np.ascontiguousarray(np.broadcast_to(block, (layers, nj, ni))).astype(F32)

    def run(obc_sides) -> dict[str, np.ndarray]:
        from tools.oracle.cuda_kernel_host_emulator import HostKernelLibrary, launch_grid

        source = GM1_FLAT_STRETCHED_MOMENTUM_SOURCE if stretched else CUDA_MOMENTUM_ADVECTION5_SOURCE
        library = HostKernelLibrary.compile(source)
        carriers = {
            "rru": np.zeros(layout["U"], F32), "rrv": np.zeros(layout["V"], F32),
            "rrw": np.zeros(layout["W"], F32), "divx": np.zeros((nk, nj, ni), F32),
            "uten": winds["uten"].copy(), "vten": winds["vten"].copy(), "wten": winds["wten"].copy(),
        }
        values: dict[str, Any] = {
            **carriers,
            # The halos CM1's bcu/bcv/bcw(.,1) filled for this family.
            "u": cm1["u3d"], "v": cm1["v3d"], "w": cm1["w3d"],
            "rho0": arrays["rho0"], "rf0": compact(columns["rf0"], 1, nk + 1),
            "rrf0": compact(columns["rrf0"], 1, nk + 1), "c2a": compact(columns["c2"], 1, nk + 1),
            "rr0": compact(columns["rr0"], 1, nk), "mh": mh, "mf": mf,
            "gz": np.ones((nj, ni), F32), "gzu": np.ones((nj, ni + 1), F32),
            "gzv": np.ones((nj + 1, ni), F32), "rgzu": np.ones((nj, ni + 1), F32),
            "rgzv": np.ones((nj + 1, ni), F32), "dzdx": np.zeros((nj, ni), F32),
            "dzdy": np.zeros((nj, ni), F32),
            "uh": np.ones((ni + 2 * NG,), F32), "vh": np.ones((nj + 2 * NG,), F32),
            "ni": ni, "nj": nj, "nk": nk, "ng": NG, "dx": 125.0, "dy": 125.0, "dz": 200.0,
            "rdz": F32(1.0) / F32(200.0),
            "smeps": CM1_R21_SMEPS_F32 if stretched else 1.0e-20,
            "obc": momentum_open_boundary_code(obc_sides),
        }
        if not stretched:
            # The uniform text reads rho0 as a compact 3-D carrier.
            values["rho0"] = compact(columns["rho0"], 1, nk)
        u_words, v_words, w_words = momentum_advection5_counts(ni, nj, nk, obc_sides)
        for name, words in (
            ("cm1_mom5_horizontal_mass_v1", nk * nj * (ni + 1) + nk * (nj + 1) * ni),
            ("cm1_mom5_vertical_mass_v1", (nk + 1) * nj * ni),
            ("cm1_mom5_divergence_v1", nk * nj * ni),
            ("cm1_mom5_advu_v1", u_words),
            ("cm1_mom5_advv_v1", v_words),
            ("cm1_mom5_advw_v1", w_words),
        ):
            kernel = library.kernel(name)
            arguments = []
            for parameter in kernel.parameters:
                value = values[parameter.name]
                if parameter.pointer:
                    arguments.append(value)
                elif parameter.base == "float":
                    arguments.append(np.float32(value))
                else:
                    arguments.append(np.uint32(value))
            grid, block = launch_grid(words, 128)
            kernel(grid, block, tuple(arguments))
        return carriers

    faces = {
        "uten": (slice(1, nk + 1), slice(NG, NG + nj), slice(NG, NG + ni + 1)),
        "vten": (slice(1, nk + 1), slice(NG, NG + nj + 1), slice(NG, NG + ni)),
        "wten": (slice(2, nk + 1), slice(NG, NG + nj), slice(NG, NG + ni)),
        "rru": (slice(1, nk + 1), slice(NG, NG + nj), slice(NG, NG + ni + 1)),
        "rrv": (slice(1, nk + 1), slice(NG, NG + nj + 1), slice(NG, NG + ni)),
        "rrw": (slice(1, nk + 2), slice(NG, NG + nj), slice(NG, NG + ni)),
    }
    gm1 = run(sides)
    edits = _inflow_edit_cells(sides, ni, nj, nk)
    for name, cells in faces.items():
        reference, candidate = cm1[name][cells], gm1[name][cells]
        if stretched or name not in edits:
            assert _differing(reference, candidate) == 0, name
            continue
        # The uniform text keeps the pre-family grouping of the open-side
        # inflow edits, (q*0.5)*(c*(rdy*vh)), where adv.F:1145-1172 writes
        # (((q*0.5)*c)*rdy)*vh (the flat image regroups them): every other
        # word is CM1's, and those cells differ by rounding only.
        mask = edits[name]
        assert _differing(reference[~mask], candidate[~mask]) == 0, name
        scale = float(np.spacing(np.max(np.abs(reference[mask])))) if mask.any() else 0.0
        assert float(np.max(np.abs(reference[mask].astype(np.float64) - candidate[mask]), initial=0.0)) <= 4.0 * scale, name
    mass = (slice(1, nk + 1), slice(NG, NG + nj), slice(NG, NG + ni))
    assert _differing(cm1["divx"][mass], gm1["divx"]) == 0
    # Non-vacuous: CM1 moved every tendency it was handed, including the
    # boundary face of a periodic side ...
    for name in ("uten", "vten", "wten"):
        assert _differing(cm1[name][faces[name]], winds[name][faces[name]]) > 0, name
    if sides[0] == 1:
        west = (slice(1, nk + 1), slice(NG, NG + nj), NG)
        assert _differing(cm1["uten"][west], winds["uten"][west]) > 0
    if sides[2] == 1:
        south = (slice(1, nk + 1), NG, slice(NG, NG + ni))
        assert _differing(cm1["vten"][south], winds["vten"][south]) > 0
    # ... and the pre-fix schedule (all four sides open) is not CM1 there.
    if sides != MOMENTUM_ALL_OPEN_SIDES:
        before = run(MOMENTUM_ALL_OPEN_SIDES)
        assert sum(_differing(cm1[n][faces[n]], before[n][faces[n]]) for n in ("uten", "vten", "wten")) > 0


# ------------------------------------------------ the budget's ndiff split


@pytest.mark.parametrize("axis", ["x", "y"])
def test_budget_ndiff_is_translation_invariant_along_a_periodic_direction(axis) -> None:
    """cm1_budget_operators' adv.F ndiff reads across a periodic side as the kernel does.

    On a periodic state shifting the winds and fluxes shifts ``hadv``/``vadv``
    word for word, boundary face included, and the two copies of the boundary
    face agree -- which the pre-fix range (faces 2..ni only) cannot satisfy.
    """

    from cm1core.cm1_budget_operators import nondiffusive_u, nondiffusive_v

    ni, nj, nk = 9, 8, 8
    sides = (1, 1, 2, 2) if axis == "x" else (2, 2, 1, 1)
    generator = np.random.default_rng(97)
    shapes = terrain_open_sound_shapes(ni, nj, nk)
    state = {name: generator.uniform(-4, 4, shapes[name]).astype(F32)
             for name in ("u3d", "v3d", "w3d")}
    fluxes = {name: generator.uniform(-4, 4, shapes[name.replace("rr", "") + "3d"]).astype(F32)
              for name in ("rru", "rrv", "rrw")}
    column = generator.uniform(0.3, 0.7, (nk + 1, 1, 1)).astype(F32)
    c2 = np.ascontiguousarray(np.broadcast_to(column, (nk + 1, nj, ni))).astype(F32)
    divh = generator.uniform(-1e-3, 1e-3, (nk, nj, ni)).astype(F32)
    divv = generator.uniform(-1e-3, 1e-3, (nk, nj, ni)).astype(F32)
    scale_u = np.ones((nk, nj, ni + 1), F32)
    scale_v = np.ones((nk, nj + 1, ni), F32)
    levels = np.full((nk,), F32(1.0 / 200.0), F32)

    def periodic(fields):
        packed = {"u3d": fields["u3d"], "v3d": fields["v3d"], "w3d": fields["w3d"]}
        _keep_periodic_faces(packed, sides, ni, nj)
        _periodic_halo(packed, sides, ni, nj)
        rr = {"u3d": fields["rru"], "v3d": fields["rrv"], "w3d": fields["rrw"]}
        _keep_periodic_faces(rr, sides, ni, nj)
        _periodic_halo(rr, sides, ni, nj)

    fields = {**state, **fluxes}
    periodic(fields)
    dx, dy = (3, 0) if axis == "x" else (0, 2)
    moved = {name: _roll(fields[name], {"u3d": "u", "rru": "u", "v3d": "v", "rrv": "v"}.get(name, "w"),
                         sides, ni, nj, dx=dx, dy=dy) for name in fields}
    divh_m = np.roll(divh, (dy, dx), axis=(1, 2))
    divv_m = np.roll(divv, (dy, dx), axis=(1, 2))

    def split(f, dh, dv):
        common = dict(rru=f["rru"], rrv=f["rrv"], rrw=f["rrw"], c2=c2, divh=dh, divv=dv,
                      rdsf_levels=levels, ni=ni, nj=nj, nk=nk, ng=NG, dx=100.0, dy=100.0)
        if axis == "x":
            return nondiffusive_u(np, u=f["u3d"], scale=scale_u, periodic_x=True, **common)
        return nondiffusive_v(np, v=f["v3d"], scale=scale_v, periodic_y=True, **common)

    base, shifted = split(fields, divh, divv), split(moved, divh_m, divv_m)
    for b, s in zip(base, shifted):
        if axis == "x":
            unique = np.roll(b[..., :ni], dx, axis=2)
            assert _differing(unique, s[..., :ni]) == 0
            assert _differing(b[..., 0], b[..., ni]) == 0
            assert np.count_nonzero(b[..., 0])
        else:
            unique = np.roll(b[:, :nj], dy, axis=1)
            assert _differing(unique, s[:, :nj]) == 0
            assert _differing(b[:, 0], b[:, nj]) == 0
            assert np.count_nonzero(b[:, 0])


# ------------------------------------------- the uniform lane's acoustic step


def test_the_periodic_boundary_faces_join_the_pressure_gradient_before_the_divergence() -> None:
    """sound.F:386-406 updates faces 1+ibw..ni+1-ibe, then :446-530 forms the divergence.

    On a periodic side ``ibw = 0`` puts faces 1 and ni+1 in that loop.  The
    uniform lane's sealed pgrad kernel omits them and a boundary-family kernel
    advances them; it must run before the column pass that forms the
    divergence, or the first and last columns read the previous small step's
    boundary face (tests/test_cartesian_periodic_column_gpu.py integrates it).
    """

    import inspect

    from cm1core import cuda_psolver3_terrain_open as stage

    text = inspect.getsource(stage.launch_cuda_psolver3_terrain_open_three_stage)
    loop = text[text.index("for n in range(1, nloop + 1):"):]
    pgrad = loop.index("pgrad(face_grid")
    faces = loop.index("_bc.enqueue_periodic_face_pressure_gradient(")
    divergence = loop.index("column(col_grid")
    assert loop.count("_bc.enqueue_periodic_face_pressure_gradient(") == 1
    assert pgrad < faces < divergence


# ------------------------------------------------------------ the device


@pytest.mark.skipif(
    __import__("os").environ.get("CM1_RUN_GPU_TESTS") != "1",
    reason="bounded GPU gate is opt-in",
)
@pytest.mark.parametrize("sides", FAMILIES, ids=str)
def test_device_launch_is_the_cpu_reference_on_every_family(sides) -> None:
    """The real launcher (its counts and its obc word) on the device, 0 words."""

    cupy = pytest.importorskip("cupy")
    from cm1core.cuda_momentum_advection5 import launch_cuda_terrain_open_momentum_advection5

    ni, nj, nk = 9, 8, 8
    arrays, metrics = _uniform_state(43, ni, nj, nk, sides)
    generator = np.random.default_rng(9)
    for name in ("uten", "vten", "wten"):
        arrays[name][:] = generator.uniform(-1, 1, arrays[name].shape).astype(F32)
    expected, expected_divx = _cpu(arrays, metrics, sides, ni, nj, nk)
    device = {name: cupy.asarray(value) for name, value in arrays.items()}
    device_metrics = {name: cupy.asarray(value) for name, value in metrics.items()}
    launch = launch_cuda_terrain_open_momentum_advection5(
        device, device_metrics, ni=ni, nj=nj, nk=nk, nrk=2, dx=100.0, dy=100.0, dz=200.0,
        lateral_sides=sides,
    )
    launch.ready_event.synchronize()
    assert launch.receipt["selection"]["lateral_boundaries"] == list(sides)
    assert _differing(expected_divx, cupy.asnumpy(launch.divx)) == 0
    for name in ("rru", "rrv", "rrw", "uten", "vten", "wten"):
        assert _differing(expected[name], cupy.asnumpy(device[name])) == 0, name
