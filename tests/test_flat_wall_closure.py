"""The closure beside a rigid side wall: the derived text, its words without a wall, its routing.

:mod:`cm1core.cuda_gm1_flat_wall_closure` derives CM1's wall branches
(turb.F:4788-4970, :4131-4232; turbtend.F's edge zeros on an open side only;
adv.F's TKE inflow edits on an open side only, bcw's mirrored TKE halo) from
the flat closure text.  Where a tile has no wall edge the derived text must
write the flat text's words exactly, so a walled domain's interior and its
open sides are what a stretched column has always run.  Beside a wall the
derived text gives CM1's words: ``evidence/oracle/walls-operators`` holds the
comparisons against CM1 r21.1's own dumps, and the last test here holds the
recorded summary to that.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import pytest

from cm1core import cuda_gm1_flat_wall_closure as wall
from cm1core.cuda_gm1_flat_stretched_closure import (
    GM1_FLAT_STRETCHED_CLOSURE_KERNEL_NAMES,
    GM1_FLAT_STRETCHED_TKE_ADVECTION_KERNEL_NAME,
    GM1_FLAT_STRETCHED_TKE_ADVECTION_SOURCE,
    flat_stretched_closure_source,
)

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence" / "oracle" / "walls-operators"
HOST = shutil.which("g++") is not None
F32 = np.float32
NI, NJ, NK = 9, 7, 10


def test_every_flat_kernel_has_a_wall_twin_with_one_more_argument() -> None:
    from tools.oracle.cuda_kernel_host_emulator import parse_signatures

    flat = parse_signatures(flat_stretched_closure_source(max_nk=16))
    derived = parse_signatures(wall.flat_wall_closure_source(max_nk=16))
    assert len(wall.GM1_FLAT_WALL_CLOSURE_KERNEL_NAMES) == len(GM1_FLAT_STRETCHED_CLOSURE_KERNEL_NAMES)
    for flat_name, wall_name in zip(GM1_FLAT_STRETCHED_CLOSURE_KERNEL_NAMES, wall.GM1_FLAT_WALL_CLOSURE_KERNEL_NAMES):
        assert wall_name == flat_name.replace("gm1_flat_", "gm1_flatwall_")
        assert [p.name for p in derived[wall_name]] == [p.name for p in flat[flat_name]] + ["walls"]
    tke_flat = parse_signatures(GM1_FLAT_STRETCHED_TKE_ADVECTION_SOURCE)[GM1_FLAT_STRETCHED_TKE_ADVECTION_KERNEL_NAME]
    tke_wall = parse_signatures(wall.GM1_FLAT_WALL_TKE_ADVECTION_SOURCE)[wall.GM1_FLAT_WALL_TKE_ADVECTION_KERNEL_NAME]
    assert [p.name for p in tke_wall] == [p.name for p in tke_flat] + ["walls"]


def test_the_derived_texts_quote_cm1s_wall_branches() -> None:
    text = wall.flat_wall_closure_source(max_nk=16)
    module = Path(wall.__file__).read_text(encoding="utf-8")
    for anchor in ("turb.F:4801-4809", "turb.F:4845-4853", "turb.F:4895-4970", "turb.F:4131-",
                   "turbtend.F:1200-1227", "bc.F:942-1143"):
        assert anchor in module
    # The kernel text itself names the CM1 lines its wall statements come from.
    for anchor in ("turb.F:4801-4809", "turb.F:4845-4853"):
        assert anchor in text
    assert "gm1_flat_" not in text
    assert wall.GM1_FLAT_WALL_CLOSURE_TEMPLATE_SHA256 != wall.GM1_FLAT_STRETCHED_CLOSURE_TEMPLATE_SHA256


@pytest.mark.parametrize(("sides", "expected"), [
    ((2, 2, 2, 2), 0),
    ((1, 1, 2, 2), 0),
    ((3, 3, 2, 2), 0x33),
    ((2, 2, 4, 4), 0x4400),
    ((3, 4, 4, 3), 0x3443),
])
def test_the_walls_word_names_rigid_sides_only(sides: tuple[int, int, int, int], expected: int) -> None:
    assert wall.wall_code(sides) == expected
    assert wall.has_rigid_wall(sides) == (expected != 0)


def test_a_tile_names_only_the_wall_edges_that_are_domain_edges() -> None:
    sides = (3, 4, 4, 3)
    corner = SimpleNamespace(ci0=0, cnx=40, cj0=0, cny=40)
    inner = SimpleNamespace(ci0=30, cnx=40, cj0=30, cny=40)
    far = SimpleNamespace(ci0=60, cnx=40, cj0=60, cny=40)
    assert wall.tile_wall_code(sides, corner, nx=100, ny=100) == 0x0403
    assert wall.tile_wall_code(sides, inner, nx=100, ny=100) == 0
    assert wall.tile_wall_code(sides, far, nx=100, ny=100) == 0x3040
    whole = SimpleNamespace(ci0=0, cnx=100, cj0=0, cny=100)
    assert wall.tile_wall_code(sides, whole, nx=100, ny=100) == wall.wall_code(sides)


# ------------------------------------------------------------------ no wall: the flat words

def _random(seed: int) -> Any:
    generator = np.random.default_rng(seed)

    def uniform(shape: tuple[int, ...], low: float = -1.0, high: float = 1.0) -> np.ndarray:
        return generator.uniform(low, high, shape).astype(F32)

    return uniform


def _inputs(seed: int) -> dict[str, Any]:
    uniform = _random(seed)
    return {
        "u": uniform((NK, NJ, NI + 1)), "v": uniform((NK, NJ + 1, NI)), "w": uniform((NK + 1, NJ, NI)),
        "rho": uniform((NK, NJ, NI), 0.8, 1.2), "rf": uniform((NK + 1, NJ, NI), 0.8, 1.2),
        "mh": uniform((NK + 2,), 0.5, 2.0), "mf": uniform((NK + 1, NJ, NI), 0.5, 2.0),
        "mhfull": uniform((NK, NJ, NI), 0.5, 2.0),
        "c1": uniform((NK + 1, NJ, NI), 0.2, 0.8), "c2": uniform((NK + 1, NJ, NI), 0.2, 0.8),
        "s11": uniform((NK, NJ, NI)), "s22": uniform((NK, NJ, NI)), "s33": uniform((NK, NJ, NI)),
        "s12": uniform((NK, NJ + 1, NI + 1)), "s13": uniform((NK + 1, NJ, NI + 1)), "s23": uniform((NK + 1, NJ + 1, NI)),
        "kmh": uniform((NK + 1, NJ, NI), 1.0, 50.0), "kmv": uniform((NK + 1, NJ, NI), 1.0, 50.0),
        "t11": uniform((NK, NJ, NI)), "t22": uniform((NK, NJ, NI)), "t33": uniform((NK, NJ, NI)),
        "t12": uniform((NK, NJ + 1, NI + 1)), "t13": uniform((NK + 1, NJ, NI + 1)), "t23": uniform((NK + 1, NJ + 1, NI)),
        "tke": uniform((NK + 1, NJ, NI), 0.0, 2.0), "sh": uniform((NK, NJ + 2, NI + 2)),
        "rhoh": uniform((NK, NJ + 2, NI + 2), 0.8, 1.2), "khhh": uniform((NK + 1, NJ + 2, NI + 2), 1.0, 50.0),
        "khv": uniform((NK + 1, NJ, NI), 1.0, 50.0), "rr": uniform((NK, NJ, NI), 0.8, 1.2),
        "sflux": uniform((NJ, NI)),
    }


def _launch(library: Any, name: str, words: int, values: Mapping[str, Any]) -> None:
    from tools.oracle.walls_operator_harness import host_call

    host_call(library.kernel(name), words, values)


def _closure_words(library: Any, prefix: str, base: Mapping[str, Any], walls: int | None) -> dict[str, np.ndarray]:
    """Every closure kernel once, as the flat launchers order them, on copies of ``base``."""

    extra = {} if walls is None else {"walls": walls}
    out: dict[str, np.ndarray] = {}
    calc = {"s11": np.zeros((NK, NJ, NI), F32), "s22": np.zeros((NK, NJ, NI), F32), "s33": np.zeros((NK, NJ, NI), F32),
            "s12": np.zeros((NK, NJ + 1, NI + 1), F32), "s13": np.zeros((NK + 1, NJ, NI + 1), F32),
            "s23": np.zeros((NK + 1, NJ + 1, NI), F32), "defv": np.zeros((NK + 1, NJ, NI), F32),
            "defh": np.zeros((NK + 1, NJ, NI), F32)}
    values = {**{k: base[k] for k in ("u", "v", "w", "rho", "rf", "mh", "mf", "c1", "c2")}, **calc,
              "ni": NI, "nj": NJ, "nk": NK, "rdx": 0.004, "rdy": 0.004, "rdz": 0.002, **extra}
    for kernel, words in (("calcdef_mass_v1", NK * (NJ + 1) * (NI + 1)), ("calcdef_face_v1", (NK - 1) * (NJ + 1) * (NI + 1)),
                          ("calcdef_reduce_v1", (NK - 1) * NJ * NI)):
        _launch(library, prefix + kernel, words, values)
    out.update({f"calcdef.{k}": v for k, v in calc.items()})
    tau = {"t11": np.zeros((NK, NJ, NI), F32), "t22": np.zeros((NK, NJ, NI), F32), "t33": np.zeros((NK, NJ, NI), F32),
           "t12": np.zeros((NK, NJ + 1, NI + 1), F32), "t13": np.zeros((NK + 1, NJ, NI + 1), F32),
           "t23": np.zeros((NK + 1, NJ + 1, NI), F32)}
    values = {**{k: base[k] for k in ("s11", "s22", "s33", "s12", "s13", "s23", "kmh", "kmv")}, **tau,
              "ni": NI, "nj": NJ, "nk": NK, **extra}
    for kernel, words in (("gettau_normal_v1", NI * NJ * NK), ("gettau_t12_v1", NK * (NJ + 1) * (NI + 1)),
                          ("gettau_t13_v1", (NK + 1) * NJ * (NI + 1)), ("gettau_t23_v1", (NK + 1) * (NJ + 1) * NI)):
        _launch(library, prefix + kernel, words, values)
    out.update({f"gettau.{k}": v for k, v in tau.items()})
    column = {"rho": base["rho"], "rf": base["rf"], "mh": base["mhfull"], "mf": base["mf"], "mhc": base["mh"],
              "w": base["w"], "tfw": base["mh"], "t2pt": np.zeros(1, F32), "ntwk": 0, "dt": 1.0,
              "ni": NI, "nj": NJ, "nk": NK, "rdx": 0.004, "rdy": 0.004, "rdz": 0.002, **extra}
    for direction, stresses, words in (("u", ("t11", "t12", "t13"), NJ * (NI + 1)),
                                       ("v", ("t12", "t22", "t23"), (NJ + 1) * NI),
                                       ("w", ("t13", "t23", "t33"), NJ * NI)):
        shape = {"u": (NK, NJ, NI + 1), "v": (NK, NJ + 1, NI), "w": (NK + 1, NJ, NI)}[direction]
        tendency = np.zeros(shape, F32)
        values = {**column, **{k: base[k] for k in stresses}, f"{direction}ten": tendency,
                  "kmv": base["kmv"], "kmh": base["kmh"]}
        if direction != "w":
            values[direction] = base[direction]
        _launch(library, prefix + f"turb{direction}_v1", words, values)
        out[f"turb{direction}"] = tendency
    ten = np.zeros((NK + 1, NJ, NI), F32)
    _launch(library, prefix + "turbt_v1", NJ * NI, {**column, "t": base["tke"], "ten": ten, "kmh": base["kmh"],
                                                     "kmv": base["kmv"]})
    out["turbt"] = ten
    target = np.zeros((NK, NJ, NI), F32)
    values = {"sh": base["sh"], "rhoh": base["rhoh"], "khhh": base["khhh"], "fx": np.zeros((NK, NJ, NI + 1), F32),
              "fy": np.zeros((NK, NJ + 1, NI), F32), "rho": base["rho"], "rr": base["rr"], "rf": base["rf"],
              "mh": base["mhfull"], "mf": base["mf"], "khv": base["khv"], "sflux": base["sflux"],
              "cp": np.zeros((NK, NJ, NI), F32), "dp": np.zeros((NK, NJ, NI), F32), "target": target,
              "ni": NI, "nj": NJ, "nk": NK, "dx": 250.0, "dy": 250.0, "dz": 500.0, "dt": 1.0, "iflux": 0, **extra}
    _launch(library, prefix + "turbs_xy_flux_v1", max(NK * NJ * (NI + 1), NK * (NJ + 1) * NI), values)
    _launch(library, prefix + "turbs_implicit_apply_v1", NJ * NI, values)
    out["turbs"] = target
    return out


@pytest.mark.skipif(not HOST, reason="needs g++ for the kernel host emulator")
@pytest.mark.parametrize("seed", [3, 17])
def test_without_a_wall_edge_the_wall_text_writes_the_flat_words(seed: int) -> None:
    from tools.oracle.walls_operator_harness import library

    base = _inputs(seed)
    flat = _closure_words(library(flat_stretched_closure_source(max_nk=16)), "gm1_flat_", base, None)
    derived = _closure_words(library(wall.flat_wall_closure_source(max_nk=16)), "gm1_flatwall_", base, 0)
    for name, words in flat.items():
        assert np.array_equal(words.view(np.uint32), derived[name].view(np.uint32)), name


@pytest.mark.skipif(not HOST, reason="needs g++ for the kernel host emulator")
def test_a_wall_moves_only_the_words_beside_it() -> None:
    """Free and no-slip walls on every side change the closure's words in the outer ring only."""

    from tools.oracle.walls_operator_harness import library

    base = _inputs(5)
    flat = _closure_words(library(flat_stretched_closure_source(max_nk=16)), "gm1_flat_", base, None)
    for sides in ((3, 3, 3, 3), (4, 4, 4, 4), (3, 4, 4, 3)):
        derived = _closure_words(library(wall.flat_wall_closure_source(max_nk=16)), "gm1_flatwall_", base,
                                 wall.wall_code(sides))
        moved = 0
        for name, words in flat.items():
            differ = words.view(np.uint32) != derived[name].view(np.uint32)
            inner = differ[..., 2:-2, 2:-2]
            assert not inner.any(), (sides, name)
            moved += int(differ.sum())
        assert moved > 0, sides


@pytest.mark.skipif(not HOST, reason="needs g++ for the kernel host emulator")
def test_without_a_wall_edge_the_wall_tke_advection_writes_the_flat_words() -> None:
    from tools.oracle.walls_operator_harness import library

    uniform = _random(11)
    ng = 3
    base = {
        "q": uniform((NK + 1, NJ, NI), 0.0, 2.0), "base": uniform((NK + 1, NJ, NI), 0.0, 2.0),
        "ten": uniform((NK + 1, NJ, NI)), "c1": uniform((NK + 1, NJ, NI), 0.2, 0.8),
        "c2": uniform((NK + 1, NJ, NI), 0.2, 0.8), "rrf0": uniform((NK + 1, NJ, NI), 0.8, 1.2),
        "rru": uniform((NK + 2, NJ + 2 * ng, NI + 1 + 2 * ng)), "rrv": uniform((NK + 2, NJ + 1 + 2 * ng, NI + 2 * ng)),
        "rrw": uniform((NK + 3, NJ + 2 * ng, NI + 2 * ng)), "divx": uniform((NK, NJ, NI)),
        "mf": uniform((NK + 3,), 0.5, 2.0), "ni": NI, "nj": NJ, "nk": NK, "ng": ng, "dx": 250.0, "dy": 250.0,
        "dttmp": 1.0, "ep": 1.0e-16, "rdz": 0.002,
    }
    for w in (0, 1):
        flat_out = np.zeros((NK + 1, NJ, NI), F32)
        wall_out = np.zeros((NK + 1, NJ, NI), F32)
        _launch(library(GM1_FLAT_STRETCHED_TKE_ADVECTION_SOURCE), GM1_FLAT_STRETCHED_TKE_ADVECTION_KERNEL_NAME,
                (NK + 1) * NJ * NI, {**base, "w": w, "out": flat_out})
        _launch(library(wall.GM1_FLAT_WALL_TKE_ADVECTION_SOURCE), wall.GM1_FLAT_WALL_TKE_ADVECTION_KERNEL_NAME,
                (NK + 1) * NJ * NI, {**base, "w": w, "out": wall_out, "walls": 0})
        assert np.array_equal(flat_out.view(np.uint32), wall_out.view(np.uint32))


# ------------------------------------------------------------------ routing

def _runtime_config(sides: tuple[int, int, int, int], stretch_z: int = 0) -> Any:
    from cm1core.cuda_boundary_families import LateralBoundarySelection

    west, east, south, north = sides
    return SimpleNamespace(grid=SimpleNamespace(stretch_z=stretch_z),
                           lateral_boundary_selection=LateralBoundarySelection(west=west, east=east, south=south,
                                                                               north=north))


@pytest.mark.parametrize(("sides", "stretch_z", "flat"), [
    ((2, 2, 2, 2), 0, False), ((1, 1, 1, 1), 0, False), ((2, 2, 1, 1), 0, False),
    ((3, 3, 2, 2), 0, True), ((2, 2, 4, 4), 0, True), ((3, 3, 1, 1), 0, True),
    ((2, 2, 2, 2), 1, True), ((4, 4, 4, 4), 1, True),
])
def test_a_rigid_wall_selects_the_flat_family(sides: tuple[int, int, int, int], stretch_z: int, flat: bool) -> None:
    from cm1core.cuda_cartesian_dynamics import runs_flat_family

    assert runs_flat_family(_runtime_config(sides, stretch_z)) is flat


class _Stream:
    def __enter__(self) -> "_Stream":
        return self

    def __exit__(self, *exc: Any) -> None:
        return None


def _dynamics_double(sides: tuple[int, int, int, int]) -> tuple[Any, list[tuple[str, Any, int]]]:
    """A CartesianDynamics with its fills recorded instead of launched."""

    from cm1core.cuda_cartesian_dynamics import CartesianDynamics

    launches: list[tuple[str, Any, int]] = []

    def kernel(name: str) -> Any:
        def launch(grid: Any, block: Any, arguments: tuple[Any, ...], stream: Any = None) -> None:
            launches.append((name, arguments[0], int(arguments[-1]) if name in ("we", "sn") else -1))
        return launch

    dynamics = object.__new__(CartesianDynamics)
    dynamics.geometry = SimpleNamespace(nx=6, ny=5, nz=4)
    dynamics.cupy = SimpleNamespace(uint32=np.uint32)
    dynamics.stream = _Stream()
    dynamics.arrays = {name: np.zeros(8, F32) for name in ("ua", "va", "wa", "u3d", "v3d", "w3d", "ppi", "pp3d")}
    dynamics.open_compiled = SimpleNamespace(kernels=[None, None, None, kernel("pwe"), kernel("psn"),
                                                      kernel("we"), kernel("sn"), kernel("rigid")])
    dynamics.config = _runtime_config(sides)
    return dynamics, launches


@pytest.mark.parametrize(("sides", "walled"), [
    ((2, 2, 2, 2), False), ((1, 1, 1, 1), False), ((1, 1, 2, 2), False),
    ((3, 3, 2, 2), True), ((2, 2, 4, 4), True), ((1, 1, 3, 3), True), ((3, 4, 4, 3), True),
])
def test_only_a_walled_domain_fills_its_initial_boundaries(sides: tuple[int, int, int, int], walled: bool) -> None:
    dynamics, _ = _dynamics_double(sides)
    assert dynamics._has_rigid_wall() is walled


def test_the_initial_fill_is_cm1s_getset_fill_on_both_wind_levels_and_both_pressures() -> None:
    """init3d.F:2147-2151: bcu(ua), bcv(va), bcw(wa,1), bcs(ppi); the current level starts from the same words."""

    dynamics, launches = _dynamics_double((3, 3, 4, 4))
    dynamics._fill_initial_boundaries()
    a = dynamics.arrays
    expected = []
    for u, v, w in (("ua", "va", "wa"), ("u3d", "v3d", "w3d")):
        expected += [("we", a[u], 1), ("sn", a[u], 1), ("sn", a[v], 2), ("we", a[v], 2),
                     ("we", a[w], 3), ("sn", a[w], 3), ("rigid", a[w], -1)]
    for name in ("ppi", "pp3d"):
        expected += [("pwe", a[name], -1), ("psn", a[name], -1)]
    assert [(n, id(arr), c) for n, arr, c in launches] == [(n, id(arr), c) for n, arr, c in expected]


def test_the_receipt_names_the_family_a_wall_runs() -> None:
    from cm1core.cartesian_door import CARTESIAN_OPERATOR_FAMILIES, cartesian_operator_family

    assert cartesian_operator_family(0) == dict(CARTESIAN_OPERATOR_FAMILIES["uniform"])
    assert cartesian_operator_family(1) == dict(CARTESIAN_OPERATOR_FAMILIES["stretched"])
    walled = cartesian_operator_family(0, rigid_walls=True)
    assert walled["acoustic"] == CARTESIAN_OPERATOR_FAMILIES["stretched"]["acoustic"]
    assert walled["closure"].startswith("cuda_gm1_flat_wall_closure.gm1_flatwall_")
    assert "mh = mf = 1" in walled["vertical_metric"]


# ------------------------------------------------------------------ the recorded comparisons

def _summary() -> list[dict[str, Any]]:
    return json.loads((EVIDENCE / "summary.json").read_text())


def test_the_record_names_the_derived_texts_it_compared() -> None:
    """The four derived texts a walled domain and the lateral step launch are the ones the rows ran.

    Each is built when its module loads (the wall closure and its TKE advection
    from the flat texts, the CM1-order roflux rescale from the sealed boundary
    text, the lateral small-step loop from the sealed flat loop), so the record
    names each by sha256; an edit to a derivation must be compared again.
    """

    import hashlib

    from cm1core import cuda_boundary_families, cuda_gm1_flat_lateral_acoustic

    record = json.loads((EVIDENCE / "gm1-texts.json").read_text())
    shipped = {
        "wall_closure_template": hashlib.sha256(wall.GM1_FLAT_WALL_CLOSURE_TEMPLATE.encode()).hexdigest(),
        "wall_tke_advection": hashlib.sha256(wall.GM1_FLAT_WALL_TKE_ADVECTION_SOURCE.encode()).hexdigest(),
        "roflux_rescale_cm1_order": hashlib.sha256(
            cuda_boundary_families.CUDA_BOUNDARY_ROFLUX_CM1_ORDER_SOURCE.encode()).hexdigest(),
        "lateral_small_step_loop": hashlib.sha256(
            cuda_gm1_flat_lateral_acoustic.lateral_stage_source().encode()).hexdigest(),
    }
    assert {name: entry["sha256"] for name, entry in record["texts"].items()} == shipped
    # The sweep writes the same record from the tree it runs in.
    from tools.oracle.walls_operator_sweep import gm1_texts

    assert gm1_texts() == record["texts"]


def _sides(case: str) -> str:
    return case.split("-")[1]


#: The cases whose sides are rigid walls and open sides only (no periodic side).
WALLED = {"x3", "x4", "y3", "y4", "all3", "all4", "mixed"}
#: What a walled domain launches: the wall text of the closure and its TKE
#: advection, and the flat family's advection, pdef, fills and acoustic stage.
SHIPPED = {("calcdef", "wall"), ("gettau", "wall"), ("turbu", "wall"), ("turbv", "wall"), ("turbw", "wall"),
           ("turbt", "wall"), ("turbs", "wall"), ("tkekm", "wall"), ("tkeadv", "wall"),
           ("advu", "flat"), ("advv", "flat"), ("advw", "flat"), ("advs", "flat"), ("pdef", "flat"),
           ("bcfamilies", "gm1"), ("sound", "flat")}


def test_every_wall_case_gives_cm1s_words_on_the_path_a_wall_runs() -> None:
    """Every operator a walled domain launches, on every walled case, both grids: CM1's words, none vacuous."""

    rows = _summary()
    seen = set()
    for row in rows:
        if (row["operator"], row["variant"]) not in SHIPPED or _sides(row["case"]) not in WALLED:
            continue
        seen.add((row["operator"], row["variant"]))
        assert row["passed"] and not row["vacuous"] and not row["error"], (row["operator"], row["case"])
        for label, output in row["outputs"].items():
            assert output["words_differ"] == 0, (row["operator"], row["case"], label)
    assert seen == SHIPPED
    grids = {row["case"].split("-")[0] for row in rows if _sides(row["case"]) in WALLED}
    assert grids == {"even", "stretched"}


def test_the_released_closure_differs_beside_a_wall() -> None:
    """The control: what 0.1.7 ran beside a wall is seen to differ, so the comparison sees the walls."""

    rows = _summary()
    for operator in ("calcdef", "gettau", "turbu", "turbv", "turbw", "turbs", "turbt", "tkeadv"):
        released = [row for row in rows if row["operator"] == operator and row["variant"] == "flat"
                    and _sides(row["case"]) in WALLED]
        assert released and any(any(o["words_differ"] for o in row["outputs"].values()) for row in released), operator


def test_without_a_wall_the_wall_text_writes_the_flat_texts_words() -> None:
    """Open and periodic cases: the wall text's comparisons are the flat text's, output by output."""

    rows = _summary()
    by_key = {(r["operator"], r["variant"], r["case"], r["dump_at"]): r for r in rows}
    compared = 0
    for (operator, variant, case, at), row in by_key.items():
        if variant != "wall" or _sides(case) not in ("open", "x1"):
            continue
        flat = by_key[(operator, "flat", case, at)]
        assert row["outputs"] == flat["outputs"], (operator, case)
        compared += 1
    assert compared >= 30


def test_the_flat_acoustic_stage_takes_every_lateral_setting() -> None:
    """irbc = 1, 2 and roflux = 1 on open, walled and periodic sides: the shipped stage gives CM1's words, and
    the stage 0.1.7 ran, which was handed no lateral step, does not."""

    rows = [r for r in _summary() if r["operator"] == "sound" and r["group"].startswith("lateral-")]
    shipped = [r for r in rows if r["variant"] == "flat"]
    released = [r for r in rows if r["variant"] == "flat-released"]
    settings = {json.dumps(r["extra"], sort_keys=True) for r in shipped}
    assert settings == {json.dumps(s, sort_keys=True) for s in
                        ({"irbc": 2}, {"irbc": 1}, {"roflux": 1}, {"irbc": 2, "roflux": 1})}
    assert {r["case"].split("-")[0] for r in shipped} == {"even", "stretched"}
    assert shipped and all(r["passed"] and not r["vacuous"] for r in shipped)
    assert released and all(not r["passed"] for r in released)
