"""The ArWen -> GM1 chain on synthetic ArWen-like frames (host only).

Every test here runs without a device.  What is covered:

* CM1's ``interp_on_restart`` column rule, vectorized over columns with their
  own heights, is the scalar transliteration to the last bit;
* a steady, horizontally and vertically uniform ArWen state is reproduced on
  the GM1 lattice and its boundary tables are exactly steady;
* the boundary target is linear in time across an interval, starts at the
  driving state and is continuous into the next interval (WRF ``wrfbdy``);
* a field linear in height is remapped through stretched, column-varying
  ArWen levels to the float32 rounding of the exact value;
* the variable mapping (T, P+PB, species names, perturbations, staggers)
  round-trips, and the GM1 boundary normal velocity is the ArWen face value;
* every named refusal fires on the input it exists for;
* the Rust extractor's frame format is the one the Python side reads.
"""

from __future__ import annotations

import json
import math
import os
import struct
import subprocess
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from cm1core import arwen_chain as chain  # noqa: E402
from cm1core.cm1_vertical_grid import uniform_vertical_levels  # noqa: E402
from cm1core.errors import CM1Refusal  # noqa: E402
from cm1core.nest_live_kernels import lattice_shape, side_shape  # noqa: E402
from cm1core.nest_vertical import build_cm1_vertical_map, cm1_vertical_interpolate  # noqa: E402

G = chain.WRF_GRAVITY_M_S2
REPO = Path(__file__).resolve().parents[1]
START = "2026-05-01_18:00:00"


def stamp(seconds: float) -> str:
    import datetime as dt

    base = dt.datetime.strptime(START, "%Y-%m-%d_%H:%M:%S")
    return (base + dt.timedelta(seconds=seconds)).strftime("%Y-%m-%d_%H:%M:%S")


# ---------------------------------------------------------------------------
# A synthetic ArWen history frame.
# ---------------------------------------------------------------------------
NX, NY, NZ, DX = 16, 16, 30, 3000.0
ZTOP = 20000.0
LAT = 35.0
FCOR = 2.0 * chain.WRF_EOMEG_S_INV * math.sin(math.radians(LAT))


def default_fields():
    return {
        "u": lambda x, y, z, t: 10.0 + 0.0 * z,
        "v": lambda x, y, z, t: 5.0 + 0.0 * z,
        "w": lambda x, y, z, t: 0.0 * z,
        "theta": lambda x, y, z, t: 300.0 + 0.0 * z,
        "qv": lambda x, y, z, t: 0.01 + 0.0 * z,
        "p": lambda x, y, z, t: 1.0e5 * np.exp(-z / 8000.0),
    }


def make_frame(root: Path, t_s: float, *, fields=None, ph_bump=0.0, hgt=0.0, mapfac=1.0,
               attributes=None, drop=(), extra=None, nz=NZ, ztop=ZTOP):
    f = default_fields()
    f.update(fields or {})
    k = np.arange(nz + 1, dtype=np.float64)
    zw_nom = ztop * (k / nz) ** 1.5
    x_mass = (np.arange(NX) + 0.5) * DX
    y_mass = (np.arange(NY) + 0.5) * DX
    bump = ph_bump * np.sin(2 * np.pi * np.arange(NX) / NX)[None, None, :] * np.sin(np.pi * k / nz)[:, None, None]
    zw = zw_nom[:, None, None] + bump + np.zeros((1, NY, 1))
    hgt_field = np.broadcast_to(np.asarray(hgt, dtype=np.float64), (NY, NX)).astype(np.float32)
    phb = G * (zw_nom[:, None, None] + hgt_field.astype(np.float64)[None]) + np.zeros((1, NY, NX))
    ph = G * (zw - zw_nom[:, None, None])
    # Mirror the chain's height arithmetic so a field linear in z is exact.
    zw32 = ((ph + phb) / G - hgt_field.astype(np.float64)[None]).astype(np.float32)
    zm32 = (np.float32(0.5) * (zw32[:-1] + zw32[1:])).astype(np.float32)
    zu = zm32.copy()
    zu = np.concatenate([zm32[:, :, :1], (np.float32(0.5) * (zm32[:, :, :-1] + zm32[:, :, 1:])).astype(np.float32), zm32[:, :, -1:]], axis=2)
    zv = np.concatenate([zm32[:, :1, :], (np.float32(0.5) * (zm32[:, :-1, :] + zm32[:, 1:, :])).astype(np.float32), zm32[:, -1:, :]], axis=1)
    X, Y = np.meshgrid(x_mass, y_mass)
    Xu, Yu = np.meshgrid(np.arange(NX + 1) * DX, y_mass)
    Xv, Yv = np.meshgrid(x_mass, np.arange(NY + 1) * DX)
    zmd, zud, zvd, zwd = (a.astype(np.float64) for a in (zm32, zu, zv, zw32))
    p_full = f["p"](X[None], Y[None], zmd, t_s)
    pb = 1.0e5 * np.exp(-zmd / 8000.0)
    arrays = {
        "U": f["u"](Xu[None], Yu[None], zud, t_s),
        "V": f["v"](Xv[None], Yv[None], zvd, t_s),
        "W": f["w"](X[None], Y[None], zwd, t_s),
        "T": f["theta"](X[None], Y[None], zmd, t_s) - 300.0,
        "PH": ph, "PHB": phb, "P": p_full - pb, "PB": pb,
        "HGT": hgt_field,
        "MAPFAC_M": np.full((NY, NX), mapfac),
        "XLAT": np.full((NY, NX), LAT),
        "PSFC": np.full((NY, NX), 1.0e5),
        "TH2": np.full((NY, NX), 300.0),
        "Q2": np.full((NY, NX), 0.01),
        "QVAPOR": f["qv"](X[None], Y[None], zmd, t_s),
    }
    for name in ("QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP"):
        arrays[name] = np.zeros((nz, NY, NX))
    arrays.update(extra or {})
    for name in drop:
        arrays.pop(name)
    attrs = {"DX": DX, "DY": DX, "MAP_PROJ": 1, "TRUELAT1": 30.0, "TRUELAT2": 60.0,
             "STAND_LON": -97.0, "MP_PHYSICS": 6, "HYBRID_OPT": 2}
    attrs.update(attributes or {})
    dims = {}
    for name, a in arrays.items():
        a = np.asarray(a)
        if a.ndim == 2:
            dims[name] = ["south_north", "west_east"]
        else:
            z = "bottom_top_stag" if a.shape[0] == nz + 1 else "bottom_top"
            y = "south_north_stag" if a.shape[1] == NY + 1 else "south_north"
            x = "west_east_stag" if a.shape[2] == NX + 1 else "west_east"
            dims[name] = [z, y, x]
    return chain.write_frame(root / f"frame-{int(t_s)}", times=stamp(t_s), attributes=attrs,
                             fields=arrays, dims=dims)


def levels(nk=20, dz=400.0):
    return uniform_vertical_levels(nk, dz)


def request(**overrides):
    base = dict(
        nx=15, ny=15, dx_m=1000.0, dy_m=1000.0, levels=levels(), dt_s=5.0, duration_s=600.0,
        component="wsm6-mp6", coriolis=1, fcor_s_inv=FCOR, start_time=stamp(0), end_time=stamp(600),
        placement=chain.ChainPlacement(parent_i0=5, parent_j0=5, refinement_ratio=3, spec_zone=1, relax_zone=4),
        terrain_relief_limit_m=50.0, map_factor_departure_limit=0.01,
    )
    base.update(overrides)
    return chain.ChainRequest(**base)


def frames(tmp_path, times=(0, 300, 600), **kwargs):
    return [make_frame(tmp_path, t, **kwargs) for t in times]


def bits(a):
    return np.asarray(a, dtype=np.float32).view(np.uint32)


def refused(code):
    return pytest.raises(CM1Refusal, match=code)


# ---------------------------------------------------------------------------
# CM1's column rule.
# ---------------------------------------------------------------------------
def _ghost(col):
    f = np.float32
    return np.concatenate([[-col[0]], col, [col[-1] + f(2.0) * (col[-1] - col[-2])]]).astype(f)


@pytest.mark.parametrize("seed", range(6))
def test_column_map_is_the_cm1_transliteration_word_for_word(seed):
    rng = np.random.default_rng(seed)
    nzold = int(rng.integers(3, 40))
    columns = []
    for _ in range(7):
        steps = rng.uniform(5.0, 900.0, size=nzold).astype(np.float64)
        columns.append(_ghost(np.cumsum(steps).astype(np.float32)))
    old = np.stack(columns, axis=1)[:, :, None]  # (nzold+2, 7, 1)
    top = float(old[nzold, :, 0].min())
    nk = int(rng.integers(2, 60))
    new = np.sort(rng.uniform(0.5, top, size=nk)).astype(np.float32)
    new = np.unique(new)
    values = rng.normal(size=(nzold, 7, 1)).astype(np.float32)
    cmap = chain.build_column_map(old, new, nzold=nzold)
    got = chain.interpolate_columns(values, cmap)
    for c in range(7):
        ref = build_cm1_vertical_map(old[:, c, 0].copy(), new, nzold=nzold)
        assert np.array_equal(cmap.lower[:, c, 0], ref.lower)
        assert np.array_equal(cmap.upper[:, c, 0], ref.upper)
        assert np.array_equal(bits(cmap.fraction[:, c, 0]), bits(ref.fraction))
        want = cm1_vertical_interpolate(values[:, c:c + 1, :].copy(), ref)
        assert np.array_equal(bits(got[:, c:c + 1, :]), bits(want))


def test_column_map_refuses_levels_above_the_last_ghost_and_unordered_columns():
    old = _ghost(np.array([100, 300, 600], dtype=np.float32))[:, None, None]
    with pytest.raises(ValueError, match="above"):
        chain.build_column_map(old, np.array([10000.0], dtype=np.float32), nzold=3)
    bad = old.copy()
    bad[2, 0, 0] = bad[1, 0, 0]
    with pytest.raises(ValueError, match="strictly increasing"):
        chain.build_column_map(bad, np.array([50.0], dtype=np.float32), nzold=3)


# ---------------------------------------------------------------------------
# Steady uniform state.
# ---------------------------------------------------------------------------
def test_a_steady_uniform_state_is_reproduced_and_its_tables_are_steady(tmp_path):
    fr = frames(tmp_path, times=(0, 300), ph_bump=40.0)
    req = request(duration_s=300.0, end_time=stamp(300))
    geometry = req.placement.geometry(fr[0], nx=15, ny=15)
    state = chain.remap_frame(fr[0], geometry, req.levels, req.component)
    lv = req.levels
    expected = {"u": 10.0, "v": 5.0, "w": 0.0, "theta": 300.0, "qv": 0.01, "qc": 0.0}
    for name, value in expected.items():
        volume = state[name]
        stagger = dict(chain.carriers_of(req.component))[name]
        assert volume.shape == lattice_shape(stagger, lv.nk, 15, 15)
        # The column rule's two-plane sum at identical planes reproduces the
        # word to within one float32 ulp; the horizontal capture of a flat
        # level is exact.  Measured, not assumed:
        err = np.max(np.abs(volume.astype(np.float64) - np.float32(value)))
        assert err <= np.spacing(np.float32(max(abs(value), 1e-30))), (name, err)
    # Two identical frames: every tendency word is exactly zero, and the value
    # rows are the initial state's own boundary rows, word for word.
    th0 = np.full(lv.nk, 300.0, np.float32)
    pi0 = state["pi"][:, 0, 0].copy()
    prepared_dir = tmp_path / "prepared"
    chain.prepare_chain(fr, req, prepared_dir)
    prepared = chain.PreparedChain.open(prepared_dir, contract=req.contract())
    tables = chain.host_interval_tables(prepared, 0, th0=th0, pi0=pi0,
                                        species=("qv", "qc", "qr", "qi", "qs", "qg"))
    for carrier, sides in tables.items():
        for side, (value, tend) in sides.items():
            assert not np.any(bits(tend)), (carrier, side)
    initial_u = prepared.initial("u")
    assert np.array_equal(bits(tables["u"]["west"][0]), bits(chain.strip(initial_u, "west", prepared.width)))


def test_steady_tables_leave_the_davies_operator_with_nothing_to_do(tmp_path):
    from cm1core.nest_boundary import BoundaryTable, apply_specified_relaxation
    from cm1core.nest_geometry import NestGeometry

    fr = frames(tmp_path, times=(0, 300))
    req = request(duration_s=300.0, end_time=stamp(300))
    geometry = req.placement.geometry(fr[0], nx=15, ny=15)
    state = chain.remap_frame(fr[0], geometry, req.levels, req.component)
    theta = state["theta"][3]
    width = 5
    tables = {}
    for side in chain.SIDES:
        rows = chain.strip(state["theta"], side, width)[3]
        value, tend = chain.interval_tables(rows, rows, 300.0)
        tables[side] = BoundaryTable(side, tuple(map(tuple, value.tolist())), tuple(map(tuple, tend.tolist())))
    out = apply_specified_relaxation(
        field=theta.tolist(), tendency=np.zeros_like(theta).tolist(), tables=tables,
        geometry=geometry, dtbc=150.0, child_dt_s=5.0,
    )
    assert np.max(np.abs(np.asarray(out))) == 0.0


# ---------------------------------------------------------------------------
# Time interpolation.
# ---------------------------------------------------------------------------
def test_the_boundary_target_is_linear_in_time_and_continuous_across_intervals(tmp_path):
    fields = {
        "u": lambda x, y, z, t: 8.0 + 2.0e-4 * x + 0.004 * t + 0.0 * z,
        "theta": lambda x, y, z, t: 300.0 + 0.003 * z - 1.0e-4 * y + 0.001 * t,
    }
    fr = frames(tmp_path, times=(0, 300, 600), fields=fields)
    req = request()
    out = tmp_path / "prepared"
    chain.prepare_chain(fr, req, out)
    prepared = chain.PreparedChain.open(out)
    lv = req.levels
    th0 = np.asarray(lv.zh[1:lv.nk + 1], dtype=np.float32) * 0 + np.float32(300.0)
    pi0 = prepared.initial("pi")[:, 0, 0]
    species = ("qv", "qc", "qr", "qi", "qs", "qg")
    t0 = chain.host_interval_tables(prepared, 0, th0=th0, pi0=pi0, species=species)
    t1 = chain.host_interval_tables(prepared, 1, th0=th0, pi0=pi0, species=species)
    for carrier, source in (("u", "u"), ("th", "theta")):
        for side in chain.SIDES:
            start = prepared.strip(0, source, side)
            mid = prepared.strip(1, source, side)
            if carrier == "th":
                start, mid = chain.perturbation(start, th0), chain.perturbation(mid, th0)
            value, tend = t0[carrier][side]
            # WRF stuff_bdy: the value is the driving state at the interval start.
            assert np.array_equal(bits(value), bits(start))
            for n in range(0, 61, 12):
                dtbc = 5.0 * n
                target = chain.relaxation_target(value, tend, dtbc)
                exact = start.astype(np.float64) + (mid.astype(np.float64) - start) * dtbc / 300.0
                scale = np.maximum(np.abs(exact), 1.0)
                assert np.max(np.abs(target - exact) / scale) < 2e-6, (carrier, side, n)
            # Continuity: the next interval starts exactly at frame 1.
            assert np.array_equal(bits(t1[carrier][side][0]), bits(mid))


def test_frame_strips_equal_the_initial_state_boundary_rows_word_for_word(tmp_path):
    fields = {"u": lambda x, y, z, t: 8.0 + 3.0e-4 * x - 1e-4 * y + 0.001 * z}
    fr = frames(tmp_path, times=(0, 300), fields=fields, ph_bump=25.0)
    req = request(duration_s=300.0, end_time=stamp(300))
    chain.prepare_chain(fr, req, tmp_path / "p")
    prepared = chain.PreparedChain.open(tmp_path / "p")
    for name, stagger in chain.carriers_of(req.component):
        full = prepared.initial(name)
        for side in chain.SIDES:
            rows = prepared.strip(0, name, side)
            assert rows.shape == side_shape(side, full.shape, prepared.width)
            assert np.array_equal(bits(rows), bits(chain.strip(full, side, prepared.width)))


# ---------------------------------------------------------------------------
# Vertical remap accuracy and variable mapping.
# ---------------------------------------------------------------------------
def test_a_field_linear_in_height_crosses_moving_eta_columns_to_float32_rounding(tmp_path):
    fields = {"theta": lambda x, y, z, t: 290.0 + 0.0045 * z, "w": lambda x, y, z, t: 0.001 * z}
    fr = make_frame(tmp_path, 0, fields=fields, ph_bump=60.0)
    req = request()
    geometry = req.placement.geometry(fr, nx=15, ny=15)
    state = chain.remap_frame(fr, geometry, req.levels, req.component)
    lv = req.levels
    zh = np.asarray(lv.zh[1:lv.nk + 1], dtype=np.float64)
    zf = np.asarray(lv.zf[1:lv.nk + 2], dtype=np.float64)
    theta_err = np.abs(state["theta"] - (290.0 + 0.0045 * zh)[:, None, None])
    # Below ArWen's first scalar level CM1 extends the column by a constant;
    # every GM1 level above it is linear interpolation of a linear field.
    first = float(fr.array("PH")[1].max() / G + ZTOP * (1 / NZ) ** 1.5) / 2  # lowest mass level, upper bound
    supported = zh > first + 200.0
    assert np.max(theta_err[supported]) < 2.0e-3  # float32 near 300 K: ulp 3e-5, heights ulp ~5e-4 m
    w_err = np.abs(state["w"] - (0.001 * zf)[:, None, None])
    assert np.max(w_err[1:]) < 1.0e-5
    assert state["_support"]["mass"]["below"] >= 0


def test_variable_mapping_round_trips(tmp_path):
    fields = {"p": lambda x, y, z, t: 9.9e4 * np.exp(-z / 7900.0) * (1.0 + 1e-3 * np.sin(x / 5000.0))}
    fr = make_frame(tmp_path, 0, fields=fields)
    req = request()
    geometry = req.placement.geometry(fr, nx=15, ny=15)
    from cm1core.nest_vertical import VerticalWindow

    window = VerticalWindow.for_stagger(geometry, "mass")
    pi = chain._parent_carrier(fr, "pi", window).astype(np.float64)
    p = fr.array("P")[:, window.j0:window.j0 + window.nj, window.i0:window.i0 + window.ni].astype(np.float64) + \
        fr.array("PB")[:, window.j0:window.j0 + window.nj, window.i0:window.i0 + window.ni]
    back = chain.CM1_P00_PA * pi ** (chain.CM1_CP / chain.CM1_RD)
    assert np.max(np.abs(back - p) / p) < 5e-7
    theta = chain._parent_carrier(fr, "theta", window)
    t = fr.array("T")[:, window.j0:window.j0 + window.nj, window.i0:window.i0 + window.ni]
    assert np.array_equal(bits((theta.astype(np.float64) - 300.0).astype(np.float32)), bits(t))
    th0 = np.linspace(299.0, 330.0, theta.shape[0]).astype(np.float32)
    tha = chain.perturbation(theta, th0)
    assert np.max(np.abs((tha + th0[:, None, None]) - theta)) <= 4 * np.spacing(np.float32(330.0))
    # Every GM1 scalar of every selectable scheme has a WRF name, and each
    # scheme's WRF option is the number in its component id.
    from cm1core.moisture_species import moisture_species_profile

    for component, option in chain.WRF_MP_PHYSICS.items():
        assert component.endswith(f"mp{option}")
        try:
            profile = moisture_species_profile(component)
        except CM1Refusal:
            continue  # transcribed but not selectable in GM1 (a named blocker)
        for name in profile.names:
            assert name in chain.WRF_SPECIES


def test_the_gm1_boundary_normal_velocity_is_the_arwen_face_value(tmp_path):
    fields = {"u": lambda x, y, z, t: 5.0 + 1.0e-3 * x + 2.0e-4 * y + 0.0 * z,
              "v": lambda x, y, z, t: -3.0 + 7.0e-4 * x - 1.0e-3 * y + 0.0 * z}
    fr = make_frame(tmp_path, 0, fields=fields)
    req = request()
    geometry = req.placement.geometry(fr, nx=15, ny=15)
    state = chain.remap_frame(fr, geometry, req.levels, req.component)
    U = fr.array("U")
    r = 3
    # Coincident faces: GM1 u face i*r sits on ArWen u face parent_i0+i.  The
    # vertical step is the same for both; compare after it on one level where
    # both planes are the same ArWen level pair.
    k = 10
    cols = [0, 15]
    for c in cols:
        ip = geometry.parent_i0 + c // r
        gm1 = state["u"][k, 1::3, c]  # centre child rows of each ArWen cell
        arwen = state["u"][k, 1::3, c]
        assert np.all(np.isfinite(gm1))
        # the face value is a copy of the vertically interpolated ArWen column
        # (normal offset exactly zero, tangential offset zero at row 1 of 3)
        assert np.max(np.abs(gm1 - arwen)) == 0.0
    # And horizontally linear u is reproduced at every GM1 u point.
    x_face = geometry.origin_x_m + np.arange(16) * 1000.0
    y_mass = geometry.origin_y_m + (np.arange(15) + 0.5) * 1000.0
    exact = 5.0 + 1.0e-3 * x_face[None, :] + 2.0e-4 * y_mass[:, None]
    assert np.max(np.abs(state["u"][k] - exact)) < 1e-4


# ---------------------------------------------------------------------------
# The prepared products.
# ---------------------------------------------------------------------------
def test_prepare_writes_a_cm1_sounding_and_hash_checked_products(tmp_path):
    fr = frames(tmp_path, times=(0, 300))
    req = request(duration_s=300.0, end_time=stamp(300))
    receipt = chain.prepare_chain(fr, req, tmp_path / "p", document={"label": "synthetic"})
    text = (tmp_path / "p" / "input_sounding").read_text().splitlines()
    assert len(text[0].split()) == 3 and float(text[0].split()[0]) == pytest.approx(1000.0)
    rows = [list(map(float, line.split())) for line in text[1:]]
    assert all(len(r) == 5 for r in rows)
    assert all(b[0] > a[0] for a, b in zip(rows, rows[1:]))
    assert rows[0][1] == pytest.approx(300.0) and rows[0][3] == pytest.approx(10.0)
    assert receipt["admission"]["steps_per_interval"] == 60
    assert receipt["authority"]["boundary_tendency"].startswith("WRF main/ndown_em.F")
    # Tampering is refused.
    arena = tmp_path / "p" / chain.BOUNDARY_ARENA
    data = bytearray(arena.read_bytes())
    data[7] ^= 1
    arena.write_bytes(bytes(data))
    with refused("GM1-CHAIN-PREPARED"):
        chain.PreparedChain.open(tmp_path / "p")


def test_a_prepared_chain_for_another_document_is_refused(tmp_path):
    fr = frames(tmp_path, times=(0, 300))
    req = request(duration_s=300.0, end_time=stamp(300))
    chain.prepare_chain(fr, req, tmp_path / "p")
    other = dict(req.contract(), dt_s=2.5)
    with refused("GM1-CHAIN-PREPARED"):
        chain.PreparedChain.open(tmp_path / "p", contract=other)
    prepared = chain.PreparedChain.open(tmp_path / "p", contract=req.contract())
    with refused("GM1-CHAIN-DURATION"):
        chain.host_interval_tables(prepared, 1, th0=np.zeros(20, np.float32),
                                   pi0=np.zeros(20, np.float32), species=("qv",))


# ---------------------------------------------------------------------------
# Refusals.
# ---------------------------------------------------------------------------
def test_refusal_frame_count_and_cadence(tmp_path):
    with refused("GM1-CHAIN-FRAME-COUNT"):
        chain.admit_frames([make_frame(tmp_path, 0)], request())
    uneven = [make_frame(tmp_path / "a", t) for t in (0, 300, 700)]
    with refused("GM1-CHAIN-CADENCE"):
        chain.admit_frames(uneven, request(end_time=stamp(700)))


def test_refusal_window_interval_step_and_duration(tmp_path):
    fr = frames(tmp_path)
    with refused("GM1-CHAIN-WINDOW"):
        chain.admit_frames(fr, request(start_time=stamp(60)))
    with refused("GM1-CHAIN-INTERVAL-STEP"):
        chain.admit_frames(fr, request(dt_s=7.0))
    with refused("GM1-CHAIN-DURATION"):
        chain.admit_frames(fr, request(duration_s=900.0))


def test_refusal_crossing_time(tmp_path):
    fr = frames(tmp_path, fields={"u": lambda x, y, z, t: 80.0 + 0.0 * z})
    with refused("GM1-CHAIN-CROSSING"):
        chain.admit_frames(fr, request())


def test_refusal_grid_disagreement_between_frames(tmp_path):
    a = make_frame(tmp_path / "a", 0)
    b = make_frame(tmp_path / "b", 300, attributes={"MP_PHYSICS": 8})
    with refused("GM1-CHAIN-FRAME-GRID"):
        chain.admit_frames([a, b], request(duration_s=300.0, end_time=stamp(300)))
    c = make_frame(tmp_path / "c", 300, hgt=1.0)
    with refused("GM1-CHAIN-FRAME-GRID"):
        chain.admit_frames([a, c], request(duration_s=300.0, end_time=stamp(300)))


def test_ground_heights_may_differ_between_saved_times_only_by_float_rounding(tmp_path):
    """Stock WRF 3.7 rewrote HGT with 2.5 mm differences between records (the Katrina file); a centimetre is
    the tolerance (``arwen_chain.HGT_TOLERANCE_M``), and anything larger is still refused with its size."""

    a = make_frame(tmp_path / "a", 0)
    b = make_frame(tmp_path / "b", 300, hgt=0.5 * chain.HGT_TOLERANCE_M)
    report = chain.admission_report([a, b], request(duration_s=300.0, end_time=stamp(300)))
    row = next(r for r in report["checks"] if r["id"] == "frame_grid")
    assert row["status"] == "pass"
    assert report["facts"]["grid"]["ground_change_m"] == pytest.approx(0.5 * chain.HGT_TOLERANCE_M, rel=1e-3)
    c = make_frame(tmp_path / "c", 300, hgt=2.0 * chain.HGT_TOLERANCE_M)
    with refused("GM1-CHAIN-FRAME-GRID"):
        chain.admit_frames([a, c], request(duration_s=300.0, end_time=stamp(300)))


def test_refusal_projection_spacing_and_placement(tmp_path):
    fr = frames(tmp_path / "ll", attributes={"MAP_PROJ": 6})
    with refused("GM1-CHAIN-PROJECTION"):
        chain.admit_frames(fr, request())
    fr = frames(tmp_path / "ok")
    with refused("GM1-CHAIN-SPACING"):
        chain.admit_frames(fr, request(dx_m=999.0))
    edge = chain.ChainPlacement(parent_i0=1, parent_j0=5, refinement_ratio=3, spec_zone=1, relax_zone=4)
    with refused("GM1-CHAIN-PLACEMENT"):
        chain.admit_frames(fr, request(placement=edge))
    even = chain.ChainPlacement(parent_i0=5, parent_j0=5, refinement_ratio=4, spec_zone=1, relax_zone=4)
    with refused("GM1-CHAIN-PLACEMENT"):
        chain.admit_frames(fr, request(placement=even, nx=16, ny=16, dx_m=750.0, dy_m=750.0))


def test_refusal_terrain_and_map_factor(tmp_path):
    rng = np.random.default_rng(3)
    rough = rng.uniform(0.0, 300.0, size=(NY, NX))
    fr = [make_frame(tmp_path / "t", t, hgt=rough) for t in (0, 300, 600)]
    with refused("GM1-CHAIN-TERRAIN"):
        chain.admit_frames(fr, request())
    # A map factor away from 1 but the same over the window: GM1 takes the history's true spacing,
    # DX/(ratio*m), and has nothing left to get wrong (GM1_CHOICES["map_scale"]).
    fr = frames(tmp_path / "m", mapfac=1.03)
    with refused("GM1-CHAIN-SPACING"):
        chain.admit_frames(fr, request())
    true = float(np.float32(DX / (3 * float(np.float32(1.03)))))
    record = chain.admit_frames(fr, request(dx_m=true, dy_m=true))
    assert record["map_factor_departure"] == 0.0
    assert abs(record["map_factor_unit_departure"] - 0.03) < 1e-6
    assert record["map_scale"]["gm1_dx_m"] == true
    # One that varies over the window by more than the limit is refused: GM1 has no map-factor terms.
    ramp = np.broadcast_to(np.linspace(1.0, 1.6, NX)[None, :], (NY, NX)).astype(np.float32)
    fr = frames(tmp_path / "r", extra={"MAPFAC_M": ramp})
    window = ramp[5:10, 5:10].astype(np.float64)
    true = float(np.float32(DX / (3 * window.mean())))
    with refused("GM1-CHAIN-MAP-FACTOR"):
        chain.admit_frames(fr, request(dx_m=true, dy_m=true))
    assert chain.chain_spacing_m(fr[0], 3, (slice(5, 10), slice(5, 10))) == true


def test_refusal_coriolis(tmp_path):
    fr = frames(tmp_path)
    with refused("GM1-CHAIN-CORIOLIS"):
        chain.admit_frames(fr, request(fcor_s_inv=5.0e-5))
    with refused("GM1-CHAIN-CORIOLIS"):
        chain.admit_frames(fr, request(coriolis=0))


def test_refusal_microphysics_species_and_variables(tmp_path):
    fr = frames(tmp_path / "mp")
    with refused("GM1-CHAIN-MICROPHYSICS"):
        chain.admit_frames(fr, request(component="morrison-mp10"))
    fr = frames(tmp_path / "sp", drop=("QGRAUP",))
    with refused("GM1-CHAIN-SPECIES"):
        chain.admit_frames(fr, request())
    fr = frames(tmp_path / "p", drop=("PB",))
    with refused("GM1-CHAIN-VARIABLE"):
        chain.admit_frames(fr, request())


def test_refusal_theta_m(tmp_path):
    fr = frames(tmp_path / "a", attributes={"USE_THETA_M": 1})
    with refused("GM1-CHAIN-THETA-M"):
        chain.admit_frames(fr, request())
    # T that really is dry theta, with THM beside it, is admitted.
    def thm(frame_root, t):
        base = make_frame(frame_root / "probe", t)
        th = base.array("T").astype(np.float64) + 300.0
        qv = base.array("QVAPOR").astype(np.float64)
        return th * (1.0 + chain.WRF_RV_OVER_RD * qv) - 300.0
    ok = [make_frame(tmp_path / "b", t, attributes={"USE_THETA_M": 1},
                     extra={"THM": thm(tmp_path / f"c{t}", t)}) for t in (0, 300, 600)]
    record = chain.admit_frames(ok, request())
    assert record["theta"]["dry_T_residual_K"] < 1e-2
    moist = [make_frame(tmp_path / "d", t, attributes={"USE_THETA_M": 1},
                        fields={"theta": lambda x, y, z, t: 305.0 + 0.0 * z},
                        extra={"THM": thm(tmp_path / f"e{t}", t)}) for t in (0, 300, 600)]
    with refused("GM1-CHAIN-THETA-M"):
        chain.admit_frames(moist, request())


def test_refusal_model_top_above_arwen(tmp_path):
    fr = frames(tmp_path, ztop=6000.0)
    req = request()
    geometry = req.placement.geometry(fr[0], nx=15, ny=15)
    with refused("GM1-CHAIN-MODEL-TOP"):
        chain.remap_frame(fr[0], geometry, req.levels, req.component)


def test_a_malformed_frame_is_refused(tmp_path):
    fr = make_frame(tmp_path, 0)
    arena = fr.directory / "fields.f32le"
    data = bytearray(arena.read_bytes())
    data[0] ^= 0xFF
    arena.write_bytes(bytes(data))
    with refused("GM1-CHAIN-FRAME"):
        chain.ArwenFrame.open(fr.directory).array("U")
    with refused("GM1-CHAIN-FRAME"):
        chain.ArwenFrame.open(tmp_path / "missing")


# ---------------------------------------------------------------------------
# The Rust extractor, when built.
# ---------------------------------------------------------------------------
EXTRACTOR = REPO / "rust/cm1-state/target/release/gm1-arwen-history-extract"


def _write_cdf2(path: Path, times: list[str], record_fields: dict, static_fields: dict, attrs: dict):
    """A minimal netCDF classic 64-bit-offset writer (the ArWen default container)."""

    def name(s):
        b = s.encode()
        return struct.pack(">I", len(b)) + b + b"\0" * (-len(b) % 4)

    dims = {"Time": 0, "DateStrLen": 19}
    for arr, dn in record_fields.values():
        for d, n in zip(dn, arr.shape[1:]):
            dims[d] = n
    for arr, dn in static_fields.values():
        for d, n in zip(dn, arr.shape):
            dims[d] = n
    dim_ids = {d: i for i, d in enumerate(dims)}

    def atts(items):
        if not items:
            return b"\0" * 8
        out = struct.pack(">II", 0x0C, len(items))
        for k, v in items.items():
            out += name(k)
            if isinstance(v, str):
                b = v.encode()
                out += struct.pack(">II", 2, len(b)) + b + b"\0" * (-len(b) % 4)
            elif isinstance(v, int):
                out += struct.pack(">IIi", 4, 1, v)
            else:
                out += struct.pack(">IIf", 5, 1, v)
        return out

    variables = [("Times", ["Time", "DateStrLen"], 2, 19)]
    variables += [(k, ["Time", *dn], 5, int(np.prod(a.shape[1:]))) for k, (a, dn) in record_fields.items()]
    variables += [(k, list(dn), 5, int(np.prod(a.shape))) for k, (a, dn) in static_fields.items()]

    def header(begins):
        out = b"CDF\x02" + struct.pack(">I", len(times))
        out += struct.pack(">II", 0x0A, len(dims)) + b"".join(name(d) + struct.pack(">I", n) for d, n in dims.items())
        out += atts(attrs)
        out += struct.pack(">II", 0x0B, len(variables))
        for (vn, dn, t, count), begin in zip(variables, begins):
            size = count * (1 if t == 2 else 4)
            out += name(vn) + struct.pack(">I", len(dn)) + b"".join(struct.pack(">I", dim_ids[d]) for d in dn)
            out += atts({}) + struct.pack(">II", t, size + (-size % 4)) + struct.pack(">Q", begin)
        return out

    length = len(header([0] * len(variables)))
    begins, cursor = [], length
    static_names = list(static_fields)
    for vn, dn, t, count in variables:
        if vn in static_names:
            begins.append(cursor)
            cursor += count * 4
    record_start = cursor
    rec_begins = []
    for vn, dn, t, count in variables:
        if vn not in static_names:
            rec_begins.append(cursor)
            size = count * (1 if t == 2 else 4)
            cursor += size + (-size % 4)
    record_size = cursor - record_start
    ordered, si, ri = [], 0, 0
    for vn, *_ in variables:
        if vn in static_names:
            ordered.append(begins[si]); si += 1
        else:
            ordered.append(rec_begins[ri]); ri += 1
    body = header(ordered)
    for vn in static_names:
        body += static_fields[vn][0].astype(">f4").tobytes()
    for r, stamp_text in enumerate(times):
        b = stamp_text.encode()
        body += b + b"\0" * (-len(b) % 4)
        for vn, (arr, dn) in record_fields.items():
            raw = arr[r].astype(">f4").tobytes()
            body += raw + b"\0" * (-len(raw) % 4)
    path.write_bytes(body)
    assert record_size > 0


@pytest.mark.skipif(not EXTRACTOR.exists(), reason="build rust/cm1-state gm1-arwen-history-extract first")
def test_rust_extractor_writes_the_frame_the_chain_reads(tmp_path):
    rng = np.random.default_rng(11)
    t = rng.normal(size=(2, 3, 4, 5)).astype(np.float32)
    u = rng.normal(size=(2, 3, 4, 6)).astype(np.float32)
    hgt = rng.normal(size=(4, 5)).astype(np.float32)
    znu = np.linspace(1, 0, 3).astype(np.float32)
    path = tmp_path / "wrfout_d01"
    _write_cdf2(
        path, [stamp(0), stamp(300)],
        {"T": (t, ["bottom_top", "south_north", "west_east"]),
         "U": (u, ["bottom_top", "south_north", "west_east_stag"]),
         "HGT": (np.stack([hgt, hgt]), ["south_north", "west_east"])},
        {"ZNU": (znu, ["bottom_top"])},
        {"DX": 3000.0, "MAP_PROJ": 1, "MP_PHYSICS": 6, "TITLE": " OUTPUT FROM ArWen"},
    )
    out = tmp_path / "frame"
    result = subprocess.run([os.fspath(EXTRACTOR), os.fspath(path), "1", os.fspath(out)],
                            capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)["times"] == stamp(300)
    frame = chain.ArwenFrame.open(out)
    assert frame.times == stamp(300)
    assert frame.attributes["MAP_PROJ"] == 1 and frame.attributes["DX"] == 3000
    assert np.array_equal(bits(frame.array("T")), bits(t[1]))
    assert np.array_equal(bits(frame.array("U")), bits(u[1]))
    assert np.array_equal(bits(frame.array("HGT")), bits(hgt))
    assert "QVAPOR" in frame.header["absent"]
