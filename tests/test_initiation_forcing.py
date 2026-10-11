"""CM1's forcing initiations iinit = 9, 10 and 12 (cm1core.initiation_forcing).

Host tests: the configuration door, CM1's defaults and time windows, the
three host transcriptions against loop-for-loop scalar transcriptions of
misclibs.F / solve1.F, and the refusals.  The device kernels are compared with
the host forms on a small padded grid when a GPU is present.
"""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cm1core.config import load_config
from cm1core.errors import CM1Refusal
from cm1core.initiation_forcing import (
    CONVERGENCE_DEFAULTS,
    FORCING_DEFAULTS,
    UFORCE_DEFAULTS,
    WNUDGE_DEFAULTS,
    InitiationForcing,
    convergence_u_host,
    horizontal_coordinates,
    resolve_initiation_forcing,
    uforce_table_host,
    wnudge_tendency_host,
)

from test_initialization_selectors import toml_document

f32 = np.float32


def write(tmp_path: Path, table: str) -> Path:
    path = tmp_path / "gpuwm-cm1.toml"
    path.write_text(toml_document(table), encoding="utf-8")
    return path


# -- the configuration door ------------------------------------------------


@pytest.mark.parametrize("iinit", [9, 10, 12])
def test_the_forcing_arms_are_admitted_with_cm1s_own_words(tmp_path: Path, iinit: int) -> None:
    config = load_config(write(tmp_path, f"[initialization]\nisnd = 5\niwnd = 2\niinit = {iinit}"))
    init = config.initialization
    assert init.iinit == iinit
    for name, value in FORCING_DEFAULTS[iinit].items():
        assert getattr(init, name) == value, name


def test_init3d_parameter_blocks_are_cm1s() -> None:
    # init3d.F:1372-1385, :1309-1315, :1339-1344, read from the pristine source.
    assert dict(WNUDGE_DEFAULTS) == {
        "xc_wnudge": None, "xr_wnudge": 10000.0, "yc_wnudge": None, "yr_wnudge": 10000.0,
        "zc_wnudge": 1500.0, "zr_wnudge": 1500.0, "alpha_wnudge": 0.5, "wmax_wnudge": 10.0,
        "t1_wnudge": 900.0, "t2_wnudge": 1200.0,
    }
    assert dict(CONVERGENCE_DEFAULTS) == {
        "dmax": -1.0e-3, "zdeep": 2000.0, "lamx": 10000.0, "lamy": 10000.0,
        "xcent": 0.0, "ycent": 0.0, "convtime": 900.0,
    }
    assert dict(UFORCE_DEFAULTS) == {
        "xc_uforce": None, "xr_uforce": 10000.0, "zr_uforce": 10000.0,
        "alpha_uforce": 0.1, "t1_uforce": 3300.0, "t2_uforce": 3600.0,
    }


def test_declared_words_reach_the_forcing_and_centre_defaults_follow_iorigin(tmp_path: Path) -> None:
    config = load_config(write(
        tmp_path,
        "[initialization]\nisnd = 5\niwnd = 2\niinit = 12\nwmax_wnudge = 15.0\nt2_wnudge = 1500.0\n"
        "zc_wnudge = 2000.0",
    ))
    forcing = resolve_initiation_forcing(config.initialization, nx=40, ny=30, dx_m=1000.0, dy_m=1000.0)
    assert forcing["wmax_wnudge"] == 15.0 and forcing["t2_wnudge"] == 1500.0 and forcing["zc_wnudge"] == 2000.0
    assert forcing.declared == frozenset({"wmax_wnudge", "t2_wnudge", "zc_wnudge"})
    # iorigin=1 (GM1's default frame): param.F:7408's centerx = 0.5*nx*dx.
    assert (forcing["xc_wnudge"], forcing["yc_wnudge"]) == (20000.0, 15000.0)
    centred = SimpleNamespace(**{**{k: getattr(config.initialization, k) for k in (
        "iinit", "declared", *WNUDGE_DEFAULTS)}, "iorigin": 2})
    forcing = resolve_initiation_forcing(centred, nx=40, ny=30, dx_m=1000.0, dy_m=1000.0)
    assert (forcing["xc_wnudge"], forcing["yc_wnudge"]) == (0.0, 0.0)


def test_no_forcing_for_the_other_arms(tmp_path: Path) -> None:
    config = load_config(write(tmp_path, "[initialization]\nisnd = 5\niwnd = 2\niinit = 1"))
    assert resolve_initiation_forcing(config.initialization, nx=8, ny=8, dx_m=1.0, dy_m=1.0) is None


def test_a_document_without_forcing_words_keeps_its_canonical_bytes(tmp_path: Path) -> None:
    # A declared optional key projects the whole table; the forcing words of
    # arms the document did not select must not join it.
    config = load_config(write(tmp_path, "[initialization]\nisnd = 5\niwnd = 2\niinit = 1\nzc_m = 1500.0"))
    table = config.canonical_dict()["initialization"]
    assert table["zc_m"] == 1500.0
    for name in table:
        assert not any(part in name for part in ("wnudge", "uforce", "convtime", "lamx", "dmax")), name
    nudged = load_config(write(tmp_path, "[initialization]\nisnd = 5\niwnd = 2\niinit = 12\nwmax_wnudge = 12.0"))
    table = nudged.canonical_dict()["initialization"]
    assert table["wmax_wnudge"] == 12.0 and table["t2_wnudge"] == 1200.0 and "lamx" not in table


@pytest.mark.parametrize(
    "words, needle",
    [
        ("xr_wnudge = 0.0", "rxrwnudge"),
        ("zdeep = -5.0", "misclibs.F:478"),
        ("lamy = 0.0", "init3d.F:1328"),
    ],
)
def test_bad_forcing_words_are_refused_by_what_they_break(tmp_path: Path, words: str, needle: str) -> None:
    with pytest.raises(CM1Refusal) as error:
        load_config(write(tmp_path, f"[initialization]\nisnd = 5\niwnd = 2\niinit = 12\n{words}"))
    assert needle in str(error.value)


@pytest.mark.parametrize("iinit, words", [(12, "t1_wnudge = 1300.0"), (12, "t1_wnudge = 1200.0"), (10, "t2_uforce = 3300.0")])
def test_a_window_without_a_ramp_runs_as_cm1_runs_it(tmp_path: Path, iinit: int, words: str) -> None:
    # sound.F:569 / solve1.F:1245 stop the forcing before mtime reaches t1, so
    # CM1's gamm stays 1 (misclibs.F:557-561, solve1.F:1247-1251): full
    # strength, then off at t2.  Nothing divides by t2 - t1.
    config = load_config(write(tmp_path, f"[initialization]\nisnd = 5\niwnd = 2\niinit = {iinit}\n{words}"))
    init = config.initialization
    fz = resolve_initiation_forcing(init, nx=40, ny=40, dx_m=1000.0, dy_m=1000.0)
    dt = 6.0
    t2 = fz["t2_wnudge"] if iinit == 12 else fz["t2_uforce"]
    active = [t for t in np.arange(0.0, t2 + 30.0, dt)
              if (fz.active_substeps(t, dt) if iinit == 12 else fz.active_held(t))]
    assert active and active[-1] < t2
    assert all(fz.gamm(t) == 1.0 for t in active)


def test_cuda_graphs_and_nests_are_refused_with_a_forcing(tmp_path: Path) -> None:
    from cm1core.cartesian_door import cuda_graph_configuration_refusal

    config = load_config(write(tmp_path, "[initialization]\nisnd = 5\niwnd = 2\niinit = 12"))
    assert "freeze the ramp" in cuda_graph_configuration_refusal(config)
    plain = load_config(write(tmp_path, "[initialization]\nisnd = 5\niwnd = 2\niinit = 1"))
    assert cuda_graph_configuration_refusal(plain) is None


# -- the time windows, in CM1's precision ----------------------------------


def forcing(iinit: int, **words: float) -> InitiationForcing:
    base = {k: (0.0 if v is None else v) for k, v in FORCING_DEFAULTS[iinit].items()}
    base.update(words)
    return InitiationForcing(iinit=iinit, words=tuple(base.items()))


def test_updraft_nudging_acts_while_mtime_plus_dt_is_within_t2() -> None:
    nudge = forcing(12)
    # sound.F:569: (mtime+dt).le.t2_wnudge.
    assert nudge.active_substeps(1194.0, 6.0)
    assert not nudge.active_substeps(1200.0, 6.0)
    # misclibs.F:557-561: gamm = 1 before t1, then linear to 0 at t2.
    assert nudge.gamm(0.0) == 1.0 and nudge.gamm(899.0) == 1.0
    assert nudge.gamm(1050.0) == pytest.approx(0.5)
    assert nudge.gamm(1194.0) == pytest.approx(0.02, abs=1e-6)


def test_convergence_uses_sngl_mtime_plus_dt_and_uforce_mtime() -> None:
    conv = forcing(9)
    assert conv.active_substeps(894.0, 6.0) and not conv.active_substeps(900.0, 6.0)
    assert not conv.active_held(0.0)
    uforce = forcing(10)
    assert uforce.active_held(3594.0) and not uforce.active_held(3600.0)
    assert not uforce.active_substeps(0.0, 6.0)
    assert uforce.gamm(3450.0) == pytest.approx(0.5)
    # init3d.F:1328: Aconv = (-0.5*Dmax)/(1/lamx**2 + 1/lamy**2).
    assert conv.aconv() == pytest.approx(0.5e-3 / 2.0e-8, rel=1e-6)


# -- host forms against scalar transcriptions -----------------------------


def grid(nx: int = 12, ny: int = 10, nz: int = 8, dx: float = 1000.0, dz: float = 500.0, iorigin: int = 2):
    coords = horizontal_coordinates(nx=nx, ny=ny, dx_m=dx, dy_m=dx, iorigin=iorigin)
    zf = (dz * np.arange(nz + 1)).astype(f32)
    zh = (0.5 * (zf[:-1] + zf[1:])).astype(f32)
    return coords, zh, zf


def test_horizontal_coordinates_are_param_F_s() -> None:
    coords = horizontal_coordinates(nx=4, ny=2, dx_m=1000.0, dy_m=500.0, iorigin=2)
    assert coords["xf"].tolist() == [-2000.0, -1000.0, 0.0, 1000.0, 2000.0]
    assert coords["xh"].tolist() == [-1500.0, -500.0, 500.0, 1500.0]
    assert coords["yh"].tolist() == [-250.0, 250.0]
    corner = horizontal_coordinates(nx=4, ny=2, dx_m=1000.0, dy_m=500.0, iorigin=1)
    assert corner["xh"].tolist() == [500.0, 1500.0, 2500.0, 3500.0]


def test_wnudge_host_form_is_get_wnudge() -> None:
    coords, zh, zf = grid()
    nudge = forcing(12, xc_wnudge=0.0, yc_wnudge=0.0, xr_wnudge=4000.0, yr_wnudge=3000.0, zr_wnudge=1500.0)
    rng = np.random.default_rng(3)
    w = rng.uniform(-3.0, 12.0, size=(len(zf), 10, 12)).astype(f32)
    got = wnudge_tendency_host(nudge, mtime_s=1000.0, xh=coords["xh"], yh=coords["yh"], zf=zf, w=w)
    # misclibs.F:563-583, loop for loop, in default REAL.
    tem = f32(nudge["alpha_wnudge"]) * f32(nudge.gamm(1000.0))
    rx, ry, rz = (f32(1.0) / f32(nudge[n]) for n in ("xr_wnudge", "yr_wnudge", "zr_wnudge"))
    halfpi = f32(f32(0.5) * f32(math.pi))
    want = np.zeros_like(w)
    for k in range(1, len(zf) - 1):          # CM1 k = 2..nk
        for j in range(10):
            for i in range(12):
                a = (coords["xh"][i] - f32(0.0)) * rx
                b = (coords["yh"][j] - f32(0.0)) * ry
                c = (zf[k] - f32(1500.0)) * rz
                beta = f32(np.sqrt(f32(f32(a * a + b * b) + c * c)))
                if beta < 1.0:
                    wmag = f32(10.0) * f32(np.cos(halfpi * beta)) ** 2
                    want[k, j, i] = tem * max(f32(wmag - w[k, j, i]), f32(0.0))
    np.testing.assert_allclose(got, want, rtol=2e-6, atol=1e-7)
    assert np.count_nonzero(got) > 20
    assert (got[0] == 0).all() and (got[-1] == 0).all()


def test_uforce_host_form_is_solve1s() -> None:
    coords, zh, zf = grid(nx=24)
    uf = forcing(10, xc_uforce=0.0, xr_uforce=8000.0, zr_uforce=2000.0)
    got = uforce_table_host(uf, xf=coords["xf"], zf=zf[:-1])
    want = np.zeros_like(got)
    for k in range(len(zf) - 1):
        for i, x in enumerate(coords["xf"]):
            if abs(x - 0.0) < 8000.0 and abs(zf[k]) < 2000.0:
                want[k, i] = f32(0.1) * np.cos(f32(0.5) * f32(math.pi) * x / f32(8000.0)) * (
                    np.cosh(f32(2.5) * zf[k] / f32(2000.0)) ** -2)
    np.testing.assert_allclose(got, want, rtol=2e-6, atol=1e-9)
    assert got[0].max() == pytest.approx(0.1) and (got[4:] == 0).all()


def test_cosh_is_the_reference_builds_correctly_rounded_coshf() -> None:
    # glibc 2.41+ (the reference CM1 build's, glibc 2.43) returns the correctly
    # rounded coshf; glibc 2.39's is an ulp off at these three arguments (and
    # ~0.8 % of all).  GM1 must give the reference build's words whatever glibc
    # it runs on.
    from cm1core.initiation_forcing import libm_f32

    x = np.array([0x3A1CC470, 0x3A4A62C2, 0x3A87C3B6], dtype=np.uint32).view(np.float32)
    got = libm_f32("coshf", x).view(np.uint32)
    assert got.tolist() == [0x3F800001, 0x3F800003, 0x3F800004]


def test_the_stock_grid_u_forcing_is_the_reference_builds_words() -> None:
    # CM1's stock 1 km supercell grid (120 x 40 levels of 500 m, iorigin = 2)
    # with iinit = 10's own words: the aiu table GM1's host form built with
    # glibc 2.43, the reference CM1's library.  Before the correctly
    # rounded coshf, a glibc 2.39 build had 76 of its 380 forced
    # words up to 4 ulp away.
    import hashlib

    init = SimpleNamespace(iinit=10, iorigin=2, declared=frozenset(), **dict(FORCING_DEFAULTS[10]))
    fz = resolve_initiation_forcing(init, nx=120, ny=120, dx_m=1000.0, dy_m=1000.0)
    c = horizontal_coordinates(nx=120, ny=120, dx_m=1000.0, dy_m=1000.0, iorigin=2)
    zf = (np.float32(500.0) * np.arange(41, dtype=np.float32)).astype(np.float32)
    aiu = uforce_table_host(fz, xf=c["xf"], zf=zf[:40])
    assert hashlib.sha256(aiu.tobytes()).hexdigest() == (
        "142f961b75818c1192c0fdb1d8dafe1d76f25a31d1fbefd8022fd45ec389f813")


def test_convergence_host_form_is_convinitu() -> None:
    coords, zh, zf = grid(nx=16, ny=14)
    conv = forcing(9, lamx=4000.0, lamy=4000.0, zdeep=2000.0)
    u0 = np.linspace(-5.0, 10.0, len(zh)).astype(f32)
    u = np.full((len(zh), 14, 17), 99.0, dtype=f32)
    got = convergence_u_host(conv, xf=coords["xf"], yh=coords["yh"], zh=zh, u0=u0, u=u)
    aconv = f32(conv.aconv())
    for k in range(len(zh)):
        for j in range(14):
            for i in range(17):
                term4 = (f32(2000.0) - zh[k]) / f32(2000.0)
                expected = f32(99.0)
                if term4 > 0:
                    dx = coords["xf"][i]
                    umo = (-(f32(2.0) * aconv * dx) / f32(4000.0) ** 2) * (
                        np.exp(-(dx / f32(4000.0)) ** 2) * np.exp(-(coords["yh"][j] / f32(4000.0)) ** 2)) * term4
                    if abs(umo) > 0.01:
                        expected = u0[k] + umo
                assert got[k, j, i] == pytest.approx(expected, rel=2e-6, abs=1e-6), (k, j, i)
    # The forcing converges: westerly on the west side, easterly on the east.
    assert got[0, 7, 4] > u0[0] and got[0, 7, 12] < u0[0]


# -- the device kernels against the host forms ----------------------------


def _gpu():
    cupy = pytest.importorskip("cupy")
    try:
        if cupy.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA device")
    except Exception:  # noqa: BLE001
        pytest.skip("no CUDA device")
    return cupy


def _stage(cupy, fz, *, nx=12, ny=10, nz=8):
    from cm1core.grid_geometry import CartesianGridGeometry
    from cm1core.initiation_forcing import InitiationForcingStage

    g = CartesianGridGeometry(nx=nx, ny=ny, nz=nz, dx_m=1000.0, dy_m=1000.0, dz_m=500.0)
    stream = cupy.cuda.Stream(non_blocking=True)

    class Registry:
        def allocate(self, name, shape, zero=False):
            return cupy.zeros(shape, dtype=cupy.float32)

        def release(self, names):
            pass

    ng = 3
    arrays = {
        "wten": cupy.asarray(np.random.default_rng(1).normal(size=(nz + 3, ny + 2 * ng, nx + 2 * ng)).astype(f32)),
        "w3d": cupy.asarray(np.random.default_rng(2).uniform(-2, 12, size=(nz + 3, ny + 2 * ng, nx + 2 * ng)).astype(f32)),
        "u3d": cupy.asarray(np.full((nz + 2, ny + 2 * ng, nx + 1 + 2 * ng), 7.0, dtype=f32)),
        "v3d": cupy.asarray(np.full((nz + 2, ny + 1 + 2 * ng, nx + 2 * ng), -3.0, dtype=f32)),
    }
    dynamics = SimpleNamespace(
        geometry=g, cupy=cupy, stream=stream, arrays=arrays, registry=Registry(),
        config=SimpleNamespace(dt_s=6.0),
        base_profiles={"u0": cupy.asarray(np.linspace(1, 8, nz).astype(f32)),
                       "v0": cupy.asarray(np.linspace(-2, 2, nz).astype(f32))},
        held={"uten": cupy.zeros((nz, ny, nx + 1), dtype=cupy.float32)},
    )
    return InitiationForcingStage(fz, dynamics), dynamics


def test_device_nudging_matches_the_host_form() -> None:
    cupy = _gpu()
    fz = forcing(12, xc_wnudge=0.0, yc_wnudge=0.0, xr_wnudge=4000.0, yr_wnudge=3000.0)
    stage, d = _stage(cupy, fz)
    stage.begin_step(1000.0)
    wten0 = cupy.asnumpy(d.arrays["wten"]).copy()
    hook = stage.substep_hook(None)
    hook.after_momentum(nrk=1, n=1, dts=1.0)
    d.stream.synchronize()
    got = cupy.asnumpy(d.arrays["wten"])
    g = d.geometry
    phys = (slice(1, g.nz + 2), slice(3, 3 + g.ny), slice(3, 3 + g.nx))
    w = cupy.asnumpy(d.arrays["w3d"])[phys]
    host = wnudge_tendency_host(fz, mtime_s=1000.0, xh=stage._host["xh"], yh=stage._host["yh"],
                                zf=stage._host["zf"], w=w)
    expected = wten0.copy()
    inner = (slice(2, g.nz + 1), slice(3, 3 + g.ny), slice(3, 3 + g.nx))
    expected[inner] = wten0[inner] + host[1:-1]
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)
    # A second substep adds to the saved tendency, not to the nudged one.
    hook.after_momentum(nrk=1, n=2, dts=1.0)
    d.stream.synchronize()
    np.testing.assert_allclose(cupy.asnumpy(d.arrays["wten"]), expected, rtol=1e-6, atol=1e-6)
    # After t2 no hook is returned.
    stage.begin_step(1200.0)
    assert stage.substep_hook(None) is None


def test_device_convergence_and_uforce_match_the_host_forms() -> None:
    cupy = _gpu()
    fz = forcing(9, lamx=4000.0, lamy=4000.0)
    stage, d = _stage(cupy, fz, nx=16, ny=14)
    stage.begin_step(0.0)
    stage.substep_hook(None).after_momentum(nrk=1, n=1, dts=1.0)
    d.stream.synchronize()
    g = d.geometry
    u = cupy.asnumpy(d.arrays["u3d"])[1:g.nz + 1, 3:3 + g.ny, 3:3 + g.nx + 1]
    host = convergence_u_host(fz, xf=stage._host["xf"], yh=stage._host["yh"], zh=stage._host["zh"],
                              u0=cupy.asnumpy(d.base_profiles["u0"]), u=np.full_like(u, 7.0))
    np.testing.assert_allclose(u, host, rtol=1e-6, atol=1e-6)
    assert (u != 7.0).any()

    uf = forcing(10, xc_uforce=0.0, xr_uforce=5000.0, zr_uforce=2000.0)
    stage, d = _stage(cupy, uf, nx=16, ny=14)
    stage.begin_step(3450.0)
    stage.held()
    d.stream.synchronize()
    table = uforce_table_host(uf, xf=stage._host["xf"], zf=stage._host["zf"][:-1])
    got = cupy.asnumpy(d.held["uten"])
    np.testing.assert_allclose(got, np.broadcast_to(f32(0.5) * table[:, None, :], got.shape), rtol=1e-6, atol=1e-9)


# -- where the storm starts, and the Settings screen ------------------------


def test_the_storm_starts_at_the_forcing_centre(tmp_path: Path) -> None:
    from cm1core.cartesian_door import bubble_center_m

    config = load_config(write(
        tmp_path,
        "[initialization]\nisnd = 5\niwnd = 2\niinit = 12\nxc_wnudge = 4000.0\nyc_wnudge = -3000.0\niorigin = 2",
    ))
    assert bubble_center_m(config) == (4000.0, -3000.0)
    centred = load_config(write(tmp_path, "[initialization]\nisnd = 5\niwnd = 2\niinit = 12\niorigin = 2"))
    assert bubble_center_m(centred) == (0.0, 0.0)
    conv = load_config(write(tmp_path, "[initialization]\nisnd = 5\niwnd = 2\niinit = 9\nxcent = 2000.0"))
    assert bubble_center_m(conv)[0] == 2000.0


def test_the_settings_screen_shows_the_selected_forcing() -> None:
    from cm1core.cm1import.setup import fortran_groups

    effective = {"iinit": 12, "nx": 120, "ny": 120, "dx": 1000.0, "dy": 1000.0, "iorigin": 2}
    groups = {g["id"]: g for g in fortran_groups(effective, {}, {})}
    assert groups["nudging"]["active"] and not groups["bubble"]["active"]
    fields = {f["key"]: f for f in groups["nudging"]["fields"]}
    assert fields["initialization.wmax_wnudge"]["value"] == 10.0
    assert fields["initialization.xc_wnudge"]["value"] == 0.0  # centerx, km
    assert fields["initialization.t2_wnudge"]["value"] == 1200.0
    assert not groups["convergence"]["active"] and not groups["uforce"]["active"]


# -- CM1's own random numbers ------------------------------------------------


def test_cm1s_random_stream_needs_no_seed_and_refuses_one(tmp_path: Path) -> None:
    config = load_config(write(
        tmp_path, "[initialization]\nisnd = 5\niwnd = 2\niinit = 1\nirandp = 1\nrandom_generator = \"cm1\""))
    assert config.initialization.random_generator == "cm1" and config.initialization.random_seed == 0
    with pytest.raises(CM1Refusal) as error:
        load_config(write(tmp_path, "[initialization]\nisnd = 5\niwnd = 2\nirandp = 1\n"
                          "random_generator = \"cm1\"\nrandom_seed = 7"))
    assert error.value.detail.code == "CM1-CFG-RANDOM-SEED-UNREAD"
    with pytest.raises(CM1Refusal):
        load_config(write(tmp_path, "[initialization]\nisnd = 5\niwnd = 2\nrandom_generator = \"fortran\""))


def test_cm1_namelists_with_the_new_arms_import_as_cm1_runs_them(tmp_path: Path) -> None:
    import tomllib

    from cm1import_support import SUPERCELL, edited, namelist_tool
    from cm1core.cm1import.coverage import build_report, import_for_report

    for iinit, irandp in (("12", "0"), ("9", "0"), ("10", "0"), ("11", "0"), ("8", "0"), ("1", "1")):
        deck = edited(SUPERCELL, tmp_path, {"iinit": iinit, "irandp": irandp}, f"n{iinit}-{irandp}")
        report = build_report(import_for_report(deck, tool=namelist_tool()))
        assert report.exit_code == 0, (iinit, [r.as_dict() for r in report.category("refused")])
        rows = {key for row in report.category("differs") for key in row.keys}
        assert "iinit" not in rows and "irandp" not in rows, (iinit, rows)
        table = tomllib.loads(report.toml)["initialization"]
        assert table["iinit"] == int(iinit)
        if iinit == "8" or irandp == "1":
            assert table["random_generator"] == "cm1" and "random_seed" not in table


def test_the_validation_record_names_cm1s_stream() -> None:
    from cm1core.gm1_idealized_residency import InitializerSelection
    from cm1core.initialization_validation import unvalidated_initialization

    cm1 = InitializerSelection(isnd=5, iwnd=2, iinit=8, random_generator=1)
    assert unvalidated_initialization(cm1)["status"] == "validated"
    gm1 = InitializerSelection(isnd=5, iwnd=2, iinit=8, random_seed=5)
    assert unvalidated_initialization(gm1)["status"] != "validated"
    nudged = InitializerSelection(isnd=5, iwnd=2, iinit=12)
    assert unvalidated_initialization(nudged)["status"] == "validated"


# -- iinit = 7, the tropical-cyclone vortex ----------------------------------


def test_the_vortex_is_admitted_on_cm1s_centred_origin_only(tmp_path: Path) -> None:
    config = load_config(write(tmp_path, "[initialization]\nisnd = 5\niwnd = 2\niinit = 7\niorigin = 2"))
    assert config.initialization.iinit == 7
    for words, code in (
        ("iorigin = 1", "CM1-CFG-VORTEX-IORIGIN"),
        ("iorigin = 2\nibalance = 1", "CM1-CFG-VORTEX-IBALANCE"),
        ("iorigin = 2\nmaintain_rh = true", "CM1-CFG-VORTEX-MAINTAIN-RH"),
    ):
        with pytest.raises(CM1Refusal) as error:
            load_config(write(tmp_path, f"[initialization]\nisnd = 5\niwnd = 2\niinit = 7\n{words}"))
        assert error.value.detail.code == code, words


def test_the_vortex_reads_fcor_only_with_icor(tmp_path: Path) -> None:
    from cm1core.cartesian_door import require_initializer_selection

    config = load_config(write(tmp_path, "[initialization]\nisnd = 5\niwnd = 2\niinit = 7\niorigin = 2"))
    selection, _ = require_initializer_selection(config)
    # param.F:1985: icor = 0 zeroes fcor, and the vortex's gradient wind reads it.
    assert selection.fcor_s_inv == 0.0
    assert selection.options()["fcor_s_inv"] == 0.0
    plain = load_config(write(tmp_path, "[initialization]\nisnd = 5\niwnd = 2\niinit = 1"))
    assert "fcor_s_inv" not in require_initializer_selection(plain)[0].options()


# -- CM1's own statements under gfortran, word for word ----------------------


def test_the_host_forms_are_cm1s_statements_word_for_word(tmp_path: Path) -> None:
    """tools/oracle/initiation_forcing_oracle.py: get_wnudge + sound.F's
    placement, convinitu/convinitv + Aconv, and solve1.F's uforce block,
    compiled from CM1's statements, against the host forms raw word for word.
    (The device kernels are held to the same words with --device on a GPU.)"""

    import shutil
    import subprocess
    import sys

    if shutil.which("gfortran") is None:
        pytest.skip("gfortran is not installed")
    root = Path(__file__).resolve().parents[1]
    out = tmp_path / "oracle.json"
    result = subprocess.run(
        [sys.executable, str(root / "tools" / "oracle" / "initiation_forcing_oracle.py"), "--json", str(out)],
        capture_output=True, text=True, env={"PYTHONPATH": str(root / "src"), "PATH": "/usr/bin:/bin"},
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    payload = __import__("json").loads(out.read_text())
    assert payload["status"] == "raw-word-identical"
    assert all(case["cm1_words_changed_by_the_forcing"] > 0 for case in payload["cases"].values())


def test_the_line_s_first_bubble_is_cm1s_literal_unless_a_word_moves_it(tmp_path: Path) -> None:
    # init3d.F:548: an undeclared (imported) iinit = 3 line starts at rjc = 3 km;
    # a declared bubble_center_y_m (what the Settings screen writes beside x,
    # and what the line-open preset's picture was made with) or
    # bubble_first_center_y_m moves it; both at once are refused.
    from cm1core.cartesian_door import require_initializer_selection

    def rjc(words: str) -> float:
        selection = require_initializer_selection(load_config(write(tmp_path, f"[initialization]\nisnd = 5\niwnd = 2\n{words}")))
        return (selection[0] if isinstance(selection, tuple) else selection).rjc_m

    assert rjc("iinit = 3") == 3000.0
    assert rjc("iinit = 3\nbubble_center_y_m = 60000.0") == 60000.0
    assert rjc("iinit = 3\nbubble_first_center_y_m = 20000.0") == 20000.0
    with pytest.raises(CM1Refusal) as error:
        rjc("iinit = 3\nbubble_center_y_m = 60000.0\nbubble_first_center_y_m = 20000.0")
    assert "init3d.F:548" in str(error.value)


def test_the_wave_refuses_the_bubble_radius_it_does_not_read(tmp_path: Path) -> None:
    with pytest.raises(CM1Refusal) as error:
        load_config(write(tmp_path, "[initialization]\nisnd = 5\niwnd = 2\niinit = 11\nbhrad_m = 7000.0"))
    assert "half_width_m" in str(error.value) and "init3d.F:1355-1356" in str(error.value)
