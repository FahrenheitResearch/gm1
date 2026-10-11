"""CM1 ``pdcomp.F`` with the open-lateral (mirrored) Poisson inversion.

Four kinds of evidence, none of which opens a CUDA context:

1. **CM1 itself.**  Pristine ``pdcomp.F``/``poiss.F``/``singleton.F`` are
   built with ``gfortran`` (``tools/oracle/pdcomp_native_oracle.py``).  The
   capture build hands every forcing, flux, perturbation wind and ``divx`` CM1
   builds to a stub ``poiss``/``advu``; those, ``thv0`` and the
   ``init3d.F`` coefficients are compared to the CPU reference **bit for
   bit**.  The solve build is CM1's own inversion (Singleton FFT), compared to
   the CPU reference's (NumPy FFT) within transform round-off.
2. **The packaged CUDA text** compiled with ``g++`` and launched thread by
   thread (index and shuffled order), compared to the CPU reference word for
   word -- including this lane's ``advu``/``advv``/``advw`` kernel text driven
   on pdcomp's buffers.
3. **Analytic Poisson problems** under CM1's lateral condition: a discrete
   Neumann cosine eigenfunction is recovered, and a general solution satisfies
   the reflecting-ghost equations at the boundary columns (and not the
   periodic ones), on uniform and stretched decks.
4. **Diagnostic purity**: the parts sum to the inversion of the summed
   forcing; no model carrier reaches a writable kernel parameter; the CPU path
   leaves every input byte unchanged.

Uniform and stretched (non-unit ``mh``/``mf``, non-half ``c1``/``c2``) decks
run through every check.
"""

from __future__ import annotations

import hashlib
import shutil

import pytest

np = pytest.importorskip("numpy")

from cm1core.cuda_pdcomp_open import (  # noqa: E402
    CUDA_PDCOMP_OPEN_SOURCE,
    PDCOMP_SCALAR_NAMES,
    PDCOMP_W_NAMES,
    PdcompSelection,
    build_pdcomp_coefficients,
    cm1_constants,
    cm1_thv0_profile,
    cpu_buoyancy_wten,
    cpu_coriolis_uv,
    cpu_gauge,
    cpu_linear_uv,
    cpu_nonlinear_fluxes,
    cpu_pdcomp,
    cpu_pgrad,
    cpu_poiss_invert,
    cpu_poiss_rhs,
    cpu_uniform_advect,
    padded_shapes,
    pdcomp_selection_defect,
)

requires_gxx = pytest.mark.skipif(shutil.which("g++") is None, reason="g++ is not installed")
requires_gfortran = pytest.mark.skipif(shutil.which("gfortran") is None, reason="gfortran is not installed")
f32 = np.float32


def _pristine():
    from tools.oracle.pdcomp_native_oracle import pristine_source_dir

    return pristine_source_dir()


requires_pristine = pytest.mark.skipif(
    _pristine() is None,
    reason="CM1 r21.1 pristine src not found (set GM1_CM1_PRISTINE_SRC)",
)


# ----------------------------------------------------------------- decks


def make_deck(*, ni=8, nj=6, nk=7, stretched=False, icor=1, imoist=1, seed=0,
              dx=500.0, dz=400.0, dt=3.0):
    """One deck, in Fortran bounds order (``native``) and as CPU inputs."""

    rng = np.random.default_rng(seed)
    ng = 3
    kk = np.arange(nk + 3, dtype=np.float64)          # Fortran k = 0..nk+2
    th0p = (300.0 + 2.5 * kk).astype(f32)
    qv0p = np.maximum(0.016 - 0.0019 * kk, 1.0e-4).astype(f32)
    rho0p = (1.2 * np.exp(-kk * dz / 9000.0)).astype(f32)
    rf0p = (1.2 * np.exp(-(kk - 0.5) * dz / 9000.0)).astype(f32)
    u0p = (1.5 * kk - 4.0).astype(f32)
    v0p = (3.0 - 0.75 * kk + 0.05 * kk * kk).astype(f32)
    if stretched:
        zf = np.concatenate(([0.0], np.cumsum(dz * (0.55 + 0.12 * np.arange(nk + 2)))))
        zf = np.concatenate(([-dz * 0.55], zf))    # index 0 -> k=0
        zh = 0.5 * (zf[:-1] + zf[1:])
        mhp = (dz / (zf[1:nk + 4] - zf[0:nk + 3])).astype(f32)[: nk + 3]
        mfp = np.ones(nk + 3, dtype=f32)
        mfp[1:] = (dz / (zh[1:nk + 3] - zh[0:nk + 2])).astype(f32)
        mfp[0] = mfp[1]
        c2p = (0.35 + 0.3 * rng.random(nk + 3)).astype(f32)
    else:
        mhp = np.ones(nk + 3, dtype=f32)
        mfp = np.ones(nk + 3, dtype=f32)
        c2p = np.full(nk + 3, 0.5, dtype=f32)
    c1p = (f32(1.0) - c2p).astype(f32)

    m_shape = (nk + 2, nj + 2 * ng, ni + 2 * ng)
    shapes = padded_shapes(ni, nj, nk)

    def col(p, shape):
        return np.ascontiguousarray(np.broadcast_to(p[: shape[0], None, None], shape), dtype=f32)

    native = {
        "uh": np.ones(ni + 2 * ng, dtype=f32), "vh": np.ones(nj + 2 * ng, dtype=f32),
        "f2d": (1.0e-4 + 2.0e-9 * (np.arange(nj + 2 * ng)[:, None] - 5) * np.ones((1, ni + 2 * ng))).astype(f32),
        "mh": col(mhp, m_shape), "mf": col(mfp, (nk + 3, *m_shape[1:])),
        "c1": col(c1p, m_shape), "c2": col(c2p, m_shape),
        "th0": col(th0p, m_shape), "qv0": col(qv0p, m_shape),
        "rho0": col(rho0p, m_shape), "rr0": col((f32(1.0) / rho0p).astype(f32), m_shape),
        "rf0": col(rf0p, m_shape),
        "u0": col(u0p, shapes["u"]), "v0": col(v0p, shapes["v"]),
        "u3d": (u0p[: nk + 2, None, None] + 3.0 * rng.standard_normal(shapes["u"])).astype(f32),
        "v3d": (v0p[: nk + 2, None, None] + 3.0 * rng.standard_normal(shapes["v"])).astype(f32),
        "w3d": (2.0 * rng.standard_normal(shapes["w"])).astype(f32),
        "pp3d": (1.0e-3 * rng.standard_normal(m_shape)).astype(f32),
        "th3d": rng.standard_normal(m_shape).astype(f32),
        "q3d": np.abs(1.0e-3 * rng.standard_normal((6, *m_shape))).astype(f32),
    }
    native["q3d"][0] = (native["qv0"] + 5.0e-4 * rng.standard_normal(m_shape)).astype(f32)
    sel = PdcompSelection(ni, nj, nk, dx, dx, dz, dt, icor=icor, imoist=imoist)
    I, J = slice(ng, ng + ni), slice(ng, ng + nj)
    phys = lambda a, n: np.ascontiguousarray(a[1:n + 1, J, I])  # noqa: E731
    thv0 = cm1_thv0_profile(th0p[1:nk + 1], qv0p[1:nk + 1])
    coeff = build_pdcomp_coefficients(
        sel, mh=mhp[1:nk + 1], mf=mfp[1:nk + 2], rho0=rho0p[1:nk + 1], rf0=rf0p[1:nk + 2], thv0=thv0)
    cpu = dict(
        u3d=native["u3d"], v3d=native["v3d"], w3d=native["w3d"],
        th3d=phys(native["th3d"], nk), pp3d=phys(native["pp3d"], nk),
        q3d=np.ascontiguousarray(native["q3d"][:, 1:nk + 1, J, I]),
        th0=th0p[1:nk + 1].copy(), qv0=qv0p[1:nk + 1].copy(),
        u0=u0p[1:nk + 1].copy(), v0=v0p[1:nk + 1].copy(),
        rho0c=rho0p[1:nk + 1].copy(), rr0c=(f32(1.0) / rho0p[1:nk + 1]).astype(f32),
        rf0c=rf0p[1:nk + 2].copy(), mhc=mhp[1:nk + 1].copy(), mfc=mfp[1:nk + 2].copy(),
        rf0=phys(native["rf0"], nk + 1), mh=phys(native["mh"], nk), mf=phys(native["mf"], nk + 1),
        c1=phys(native["c1"], nk + 1), c2=phys(native["c2"], nk + 1),
        uh=native["uh"], vh=native["vh"],
        f2d=np.ascontiguousarray(native["f2d"][ng - 1:ng + nj + 1, ng - 1:ng + ni + 1]),
    )
    metrics = dict(
        rho0_3d=phys(native["rho0"], nk), c2=cpu["c2"],
        gzu=np.ones((nj, ni + 1), dtype=f32), gzv=np.ones((nj + 1, ni), dtype=f32),
        rrf0=(f32(1.0) / cpu["rf0"]).astype(f32), gz=np.ones((nj, ni), dtype=f32),
    )
    deck = dict(ni=ni, nj=nj, nk=nk, dx=dx, dy=dx, dz=dz, dt=dt, icor=icor, imoist=imoist,
                iice=1, wbc=2, ebc=2, sbc=2, nbc=2)
    return dict(deck=deck, native=native, cpu=cpu, sel=sel, coeff=coeff, metrics=metrics)


DECKS = {"uniform": dict(stretched=False), "stretched": dict(stretched=True, seed=3)}


def _bits(a):
    return np.ascontiguousarray(a, dtype=f32).view(np.uint32)


def assert_bitwise(actual, expected, what):
    a, e = _bits(actual), _bits(expected)
    assert a.shape == e.shape, what
    bad = np.flatnonzero(a.reshape(-1) != e.reshape(-1))
    assert bad.size == 0, f"{what}: {bad.size} words differ, first at {bad[:5]}"


# ------------------------------------------------------ 1. CM1 natively


@pytest.fixture(scope="module")
def native_runs(tmp_path_factory):
    cache = {}

    def run(name, mode):
        key = (name, mode)
        if key not in cache:
            from tools.oracle.pdcomp_native_oracle import run_native_pdcomp

            d = make_deck(**DECKS[name])
            cache[key] = (d, run_native_pdcomp(d["deck"], d["native"], tmp_path_factory.mktemp(f"{name}-{mode}"),
                                                mode=mode, pristine=_pristine()))
        return cache[key]

    return run


@requires_gfortran
@requires_pristine
@pytest.mark.parametrize("name", sorted(DECKS))
def test_cm1_coefficients_and_thv0_are_the_cpu_reference_bit_for_bit(native_runs, name):
    d, out = native_runs(name, "capture")
    s, nk = d["sel"], d["sel"].nk
    ng = 3
    thv0 = out["out_thv0"].reshape(nk + 2, s.nj + 2 * ng, s.ni + 2 * ng)[1:nk + 1, 0, 0]
    assert_bitwise(d["coeff"].thv0, thv0, "thv0 (base.F:2130)")
    for name_, ref in (("cfa", d["coeff"].cfa), ("cfc", d["coeff"].cfc), ("ad1", d["coeff"].d1), ("ad2", d["coeff"].d2)):
        assert_bitwise(ref, out[f"out_{name_}"][1:nk + 1], name_)
    cfb = out["out_cfb"].reshape(nk + 2, s.jpe, s.ipe)[1:nk + 1]
    assert (s.ipe, s.jpe) == (2 * s.ni, 2 * s.nj)
    assert_bitwise(d["coeff"].cfb, cfb, "cfb (init3d.F:1687-1689, mirrored modes)")


@requires_gfortran
@requires_pristine
@pytest.mark.parametrize("name", sorted(DECKS))
def test_cm1_pdcomp_forcings_are_the_cpu_reference_bit_for_bit(native_runs, name):
    d, out = native_runs(name, "capture")
    s, c = d["sel"], d["cpu"]
    ni, nj, nk = s.ni, s.nj, s.nk
    sh = padded_shapes(ni, nj, nk)
    cap = lambda key, st: out[key].reshape(sh[st])  # noqa: E731

    wten = cpu_buoyancy_wten(s, th3d=c["th3d"], q3d=c["q3d"], th0=c["th0"], qv0=c["qv0"], c1=c["c1"], c2=c["c2"])
    region_w = (slice(1, nk + 2), slice(3, nj + 3), slice(3, ni + 3))
    assert_bitwise(wten[region_w], cap("cap_wten_1", "w")[region_w], "buoyancy wten (pdcomp.F:104-117)")

    region_u = (slice(1, nk + 1), slice(3, nj + 4), slice(3, ni + 4))
    uten, vten = cpu_linear_uv(s, w3d=c["w3d"], rf0c=c["rf0c"], mfc=c["mfc"], rr0c=c["rr0c"], u0=c["u0"], v0=c["v0"])
    assert_bitwise(uten[region_u], cap("cap_uten_2", "u")[region_u], "linear uten (pdcomp.F:196-236)")
    assert_bitwise(vten[region_u], cap("cap_vten_2", "v")[region_u], "linear vten")

    uten, vten = cpu_coriolis_uv(s, u3d=c["u3d"], v3d=c["v3d"], f2d=c["f2d"])
    assert_bitwise(uten[region_u], cap("cap_uten_4", "u")[region_u], "Coriolis uten (pdcomp.F:373-386)")
    assert_bitwise(vten[region_u], cap("cap_vten_4", "v")[region_u], "Coriolis vten")

    fl = cpu_nonlinear_fluxes(s, u3d=c["u3d"], v3d=c["v3d"], w3d=c["w3d"], rho0c=c["rho0c"], rf0c=c["rf0c"],
                              mhc=c["mhc"], u0=c["u0"], v0=c["v0"], uh=c["uh"], vh=c["vh"])
    k_all = slice(1, nk + 1)
    assert_bitwise(fl["rru"][k_all], cap("cap_rru", "u")[k_all], "rru (pdcomp.F:275)")
    assert_bitwise(fl["uten1"][k_all], cap("cap_uten1", "u")[k_all], "u' (pdcomp.F:276)")
    assert_bitwise(fl["rrv"][k_all], cap("cap_rrv", "v")[k_all], "rrv (pdcomp.F:281)")
    assert_bitwise(fl["vten1"][k_all], cap("cap_vten1", "v")[k_all], "v' (pdcomp.F:282)")
    rw = (slice(1, nk + 2), slice(2, nj + 4), slice(2, ni + 4))
    assert_bitwise(fl["rrw"][rw], cap("cap_rrw", "w")[rw], "rrw (pdcomp.F:285-297)")
    divx = out["cap_divx"].reshape(nk + 2, nj + 6, ni + 6)[1:nk + 1, 3:nj + 3, 3:ni + 3]
    assert_bitwise(fl["divx"], divx, "divx (pdcomp.F:306-309)")


def _cpu_run(d, advect=None):
    s, c = d["sel"], d["cpu"]
    return cpu_pdcomp(s, d["coeff"], **c, advect=advect or cpu_uniform_advect(d["metrics"], s))


@requires_gfortran
@requires_pristine
@pytest.mark.parametrize("name", sorted(DECKS))
def test_cm1_inversion_is_the_cpu_inversion_to_transform_roundoff(native_runs, name):
    d, out = native_runs(name, "solve")
    s = d["sel"]
    ni, nj, nk = s.ni, s.nj, s.nk
    pdiag = out["out_pdiag"].reshape(4, nk + 2, nj + 6, ni + 6)[:, 1:nk + 1, 3:nj + 3, 3:ni + 3]
    ours = _cpu_run(d, advect=lambda arrays, divx: None)   # the native build's advu adds nothing
    for slot, name_ in enumerate(PDCOMP_SCALAR_NAMES):
        cm1, mine = pdiag[slot].astype(np.float64), ours[name_].astype(np.float64)
        scale = max(float(np.max(np.abs(cm1))), 1.0e-30)
        err = float(np.max(np.abs(cm1 - mine))) / scale
        if name_ == "pipdn":
            assert float(np.max(np.abs(cm1))) < 1.0e-6 and float(np.max(np.abs(mine))) < 1.0e-6
            continue
        # Singleton (CM1) and NumPy's FFT differ only in round-off; the vertical
        # recurrence, mirror and gauge are the same statements.
        assert err < 2.0e-6, f"{name_}: relative difference {err:.3e} from CM1"
    # pdcomp.F:168-191: buoyancy's top-level mean is pp3d's.
    top = lambda a: float(np.add.accumulate(np.concatenate(([0.0], a[nk - 1].astype(np.float64).ravel())))[-1]) / (ni * nj)  # noqa: E731
    assert abs(top(ours["pipb"]) - top(d["cpu"]["pp3d"])) <= 4 * np.finfo(np.float32).eps * max(1e-3, abs(top(d["cpu"]["pp3d"])))


# ------------------------------------------------- 2. packaged kernel text


@pytest.fixture(scope="module")
def harness(tmp_path_factory):
    from tools.oracle.pdcomp_open_oracle import KernelHarness

    return KernelHarness(CUDA_PDCOMP_OPEN_SOURCE, tmp_path_factory.mktemp("pdcomp-gxx"))


@requires_gxx
@pytest.mark.parametrize("order", ["index", "shuffled"])
@pytest.mark.parametrize("name", sorted(DECKS))
@pytest.mark.parametrize("imoist", [1, 0])
def test_every_pdcomp_kernel_is_its_cpu_reference_word_for_word(harness, name, order, imoist):
    d = make_deck(**DECKS[name], imoist=imoist)
    s, c, co = d["sel"], d["cpu"], d["coeff"]
    ni, nj, nk = s.ni, s.nj, s.nk
    sh = padded_shapes(ni, nj, nk)
    const = cm1_constants()
    rdx = f32(f32(1.0) / f32(s.dx)); rdy = f32(f32(1.0) / f32(s.dy)); rdz = f32(f32(1.0) / f32(s.dz))
    rdt = f32(f32(1.0) / f32(s.dt))

    wten = np.zeros(sh["w"], dtype=f32)
    # base.F's qc0/qi0: exact zero on these decks' base state (pdcomp.F:104).
    zero = np.zeros_like(np.asarray(c["qv0"], dtype=f32))
    harness.call("cm1_pdcomp_buoyancy_v1", [c["th3d"], c["q3d"], c["th0"], c["qv0"], zero, zero.copy(), c["c1"], c["c2"],
                 wten, ni, nj, nk, s.imoist, s.iice, s.nqv, s.nql1, s.nql2, s.nqs1, s.nqs2, const.g, const.repsm1],
                 count=(nk - 1) * nj * ni, order=order)
    assert_bitwise(wten, cpu_buoyancy_wten(s, th3d=c["th3d"], q3d=c["q3d"], th0=c["th0"], qv0=c["qv0"], c1=c["c1"], c2=c["c2"]), "buoyancy")
    if s.imoist == 1 and order == "index":
        # A base state with cloud (base.F:347 qc0 = qt_mb - qv0; isnd = 10's qi0).
        qv0 = np.asarray(c["qv0"], dtype=f32)
        qc0 = (f32(0.020) - qv0).astype(f32)
        qi0 = np.full_like(qv0, f32(0.005))
        cloudy = np.zeros(sh["w"], dtype=f32)
        harness.call("cm1_pdcomp_buoyancy_v1", [c["th3d"], c["q3d"], c["th0"], c["qv0"], qc0, qi0, c["c1"], c["c2"],
                     cloudy, ni, nj, nk, s.imoist, s.iice, s.nqv, s.nql1, s.nql2, s.nqs1, s.nqs2, const.g,
                     const.repsm1], count=(nk - 1) * nj * ni, order=order)
        assert_bitwise(cloudy, cpu_buoyancy_wten(s, th3d=c["th3d"], q3d=c["q3d"], th0=c["th0"], qv0=c["qv0"],
                                                 c1=c["c1"], c2=c["c2"], qc0=qc0, qi0=qi0), "buoyancy with qc0/qi0")
        assert cloudy.tobytes() != wten.tobytes()

    uten, vten = np.zeros(sh["u"], dtype=f32), np.zeros(sh["v"], dtype=f32)
    harness.call("cm1_pdcomp_linear_v1", [c["w3d"], c["rf0c"], c["mfc"], c["rr0c"], c["u0"], c["v0"], uten, vten, ni, nj, nk, rdz],
                 count=nk * (nj + 1) * (ni + 1), order=order)
    ru, rv = cpu_linear_uv(s, w3d=c["w3d"], rf0c=c["rf0c"], mfc=c["mfc"], rr0c=c["rr0c"], u0=c["u0"], v0=c["v0"])
    assert_bitwise(uten, ru, "linear uten"); assert_bitwise(vten, rv, "linear vten")

    cu, cv = np.zeros(sh["u"], dtype=f32), np.zeros(sh["v"], dtype=f32)
    harness.call("cm1_pdcomp_coriolis_v1", [c["u3d"], c["v3d"], c["f2d"], cu, cv, ni, nj, nk],
                 count=nk * (nj + 1) * (ni + 1), order=order)
    ru, rv = cpu_coriolis_uv(s, u3d=c["u3d"], v3d=c["v3d"], f2d=c["f2d"])
    assert_bitwise(cu, ru, "Coriolis uten"); assert_bitwise(cv, rv, "Coriolis vten")

    fl = cpu_nonlinear_fluxes(s, u3d=c["u3d"], v3d=c["v3d"], w3d=c["w3d"], rho0c=c["rho0c"], rf0c=c["rf0c"],
                              mhc=c["mhc"], u0=c["u0"], v0=c["v0"], uh=c["uh"], vh=c["vh"])
    rru, u1 = np.zeros(sh["u"], dtype=f32), np.zeros(sh["u"], dtype=f32)
    harness.call("cm1_pdcomp_nl_uv_v1", [c["u3d"], c["rho0c"], c["u0"], rru, u1, sh["u"][1], sh["u"][2], nk],
                 count=nk * sh["u"][1] * sh["u"][2], order=order)
    rrv, v1 = np.zeros(sh["v"], dtype=f32), np.zeros(sh["v"], dtype=f32)
    harness.call("cm1_pdcomp_nl_uv_v1", [c["v3d"], c["rho0c"], c["v0"], rrv, v1, sh["v"][1], sh["v"][2], nk],
                 count=nk * sh["v"][1] * sh["v"][2], order=order)
    rrw = np.zeros(sh["w"], dtype=f32)
    harness.call("cm1_pdcomp_nl_w_v1", [c["w3d"], c["rf0c"], rrw, ni, nj, nk], count=(nk - 1) * (nj + 2) * (ni + 2), order=order)
    for got, key in ((rru, "rru"), (u1, "uten1"), (rrv, "rrv"), (v1, "vten1"), (rrw, "rrw")):
        assert_bitwise(got, fl[key], key)
    divx = np.zeros((nk, nj, ni), dtype=f32)
    harness.call("cm1_pdcomp_divx_v1", [rru, rrv, rrw, c["uh"], c["vh"], c["mhc"], divx, ni, nj, nk, rdx, rdy, rdz, const.smeps],
                 count=nk * nj * ni, order=order)
    assert_bitwise(divx, fl["divx"], "divx")

    rhs = np.zeros((nk, co.jpe, co.ipe), dtype=f32)
    harness.call("cm1_pdcomp_poiss_rhs_v1", [cu, cv, wten, c["rho0c"], c["rf0"], c["mh"], co.d1, c["uh"], c["vh"], rhs,
                 ni, nj, nk, co.ipe, co.jpe, s.imirror, s.jmirror, rdx, rdy, rdz, rdt], count=nk * nj * ni, order=order)
    ref_rhs = cpu_poiss_rhs(s, co, uten=cu, vten=cv, wten=wten, rho0c=c["rho0c"], rf0=c["rf0"], mh=c["mh"], uh=c["uh"], vh=c["vh"])
    assert_bitwise(rhs, ref_rhs, "poiss rhs with mirror (poiss.F:62-114)")

    physical_real = np.random.default_rng(9).standard_normal((nk, co.jpe, co.ipe)).astype(f32)
    p = np.zeros((nk, nj, ni), dtype=f32)
    harness.call("cm1_pdcomp_poiss_store_v1", [physical_real, co.d2, p, ni, nj, nk, co.ipe, co.jpe], count=nk * nj * ni, order=order)
    assert_bitwise(p, (physical_real[:, :nj, :ni] * co.d2[:, None, None]).astype(f32), "store (poiss.F:207-211)")

    for use in (1, 0):
        tem = np.zeros(1, dtype=f32)
        harness.call("cm1_pdcomp_gauge_v1", [p, c["pp3d"], tem, ni, nj, nk, use], count=None)
        expected = cpu_gauge(s, p, c["pp3d"] if use else None)
        got = p.copy()
        harness.call("cm1_pdcomp_add_v1", [got, tem[0], got.size], count=got.size, order=order)
        assert_bitwise(got, expected, f"gauge use_pp3d={use}")

    g = np.zeros((nk + 1, nj, ni), dtype=f32)
    harness.call("cm1_pdcomp_pgrad_v1", [p, c["c1"], c["c2"], co.thv0, c["mf"], g, ni, nj, nk, const.cp, rdz],
                 count=(nk - 1) * nj * ni, order=order)
    assert_bitwise(g, cpu_pgrad(s, p, c1=c["c1"], c2=c["c2"], thv0=co.thv0, mf=c["mf"]), "pgrad (writeout.F:8497-8505)")


@requires_gxx
def test_the_lanes_advection_kernel_text_on_pdcomp_buffers_is_the_cpu_nonlinear_forcing(tmp_path):
    """pdcomp.F:317-326 through this lane's advu/advv/advw text, not a copy."""

    from cm1core.cuda_momentum_advection5 import CUDA_MOMENTUM_ADVECTION5_SOURCE, momentum_open_boundary_code
    from tools.oracle.pdcomp_open_oracle import KernelHarness

    adv = KernelHarness(CUDA_MOMENTUM_ADVECTION5_SOURCE, tmp_path)
    d = make_deck(ni=9, nj=8, nk=9, seed=5)
    s, c = d["sel"], d["cpu"]
    obc = momentum_open_boundary_code((s.wbc, s.ebc, s.sbc, s.nbc))
    ni, nj, nk = s.ni, s.nj, s.nk
    sh = padded_shapes(ni, nj, nk)
    m = d["metrics"]
    fl = cpu_nonlinear_fluxes(s, u3d=c["u3d"], v3d=c["v3d"], w3d=c["w3d"], rho0c=c["rho0c"], rf0c=c["rf0c"],
                              mhc=c["mhc"], u0=c["u0"], v0=c["v0"], uh=c["uh"], vh=c["vh"])
    ref = {"u3d": fl["uten1"], "v3d": fl["vten1"], "w3d": c["w3d"], "rru": fl["rru"], "rrv": fl["rrv"], "rrw": fl["rrw"],
           "uten": np.zeros(sh["u"], dtype=f32), "vten": np.zeros(sh["v"], dtype=f32), "wten": np.zeros(sh["w"], dtype=f32),
           "uh": c["uh"], "vh": c["vh"]}
    cpu_uniform_advect(m, s)(ref, fl["divx"])
    dev = {k: np.zeros(sh[st], dtype=f32) for k, st in (("uten", "u"), ("vten", "v"), ("wten", "w"))}
    dx, dy, dz = f32(s.dx), f32(s.dy), f32(s.dz)
    adv.call("cm1_mom5_advu_v1", [fl["uten1"], fl["rru"], fl["rrv"], fl["rrw"], fl["divx"], m["rho0_3d"], m["c2"], m["gzu"], c["vh"], dev["uten"], ni, nj, nk, 3, dx, dy, dz, obc], count=None)
    adv.call("cm1_mom5_advv_v1", [fl["vten1"], fl["rru"], fl["rrv"], fl["rrw"], fl["divx"], m["rho0_3d"], m["c2"], m["gzv"], c["uh"], dev["vten"], ni, nj, nk, 3, dx, dy, dz, obc], count=None)
    adv.call("cm1_mom5_advw_v1", [c["w3d"], fl["rru"], fl["rrv"], fl["rrw"], fl["divx"], m["c2"], m["rrf0"], m["gz"], c["uh"], c["vh"], dev["wten"], ni, nj, nk, 3, dx, dy, dz, obc], count=None)
    for key in ("uten", "vten", "wten"):
        assert_bitwise(dev[key], ref[key], f"nonlinear {key}")
    assert float(np.max(np.abs(ref["uten"]))) > 0.0


# --------------------------------------------- 3. analytic, CM1's lateral BC


def _operator(coeff, sel, p):
    """init3d.F's discrete operator on a (nk, nj, ni) field with reflecting ghosts."""

    nk, nj, ni = p.shape
    q = p.astype(np.float64)
    dx2 = float(f32(sel.dx) * f32(sel.dx))
    g = np.pad(q, ((0, 0), (1, 1), (1, 1)), mode="edge")   # p(0)=p(1): even reflection
    lap = (g[:, 1:-1, 2:] + g[:, 1:-1, :-2] + g[:, 2:, 1:-1] + g[:, :-2, 1:-1] - 4.0 * q) / dx2
    cfa, cfc = coeff.cfa.astype(np.float64), coeff.cfc.astype(np.float64)
    vert = -(cfa + cfc)[:, None, None] * q
    vert[1:] += cfa[1:, None, None] * q[:-1]
    vert[:-1] += cfc[:-1, None, None] * q[1:]
    return lap + vert


@pytest.mark.parametrize("name", sorted(DECKS))
@pytest.mark.parametrize("modes", [(1, 0), (0, 2), (3, 1), (7, 5)])
def test_a_neumann_cosine_eigenfunction_is_recovered(name, modes):
    d = make_deck(**DECKS[name])
    s, co = d["sel"], d["coeff"]
    ni, nj, nk = s.ni, s.nj, s.nk
    m, n = modes
    i = np.arange(ni) + 0.5
    j = np.arange(nj) + 0.5
    zk = np.sin(np.pi * (np.arange(nk) + 1.0) / (nk + 1.0)) + 0.3
    p = (np.cos(np.pi * m * j / nj)[:, None] * np.cos(np.pi * n * i / ni)[None, :])[None] * zk[:, None, None]
    rhs = _operator(co, s, p).astype(f32)
    full = np.zeros((nk, co.jpe, co.ipe), dtype=f32)
    full[:, :nj, :ni] = rhs
    full[:, :nj, ni:] = rhs[:, :, ::-1]
    full[:, nj:, :ni] = rhs[:, ::-1, :]
    full[:, nj:, ni:] = rhs[:, ::-1, ::-1]
    got = cpu_poiss_invert(s, co, full).astype(np.float64)
    err = float(np.max(np.abs(got - p))) / float(np.max(np.abs(p)))
    assert err < 5.0e-4, f"mode {modes}: relative error {err:.2e}"


@pytest.mark.parametrize("name", sorted(DECKS))
def test_a_general_solution_obeys_reflecting_not_periodic_boundaries(name):
    d = make_deck(**DECKS[name])
    s, co = d["sel"], d["coeff"]
    ni, nj, nk = s.ni, s.nj, s.nk
    rng = np.random.default_rng(11)
    src = np.zeros((nk, nj, ni))
    src[:, 1:3, 0:2] = rng.standard_normal((nk, 2, 2))      # a storm against the west face
    src -= src.mean(axis=(1, 2), keepdims=True)
    src *= 1.0e-6
    full = np.zeros((nk, co.jpe, co.ipe), dtype=f32)
    live = src.astype(f32)
    full[:, :nj, :ni] = live
    full[:, :nj, ni:] = live[:, :, ::-1]
    full[:, nj:, :ni] = live[:, ::-1, :]
    full[:, nj:, ni:] = live[:, ::-1, ::-1]
    p = cpu_poiss_invert(s, co, full)
    res = (_operator(co, s, p) - src)[: nk - 1]
    assert float(np.max(np.abs(res))) < 2.0e-3 * float(np.max(np.abs(src)))
    # the periodic ghost (p(0)=p(ni)) is not what this solution satisfies
    q = p.astype(np.float64)
    gp = np.pad(q, ((0, 0), (1, 1), (1, 1)), mode="wrap")
    dx2 = float(f32(s.dx) * f32(s.dx))
    lap = (gp[:, 1:-1, 2:] + gp[:, 1:-1, :-2] + gp[:, 2:, 1:-1] + gp[:, :-2, 1:-1] - 4.0 * q) / dx2
    vertical = _operator(co, s, q) - _horizontal_reflecting(q, dx2)
    periodic = lap + vertical - src
    assert float(np.max(np.abs(periodic[: nk - 1, :, [0, ni - 1]]))) > 50.0 * float(np.max(np.abs(res)))


def _horizontal_reflecting(q, dx2):
    g = np.pad(q, ((0, 0), (1, 1), (1, 1)), mode="edge")
    return (g[:, 1:-1, 2:] + g[:, 1:-1, :-2] + g[:, 2:, 1:-1] + g[:, :-2, 1:-1] - 4.0 * q) / dx2


# ------------------------------------------------------ 4. diagnostic purity


@pytest.mark.parametrize("name", sorted(DECKS))
def test_the_parts_sum_to_the_inversion_of_the_summed_forcing(name):
    d = make_deck(**DECKS[name])
    s, c, co = d["sel"], d["cpu"], d["coeff"]
    parts = _cpu_run(d)
    wb = cpu_buoyancy_wten(s, th3d=c["th3d"], q3d=c["q3d"], th0=c["th0"], qv0=c["qv0"], c1=c["c1"], c2=c["c2"])
    ul, vl = cpu_linear_uv(s, w3d=c["w3d"], rf0c=c["rf0c"], mfc=c["mfc"], rr0c=c["rr0c"], u0=c["u0"], v0=c["v0"])
    uc, vc = cpu_coriolis_uv(s, u3d=c["u3d"], v3d=c["v3d"], f2d=c["f2d"])
    fl = cpu_nonlinear_fluxes(s, u3d=c["u3d"], v3d=c["v3d"], w3d=c["w3d"], rho0c=c["rho0c"], rf0c=c["rf0c"],
                              mhc=c["mhc"], u0=c["u0"], v0=c["v0"], uh=c["uh"], vh=c["vh"])
    sh = padded_shapes(s.ni, s.nj, s.nk)
    a = {"u3d": fl["uten1"], "v3d": fl["vten1"], "w3d": c["w3d"], "rru": fl["rru"], "rrv": fl["rrv"], "rrw": fl["rrw"],
         "uten": np.zeros(sh["u"], dtype=f32), "vten": np.zeros(sh["v"], dtype=f32), "wten": np.zeros(sh["w"], dtype=f32),
         "uh": c["uh"], "vh": c["vh"]}
    cpu_uniform_advect(d["metrics"], s)(a, fl["divx"])
    total = cpu_gauge(s, cpu_poiss_invert(s, co, cpu_poiss_rhs(
        s, co, uten=(ul + a["uten"] + uc).astype(f32), vten=(vl + a["vten"] + vc).astype(f32),
        wten=(wb + a["wten"]).astype(f32), rho0c=c["rho0c"], rf0=c["rf0"], mh=c["mh"], uh=c["uh"], vh=c["vh"])), c["pp3d"])
    summed = sum(parts[n].astype(np.float64) for n in s.names)
    scale = float(np.max(np.abs(total)))
    assert float(np.max(np.abs(summed - total))) < 1.0e-5 * scale
    # each gradient is the vertical difference of its own part
    for n, w in zip(s.names, s.w_names):
        assert parts[w].shape == (s.nk + 1, s.nj, s.ni) and not np.any(parts[w][[0, s.nk]])


def test_no_model_carrier_reaches_a_writable_kernel_parameter():
    from cm1core.cuda_momentum_advection5 import CUDA_MOMENTUM_ADVECTION5_SOURCE
    from cm1core.cuda_gm1_child_stretched_momentum import _stretched_source
    import re

    model = {"th3d", "q3d", "th0", "qv0", "c1", "c2", "w3d", "rf0c", "mfc", "rr0c", "u0", "v0", "u3d", "v3d",
             "f2d", "s3d", "s0", "rho0c", "uh", "vh", "mhc", "rf0", "mh", "d1", "d2", "thv0c", "mf", "pp3d", "physical_real"}
    sig = re.compile(r'extern "C" __global__ void (\w+)\(([^)]*)\)')
    seen = 0
    for name, params in sig.findall(CUDA_PDCOMP_OPEN_SOURCE):
        for raw in params.split(","):
            if "*" in raw:
                ident = raw.split("*")[-1].strip()
                if ident in model:
                    assert raw.strip().startswith("const"), f"{name}: {ident} is writable"
                    seen += 1
    assert seen > 30
    # the reused advection kernels read the model w3d through a const pointer
    for source in (CUDA_MOMENTUM_ADVECTION5_SOURCE, _stretched_source()):
        for name, params in sig.findall(source):
            if name.endswith("advw_v1"):
                assert params.split(",")[0].strip().startswith("const float*")


def test_the_cpu_decomposition_changes_no_input_byte():
    d = make_deck(stretched=True, seed=7)
    before = {k: hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()
              for k, v in d["cpu"].items() if isinstance(v, np.ndarray)}
    out = _cpu_run(d)
    after = {k: hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()
             for k, v in d["cpu"].items() if isinstance(v, np.ndarray)}
    assert before == after
    assert set(out) == set(PDCOMP_SCALAR_NAMES) | set(PDCOMP_W_NAMES)


# ----------------------------------------------------------------- refusals


def test_the_admitted_selection_and_each_refusal_by_name():
    base = dict(ni=8, nj=8, nk=8, dx=250.0, dy=250.0, dz=200.0, dt=1.0)
    assert pdcomp_selection_defect(PdcompSelection(**base)) is None
    assert "grid.terrain" in pdcomp_selection_defect(PdcompSelection(**base, terrain=True))
    assert "dx_m" in pdcomp_selection_defect(PdcompSelection(**{**base, "dy": 300.0}))
    assert "boundaries" in pdcomp_selection_defect(PdcompSelection(**base, wbc=1, ebc=1))
    assert "boundaries" in pdcomp_selection_defect(PdcompSelection(**base, sbc=4, nbc=4))
    # A free-slip wall keeps the mirrored inversion, and advu/advv/advw now
    # take the run's own family, so it is admitted.
    assert pdcomp_selection_defect(PdcompSelection(**base, wbc=3, ebc=3, sbc=2, nbc=3)) is None
    assert "psolver" in pdcomp_selection_defect(PdcompSelection(**base, psolver=5))
    open_ = PdcompSelection(**base)
    assert (open_.imirror, open_.jmirror, open_.ipe, open_.jpe) == (1, 1, 16, 16)


# ------------------------------------------------ 5. device (GPU nodes only)


def _device_test_requested() -> bool:
    # A GPU is taken only on purpose, never because CuPy happens to import.
    import os

    return os.environ.get("GM1_PDCOMP_DEVICE_TEST") == "1"


@pytest.mark.skipif(not _device_test_requested(),
                    reason="device differential: set GM1_PDCOMP_DEVICE_TEST=1 on a GPU node")
@pytest.mark.parametrize("name", sorted(DECKS))
def test_device_pdcomp_is_the_cpu_reference(name):
    """launch_cuda_pdcomp on the device vs cpu_pdcomp: equal to cuFFT round-off."""

    import cupy as cp

    from cm1core.cuda_momentum_advection5 import compile_cuda_momentum_advection5
    from cm1core.cuda_pdcomp_open import device_advect, launch_cuda_pdcomp, prepare_cuda_pdcomp

    d = make_deck(**DECKS[name])
    s, c = d["sel"], d["cpu"]
    host_metrics = dict(d["metrics"], rf0=c["rf0"], mh=c["mh"], mf=c["mf"], c1=c["c1"],
                        rr0=np.ascontiguousarray(np.broadcast_to(c["rr0c"][:, None, None], c["mh"].shape)))
    stream = cp.cuda.Stream()
    dev = {k: cp.asarray(v) for k, v in host_metrics.items()}
    base = {k: cp.asarray(c[k]) for k in ("th0", "qv0", "u0", "v0")}
    plan = prepare_cuda_pdcomp(s, d["coeff"], metrics=dev, base_profiles=base,
                               f2d=cp.asarray(c["f2d"]), stream=stream)
    advect = device_advect(compile_cuda_momentum_advection5(), dev, s, vertical=None, stream=stream)
    out = launch_cuda_pdcomp(
        plan, u3d=cp.asarray(c["u3d"]), v3d=cp.asarray(c["v3d"]), w3d=cp.asarray(c["w3d"]),
        th3d=cp.asarray(c["th3d"]), pp3d=cp.asarray(c["pp3d"]), q3d=cp.asarray(c["q3d"]),
        uh=cp.asarray(c["uh"]), vh=cp.asarray(c["vh"]), advect=advect, stream=stream)
    ref = _cpu_run(d)
    assert set(out) == set(ref)
    for key, value in ref.items():
        got = cp.asnumpy(out[key]).astype(np.float64)
        scale = max(float(np.max(np.abs(value))), 1.0e-30)
        assert float(np.max(np.abs(got - value))) / scale < 2.0e-6, key
