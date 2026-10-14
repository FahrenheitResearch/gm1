"""Precipitable water and the liquid water paths on stretched levels.

CM1 r21.1 sums ``rho*q*dz*rmh(k)`` over the column (``src/writeout.F:2799``,
``:2836``, ``:2874``): ``dz*rmh(k)`` is the layer's own depth.  On even levels
``rmh`` is one.  GM1 summed ``rho*q*dz`` on every column, so on stretched
levels ``pwat``, ``lwp`` and ``cwp`` were scaled by the wrong depths (42 % too
much precipitable water on the dry oracle's stretched column).  These tests
hold the three sums to CM1's statement on a stretched column and to the
earlier expression, word for word, on even levels.
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cm1core.cartesian_output_products import CartesianOutputProducts
from cm1core.cm1_vertical_grid import build_vertical_levels, uniform_vertical_levels

F32 = np.float32


def _products(levels, nj: int = 3, ni: int = 4):
    nk = levels.nk
    rng = np.random.default_rng(1012)
    rho = (1.1 * np.exp(-np.arange(nk)[:, None, None] / 9.0) * (1.0 + 0.01 * rng.random((nk, nj, ni)))).astype(F32)
    q = (0.01 * rng.random((3, nk, nj, ni))).astype(F32)
    fields = {"rho": rho, "q3d": q}
    grid = SimpleNamespace(dz_m=levels.dz_reference_m, nz=nk, stretch_z=levels.stretch_z, levels=levels)
    selection = SimpleNamespace(on=lambda key: key in ("output_pwat", "output_lwp"))
    products = object.__new__(CartesianOutputProducts)
    products.cp = np
    products.grid = grid
    products.selection = selection
    products.stream = contextlib.nullcontext()
    products.runtime = SimpleNamespace(dynamics=SimpleNamespace(fields=fields, base_profiles={}))
    return products, rho, q


def test_on_stretched_levels_each_layer_counts_with_its_own_depth() -> None:
    levels = build_vertical_levels(stretch_z=1, nk=40, dz_m=200.0, ztop_m=8000.0, str_bot_m=1000.0,
                                   str_top_m=5000.0, dz_bot_m=100.0, dz_top_m=300.0)
    products, rho, q = _products(levels)
    out = products._derived()
    dz = F32(levels.dz_reference_m)
    rmh = np.asarray(levels.rmh[1 : levels.nk + 1], dtype=F32)[:, None, None]
    # dz*rmh(k) is the depth of layer k: 100 m at the ground, 300 m aloft, never the 200 m reference.
    depths = np.diff(np.asarray(levels.face_heights_m))
    assert np.allclose((dz * rmh)[:, 0, 0], depths, rtol=1e-5)
    assert depths.min() < 101.0 and depths.max() > 299.0
    want = {
        "pwat": (rho * q[0] * dz * rmh).sum(axis=0) * F32(1.0e-3),
        "lwp": (rho * (q[1] + q[2]) * dz * rmh).sum(axis=0),
        "cwp": (rho * q[1] * dz * rmh).sum(axis=0),
    }
    for name, value in want.items():
        assert out[name].dtype == np.float32
        assert np.array_equal(out[name].view(np.uint32), value.astype(F32).view(np.uint32)), name
    # In double precision it is the integral of rho*qv over the real layer depths, in metres of water.
    exact = (rho.astype(np.float64) * q[0] * depths[:, None, None]).sum(axis=0) / 1000.0
    assert np.allclose(out["pwat"], exact, rtol=2e-6)
    # The earlier sum, with the reference dz for every layer, is not that integral.
    earlier = (rho * q[0] * dz).sum(axis=0) * F32(1.0e-3)
    assert np.abs(earlier / exact - 1.0).min() > 0.05


def test_on_even_levels_the_sums_are_the_words_they_always_were() -> None:
    levels = uniform_vertical_levels(40, 500.0)
    products, rho, q = _products(levels)
    out = products._derived()
    dz = F32(500.0)
    earlier = {
        "pwat": ((rho * q[0] * dz).sum(axis=0) * F32(1.0e-3)).astype(F32),
        "lwp": (rho * (q[1] + q[2]) * dz).sum(axis=0).astype(F32),
        "cwp": (rho * q[1] * dz).sum(axis=0).astype(F32),
    }
    for name, value in earlier.items():
        assert np.array_equal(out[name].view(np.uint32), value.view(np.uint32)), name


def test_precipitable_water_on_stretched_levels_matches_cm1_on_the_device() -> None:
    """CM1 r21.1 against GM1 on an RTX 4090, before and after, where the evidence is present."""

    record = Path(__file__).resolve().parents[1] / "evidence" / "oracle" / "stretched-water-path" / "pwat.json"
    if not record.is_file():
        pytest.skip("the development evidence is not distributed")
    cases = json.loads(record.read_text(encoding="utf-8"))["cases"]
    assert min(cases["stretched-column-32"]["before"]) > 0.4
    assert len(cases) == 4
    for name, row in cases.items():
        assert max(row["after"]) < 1.0e-6, name
