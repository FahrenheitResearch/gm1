"""The synthetic vortices of ``tools/criterion_v2/synthetic_core_radius.py`` are the textbook ones.

The diagnosis of criterion v1's core-radius rule rests on these profiles and on
v1's own host sample; this pins the closed forms (Burgers-Rott, Sullivan) and
checks that a resolved vortex through ``reference_sample`` gives the continuous
half-peak radius, and an unresolved one the one-corner floor.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _tool():
    name = "criterion_v2_synthetic_core_radius"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "tools" / "criterion_v2" / "synthetic_core_radius.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_burgers_rott_closed_forms() -> None:
    m = _tool()
    # RMW/r0 solves exp(x^2) = 1 + 2 x^2; half-peak radius is r0 sqrt(ln 2).
    assert abs(m.BURGERS_RMW_OVER_R0 - 1.1209064) < 1e-6
    ref = m.analytic_reference(m.Flow("burgers", 80.0, 60.0, False))
    assert abs(ref["half_peak_equiv_radius_over_rmw"] - math.sqrt(math.log(2.0)) / 1.1209064) < 1e-4
    v = m.Vortex("burgers", 80.0, 60.0)
    assert abs(float(v.vt(np.array([60.0]))[0]) - 80.0) < 1e-9
    assert abs(v.zeta_peak - float(v.zeta(np.array([0.0]))[0])) < 1e-12


def test_sullivan_closed_forms() -> None:
    m = _tool()
    t = m.sullivan_table()
    assert abs(t.H_inf - 37.905) < 2e-3          # Sullivan (1959)
    assert abs(t.x_ring - 2.8214) < 2e-3         # root of 3(1 - e^-x) = x
    assert abs(1.0 / t.hp_ring - 0.126) < 1e-3   # axis / ring vorticity
    v = m.Vortex("sullivan", 60.0, 50.0)
    r = np.linspace(1.0, 200.0, 20000)
    vt = v.vt(r)
    assert abs(float(r[np.argmax(vt)]) - 50.0) < 0.05 and abs(float(vt.max()) - 60.0) < 1e-3


def test_resolved_and_unresolved_samples_through_v1() -> None:
    m = _tool()
    c = m.CRITERION
    # Resolved: RMW 75 m on an 8 m grid gives the continuous half-peak radius.
    flow = m.Flow("burgers", 100.0, 75.0, False)
    s = m.sample(flow, m.make_plan(8.0), 1200.0, 1200.0, 0.0)
    r_half = m.analytic_reference(flow)["half_peak_equiv_radius_m"]
    assert abs(s["r_core"] - r_half) / r_half < 0.03
    assert s["r_core"] >= c["core_radius_min_m"]
    # Unresolved: RMW 20 m on a 100 m grid is one corner, the floor dx/sqrt(pi) > 40 m.
    s = m.sample(m.Flow("burgers", 100.0, 20.0, False), m.make_plan(100.0), 1200.0, 1200.0, 0.0)
    assert s["ncore"] == 1 and abs(s["r_core"] - 100.0 / math.sqrt(math.pi)) < 1e-9
    # The corner stencil is the midpoint circulation of a dx box: 4 v(dx/2)/dx.
    v = m.Vortex("burgers", 100.0, 20.0)
    assert abs(s["zeta_max"] - 4.0 * float(v.vt(np.array([50.0]))[0]) / 100.0) < 1e-4
