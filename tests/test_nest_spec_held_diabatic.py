"""The specified rows get back the eqtset=2 held diabatic rate the removal takes out.

CM1 adds the previous step's microphysics rates to the held tendencies
(solve1.F:1748-1751) and subtracts ``dt*rate`` after the final RK stage
(solve2.F:1561-1576).  A nest child's specified rows advance by the boundary
tendency alone (WRF's spec_bdy_scalar), so they never received the rate; before
this fix the removal still subtracted it and, on condensing outflow rows, built
0.4-1 kg/kg hydrometeor columns in every nested run.  These tests use numpy in
place of cupy: the method is array arithmetic on views, nothing device-only.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")

from cm1core import nest_live  # noqa: E402

NZ, NY, NX, SZ = 4, 9, 11, 1
DT = 0.8


def _forcing(spec_zone=SZ):
    forcing = object.__new__(nest_live.NestLateralForcing)
    forcing.cp = np
    forcing.spec_zone = spec_zone
    forcing.carriers = nest_live._carriers(10)
    return forcing


def _dynamics(seed=3):
    rng = np.random.default_rng(seed)
    shape = (NZ, NY, NX)
    fields = {
        "pp3d": rng.standard_normal(shape).astype(np.float32) * np.float32(1e-3),
        "th3d": rng.standard_normal(shape).astype(np.float32),
        "q3d": np.abs(rng.standard_normal((10, *shape))).astype(np.float32) * np.float32(1e-3),
    }
    for name in ("qpten", "qtten", "qvten", "qcten"):
        fields[name] = rng.standard_normal(shape).astype(np.float32) * np.float32(1e-3)
    return SimpleNamespace(
        fields=fields,
        geometry=SimpleNamespace(nx=NX, ny=NY, nz=NZ),
        config=SimpleNamespace(dt_s=DT),
    )


def _spec_mask(spec_zone):
    mask = np.zeros((NY, NX), bool)
    mask[:spec_zone, :] = mask[-spec_zone:, :] = True
    mask[:, :spec_zone] = mask[:, -spec_zone:] = True
    return mask


@pytest.mark.parametrize("spec_zone", [1, 2])
def test_only_the_specified_rows_get_dt_times_the_rate_once(spec_zone):
    forcing, dyn = _forcing(spec_zone), _dynamics()
    before = {k: np.array(v, copy=True) for k, v in dyn.fields.items()}
    forcing.restore_held_diabatic(dyn)
    mask = _spec_mask(spec_zone)
    pairs = (("pp3d", None, "qpten"), ("th3d", None, "qtten"), ("q3d", 0, "qvten"), ("q3d", 1, "qcten"))
    dt = np.float32(DT)
    for name, row, rate in pairs:
        old = before[name] if row is None else before[name][row]
        new = dyn.fields[name] if row is None else dyn.fields[name][row]
        want = np.where(mask, old + dt * before[rate], old)
        assert np.array_equal(new, want), name
    # every other moisture row is untouched
    assert np.array_equal(dyn.fields["q3d"][2:], before["q3d"][2:])


def test_cm1s_removal_then_returns_the_specified_value_to_rounding():
    forcing, dyn = _forcing(), _dynamics(seed=11)
    spec = np.array(dyn.fields["q3d"][1], copy=True)
    forcing.restore_held_diabatic(dyn)
    dt = np.float32(DT)
    removed = dyn.fields["q3d"][1] - dt * dyn.fields["qcten"]      # solve2.F:1575
    mask = _spec_mask(SZ)
    assert np.allclose(removed[:, mask], spec[:, mask], rtol=0, atol=4 * np.spacing(np.float32(1e-2)))


def test_the_restore_runs_after_the_final_stage_only(monkeypatch):
    forcing = _forcing()
    forcing.tables = {c.name: {s: (None, None) for s in nest_live.SIDES} for c in forcing.carriers}
    forcing.module, forcing.width, forcing.stream = None, 5, None
    forcing.scalar_specifications = 0
    monkeypatch.setattr(nest_live, "launch_spec_apply", lambda *a, **k: None)
    calls = []
    monkeypatch.setattr(nest_live.NestLateralForcing, "restore_held_diabatic",
                        lambda self, d: calls.append(d))

    class _Stream:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    forcing.stream = _Stream()
    dyn = _dynamics()
    dyn.fields["tha"] = dyn.fields["th3d"]
    dyn.fields["ppi"] = dyn.fields["pp3d"]
    dyn.fields["qa"] = dyn.fields["q3d"]
    for nrk in (1, 2):
        forcing.specify_scalars(dyn, nrk)
    assert calls == []
    forcing.specify_scalars(dyn, 3)
    assert calls == [dyn]


def test_a_rate_on_another_lattice_is_refused():
    forcing, dyn = _forcing(), _dynamics()
    dyn.fields["qtten"] = dyn.fields["qtten"][:, :-1, :]
    with pytest.raises(RuntimeError, match="does not share"):
        forcing.restore_held_diabatic(dyn)
