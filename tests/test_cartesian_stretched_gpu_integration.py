"""The stretched Cartesian lane, executed on a real CUDA device.

``grid.stretch_z = 1`` became selectable from the Cartesian door without any
test ever integrating a step with it, and the first real run stopped inside the
acoustic stage: the lane prices ``sound.F:659-730``'s six ``tri_*`` column
reports at one word each -- they are 762 MB of pure diagnostic on the
tornado-grade domain and nothing downstream reads them -- while the stretched
acoustic successor was only ever compiled in the variant that stores into them,
and held its caller to the pressure extent it never allocated.  Host admission
could not see that, because the extents the lane allocates and the extents the
kernels require only meet on a device.

So these run.  Every check below launches the real compiled families on the
real device and reads the words back:

* the profile-bound family and the uniform family, given the *same* physical
  column, start from the same words and agree within ``FAMILY_AGREEMENT`` over
  real large steps.  They agreed raw bit for raw bit while the profile family
  bound the terrain texts; it now runs CM1's ``terrain_flag=.false.``
  statements, which the uniform family's terrain-coupled acoustic and
  buoyancy statements (with ``gz = 1``) do not reproduce to the last bit;
* on that column both families bind a declared ``alph``/``kdiv``
  (``sound.F:81-82,756,782``) to the same words, which are not the
  defaults' words;
* the tornado-grade 20 m/15 km column integrates, stays finite and positive,
  keeps its dry-mass ledger, and moves;
* the acoustic image and the column workspace must describe the same extents,
  which is the defect itself; and
* the two compiled acoustic variants integrate to identical words, so the
  compact one this lane runs is the retaining one's arithmetic.

The last two swap the image the acoustic phase *launches* -- on a stretched
run that is ``stretched_compiled["psolver"]``, not ``psolver_compiled`` -- on
both vertical families, and read back from the stage receipt that the swapped
image is the one that ran.  ``tests/test_cartesian_acoustic_image_identity.py``
runs the same two checks through the host stand-in, loading this file's two
helpers from it.  This file imports nothing from the repository outside
``src``, so copying it onto another tree needs no other file.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import pytest


cupy = pytest.importorskip("cupy")
pytest.importorskip("gpuwm")


def _device_available() -> bool:
    try:
        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:  # pragma: no cover - exercised only without a driver
        return False


pytestmark = pytest.mark.skipif(
    not _device_available(),
    reason="the Cartesian large step requires a CUDA device",
)

from cm1core.cartesian_door import (  # noqa: E402
    build_runtime_config,
    require_initializer_selection,
    require_moving_frame,
    selected_capacity_profile,
)
from cm1core.config import load_config  # noqa: E402
from cm1core.gm1_idealized_residency import IdealizedSource  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
CRATE = ROOT / "rust/cm1-state"
TORNADO_TOML = ROOT / "examples/cartesian-stretched-tornado-grade-grid.toml"

# Small enough to compile, allocate and step inside a test, and still a real
# domain: the open lateral boundary reads three columns in from each side and
# the fifth-order stencils reach three more.
NX = NY = 32
STEPS = 4
PROGNOSTIC = ("u3d", "v3d", "w3d", "th3d", "pp3d", "ppi", "rho", "prs", "q3d")


@pytest.fixture(scope="module")
def idealized_library() -> Path:
    """The Rust base-state initializer this runtime opens with ``ctypes``.

    ``param.F:7814-8139``'s coordinate is *passed* to it -- a stretched run
    builds its base state at the generated levels, not at ``0.5*dz`` -- so a
    stale build is not usable here and the crate is built when nothing current
    is present.  ``GM1_CM1_STATE_LIBRARY`` names a library built elsewhere and
    is honoured first, as the host-side tests honour it; otherwise the
    repository's own gitignored target directory is reused so a second run
    costs nothing.
    """

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


def _deck(tmp_path: Path, name: str, replacements: dict[str, str]) -> Path:
    """The shipped tornado-grade deck, shrunk to a testable domain.

    Every clause but the ones named here is the published document's own, so
    what runs is the configuration a user selects.
    """

    lines = TORNADO_TOML.read_text().splitlines()
    pending = dict(replacements)
    out = []
    for line in lines:
        key = line.split("=", 1)[0].strip() if "=" in line and not line.startswith("#") else None
        if key is not None and key in pending:
            out.append(pending.pop(key))
        else:
            out.append(line)
    assert not pending, f"deck clauses never matched: {sorted(pending)}"
    path = tmp_path / name
    path.write_text("\n".join(out) + "\n")
    return path


SMALL = {
    "nx": f"nx = {NX}",
    "ny": f"ny = {NY}",
    "duration_s": "duration_s = 60.0",
    "history_interval_s": "history_interval_s = 30.0",
    "restart_interval_s": "restart_interval_s = 60.0",
    "memory_budget_mib": "memory_budget_mib = 4096",
}

# param.F:7819-7953 with dz_bot = dz_top generates param.F:7789-7811's uniform
# column exactly: nk1 = nint(2000/200) = 10, nk3 = nint(7000/200) = 35, and
# nk2 = 75-45 = 30 satisfies param.F:7867-7877 because 30*200 = 8000-2000.
UNIFORM_COLUMN = {
    **SMALL,
    "dz_m": "dz_m = 200.0",
    "str_bot_m": "str_bot_m = 2000.0",
    "str_top_m": "str_top_m = 8000.0",
    "dz_bot_m": "dz_bot_m = 200.0",
    "dz_top_m": "dz_top_m = 200.0",
}

# The same 200 m column, reached through param.F:7789-7811 instead, so the
# uniform operator family runs.  param.F:2608 makes ztop = dz*nk there, which
# is the 15 km top the stretched deck declares.
UNIFORM_FAMILY = {
    **SMALL,
    "dz_m": "dz_m = 200.0",
    "stretch_z": "stretch_z = 0",
    "ztop_m": "# ztop_m: param.F:2608 derives it from dz*nk on this branch",
    "str_bot_m": "",
    "str_top_m": "",
    "dz_bot_m": "",
    "dz_top_m": "",
}


class _Run:
    """One built runtime, its initial words, and its stepped words."""

    def __init__(self, deck: Path, library: Path, *, steps: int, mutate=None) -> None:
        config = load_config(deck)
        runtime_config = build_runtime_config(config)
        self.grid = runtime_config.grid
        selection, _ = require_initializer_selection(config)
        frame = require_moving_frame(config)
        assert not selection.reads_input_sounding, "base.F:363-470 reads no file"
        stream = cupy.cuda.Stream(non_blocking=True)
        with IdealizedSource(
            library, b"", self.grid, selection=selection, moving_frame=frame
        ) as source:
            with stream:
                from cm1core.cuda_cartesian_runtime import CudaCartesianRuntime

                runtime = CudaCartesianRuntime(
                    source,
                    runtime_config,
                    cupy=cupy,
                    stream=stream,
                    capacity_profile=selected_capacity_profile(config),
                )
                self.installed = None if mutate is None else mutate(runtime)
                self.initial = self._read(runtime)
                self.mass_initial = runtime.dynamics.mass1_f64
                for _ in range(steps):
                    runtime.step()
                runtime.synchronize()
                self.final = self._read(runtime)
                self.mass_final = runtime.dynamics._initial_dry_mass()
                self.checks = runtime.field_checks()
                self.receipts = {
                    name: dict(value)
                    for name, value in runtime.dynamics.last_receipts.items()
                }

    @staticmethod
    def _read(runtime) -> dict[str, np.ndarray]:
        return {
            name: np.ascontiguousarray(cupy.asnumpy(runtime.dynamics.fields[name]))
            for name in PROGNOSTIC
        }



#: How closely the two families must agree on one physical column, as a
#: fraction of how far the uniform family moved each field in the run.  They
#: are no longer one set of words: the profile family runs CM1's
#: ``terrain_flag=.false.`` statements (on this column ``mh = mf = 1``, so
#: exactly CM1's ``stretch_z = 0`` words), while the uniform family still runs
#: the terrain-coupled acoustic statements with ``gz = 1``
#: (``sound.F:147-183`` against the flat ``:119-143``) and the terrain buoyancy
#: term (``solve2.F:971-1008`` with ``gxu = 0``).  Measured on an RTX 4090
#: over 4 steps: the largest ratio was 0.0097 (``u3d``), 0.0058 with Morrison
#: (``prs``), 0.0025 with declared ``alph``/``kdiv``; a declared-versus-default
#: ``alph``/``kdiv`` pair differs by 68x the movement.  The bound is twice the
#: largest measured ratio.
FAMILY_AGREEMENT = 0.02


def _families_agree(stretched: "_Run", uniform: "_Run", label: str) -> dict[str, float]:
    ratios: dict[str, float] = {}
    for name in PROGNOSTIC:
        assert stretched.initial[name].tobytes() == uniform.initial[name].tobytes(), (
            f"the two lanes did not start from the same base state in {name}"
        )
        final = uniform.final[name].astype(np.float64)
        moved = float(np.abs(final - uniform.initial[name].astype(np.float64)).max())
        gap = float(np.abs(stretched.final[name].astype(np.float64) - final).max())
        assert np.all(np.isfinite(stretched.final[name])), name
        if moved == 0.0:
            assert gap == 0.0, f"{name}: {label}: the uniform family never moved it, the profile family did"
            continue
        ratios[name] = gap / moved
        assert gap <= FAMILY_AGREEMENT * moved, (
            f"{name} differs by {gap:.3e} after {STEPS} large steps ({label}), "
            f"{gap / moved:.4f} of the {moved:.3e} the run moved it"
        )
    return ratios


def test_the_profile_family_reproduces_the_uniform_family_on_a_uniform_column(
    tmp_path: Path, idealized_library: Path
) -> None:
    """``rdsf(k)`` and ``rdz*mh`` are one number, executed rather than argued.

    ``init_terrain.F:292-298`` builds ``rdsf(k) = 1/(sigmaf(k+1)-sigmaf(k))``
    and ``param.F:8170`` makes ``mh(k) = dz/(zf(k+1)-zf(k))``, so on a column
    whose layers are all ``dz`` the profile-bound family's divisor and the
    uniform family's ``1/dz`` are the same word.  Give both families that
    column and they integrate to the same state within
    :data:`FAMILY_AGREEMENT` -- not to the same words, because the uniform
    family's acoustic and buoyancy statements are the terrain branch's with
    ``gz = 1`` while the profile family runs CM1's flat ones
    (``tests/test_cm1_flat_stretched_native_words.py`` holds those to CM1's
    own objects word for word).
    """

    stretched = _Run(
        _deck(tmp_path, "profile-uniform-column.toml", UNIFORM_COLUMN),
        idealized_library,
        steps=STEPS,
    )
    uniform = _Run(
        _deck(tmp_path, "uniform-family.toml", UNIFORM_FAMILY),
        idealized_library,
        steps=STEPS,
    )
    assert stretched.grid.stretch_z == 1 and uniform.grid.stretch_z == 0
    faces = np.asarray(stretched.grid.levels.zf, dtype=np.float64)
    spacing = np.diff(faces[1 : stretched.grid.nz + 2])
    assert np.all(spacing == 200.0), "the generated column is not the uniform one"
    # The precondition, stated separately so a failure below is diagnosable:
    # on this column the divisor the two families carry is literally one word.
    levels = stretched.grid.levels
    rdsf = np.asarray(levels.rdsf, dtype=np.float32)
    rdz = np.float32(np.float32(1.0) / np.float32(uniform.grid.dz_m))
    assert (
        rdsf[1 : levels.nk + 1].view(np.uint32).tolist()
        == [int(rdz.view(np.uint32))] * levels.nk
    ), "rdsf(k) and 1/dz are not the same binary32 word on this column"

    _families_agree(stretched, uniform, "one physical column")
    # A comparison of two states that never moved would pass for free.
    assert not np.array_equal(stretched.initial["w3d"], stretched.final["w3d"])


#: ``src/param.F:3420-3474``: ptype=5, numq=10, the tornado decks' component.
MORRISON = {"component": 'component = "morrison-mp10"'}


def _seed_hydrometeors_with_numbers(runtime) -> None:
    """A precipitating blob carrying every Morrison number concentration.

    The idealized initializer publishes CM1's six mass rows and leaves the four
    number rows at zero (``init3d.F``), and a zero row is not moved by any
    advection.  Each number row is seeded with its own mass row, in CM1's
    ``mp_driver.F:546-558`` order (nci-qi, ncs-qs, ncr-qr, ncg-qg), so the
    microphysics keeps both and the dycore has something to transport.
    """

    fields = runtime.dynamics.fields
    q3d, qa = fields["q3d"], fields["qa"]
    assert q3d.shape[0] == 10 and qa.shape[0] == 10
    nz, ny, nx = (int(v) for v in q3d.shape[1:])
    z = cupy.arange(nz, dtype=cupy.float32)[:, None, None]
    y = cupy.arange(ny, dtype=cupy.float32)[None, :, None]
    x = cupy.arange(nx, dtype=cupy.float32)[None, None, :]
    blob = cupy.exp(
        -(((x - nx / 2) / 5.0) ** 2 + ((y - ny / 2) / 5.0) ** 2 + ((z - 20.0) / 6.0) ** 2)
    ).astype(cupy.float32)
    for mass_row, number_row, mass, number in (
        (3, 6, 2.0e-5, 5.0e4), (4, 7, 5.0e-4, 2.0e5), (2, 8, 1.0e-3, 2.0e4), (5, 9, 5.0e-4, 1.0e3),
    ):
        for carrier in (q3d, qa):
            carrier[mass_row] = (cupy.float32(mass) * blob).astype(cupy.float32)
            carrier[number_row] = (cupy.float32(number) * blob).astype(cupy.float32)


def test_the_profile_family_carries_every_morrison_row_the_uniform_family_does(
    tmp_path: Path, idealized_library: Path
) -> None:
    """``solve2.F:1435`` advects n=1,numq; on one column the two families agree on all ten.

    The profile-bound scalar stage looped six rows, so on a ``stretch_z=1`` grid
    a Morrison run never advected or pdef-corrected nci, ncs, ncr or ncg while
    the uniform family did.  On the uniform column the families share every
    divisor, so any row one of them skips separates the two states by far
    more than :data:`FAMILY_AGREEMENT`.
    """

    stretched = _Run(
        _deck(tmp_path, "morrison-profile-uniform-column.toml", {**UNIFORM_COLUMN, **MORRISON}),
        idealized_library,
        steps=STEPS,
        mutate=_seed_hydrometeors_with_numbers,
    )
    uniform = _Run(
        _deck(tmp_path, "morrison-uniform-family.toml", {**UNIFORM_FAMILY, **MORRISON}),
        idealized_library,
        steps=STEPS,
        mutate=_seed_hydrometeors_with_numbers,
    )
    assert stretched.grid.stretch_z == 1 and uniform.grid.stretch_z == 0
    assert stretched.initial["q3d"].shape[0] == 10
    _families_agree(stretched, uniform, "Morrison on one physical column")
    for row in range(6, 10):
        # A skipped row would stand still in one family and move in the other.
        moved = np.abs(uniform.final["q3d"][row].astype(np.float64) - uniform.initial["q3d"][row]).max()
        gap = np.abs(stretched.final["q3d"][row].astype(np.float64) - uniform.final["q3d"][row]).max()
        assert gap <= FAMILY_AGREEMENT * moved, (row, gap, moved)
    for row in range(6, 10):
        final, initial = stretched.final["q3d"][row], stretched.initial["q3d"][row]
        assert np.any(final != 0.0), f"q3d[{row}] was never seeded or was removed"
        assert not np.array_equal(final, initial), f"q3d[{row}] never moved"
    receipt = stretched.receipts["scalar"]
    assert receipt["numq"] == 10 and len(receipt["pdef_launches"]) == 10


#: README.namelist:751-759's param3 reals, away from CM1's 0.60/0.10, declared
#: beside ``psolver`` in the deck's own ``[dynamics]`` table.
DECLARED_ACOUSTIC_REALS = {"psolver": "psolver = 3\nalph = 0.55\nkdiv = 0.35"}


def test_the_profile_family_binds_the_declared_alph_and_kdiv_as_the_uniform_family_does(
    tmp_path: Path, idealized_library: Path
) -> None:
    """``sound.F:81-82,756,782`` read ``alph`` and ``kdiv`` on every grid.

    The profile-bound acoustic stage used to carry 0.60/0.10 as constants, so a
    stretched run that declared anything else integrated CM1's defaults while
    the uniform family integrated the declaration.  On the uniform column the
    two families share every divisor, so with the same declared reals they must
    agree within :data:`FAMILY_AGREEMENT` -- and differ from the defaults' by
    far more than that.
    """

    stretched = _Run(
        _deck(tmp_path, "reals-profile-uniform-column.toml",
              {**UNIFORM_COLUMN, **DECLARED_ACOUSTIC_REALS}),
        idealized_library,
        steps=STEPS,
    )
    uniform = _Run(
        _deck(tmp_path, "reals-uniform-family.toml", {**UNIFORM_FAMILY, **DECLARED_ACOUSTIC_REALS}),
        idealized_library,
        steps=STEPS,
    )
    defaults = _Run(
        _deck(tmp_path, "defaults-profile-uniform-column.toml", UNIFORM_COLUMN),
        idealized_library,
        steps=STEPS,
    )
    assert stretched.grid.stretch_z == 1 and uniform.grid.stretch_z == 0
    # Words first, so the tree that ignored the declaration fails on the
    # integrated state itself rather than on a receipt key it never wrote.
    _families_agree(stretched, uniform, "alph=0.55, kdiv=0.35 on one column")
    # The declaration must move the state far more than the two families
    # differ, or the agreement above would not show that both bound it.
    for name in ("u3d", "v3d", "pp3d"):
        declared = np.abs(stretched.final[name].astype(np.float64) - defaults.final[name]).max()
        families = np.abs(stretched.final[name].astype(np.float64) - uniform.final[name]).max()
        assert declared > 50.0 * families, (name, declared, families)
    assert any(
        stretched.final[name].tobytes() != defaults.final[name].tobytes()
        for name in ("u3d", "v3d", "w3d", "pp3d")
    ), "the declared reals integrated to the default reals' words"
    assert stretched.receipts["acoustic"]["acoustic_reals"] == {
        "alph": float(np.float32(0.55)), "kdiv": float(np.float32(0.35)),
    }
    assert defaults.receipts["acoustic"]["acoustic_reals"] == {
        "alph": float(np.float32(0.60)), "kdiv": float(np.float32(0.10)),
    }


def test_a_stretched_column_integrates_real_large_steps(
    tmp_path: Path, idealized_library: Path
) -> None:
    """The 20 m/15 km column this coordinate exists to provide, stepped.

    What is asserted is the completed state: finite everywhere, positive
    density and pressure, no negative moisture or TKE, an advective Courant
    number below ``misclibs.F:2714``'s stop, a dry-mass ledger that still
    balances when it is re-integrated over the generated layer depths, and a
    warm bubble that has actually started to lift.
    """

    run = _Run(
        _deck(tmp_path, "tornado-grade-small.toml", SMALL),
        idealized_library,
        steps=STEPS,
    )
    levels = run.grid.levels
    assert run.grid.stretch_z == 1
    assert float(levels.lowest_mass_level_m) == 10.0
    assert float(levels.ztop_m) == 15000.0
    assert levels.levels_below_m(1000.0) == 22

    courant = run.checks["advective_courant"]
    assert courant["stop_reason"] is None
    assert 0.0 < float(courant["number"]) < float(courant["stop"])
    for name in PROGNOSTIC:
        assert np.all(np.isfinite(run.final[name])), f"{name} left the real line"
    assert run.final["rho"].min() > 0.0 and run.final["prs"].min() > 0.0
    assert run.final["q3d"].min() >= 0.0

    # misclibs.F:2584-2718's ledger is an integral over the layer depths, so it
    # is computed with this column's own mh and not with dz.
    assert run.mass_initial > 0.0
    assert abs(run.mass_final - run.mass_initial) / run.mass_initial < 1.0e-6

    # The state moved, and it moved where a warm bubble moves it.
    assert not np.array_equal(run.initial["th3d"], run.final["th3d"])
    assert np.abs(run.final["w3d"]).max() > 0.0
    assert run.final["th3d"].max() > 0.5

    acoustic = run.receipts["acoustic"]
    assert acoustic["stretch_z"] == 1
    assert acoustic["scalar_dz_fallback"] is False
    assert acoustic["source_valid_padded_rho0"] is True
    assert acoustic["column_diagnostics_retained"] is False
    assert acoustic["domain"] == {
        "ni": NX,
        "nj": NY,
        "nk": run.grid.nz,
        "dx_m": run.grid.dx_m,
        "dy_m": run.grid.dy_m,
        "ztop_m": 15000.0,
        "compact_validation": False,
    }


def install_storing_acoustic_image(dynamics: Any, *, widen: bool) -> str:
    """Make the acoustic phase launch the storing image; return its core source digest.

    ``CartesianDynamics.acoustic`` launches ``stretched_compiled["psolver"]``
    on a profile-bound run and ``psolver_compiled`` otherwise, so that is what
    is replaced.  ``widen`` also gives the six column reports their
    ``terrain_open_sound_shapes`` extent, which the phase allocates on its
    next entry (they are released after every acoustic stage).
    """

    from cm1core.cuda_psolver3_terrain_open import (
        COLUMN_DIAGNOSTIC_NAMES,
        compile_cuda_psolver3_terrain_open,
        terrain_open_sound_shapes,
    )

    if dynamics.stretched:
        # The stretched Cartesian column is flat (terrain_flag=.false.), so its
        # storing twin is the flat image's.
        from cm1core.cuda_gm1_child_psolver3_open import (
            compile_cuda_gm1_flat_stretched_psolver3_open,
        )

        compact = dynamics.stretched_compiled["psolver"]
        storing = compile_cuda_gm1_flat_stretched_psolver3_open(retain_column_diagnostics=True)
        dynamics.stretched_compiled["psolver"] = storing
        digest = storing.core.source_sha256
    else:
        compact = dynamics.psolver_compiled
        storing = compile_cuda_psolver3_terrain_open(retain_column_diagnostics=True)
        dynamics.psolver_compiled = storing
        digest = storing.source_sha256
    if compact.retain_column_diagnostics is not False or storing.retain_column_diagnostics is not True:
        raise AssertionError("the lane's acoustic image was not the compact one it swapped out")
    if widen:
        g = dynamics.geometry
        shapes = terrain_open_sound_shapes(g.nx, g.ny, g.nz)
        dynamics._padded_shapes.update({name: shapes[name] for name in COLUMN_DIAGNOSTIC_NAMES})
    return digest


def launched_acoustic_image(receipt: dict[str, Any]) -> dict[str, Any]:
    """Which image an acoustic stage receipt says ran, on either family."""

    if "kernels" in receipt:  # cuda_gm1_child_psolver3_open's stage receipt
        return {
            "family": "stretched",
            "column_diagnostics_retained": bool(receipt["column_diagnostics_retained"]),
            "core_source_sha256": receipt["kernels"]["core_source_sha256"],
        }
    # cuda_psolver3_terrain_open's receipt names an omitted report only when
    # the compact image ran.
    return {
        "family": "uniform",
        "column_diagnostics_retained": bool(receipt.get("column_diagnostics", {}).get("retained", True)),
        "core_source_sha256": receipt["kernel"]["source_sha256"],
    }


#: The acoustic-image checks on both operator families: the uniform family
#: launches ``psolver_compiled`` and the profile-bound one
#: ``stretched_compiled["psolver"]``.
ACOUSTIC_FAMILIES = {"uniform": UNIFORM_FAMILY, "stretched": SMALL}


@pytest.mark.parametrize("family", tuple(ACOUSTIC_FAMILIES))
def test_a_storing_acoustic_image_refuses_the_compact_column_workspace(
    tmp_path: Path, idealized_library: Path, family: str
) -> None:
    """The defect itself, pinned so it cannot come back silently.

    ``sound.F:659-730``'s tridiagonal coefficients are written per column at
    the pressure extent by the storing variant and not written at all by the
    compact one.  Pairing the storing image with the one-word workspace this
    lane allocates must stop at the array that would be overrun, by name,
    before a kernel is enqueued -- not corrupt ``nk*nj*ni`` words of whatever
    follows ``tri_aa`` in the pool.  The swap replaces the image this family's
    acoustic phase launches; the compact image never refuses this workspace,
    so a swap that missed it would not raise.
    """

    deck = _deck(tmp_path, f"{family}-mismatch.toml", ACOUSTIC_FAMILIES[family])
    with pytest.raises(ValueError) as caught:
        _Run(
            deck,
            idealized_library,
            steps=1,
            mutate=lambda runtime: install_storing_acoustic_image(
                runtime.dynamics, widen=False
            ),
        )
    message = str(caught.value)
    assert "tri_aa" in message
    assert "invalid dtype, shape, or layout" in message


@pytest.mark.parametrize("family", tuple(ACOUSTIC_FAMILIES))
def test_the_two_acoustic_images_integrate_to_identical_words(
    tmp_path: Path, idealized_library: Path, family: str
) -> None:
    """Dropping the column report changes the report and nothing else.

    ``ec``/``ed`` are the live elimination storage in both variants; the six
    ``tri_*`` arrays are written after the coefficients exist and read only for
    the residual line.  So the lane's compact image must integrate to exactly
    the words the storing image integrates to, and this is the check that would
    fail if the diagnostic-only slice ever stopped being diagnostic-only.  The
    stage receipts must name the compact image and exactly the storing image
    that was installed, or the comparison compared an image with itself.
    """

    compact = _Run(
        _deck(tmp_path, f"{family}-compact.toml", ACOUSTIC_FAMILIES[family]),
        idealized_library,
        steps=STEPS,
    )
    storing = _Run(
        _deck(tmp_path, f"{family}-storing.toml", ACOUSTIC_FAMILIES[family]),
        idealized_library,
        steps=STEPS,
        mutate=lambda runtime: install_storing_acoustic_image(
            runtime.dynamics, widen=True
        ),
    )
    assert compact.grid.stretch_z == storing.grid.stretch_z == (
        1 if family == "stretched" else 0
    )
    ran_compact = launched_acoustic_image(compact.receipts["acoustic"])
    ran_storing = launched_acoustic_image(storing.receipts["acoustic"])
    assert ran_compact["family"] == ran_storing["family"] == family
    assert ran_compact["column_diagnostics_retained"] is False
    assert ran_storing["column_diagnostics_retained"] is True
    assert ran_storing["core_source_sha256"] == storing.installed
    assert ran_compact["core_source_sha256"] != ran_storing["core_source_sha256"]
    for name in PROGNOSTIC:
        assert compact.final[name].tobytes() == storing.final[name].tobytes(), (
            f"{name} differs between the compact and storing acoustic images"
        )
    assert not np.array_equal(compact.initial["w3d"], compact.final["w3d"])
