"""CM1 rotation diagnostics, surface swaths and the tornado report.

The proof this file exists for is the one the project's signature failure
needs: a TOML document reaches the call site and changes what the runtime
computes and publishes.  ``test_toml_selection_reaches_the_call_site``
writes the document, loads it through the real configuration door, binds the
real runtime contract, and drives the real
:class:`cm1core.cartesian_tornado_diagnostics.CartesianTornadoDiagnostics`
over a host runtime whose array module is NumPy instead of CuPy.  No kernel
is mocked and no diagnostic is reimplemented for the test: the module under
test is array-module agnostic, so the host run is the same source the device
runs.

The one thing a host cannot do is execute a CUDA kernel.  Nothing in this
family is a CUDA kernel: every operation is an IEEE-754 binary32 add,
subtract, multiply, divide, maximum, minimum or square root on identically
shaped operands, and the only accumulation is CM1's own serial ``k`` loop, so
NumPy and CuPy produce the same words.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import cm1_marks

from cm1core.cartesian_door import build_runtime_config
from cm1core.cartesian_history import cartesian_history_profile
from cm1core.cartesian_tornado_diagnostics import (
    CartesianTornadoDiagnostics,
    tornado_carrier_table,
    tornado_diagnostic_bytes,
)
from cm1core.config import load_config, tornado_diagnostic_selection
from cm1core.errors import CM1Refusal
from cm1core.grid_geometry import CartesianGridGeometry
from cm1core.gm1_runtime import RuntimeConfig
from cm1core.tornado_diagnostics import (
    CM1_KARMAN,
    EF_SCALE_WIND_M_S,
    NEAR_SURFACE_VORTEX_S_INV,
    OUTPUT_SWITCH_NAMES,
    SOURCE_ANCHORS,
    SWATH_INITIAL_VALUES,
    SWATH_NAMES,
    UH_ROTATING_UPDRAFT_M2_S2,
    UH_STRONG_ROTATION_M2_S2,
    UPDRAFT_HELICITY_BOTTOM_M,
    UPDRAFT_HELICITY_TOP_M,
    VORTEX_MINIMUM_DURATION_S,
    GroundAnchor,
    LowerBoundaryState,
    SurfaceSwaths,
    TornadoDiagnosticSelection,
    VerticalColumn,
    potential_vorticity,
    resolution_verdict,
    surface_layer_state,
    tornado_report,
    updraft_helicity,
    vertical_column,
    vertical_vorticity_level,
    vorticity_components,
    wind_band,
)

ROOT = Path(__file__).resolve().parents[1]


def _authority_src() -> Path:
    """The read-only CM1 r21.1 tree, wherever this checkout can reach it."""

    override = os.environ.get("GM1_CM1_AUTHORITY_ROOT")
    candidates = [Path(override)] if override else []
    candidates.append(ROOT / "work" / "native-reference" / "cm1r21.1")
    for candidate in candidates:
        if (candidate / "src" / "misclibs.F").is_file():
            return candidate / "src"
    pytest.skip(
        "the CM1 r21.1 authority tree is not present in this checkout; set "
        "GM1_CM1_AUTHORITY_ROOT to the extracted release to check the "
        "transcription against its source"
    )


# ---------------------------------------------------------------------------
# A uniform column and an analytic wind field
# ---------------------------------------------------------------------------

def _column(nz: int, dz: float) -> VerticalColumn:
    zh = tuple((k + 0.5) * dz for k in range(nz))
    zf = tuple(k * dz for k in range(nz + 1))
    one = tuple(1.0 for _ in range(nz))
    return VerticalColumn(
        zh, zf, one, one + (1.0,), 1.0 / dz, (-2.0, 3.0, -1.0), (2.0, -3.0, 1.0)
    )


def _solid_body(nz: int, ny: int, nx: int, dx: float, dy: float, omega: float, w0: float):
    """Solid-body rotation with a uniform updraft.

    ``u = -omega*y`` and ``v = +omega*x`` give exactly ``zeta = 2*omega``
    under CM1's centred differences, because a centred difference of a linear
    field is exact.  With ``dx = dy = 256`` and ``omega = 2**-8`` every
    intermediate is a power of two, so the whole integral is exact in
    binary32 and can be compared by hand.
    """

    u = np.empty((nz, ny, nx + 1), dtype=np.float32)
    v = np.empty((nz, ny + 1, nx), dtype=np.float32)
    for i in range(nx + 1):
        u[:, :, i] = np.float32(-omega * ((np.arange(ny) + 0.5) * dy))[None, :]
    for j in range(ny + 1):
        v[:, j, :] = np.float32(omega * ((np.arange(nx) + 0.5) * dx))[None, :]
    w = np.full((nz + 1, ny, nx), np.float32(w0), dtype=np.float32)
    return u, v, w


# ---------------------------------------------------------------------------
# calcuh -- misclibs.F:1417-1532
# ---------------------------------------------------------------------------

def test_updraft_helicity_matches_a_hand_computed_analytic_field() -> None:
    """One field whose updraft helicity can be written down by hand.

    dx = dy = dz = 256 m and omega = 2**-8 s-1 give zeta = 2*omega = 0.0078125
    exactly, and w = 8 m/s everywhere.  CM1's layer test (misclibs.F:1520)
    admits the twelve scalar levels whose zh lies in [2000, 5000]: k = 8..19,
    with zh = 2176 .. 4992 m.  Eleven of those layers are wholly inside the
    band and contribute 256 m each; the twelfth, zf = 4864..5120, contributes
    the clipped 5000 - 4864 = 136 m (misclibs.F:1526).  The integral is
    therefore (11*256 + 136) * 8 * 0.0078125 = 2952 * 0.0625 = 184.5 m2/s2,
    and every intermediate is exact in binary32.
    """

    nz, ny, nx, dz = 24, 10, 10, 256.0
    column = _column(nz, dz)
    u, v, w = _solid_body(nz, ny, nx, 256.0, 256.0, 2.0 ** -8, 8.0)
    plane = updraft_helicity(
        np, u=u, v=v, w=w, column=column,
        rdx=np.float32(1.0 / 256.0), rdy=np.float32(1.0 / 256.0),
    )

    admitted = [k for k in range(nz) if 2000.0 <= column.zh_m[k] <= 5000.0]
    assert admitted == list(range(8, 20))
    thickness = sum(
        min(column.zf_m[k + 1], 5000.0) - max(column.zf_m[k], 2000.0) for k in admitted
    )
    assert thickness == 11 * 256 + 136 == 2952.0
    assert plane.dtype == np.float32
    assert plane.shape == (ny, nx)
    # Bit-exact, not approximately: every operand is a power of two.
    assert np.all(plane == np.float32(184.5))


def test_updraft_helicity_keeps_cm1s_two_clips() -> None:
    """misclibs.F:1523-1525 clips both w and zeta at zero.

    Only a cyclonically rotating updraft contributes.  A downdraft over
    cyclonic rotation and an updraft over anticyclonic rotation both have a
    positive product of the *unclipped* terms in one case and a negative one
    in the other; CM1 counts neither, and neither does this.
    """

    nz, ny, nx, dz = 24, 10, 10, 256.0
    column = _column(nz, dz)
    kwargs = {"rdx": np.float32(1.0 / 256.0), "rdy": np.float32(1.0 / 256.0)}

    u, v, w = _solid_body(nz, ny, nx, 256.0, 256.0, -(2.0 ** -8), 8.0)
    anticyclonic = updraft_helicity(np, u=u, v=v, w=w, column=column, **kwargs)
    assert np.all(anticyclonic == np.float32(0.0))

    u, v, w = _solid_body(nz, ny, nx, 256.0, 256.0, 2.0 ** -8, -8.0)
    downdraft = updraft_helicity(np, u=u, v=v, w=w, column=column, **kwargs)
    assert np.all(downdraft == np.float32(0.0))

    u, v, w = _solid_body(nz, ny, nx, 256.0, 256.0, -(2.0 ** -8), -8.0)
    both_negative = updraft_helicity(np, u=u, v=v, w=w, column=column, **kwargs)
    assert np.all(both_negative == np.float32(0.0))


def test_updraft_helicity_integrates_only_cm1s_layer() -> None:
    """A rotating updraft outside 2-5 km AGL contributes nothing."""

    nz, ny, nx, dz = 24, 10, 10, 256.0
    column = _column(nz, dz)
    u, v, _ = _solid_body(nz, ny, nx, 256.0, 256.0, 2.0 ** -8, 8.0)
    w = np.zeros((nz + 1, ny, nx), dtype=np.float32)
    # Everything below the band and above it, and nothing inside it.
    w[:8] = np.float32(40.0)
    w[21:] = np.float32(40.0)
    plane = updraft_helicity(
        np, u=u, v=v, w=w, column=column,
        rdx=np.float32(1.0 / 256.0), rdy=np.float32(1.0 / 256.0),
    )
    assert np.all(plane == np.float32(0.0))


def test_updraft_helicity_transcribes_cm1s_own_layer_constants() -> None:
    src = _authority_src()
    text = (src / "misclibs.F").read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    # The layer's bottom and top, 2000 m and 5000 m, as parameters.
    assert cm1_marks.has(lines[1440], "203c100b0a53b8d9/26")
    assert cm1_marks.has(lines[1441], "801ea49d3a1a2084/26")
    assert UPDRAFT_HELICITY_BOTTOM_M == 2000.0
    assert UPDRAFT_HELICITY_TOP_M == 5000.0
    assert "subroutine calcuh" in lines[1416]
    assert "Kain et al, 2008, WAF, p 931" in lines[1422]
    # The integrand: the clipped mean w, and the sum of overlap thickness times wbar times zbar.
    assert cm1_marks.has(lines[1522], "969e6483ca1ca471/41")
    assert cm1_marks.has(lines[1525], "13de0870238e48b6/67")
    assert SOURCE_ANCHORS["updraft_helicity"] == "src/misclibs.F:1417-1532"


# ---------------------------------------------------------------------------
# calcvort -- misclibs.F:1540-2174
# ---------------------------------------------------------------------------

def test_vorticity_components_match_analytic_linear_fields() -> None:
    """Three linear fields with known, exactly representable vorticity.

    ``w = a*y`` gives ``xvort = dw/dy = a`` with ``dv/dz = 0``;
    ``u = b*z`` gives ``yvort = du/dz = b`` with ``dw/dx = 0``; solid-body
    rotation gives ``zvort = 2*omega``.  Centred differences of linear fields
    are exact, so these are equalities and not tolerances.
    """

    nz, ny, nx, d = 12, 10, 10, 256.0
    column = _column(nz, d)
    rdx = rdy = np.float32(1.0 / d)
    a, b, omega = 2.0 ** -6, 2.0 ** -7, 2.0 ** -8

    u = np.zeros((nz, ny, nx + 1), dtype=np.float32)
    v = np.zeros((nz, ny + 1, nx), dtype=np.float32)
    w = np.empty((nz + 1, ny, nx), dtype=np.float32)
    for j in range(ny):
        w[:, j, :] = np.float32(a * ((j + 0.5) * d))
    for k in range(nz):
        u[k] = np.float32(b * ((k + 0.5) * d))
    for j in range(ny + 1):
        v[:, j, :] = np.float32(omega * ((np.arange(nx) + 0.5) * d))
    for i in range(nx + 1):
        u[:, :, i] += np.float32(-omega * ((np.arange(ny) + 0.5) * d))[None, :]

    out = vorticity_components(
        np, u=u, v=v, w=w, column=column, rdx=rdx, rdy=rdy,
        lower=LowerBoundaryState(bbc=1, tbc=1),
    )
    assert np.all(out["xvort"] == np.float32(a))
    assert np.all(out["yvort"] == np.float32(b))
    assert np.all(out["zvort"] == np.float32(2.0 * omega))


def test_potential_vorticity_matches_an_analytic_stratified_vortex() -> None:
    """``pv = rr * zvort * d(theta)/dz`` when the horizontal terms vanish.

    A horizontally uniform theta with a constant lapse makes both horizontal
    gradients exactly zero and the vertical one exactly the lapse -- including
    at the two ends, where CM1's Fornberg quadratic (param.F:9042-9052) is
    exact for a linear function.
    """

    nz, ny, nx, d = 12, 10, 10, 256.0
    column = _column(nz, d)
    omega, lapse, density = 2.0 ** -8, 2.0 ** -10, 1.25
    u, v, w = _solid_body(nz, ny, nx, d, d, omega, 0.0)
    theta = np.empty((nz, ny, nx), dtype=np.float32)
    for k in range(nz):
        theta[k] = np.float32(300.0 + lapse * ((k + 0.5) * d))
    rr = np.full((nz, ny, nx), np.float32(1.0 / density), dtype=np.float32)

    vort = vorticity_components(
        np, u=u, v=v, w=w, column=column,
        rdx=np.float32(1.0 / d), rdy=np.float32(1.0 / d),
        lower=LowerBoundaryState(bbc=1, tbc=1),
    )
    pv = potential_vorticity(
        np, theta=theta, rr=rr, vorticity=vort, column=column,
        rdx=np.float32(1.0 / d), rdy=np.float32(1.0 / d),
    )
    expected = np.float32(np.float32(2.0 * omega) * np.float32(lapse)) * np.float32(1.0 / density)
    assert np.allclose(pv, expected, rtol=0, atol=1e-12)


def test_lower_boundary_branches_differ_only_at_the_lowest_level() -> None:
    """``bbc`` changes ``xvort``/``yvort`` at level one and nothing else.

    misclibs.F:1634-1659 replaces only ``tem(:,:,1)``, and misclibs.F:2165-2172
    replaces only ``xvort(:,:,1)`` and ``yvort(:,:,1)``.  ``zvort`` has no
    lower-boundary branch at all, so it must be identical across the three.
    """

    nz, ny, nx, d = 12, 10, 10, 256.0
    column = _column(nz, d)
    rng = np.random.default_rng(11)
    u = rng.standard_normal((nz, ny, nx + 1)).astype(np.float32)
    v = rng.standard_normal((nz, ny + 1, nx)).astype(np.float32)
    w = rng.standard_normal((nz + 1, ny, nx)).astype(np.float32)
    common = {
        "u": u, "v": v, "w": w, "column": column,
        "rdx": np.float32(1.0 / d), "rdy": np.float32(1.0 / d),
    }
    runs = {
        bbc: vorticity_components(
            np, **common,
            lower=LowerBoundaryState(
                bbc=bbc, tbc=1, u_move_m_s=7.0, v_move_m_s=-3.0, cnstcd=0.005
            ),
        )
        for bbc in (1, 2, 3)
    }
    for bbc in (2, 3):
        assert np.array_equal(runs[bbc]["zvort"], runs[1]["zvort"])
        for name in ("xvort", "yvort"):
            assert np.array_equal(runs[bbc][name][1:], runs[1][name][1:])
            assert not np.array_equal(runs[bbc][name][0], runs[1][name][0])
    assert not np.array_equal(runs[3]["xvort"][0], runs[2]["xvort"][0])


def test_semi_slip_lowest_level_is_cm1s_log_layer_equation() -> None:
    """misclibs.F:2165-2172, term for term, against the surface layer it reads."""

    nz, ny, nx, d = 12, 8, 8, 200.0
    column = _column(nz, d)
    rng = np.random.default_rng(5)
    u = rng.standard_normal((nz, ny, nx + 1)).astype(np.float32) * np.float32(6.0)
    v = rng.standard_normal((nz, ny + 1, nx)).astype(np.float32) * np.float32(6.0)
    w = rng.standard_normal((nz + 1, ny, nx)).astype(np.float32)
    lower = LowerBoundaryState(
        bbc=3, tbc=1, u_move_m_s=12.5, v_move_m_s=3.0, cecd=1, cnstcd=0.005
    )
    out = vorticity_components(
        np, u=u, v=v, w=w, column=column,
        rdx=np.float32(1.0 / d), rdy=np.float32(1.0 / d), lower=lower,
    )
    surface = surface_layer_state(np, u=u, v=v, lower=lower, za_m=column.zh_m[0])
    scale = surface["ust"] / np.float32(CM1_KARMAN * (column.zh_m[0] + surface["znt"]))
    speed = np.maximum(surface["s1"], np.float32(0.01))
    assert np.array_equal(out["xvort"][0], np.float32(-1.0) * scale * (surface["v1"] / speed))
    assert np.array_equal(out["yvort"][0], scale * (surface["u1"] / speed))
    # getcecd's own constant-Cd roughness and friction velocity.
    assert surface["znt"] == pytest.approx(10.0 / (math.exp(CM1_KARMAN / math.sqrt(0.005)) - 1.0))
    assert float(np.min(surface["ust"])) >= 1.0e-6


def test_vorticity_refuses_an_unported_boundary_by_value() -> None:
    nz, ny, nx, d = 12, 8, 8, 200.0
    column = _column(nz, d)
    u = np.zeros((nz, ny, nx + 1), dtype=np.float32)
    v = np.zeros((nz, ny + 1, nx), dtype=np.float32)
    w = np.zeros((nz + 1, ny, nx), dtype=np.float32)
    with pytest.raises(ValueError, match="bbc 1, 2 and 3"):
        vorticity_components(
            np, u=u, v=v, w=w, column=column,
            rdx=np.float32(1.0 / d), rdy=np.float32(1.0 / d),
            lower=LowerBoundaryState(bbc=4),
        )


def test_lowest_level_vertical_vorticity_is_where_a_tornado_shows_up() -> None:
    """Level zero of ``zvort`` is the lowest model level and nothing else."""

    nz, ny, nx, d = 12, 10, 10, 256.0
    column = _column(nz, d)
    u, v, w = _solid_body(nz, ny, nx, d, d, 2.0 ** -8, 0.0)
    # Spin the lowest level twice as fast as the rest.
    v[0] = v[0] * np.float32(2.0)
    u[0] = u[0] * np.float32(2.0)
    out = vorticity_components(
        np, u=u, v=v, w=w, column=column,
        rdx=np.float32(1.0 / d), rdy=np.float32(1.0 / d),
        lower=LowerBoundaryState(bbc=1, tbc=1),
    )
    plane = vertical_vorticity_level(
        np, u=u, v=v, level=0, rdx=np.float32(1.0 / d), rdy=np.float32(1.0 / d)
    )
    assert np.array_equal(plane, out["zvort"][0])
    assert np.all(plane == np.float32(4.0 * (2.0 ** -8)))
    assert np.all(out["zvort"][1] == np.float32(2.0 * (2.0 ** -8)))


# ---------------------------------------------------------------------------
# Swaths -- solve3.F:855-1136, init3d.F:181-203
# ---------------------------------------------------------------------------

class _Grid:
    def __init__(self, nx: int, ny: int, nz: int, d: float) -> None:
        self.nx, self.ny, self.nz = nx, ny, nz
        self.dx_m = self.dy_m = self.dz_m = d
        self.ztop_m = nz * d
        self.vertical = None

    @property
    def surface_shape(self) -> tuple[int, int]:
        return self.ny, self.nx


def _swaths(selection: TornadoDiagnosticSelection, *, steps: int = 40,
            umove: float = 0.0, d: float = 256.0):
    grid = _Grid(10, 10, 24, d)
    column = _column(grid.nz, d)
    anchor = GroundAnchor(
        dx_m=d, dy_m=d, u_move_m_s=umove, v_move_m_s=0.0, dt_s=1.0, total_steps=steps
    )
    return SurfaceSwaths(
        np, grid=grid, selection=selection, anchor=anchor, column=column
    ), grid, column, anchor


def test_swaths_start_at_cm1s_own_initial_values() -> None:
    selection = TornadoDiagnosticSelection(
        **{f"output_{name}": 1 for name in SWATH_NAMES}
    )
    swaths, _, _, _ = _swaths(selection)
    assert set(swaths.names) == set(SWATH_NAMES)
    for name in SWATH_NAMES:
        assert np.all(swaths.frame[name] == np.float32(SWATH_INITIAL_VALUES[name]))
    # init3d.F:191-201, by value.
    assert SWATH_INITIAL_VALUES["sps"] == 200000.0
    assert SWATH_INITIAL_VALUES["svs"] == -1000.0
    assert SWATH_INITIAL_VALUES["sus"] == -1000.0
    assert SWATH_INITIAL_VALUES["sws"] == 0.0


def test_swaths_are_running_extrema_over_every_large_step() -> None:
    """A peak that happens between two history frames still reaches the swath."""

    selection = TornadoDiagnosticSelection(output_sws=1, output_svs=1, output_sps=1)
    swaths, grid, column, _ = _swaths(selection, steps=40)
    nz, ny, nx = grid.nz, grid.ny, grid.nx
    u = np.zeros((nz, ny, nx + 1), dtype=np.float32)
    v = np.zeros((nz, ny + 1, nx), dtype=np.float32)
    w = np.zeros((nz + 1, ny, nx), dtype=np.float32)
    prs = np.full((nz, ny, nx), np.float32(100000.0), dtype=np.float32)
    prs0 = np.full((ny, nx), np.float32(100000.0), dtype=np.float32)
    q3d = np.zeros((6, nz, ny, nx), dtype=np.float32)
    for step in range(1, 41):
        spike = step == 17
        uu = u.copy()
        uu[0] = np.float32(60.0 if spike else 1.0)
        px = prs.copy()
        if spike:
            px[0, 5, 5] = np.float32(100000.0 - 3500.0)
        swaths.update(
            step=step, u=uu, v=v, w=w, prs=px, prs0_column=prs0, q3d=q3d,
            updraft_helicity_plane=None, domainlocx_m=0.0, domainlocy_m=0.0,
        )
    extrema = swaths.extrema()
    assert extrema["sws"] == pytest.approx(60.0)
    assert extrema["sps"] == pytest.approx(-3500.0)
    assert swaths.updates == 40
    series = swaths.host_series()
    assert series["sws_max"][16] == pytest.approx(60.0)
    assert series["sws_max"][15] == pytest.approx(1.0)


def test_swath_refuses_a_step_past_the_run_it_was_sized_for() -> None:
    selection = TornadoDiagnosticSelection(output_sws=1)
    swaths, grid, _, _ = _swaths(selection, steps=4)
    nz, ny, nx = grid.nz, grid.ny, grid.nx
    kwargs = {
        "u": np.zeros((nz, ny, nx + 1), dtype=np.float32),
        "v": np.zeros((nz, ny + 1, nx), dtype=np.float32),
        "w": np.zeros((nz + 1, ny, nx), dtype=np.float32),
        "prs": np.zeros((nz, ny, nx), dtype=np.float32),
        "prs0_column": np.zeros((ny, nx), dtype=np.float32),
        "q3d": np.zeros((6, nz, ny, nx), dtype=np.float32),
        "updraft_helicity_plane": None,
        "domainlocx_m": 0.0, "domainlocy_m": 0.0,
    }
    for step in range(1, 5):
        swaths.update(step=step, **kwargs)
    with pytest.raises(ValueError, match="ground the domain had already left"):
        swaths.update(step=5, **kwargs)


def test_ground_swath_is_placed_by_the_domain_location_integral() -> None:
    """A whole-cell displacement lands the swath exactly, with no residual.

    CM1 advects the accumulated maximum with three WENO stages per step
    (adv_routines.F:8884-9178), which diffuses the peak.  GM1 places it from
    cm1.F:3512-3513's own integral, so a domain that has moved a whole number
    of cells puts the swath maximum on exactly the ground cell it happened at.
    """

    d = 256.0
    selection = TornadoDiagnosticSelection(output_svs=1)
    swaths, grid, _, anchor = _swaths(selection, steps=4, umove=d, d=d)
    assert anchor.imove == 1
    # dt is 1 s and umove is one cell per second, so every update is exact.
    # One cell of margin each way is deliberate; GroundAnchor.canvas_shape
    # says why.
    assert anchor.canvas_shape(grid.ny, grid.nx) == (grid.ny + 1, grid.nx + 5)
    nz, ny, nx = grid.nz, grid.ny, grid.nx
    base_u = np.zeros((nz, ny, nx + 1), dtype=np.float32)
    base_v = np.zeros((nz, ny + 1, nx), dtype=np.float32)
    kwargs = {
        "w": np.zeros((nz + 1, ny, nx), dtype=np.float32),
        "prs": np.zeros((nz, ny, nx), dtype=np.float32),
        "prs0_column": np.zeros((ny, nx), dtype=np.float32),
        "q3d": np.zeros((6, nz, ny, nx), dtype=np.float32),
        "updraft_helicity_plane": None,
    }
    # One fixed ground column: as the grid slides east one cell per step, the
    # vortex sits one cell further west in the frame each step.
    for step in range(1, 5):
        v = base_v.copy()
        frame_column = 6 - step
        v[0, :, frame_column] = np.float32(80.0)
        v[0, :, frame_column - 1] = np.float32(-80.0)
        swaths.update(
            step=step, u=base_u, v=v, domainlocx_m=d * step, domainlocy_m=0.0, **kwargs
        )
    assert swaths.max_residual_m == (0.0, 0.0)
    # The moving-frame swath smears the vortex over four columns; the
    # ground-anchored one keeps it on a single ground column.
    frame_columns = {
        int(i) for i in np.where(swaths.frame["svs"].max(axis=0) > 0.1)[0]
    }
    ground_columns = {
        int(i) for i in np.where(swaths.ground["svs"].max(axis=0) > 0.1)[0]
    }
    assert len(frame_columns) == 4
    assert len(ground_columns) == 1


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------

def _run_report(*, d: float, dz: float, vortex: float, wind: float, steps: int,
                vortex_steps: range, dt: float = 1.0):
    grid = _Grid(10, 10, 24, d)
    grid.dz_m = dz
    grid.ztop_m = grid.nz * dz
    column = _column(grid.nz, dz)
    anchor = GroundAnchor(
        dx_m=d, dy_m=d, u_move_m_s=0.0, v_move_m_s=0.0, dt_s=dt, total_steps=steps
    )
    selection = TornadoDiagnosticSelection(
        output_sws=1, output_svs=1, output_sps=1, output_shs=1, tornado_report=True
    )
    swaths = SurfaceSwaths(
        np, grid=grid, selection=selection, anchor=anchor, column=column
    )
    nz, ny, nx = grid.nz, grid.ny, grid.nx
    u = np.zeros((nz, ny, nx + 1), dtype=np.float32)
    v = np.zeros((nz, ny + 1, nx), dtype=np.float32)
    w = np.zeros((nz + 1, ny, nx), dtype=np.float32)
    prs = np.full((nz, ny, nx), np.float32(100000.0), dtype=np.float32)
    prs0 = np.full((ny, nx), np.float32(100000.0), dtype=np.float32)
    q3d = np.zeros((6, nz, ny, nx), dtype=np.float32)
    uh = np.full((ny, nx), np.float32(140.0), dtype=np.float32)
    for step in range(1, steps + 1):
        vv, uu = v.copy(), u.copy()
        if step in vortex_steps:
            vv[0, :, 5] = np.float32(0.5 * vortex * d)
            vv[0, :, 4] = np.float32(-0.5 * vortex * d)
            uu[0] = np.float32(wind)
        swaths.update(
            step=step, u=uu, v=vv, w=w, prs=prs, prs0_column=prs0, q3d=q3d,
            updraft_helicity_plane=uh, domainlocx_m=0.0, domainlocy_m=0.0,
        )
    return tornado_report(
        swaths=swaths, grid=grid, column=column, lower=LowerBoundaryState(bbc=3)
    )


def test_a_mesocyclone_is_never_reported_as_a_tornado() -> None:
    """Strong rotation aloft with no near-surface vortex stays a mesocyclone."""

    report = _run_report(
        d=1000.0, dz=250.0, vortex=0.0, wind=20.0, steps=300, vortex_steps=range(0)
    )
    assert report["verdict"] == "strong mid-level rotation aloft, no near-surface vortex"
    assert report["rotation_aloft"]["label"] == "strong mid-level rotation"
    assert report["rotation_aloft"]["is_a_tornado"] is False
    assert "tornado" not in report["verdict"]
    assert report["headline"]["peak_updraft_helicity_m2_s2"] == pytest.approx(140.0)
    json.dumps(report, allow_nan=False)


def test_a_resolved_vortex_on_a_coarse_grid_is_not_called_a_tornado() -> None:
    """A kilometre grid cannot hold a tornado, whatever the vorticity says."""

    report = _run_report(
        d=1000.0, dz=250.0, vortex=0.4, wind=70.0, steps=400,
        vortex_steps=range(50, 351),
    )
    assert report["headline"]["peak_near_surface_vertical_vorticity_s_inv"] > NEAR_SURFACE_VORTEX_S_INV
    assert report["near_surface_vortex"]["label"] == "sustained near-surface vortex"
    assert report["resolution"]["can_hold_a_tornado"] is False
    assert report["verdict"] == "near-surface vortex, not resolvable as a tornado on this grid"
    assert "cannot resolve a tornado" in report["statement"]
    # The EF band is reported for the swath's own peak wind and explicitly
    # refused as a rating: a lowest-level grid-volume mean is not a
    # three-second gust at ten metres.
    peak = report["headline"]["peak_near_surface_wind_m_s"]
    assert report["wind_band"] == wind_band(peak)
    assert report["wind_band"]["band"] is not None
    assert report["wind_band"]["is_an_ef_rating"] is False
    assert "three-second gust" in report["wind_band"]["why_not"]


def test_a_resolving_grid_reports_a_tornado_like_vortex() -> None:
    """Thirty-metre spacing and a thirty-metre lowest layer can hold one."""

    report = _run_report(
        d=30.0, dz=30.0, vortex=0.4, wind=70.0, steps=400, vortex_steps=range(50, 351)
    )
    assert report["resolution"]["can_hold_a_tornado"] is True
    assert report["resolution"]["effective_resolution_m"] == pytest.approx(180.0)
    assert report["verdict"] == "tornado-like vortex"
    episodes = report["near_surface_vortex"]["episodes"]
    assert len(episodes) == 1
    assert episodes[0]["duration_s"] == pytest.approx(301.0)
    assert report["near_surface_vortex"]["sustained_episodes"] == 1
    track = report["track"]["rows"]
    assert track and all(row["vertical_vorticity_s_inv"] >= NEAR_SURFACE_VORTEX_S_INV
                         for row in track)
    assert report["lower_boundary"]["generates_surface_vorticity"] is True


def test_a_brief_vortex_is_not_sustained() -> None:
    brief_steps = int(VORTEX_MINIMUM_DURATION_S) - 20
    report = _run_report(
        d=30.0, dz=30.0, vortex=0.4, wind=70.0, steps=400,
        vortex_steps=range(50, 50 + brief_steps),
    )
    assert report["near_surface_vortex"]["label"] == "transient near-surface vortex"
    assert report["near_surface_vortex"]["sustained_episodes"] == 0
    assert report["verdict"] != "tornado-like vortex"


def test_thresholds_are_the_literatures_and_are_stated() -> None:
    assert UH_ROTATING_UPDRAFT_M2_S2 == 25.0
    assert UH_STRONG_ROTATION_M2_S2 == 100.0
    assert NEAR_SURFACE_VORTEX_S_INV == 0.1
    assert VORTEX_MINIMUM_DURATION_S == 120.0
    assert EF_SCALE_WIND_M_S[0] == (0, 29.0)
    assert EF_SCALE_WIND_M_S[-1] == (5, 89.0)
    assert wind_band(28.0)["band"] is None
    assert wind_band(29.0)["band"] == 0
    assert wind_band(200.0)["band"] == 5
    for speed in (0.0, 30.0, 95.0):
        assert wind_band(speed)["is_an_ef_rating"] is False
    coarse = resolution_verdict(_Grid(10, 10, 24, 1000.0), _column(24, 250.0))
    assert coarse["can_hold_a_tornado"] is False
    fine = resolution_verdict(_Grid(10, 10, 24, 30.0), _column(24, 30.0))
    assert fine["can_hold_a_tornado"] is True
    # Fine horizontally but with a deep lowest layer: a tornado is a surface
    # phenomenon, so this is still refused.
    mixed = resolution_verdict(_Grid(10, 10, 24, 30.0), _column(24, 250.0))
    assert mixed["resolves_a_tornado_core"] is True
    assert mixed["resolves_the_near_ground_layer"] is False
    assert mixed["can_hold_a_tornado"] is False


# ---------------------------------------------------------------------------
# A host runtime: the real call site with NumPy in CuPy's place
# ---------------------------------------------------------------------------

class _HostRegistry:
    """The device registry's contract, satisfied on the host."""

    def __init__(self) -> None:
        self._buffers: dict[str, object] = {}
        self.released: list[str] = []

    @property
    def buffers(self):
        return self._buffers

    @property
    def registered_allocation_bytes(self) -> int:
        return sum(int(value.nbytes) for value in self._buffers.values())

    def allocate(self, name: str, shape, *, zero: bool = False, dtype=None):
        if name in self._buffers:
            raise ValueError(f"semantic buffer name is already registered: {name!r}")
        dtype = np.float32 if dtype is None else dtype
        value = np.zeros(shape, dtype=dtype) if zero else np.empty(shape, dtype=dtype)
        self._buffers[name] = value
        return value

    def release(self, names) -> None:
        for name in names:
            del self._buffers[name]
            self.released.append(name)

    def upload(self, source, requests):
        out = {}
        for request in requests:
            group, carrier = request.source.split(".")
            value = np.asarray(source.groups[group][carrier], dtype=np.float32)
            self._buffers[request.name] = value
            out[request.name] = value
        return out


class _NullStream:
    def __enter__(self):
        return self

    def __exit__(self, *exception):
        return False

    def synchronize(self) -> None:
        return None


def _host_runtime(runtime_config: RuntimeConfig, *, seed: int = 3):
    g = runtime_config.grid
    nz, ny, nx = g.nz, g.ny, g.nx
    rng = np.random.default_rng(seed)
    fields = {
        "u3d": rng.standard_normal((nz, ny, nx + 1)).astype(np.float32) * np.float32(8.0),
        "v3d": rng.standard_normal((nz, ny + 1, nx)).astype(np.float32) * np.float32(8.0),
        "w3d": rng.standard_normal((nz + 1, ny, nx)).astype(np.float32) * np.float32(4.0),
        "prs": np.full((nz, ny, nx), np.float32(95000.0), dtype=np.float32),
        "rr": np.full((nz, ny, nx), np.float32(1.0 / 1.1), dtype=np.float32),
        "th0": np.full((nz, ny, nx), np.float32(300.0), dtype=np.float32),
        "th3d": rng.standard_normal((nz, ny, nx)).astype(np.float32),
        "q3d": np.abs(rng.standard_normal((6, nz, ny, nx)).astype(np.float32)) * np.float32(1e-3),
    }
    source = SimpleNamespace(groups={
        "terrain_open": {"prs0": np.full((nz,), 95000.0, dtype=np.float32)}
    })
    return SimpleNamespace(
        cp=np,
        stream=_NullStream(),
        config=runtime_config,
        registry=_HostRegistry(),
        dynamics=SimpleNamespace(fields=fields, source=source),
        clock=SimpleNamespace(domainlocx_m=0.0, domainlocy_m=0.0, completed_steps=0),
        tornado=None,
    )


_BASE_DOCUMENT = (ROOT / "examples" / "cartesian-supercell-translating-frame.toml").read_text(
    encoding="utf-8"
)
_DIAGNOSTICS_BLOCK = """
[diagnostics]
output_uh = 1
output_vort = 1
output_pv = 1
output_sws = 1
output_svs = 1
output_sps = 1
output_srs = 1
output_sgs = 1
output_sus = 1
output_shs = 1
tornado_report = true
"""


def _document(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_toml_selection_reaches_the_call_site(tmp_path: Path) -> None:
    """From config text to changed behaviour, through the real door.

    Nothing here is a stand-in for the production path except the array
    module: the document is parsed by ``load_config``, bound by
    ``build_runtime_config``, and driven through the real
    ``CartesianTornadoDiagnostics``.
    """

    off = load_config(_document(tmp_path, "off.toml", _BASE_DOCUMENT))
    on = load_config(_document(tmp_path, "on.toml", _BASE_DOCUMENT + _DIAGNOSTICS_BLOCK))
    # The rotation/swath switches are their own [diagnostics] family on
    # ``CM1Config.tornado_diagnostics`` since a0cee5e (sota/uhdiag merged
    # beside the getcape planes, which keep ``CM1Config.diagnostics``).
    assert off.tornado_diagnostics is None
    assert on.tornado_diagnostics is not None
    assert on.diagnostics is None

    runtime_off = build_runtime_config(off)
    runtime_on = build_runtime_config(on)
    assert runtime_off.diagnostics.active is False
    assert runtime_on.diagnostics.active is True
    assert all(getattr(runtime_on.diagnostics, name) == 1 for name in OUTPUT_SWITCH_NAMES)

    # The document with no table is byte-identical everywhere it is sealed.
    assert "diagnostics" not in off.canonical_dict()
    assert "diagnostics" not in runtime_off.record()
    assert "diagnostics" in on.canonical_dict()
    assert runtime_on.record()["diagnostics"]["output_uh"] == 1
    assert off.sha256 != on.sha256

    # The call site, with NumPy in CuPy's place.
    host_off = _host_runtime(runtime_off)
    with pytest.raises(ValueError, match="all-off diagnostic selection"):
        CartesianTornadoDiagnostics(host_off, total_steps=8)

    host_on = _host_runtime(runtime_on)
    tornado = CartesianTornadoDiagnostics(host_on, total_steps=8)
    host_on.tornado = tornado
    for step in range(1, 9):
        host_on.clock.domainlocx_m = runtime_on.u_move_m_s * runtime_on.dt_s * step
        host_on.clock.domainlocy_m = runtime_on.v_move_m_s * runtime_on.dt_s * step
        tornado.update(step)
    tornado.refresh(8)

    published = tornado.published_carriers()
    for name in ("uh", "xvort", "yvort", "zvort", "zvort_sfc", "pv"):
        assert name in published
        assert np.all(np.isfinite(published[name]))
    for name in SWATH_NAMES:
        assert f"swath.{name}" in published
        assert f"ground.{name}" in published

    # The published field is the same number the pure function gives.
    column = vertical_column(runtime_on.grid)
    expected = updraft_helicity(
        np,
        u=host_on.dynamics.fields["u3d"],
        v=host_on.dynamics.fields["v3d"],
        w=host_on.dynamics.fields["w3d"],
        column=column,
        rdx=np.float32(1.0 / runtime_on.grid.dx_m),
        rdy=np.float32(1.0 / runtime_on.grid.dy_m),
    )
    assert np.array_equal(published["uh"], expected)

    # The history envelope carries them, and only when selected.
    names_off = [c.name for c in cartesian_history_profile(runtime_off).carriers]
    names_on = [
        c.name
        for c in cartesian_history_profile(
            runtime_on, tornado_carriers=tornado.carrier_table()
        ).carriers
    ]
    # The twelve fixed carriers, then CM1's default &param9 rows appended
    # after them (7183ce6); none of the rotation/swath names is among them.
    assert names_off[:12] == _FIXED_HISTORY_CARRIERS
    assert names_off == _DEFAULT_HISTORY_CARRIERS
    assert set(names_off) < set(names_on)
    added = set(names_on) - set(names_off)
    assert {"uh", "xvort", "yvort", "zvort", "zvort_sfc", "pv"} <= added
    assert {f"{name}2" for name in SWATH_NAMES} <= added

    # And the run can state what it produced.
    record = tornado.record()
    assert record["selection"]["output_uh"] == 1
    assert record["tornado"]["verdict"]
    assert record["tornado"]["resolution"]["can_hold_a_tornado"] is False
    frame = tornado.frame_row(8)
    assert "near_surface_vertical_vorticity_max_s_inv" in frame
    json.dumps(record, allow_nan=False)

    # Every transient the vorticity routine took was handed back.
    leaked = [name for name in host_on.registry.buffers if ".transient." in name]
    assert leaked == []


#: The twelve carriers every history frame has always published, in order.
_FIXED_HISTORY_CARRIERS = [
    "closure.tkea", "pp3d", "prs", "q3d", "rf", "rho", "terrain_msl",
    "th3d", "u3d", "v3d", "w3d", "zf",
]
#: The default frame.  7183ce6 (int/output) appended CM1's default &param9
#: selection AFTER the fixed twelve, so their arena offsets did not move;
#: 26a8abf dropped dbz/cref from that default (one output_dbz selector, off),
#: and a0cee5e (sota/uhdiag) moved the output_s*s swaths into [diagnostics],
#: whose absent table is all-off -- so no rotation or swath plane is here.
_DEFAULT_HISTORY_CARRIERS = _FIXED_HISTORY_CARRIERS + [
    "rain", "prate", "snownc", "graupelnc", "sr", "th", "zhval",
    "pi0", "th0", "prs0", "qv0", "u0", "v0",
]


def test_the_default_document_keeps_every_sealed_identity(tmp_path: Path) -> None:
    """No [diagnostics] table means no change to anything a receipt seals."""

    off = load_config(_document(tmp_path, "off.toml", _BASE_DOCUMENT))
    runtime = build_runtime_config(off)
    assert tornado_diagnostic_selection(off).active is False
    assert tornado_carrier_table(runtime) == ()
    assert tornado_diagnostic_bytes(runtime, total_steps=14400) == {
        "persistent_bytes": 0, "transient_peak_bytes": 0, "total_peak_bytes": 0,
    }
    profile = cartesian_history_profile(runtime)
    assert [carrier.name for carrier in profile.carriers] == _DEFAULT_HISTORY_CARRIERS
    # The sealed bytes of the example document are unchanged by the existence
    # of the switches: an absent optional table is not projected.
    assert off.sha256 == load_config(
        ROOT / "examples" / "cartesian-supercell-translating-frame.toml"
    ).sha256


# ---------------------------------------------------------------------------
# Refusals, by key and by value
# ---------------------------------------------------------------------------

def _diag(**overrides) -> str:
    values = {name: 0 for name in OUTPUT_SWITCH_NAMES}
    values["tornado_report"] = "false"
    values.update({key: value for key, value in overrides.items()})
    body = "\n".join(f"{key} = {value}" for key, value in values.items())
    return "\n[diagnostics]\n" + body + "\n"


def test_a_switch_outside_cm1s_two_values_is_refused(tmp_path: Path) -> None:
    text = _BASE_DOCUMENT + _diag(output_uh=2)
    with pytest.raises(CM1Refusal) as error:
        load_config(_document(tmp_path, "two.toml", text))
    assert "diagnostics.output_uh" in str(error.value)
    assert "param.F:4677" in str(error.value)


def test_an_unknown_diagnostic_key_is_refused_by_name(tmp_path: Path) -> None:
    text = _BASE_DOCUMENT + _diag() + "output_tornado = 1\n"
    with pytest.raises(CM1Refusal) as error:
        load_config(_document(tmp_path, "unknown.toml", text))
    assert "output_tornado" in str(error.value)


def test_a_partial_diagnostics_table_is_refused(tmp_path: Path) -> None:
    text = _BASE_DOCUMENT + "\n[diagnostics]\noutput_uh = 1\n"
    with pytest.raises(CM1Refusal) as error:
        load_config(_document(tmp_path, "partial.toml", text))
    assert "diagnostics" in str(error.value)


def test_updraft_helicity_is_refused_on_a_grid_that_misses_the_layer(tmp_path: Path) -> None:
    text = (
        _BASE_DOCUMENT.replace("nz = 40", "nz = 8").replace("dz_m = 500.0", "dz_m = 100.0")
        + _diag(output_uh=1)
    )
    with pytest.raises(CM1Refusal) as error:
        load_config(_document(tmp_path, "shallow.toml", text))
    message = str(error.value)
    assert "updraft-helicity layer" in message
    assert "misclibs.F:1441-1442" in message
    assert "partial layer" in message or "less of the layer" in message


def test_the_five_kilometre_updraft_swath_is_refused_on_a_shallow_grid(tmp_path: Path) -> None:
    text = (
        _BASE_DOCUMENT.replace("nz = 40", "nz = 8").replace("dz_m = 500.0", "dz_m = 100.0")
        + _diag(output_sus=1)
    )
    with pytest.raises(CM1Refusal) as error:
        load_config(_document(tmp_path, "sus.toml", text))
    assert "solve3.F:1088" in str(error.value)


def test_a_tornado_report_without_its_headline_swaths_is_refused(tmp_path: Path) -> None:
    text = _BASE_DOCUMENT + _diag(output_svs=1, tornado_report="true")
    with pytest.raises(CM1Refusal) as error:
        load_config(_document(tmp_path, "report.toml", text))
    message = str(error.value)
    assert "tornado_report" in message
    assert "output_svs, output_sws and output_sps" in message


def test_the_runtime_contract_refuses_the_same_selections() -> None:
    """A programmatic caller that bypasses the door is refused too."""

    grid = CartesianGridGeometry(nx=20, ny=20, nz=8, dx_m=1000.0, dy_m=1000.0, dz_m=100.0)
    with pytest.raises(ValueError, match="updraft-helicity layer"):
        RuntimeConfig(
            grid, dt_s=0.05, acoustic_substeps=6, memory_budget_bytes=1 << 30,
            diagnostics=TornadoDiagnosticSelection(output_uh=1),
        )


def test_a_switch_value_that_is_not_an_integer_is_refused() -> None:
    with pytest.raises(TypeError, match="integer 0/1"):
        TornadoDiagnosticSelection(output_uh=True)
    with pytest.raises(ValueError, match="param.F:4677"):
        TornadoDiagnosticSelection(output_uh=3)


# ---------------------------------------------------------------------------
# Authority
# ---------------------------------------------------------------------------

def test_every_source_anchor_names_a_line_the_authority_has() -> None:
    src = _authority_src()
    for key, anchor in SOURCE_ANCHORS.items():
        head = anchor.split(" ")[0].split(";")[0]
        relative, _, span = head.partition(":")
        path = src.parent / relative
        assert path.is_file(), f"{key} names {relative}, which the authority does not have"
        if not span:
            continue
        first = int(span.split("-")[0])
        last = int(span.split("-")[-1])
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        assert 1 <= first <= last <= len(lines), f"{key} names {anchor}, outside the file"


# ---------------------------------------------------------------------------
# GPU-gated: the same source with CuPy in NumPy's place
# ---------------------------------------------------------------------------

def test_cupy_reproduces_the_host_words_only_with_mutex() -> None:
    """The device path and the host path must give the same binary32 words.

    Every operation in this family is an IEEE-754 binary32 add, subtract,
    multiply, divide, maximum, minimum or square root on identically shaped
    operands, and the only accumulation is CM1's own serial ``k`` loop, so
    there is no reduction order for a backend to choose.  This asserts that
    by comparing raw words, and it is the one test here that needs a device.

    It is gated behind an explicit environment variable because the GPU may
    be running production storms; it must only be set while holding the GPU
    mutex.
    """

    if os.environ.get("CM1CORE_RUN_CUDA_TORNADO_DIAGNOSTICS_TEST") != "1":
        pytest.skip(
            "set CM1CORE_RUN_CUDA_TORNADO_DIAGNOSTICS_TEST=1 only while "
            "holding the GPU mutex"
        )
    import cupy

    nz, ny, nx, d = 16, 24, 24, 256.0
    column = _column(nz, d)
    rng = np.random.default_rng(19)
    host = {
        "u": rng.standard_normal((nz, ny, nx + 1)).astype(np.float32) * np.float32(9.0),
        "v": rng.standard_normal((nz, ny + 1, nx)).astype(np.float32) * np.float32(9.0),
        "w": rng.standard_normal((nz + 1, ny, nx)).astype(np.float32) * np.float32(5.0),
        "theta": (300.0 + rng.standard_normal((nz, ny, nx))).astype(np.float32),
        "rr": (1.0 / 1.1 + 0.01 * rng.standard_normal((nz, ny, nx))).astype(np.float32),
    }
    device = {name: cupy.asarray(value) for name, value in host.items()}
    lower = LowerBoundaryState(
        bbc=3, tbc=1, u_move_m_s=12.5, v_move_m_s=3.0, cecd=1, cnstcd=0.005
    )
    spacing = {"rdx": 1.0 / d, "rdy": 1.0 / d}

    for module, arrays in ((np, host), (cupy, device)):
        rdx, rdy = module.float32(spacing["rdx"]), module.float32(spacing["rdy"])
        uh = updraft_helicity(
            module, u=arrays["u"], v=arrays["v"], w=arrays["w"], column=column,
            rdx=rdx, rdy=rdy,
        )
        vort = vorticity_components(
            module, u=arrays["u"], v=arrays["v"], w=arrays["w"], column=column,
            rdx=rdx, rdy=rdy, lower=lower,
        )
        pv = potential_vorticity(
            module, theta=arrays["theta"], rr=arrays["rr"], vorticity=vort,
            column=column, rdx=rdx, rdy=rdy,
        )
        arrays["out"] = {"uh": uh, "pv": pv, **vort}

    for name in ("uh", "xvort", "yvort", "zvort", "pv"):
        expected = host["out"][name]
        actual = cupy.asnumpy(device["out"][name])
        assert np.array_equal(
            actual.view(np.uint32), expected.view(np.uint32)
        ), f"{name} differs between the host and the device"
