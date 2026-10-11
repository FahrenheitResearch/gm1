from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess

import numpy as np
import pytest

from cm1core.gm1_idealized_in_memory import (
    IdealizedGeometry,
    RustIdealizedCarrierMap,
)
from authority_trees import requires_cm1_pristine


ROOT = Path(__file__).resolve().parents[1]
CRATE = ROOT / "rust/cm1-state"
SOUNDING = ROOT / "upstream/cm1-21.1/pristine/run/config_files/sea_breeze/input_sounding"


@pytest.fixture(scope="module")
def idealized_library(tmp_path_factory: pytest.TempPathFactory) -> Path:
    target = tmp_path_factory.mktemp("gm1-idealized-cdylib")
    subprocess.run(
        ["cargo", "build", "--release", "--lib", "--target-dir", str(target)],
        cwd=CRATE,
        check=True,
    )
    names = ("cm1_state.dll", "libcm1_state.so", "libcm1_state.dylib")
    found = [target / "release" / name for name in names if (target / "release" / name).is_file()]
    assert len(found) == 1
    return found[0]


def geometry() -> IdealizedGeometry:
    return IdealizedGeometry(
        (4, 2, 3),
        (1000.0, 1000.0, 500.0),
        (0.0, 0.0),
        (1500.0, 1000.0),
    )


def raw_sha(value: np.ndarray) -> str:
    return hashlib.sha256(value.astype("<f4", copy=False).tobytes(order="C")).hexdigest()


@requires_cm1_pristine
def test_zero_disk_views_are_exact_bf115c3_carriers(idealized_library: Path) -> None:
    with RustIdealizedCarrierMap(
        idealized_library,
        SOUNDING.read_bytes(),
        iinit=1,
        geometry=geometry(),
    ) as owner:
        assert owner.carrier("terrain_open", "pi0").shape == (6, 8, 9)
        assert owner.carrier("base_wind", "u0").shape == (4, 2, 4)
        assert owner.carrier("moist", "qa").shape == (6, 6, 8, 9)
        expected = {
            ("terrain_open", "pi0"): "7be5ea48e48b6e41ada959266dcbec2fb1c635de3fd975585645067cb29a5f93",
            # init3d.F:1735-1741.  run/namelist.input:80 and all 23 shipped CM1
            # decks select ibalance=0, so the initial pressure perturbation is
            # exactly zero; it used to be the ibalance=1 hydrostatic field.
            ("terrain_open", "ppi"): "4b48f21a4b7a02bfbec19ef880a967a02334a3cdcef8ae83de2ef327ba8bc5dd",
            ("terrain_open", "tha"): "b1e69daba32360f140e6f4d0b459b90e8a711bc9cf794f920288492a2a610bc4",
            ("terrain_open", "ua"): "155e437b946ac82ae591ff382b8d19efda9397b2282672dbabd91ec31ce8a651",
            ("moist", "qa"): "fd2ade272766091b268e77a7457c8062292f497266befc3dbea60cc82db16a5a",
        }
        for key, digest in expected.items():
            assert raw_sha(owner.carrier(*key)) == digest
        assert not owner.carrier("terrain_open", "pi0").flags.writeable


@requires_cm1_pristine
def test_iinit_zero_and_one_differ_only_in_initialized_theta_family(
    idealized_library: Path,
) -> None:
    with RustIdealizedCarrierMap(
        idealized_library, SOUNDING.read_bytes(), iinit=0, geometry=geometry()
    ) as cold, RustIdealizedCarrierMap(
        idealized_library, SOUNDING.read_bytes(), iinit=1, geometry=geometry()
    ) as bubble:
        assert raw_sha(cold.carrier("terrain_open", "pi0")) == raw_sha(
            bubble.carrier("terrain_open", "pi0")
        )
        assert raw_sha(cold.carrier("terrain_open", "tha")) != raw_sha(
            bubble.carrier("terrain_open", "tha")
        )
        assert np.count_nonzero(cold.carrier("terrain_open", "tha")) == 0
        assert np.count_nonzero(bubble.carrier("terrain_open", "tha")) > 0


@requires_cm1_pristine
def test_unsupported_iinit_and_geometry_fail_closed(idealized_library: Path) -> None:
    with pytest.raises(ValueError, match="iinit=0 or iinit=1"):
        RustIdealizedCarrierMap(
            idealized_library, SOUNDING.read_bytes(), iinit=2, geometry=geometry()
        )
    with pytest.raises(ValueError, match="nz>=4"):
        RustIdealizedCarrierMap(
            idealized_library,
            SOUNDING.read_bytes(),
            iinit=1,
            geometry=IdealizedGeometry(
                (3, 2, 3), (1000.0, 1000.0, 500.0), (0.0, 0.0), (1500.0, 1000.0)
            ),
        )


# ------------------------------------------------- base.F:2661-2674 frame


@requires_cm1_pristine
def test_a_zero_frame_is_raw_bit_identical_on_both_lanes(
    idealized_library: Path,
) -> None:
    """Zero must not be a code path: it is the default of a live option."""

    for isnd, iwnd, sounding in ((7, 0, SOUNDING.read_bytes()), (5, 2, b"")):
        with RustIdealizedCarrierMap(
            idealized_library, sounding, iinit=1, geometry=geometry(),
            isnd=isnd, iwnd=iwnd,
        ) as still, RustIdealizedCarrierMap(
            idealized_library, sounding, iinit=1, geometry=geometry(),
            isnd=isnd, iwnd=iwnd, u_move_m_s=0.0, v_move_m_s=0.0,
        ) as zero:
            for group, carriers in still.groups.items():
                for name in carriers:
                    assert raw_sha(still.carrier(group, name)) == raw_sha(
                        zero.carrier(group, name)
                    ), f"isnd={isnd} {group}.{name}"
            assert zero.u_move_m_s == 0.0
            assert zero.v_move_m_s == 0.0


@pytest.mark.parametrize(
    ("isnd", "iwnd", "reads_sounding"),
    (pytest.param(7, 0, True, marks=requires_cm1_pristine), (5, 2, False)),
)
def test_the_frame_offsets_every_level_of_both_staggered_winds(
    idealized_library: Path, isnd: int, iwnd: int, reads_sounding: bool
) -> None:
    """CM1 base.F:2667-2668 on both admitted base-state lanes.

    The tornado run reads an observed profile through isnd=7 and the deck
    reproduction uses the isnd=5 analytic hodograph, so both have to honour
    the frame or one of them silently integrates a different atmosphere.
    """

    umove, vmove = 12.5, 3.0
    sounding = SOUNDING.read_bytes() if reads_sounding else b""
    with RustIdealizedCarrierMap(
        idealized_library, sounding, iinit=1, geometry=geometry(),
        isnd=isnd, iwnd=iwnd,
    ) as still, RustIdealizedCarrierMap(
        idealized_library, sounding, iinit=1, geometry=geometry(),
        isnd=isnd, iwnd=iwnd, u_move_m_s=umove, v_move_m_s=vmove,
    ) as moving:
        # u lives on nx+1 faces and v on ny+1; both, every level.
        u_still = still.carrier("base_wind", "u0")
        u_moved = moving.carrier("base_wind", "u0")
        v_still = still.carrier("base_wind", "v0")
        v_moved = moving.carrier("base_wind", "v0")
        assert u_still.shape == (4, 2, 4)
        assert v_still.shape == (4, 3, 3)
        assert np.array_equal(
            u_moved.view(np.uint32),
            (u_still - np.float32(umove)).astype("<f4").view(np.uint32),
        )
        assert np.array_equal(
            v_moved.view(np.uint32),
            (v_still - np.float32(vmove)).astype("<f4").view(np.uint32),
        )
        # Every level moved, not just the ones that happened to be nonzero.
        for level in range(u_still.shape[0]):
            assert not np.array_equal(u_moved[level], u_still[level])
            assert not np.array_equal(v_moved[level], v_still[level])

        # init3d.F:210-219 copies u0/v0 into ua/u3d/va/v3d, so the initial
        # condition carries the same offset.  A base state that moved while
        # ua did not would be a reference profile that is not the sounding's.
        for name, offset, faces in (
            ("ua", umove, 4 * 2 * 4), ("u3d", umove, 4 * 2 * 4),
            ("va", vmove, 4 * 3 * 3), ("v3d", vmove, 4 * 3 * 3),
        ):
            a = still.carrier("terrain_open", name)
            b = moving.carrier("terrain_open", name)
            # These carriers are padded and the halo stays at allocation
            # zero, so "written" is the union of the two runs' nonzero
            # words: the pinned isnd=7 fixture is calm, which leaves the
            # fixed run's written words at zero and only the moving run's
            # nonzero.
            written = (a != np.float32(0.0)) | (b != np.float32(0.0))
            # The lateral halo is filled by the bc.F wbc=2 zero-gradient
            # copy, so strictly more than the physical faces must move.
            assert int(written.sum()) > faces, name
            assert np.array_equal(
                b[written].view(np.uint32),
                (a[written] - np.float32(offset)).astype("<f4").view(np.uint32),
            ), name
            assert np.array_equal(b[~written], a[~written]), name

        assert moving.u_move_m_s == umove
        assert moving.v_move_m_s == vmove


@requires_cm1_pristine
def test_the_frame_velocity_is_refused_by_key_and_value(
    idealized_library: Path,
) -> None:
    for key, u, v in (
        ("u_move_m_s", float("nan"), 0.0),
        ("u_move_m_s", float("inf"), 0.0),
        ("v_move_m_s", 0.0, float("-inf")),
    ):
        with pytest.raises(ValueError, match=f"{key}=.*not finite"):
            RustIdealizedCarrierMap(
                idealized_library, SOUNDING.read_bytes(), iinit=1,
                geometry=geometry(), u_move_m_s=u, v_move_m_s=v,
            )
    for key, u, v in (
        ("u_move_m_s", 500.0, 0.0),
        ("v_move_m_s", 0.0, -1.0e6),
    ):
        with pytest.raises(ValueError, match=f"{key}=.*exceeds"):
            RustIdealizedCarrierMap(
                idealized_library, SOUNDING.read_bytes(), iinit=1,
                geometry=geometry(), u_move_m_s=u, v_move_m_s=v,
            )
    with pytest.raises(TypeError, match="u_move_m_s must be a float"):
        RustIdealizedCarrierMap(
            idealized_library, SOUNDING.read_bytes(), iinit=1,
            geometry=geometry(), u_move_m_s="12.5",
        )


def test_the_frame_reaches_the_wind_carriers_and_nothing_else(
    idealized_library: Path,
) -> None:
    """Trace the offset: a half-applied frame is the failure to catch.

    A base state that moved while the initial condition did not - or the
    reverse - is a reference profile that is not the declared sounding's,
    and nothing downstream would say so.  This enumerates every carrier the
    initializer emits and names the complete set the frame may touch.
    """

    with RustIdealizedCarrierMap(
        idealized_library, b"", iinit=1, geometry=geometry(), isnd=5, iwnd=2,
    ) as still, RustIdealizedCarrierMap(
        idealized_library, b"", iinit=1, geometry=geometry(), isnd=5, iwnd=2,
        u_move_m_s=12.5, v_move_m_s=3.0,
    ) as moving:
        changed = {
            f"{group}.{name}"
            for group, carriers in still.groups.items()
            for name in carriers
            if not np.array_equal(
                still.carrier(group, name).view(np.uint32),
                moving.carrier(group, name).view(np.uint32),
            )
        }
        assert changed == {
            # base.F:2667-2668 subtracts the frame from the base winds ...
            "base_wind.u0",
            "base_wind.v0",
            # ... and init3d.F:210-219 copies those into the two prognostic
            # wind pairs, which is the whole of the change of variable.
            "terrain_open.ua",
            "terrain_open.u3d",
            "terrain_open.va",
            "terrain_open.v3d",
            # bc.F's lower boundary sets w = u*dzdx + v*dzdy.  On this flat
            # grid both slopes are zero, so w stays numerically zero; only
            # the sign of that zero follows the sign of the frame-relative
            # wind.  It is listed here because it is a real raw-bit change,
            # and asserted below to be nothing more than a signed zero.
            "terrain_open.wa",
            "terrain_open.w3d",
        }
        for name in ("wa", "w3d"):
            a = still.carrier("terrain_open", name)
            b = moving.carrier("terrain_open", name)
            assert np.count_nonzero(a) == 0
            assert np.count_nonzero(b) == 0
            assert np.array_equal(a, b)  # numerically identical


@requires_cm1_pristine
def test_a_declared_bubble_centre_moves_the_perturbation_init3d_builds(
    idealized_library: Path, tmp_path: Path
) -> None:
    """From configuration text to a different ``tha`` field, end to end.

    ``init3d.F:456-457`` sets the warm bubble's ``ric``/``rjc`` from
    ``param.F:7408``'s domain centre, and ``[initialization]
    bubble_center_x_m``/``_y_m`` are those two words.  The whole point of
    declaring them is that the perturbation lands somewhere else, so this
    reads the perturbation the real initializer built and checks that it did.

    The path exercised here is the one the door drives: TOML text ->
    ``load_config`` -> ``require_bubble_center`` -> ``IdealizedSource`` ->
    the Rust ``build_warm_bubble``.
    """

    from cm1core.cartesian_door import require_bubble_center, require_grid_extent
    from cm1core.config import load_config
    from cm1core.gm1_idealized_residency import IdealizedSource

    base = (ROOT / "examples/cartesian-uniform-grid.toml").read_text()
    # A 64 x 64 grid at 500 m: 32 km square, so a 10 km bubble fits anywhere
    # between 10 km and 22 km on either axis.
    base = base.replace("nx = 96", "nx = 64").replace("ny = 96", "ny = 64")

    def perturbation(document: str) -> np.ndarray:
        path = tmp_path / f"deck-{abs(hash(document))}.toml"
        path.write_text(document)
        config = load_config(path)
        grid = require_grid_extent(config)
        with IdealizedSource(
            idealized_library,
            SOUNDING.read_bytes(),
            grid,
            bubble_center_xy_m=require_bubble_center(config, grid),
        ) as source:
            return np.array(
                source.field("terrain_open.tha", layout="physical"), copy=True
            )

    centred = perturbation(base)
    moved = perturbation(
        base + "\n[initialization]\nisnd = 7\niwnd = 0\n"
        "bubble_center_x_m = 12000.0\nbubble_center_y_m = 20000.0\n"
    )
    assert centred.shape == moved.shape
    assert not np.array_equal(centred, moved)
    # Both are the same 1.0 K cos-squared bubble: only its position moved.
    assert float(moved.max()) == pytest.approx(float(centred.max()), abs=1e-6)

    def peak_xy(field: np.ndarray) -> tuple[float, float]:
        # The physical crop is C-order [k, j, i] with no halo, and CM1's
        # xh(i) on this uniform grid is (i + 0.5) * dx from an origin of 0.
        _, j, i = np.unravel_index(int(field.argmax()), field.shape)
        return ((i + 0.5) * 500.0, (j + 0.5) * 500.0)

    # param.F:7408 with minx = 0: centerx = centery = 0.5 * 64 * 500 = 16000.
    assert peak_xy(centred) == pytest.approx((16_000.0, 16_000.0), abs=250.0)
    assert peak_xy(moved) == pytest.approx((12_000.0, 20_000.0), abs=250.0)
