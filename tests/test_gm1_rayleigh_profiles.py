from __future__ import annotations

import ctypes
import hashlib
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from cm1core.gm1_idealized_in_memory import IdealizedGeometry, RustIdealizedCarrierMap
from cm1core.gm1_rayleigh_profiles import (
    RAYLEIGH_PARAMETER_NAMES,
    _configure,
    _last_error,
    build_uniform_rayleigh_profiles,
)
from authority_trees import requires_cm1_pristine


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def rayleigh_library(tmp_path_factory: pytest.TempPathFactory) -> Path:
    supplied = os.environ.get("GM1_RAYLEIGH_LIBRARY")
    if supplied:
        return Path(supplied).resolve(strict=True)
    target = tmp_path_factory.mktemp("gm1-rayleigh-cdylib")
    subprocess.run([
        "cargo", "build", "--release", "--lib", "--manifest-path",
        str(ROOT / "rust/cm1-state/Cargo.toml"), "--target-dir", str(target),
    ], check=True)
    name = "cm1_state.dll" if sys.platform == "win32" else "libcm1_state.so"
    return (target / "release" / name).resolve(strict=True)


def test_binding_import_is_free_of_numerical_runtimes() -> None:
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    subprocess.run([
        sys.executable, "-c",
        "import sys; import cm1core.gm1_rayleigh_profiles; "
        "assert 'numpy' not in sys.modules; assert 'cupy' not in sys.modules; "
        "assert 'gpuwm' not in sys.modules",
    ], check=True, env=env)


@pytest.mark.parametrize("nz,dz,mass_native,face_native", [
    (237, 75.0,
     "234fa7317871742a34cc2c7a20a5a39fb2ed28dc13ef67d34207b473a53b4e6f",
     "8733989b9b3445f9cb9abdf01ca8660158e456aa3c1d5018da47b25ebd016a49"),
    (75, 200.0,
     "8099fc9a112da0260c8798d2d46726f74c8f81393774c9aa6129befb4441cc9b",
     "8b672d0eb23bb0fbd270543f168fb9a67bae0190eb3638d503380398016fc740"),
    (237, float(np.float32(75.1)),
     "dee566c2b13dec7daad9a7dd7eca9d3f399c93d53eae5560023fc8ede75afbe6",
     "cd46431a655a84fd7fd178a2a5905c3d365d001fdacd91dfbdd5bc9c6f2168b6"),
])
def test_coefficients_match_the_measured_target_libm_bits(
    rayleigh_library: Path, nz: int, dz: float, mass_native: str, face_native: str,
) -> None:
    profiles = build_uniform_rayleigh_profiles(rayleigh_library, nz=nz, dz_m=dz)
    # Linux matches every native O0/O2/repeat word.  Windows has one measured
    # UCRT cosine difference at 237/75 mass index210; preserve that observation
    # as a target-libm regression, never relabel it native raw-bit identity.
    expected_mass = mass_native
    if sys.platform == "win32" and nz == 237 and dz == 75.0:
        expected_mass = "97333bbf015ca79082904ed070f4d345fde6ad6ff1f6e9f5a6ab9be2f0cf829b"
        assert profiles.tauh.view(np.uint32)[210] == 0x3E82E9B7
    for name, count, digest in (("tauh", nz, expected_mass), ("taus", nz, expected_mass),
                                ("tauf", nz + 1, face_native)):
        value = getattr(profiles, name)
        assert value.shape == (count,)
        assert value.dtype == np.dtype("<f4")
        assert not value.flags.writeable
        assert value.flags.c_contiguous
        assert hashlib.sha256(value.tobytes()).hexdigest() == digest
        assert profiles.receipt["profile_sha256"][name] == digest
    assert tuple(profiles.parameters) == RAYLEIGH_PARAMETER_NAMES
    assert profiles.receipt["alpha_applied_to_coefficients"] is False
    assert profiles.receipt["python_scientific_arithmetic"] is False


def test_rust_parameters_and_profiles_cannot_be_accidentally_mutated(rayleigh_library: Path) -> None:
    profiles = build_uniform_rayleigh_profiles(
        rayleigh_library, nz=8, dz_m=75.0, damping_depth_m=120.0,
    )
    assert dict(profiles.parameters) == {
        "dz_m": 75.0, "model_top_m": 600.0, "damping_start_m": 480.0,
        "damping_depth_m": 120.0,
        "damping_rate_s_inv": 0.0033333334140479565,
    }
    with pytest.raises(TypeError):
        profiles.parameters["damping_rate_s_inv"] = 0.0
    with pytest.raises(ValueError):
        profiles.tauh[0] = 1.0
    profiles.receipt["effective_f32_parameters"]["damping_rate_s_inv"] = 0.0
    assert profiles.parameters["damping_rate_s_inv"] == 0.0033333334140479565


def test_zero_depth_and_alpha_are_source_dispositions(rayleigh_library: Path) -> None:
    inactive = build_uniform_rayleigh_profiles(
        rayleigh_library, nz=8, dz_m=75.0, damping_depth_m=0.0,
    )
    assert all(np.all(getattr(inactive, name).view(np.uint32) == 0)
               for name in ("tauh", "taus", "tauf"))
    zero_rate = build_uniform_rayleigh_profiles(
        rayleigh_library, nz=8, dz_m=75.0, damping_depth_m=120.0, damping_rate_s_inv=0.0,
    )
    normal_rate = build_uniform_rayleigh_profiles(
        rayleigh_library, nz=8, dz_m=75.0, damping_depth_m=120.0,
    )
    assert zero_rate.receipt["profile_sha256"] == normal_rate.receipt["profile_sha256"]
    assert normal_rate.tauf[-1] == np.float32(1.0)


@pytest.mark.parametrize("changes,message", [
    ({"nz": 0}, "positive CM1 integer"),
    ({"nz": 2**64}, "positive CM1 integer"),
    ({"nz": True}, "positive CM1 integer"),
    ({"dz_m": 0.0}, "remain positive"),
    ({"dz_m": float("nan")}, "finite"),
    ({"dz_m": float(np.finfo(np.float32).max)}, "overflows"),
    ({"damping_depth_m": 400.0}, "half the model top"),
    ({"damping_depth_m": -1.0}, "nonnegative"),
    ({"damping_rate_s_inv": float("inf")}, "finite"),
])
def test_invalid_scientific_inputs_refuse(rayleigh_library: Path, changes: dict, message: str) -> None:
    kwargs = dict(nz=8, dz_m=75.0, damping_depth_m=120.0)
    kwargs.update(changes)
    with pytest.raises(ValueError, match=message):
        build_uniform_rayleigh_profiles(rayleigh_library, **kwargs)


def test_c_abi_refusal_does_not_partially_overwrite_outputs(rayleigh_library: Path) -> None:
    library = ctypes.CDLL(str(rayleigh_library))
    _configure(library)
    values = [np.full(count, -17.0, dtype=np.float32) for count in (8, 8, 9, 5)]
    f32p = ctypes.POINTER(ctypes.c_float)
    pointers = [value.ctypes.data_as(f32p) for value in values]
    for alias, short, depth, message in ((True, False, 120.0, "overlap"),
                                         (False, True, 120.0, "lengths"),
                                         (False, False, 400.0, "half the model top")):
        status = library.gm1_rayleigh_profiles_fill(
            8, 75.0, depth, 0.0033333333333,
            pointers[0], 7 if short else 8,
            pointers[0] if alias else pointers[1], 8,
            pointers[2], 9, pointers[3], 5,
        )
        assert status == -1
        assert message in _last_error(library)
        assert all(np.all(value.view(np.uint32) == np.float32(-17.0).view(np.uint32))
                   for value in values)


@requires_cm1_pristine
def test_added_exports_preserve_the_existing_initializer_inventory(rayleigh_library: Path) -> None:
    sounding = (ROOT / "upstream/cm1-21.1/pristine/run/config_files/sea_breeze/input_sounding").read_bytes()
    geometry = IdealizedGeometry((4, 2, 3), (1000.0, 1000.0, 500.0), (0.0, 0.0), (1500.0, 1000.0))
    with RustIdealizedCarrierMap(rayleigh_library, sounding, iinit=1, geometry=geometry) as owner:
        assert {name: len(group) for name, group in owner.groups.items()} == {
            "terrain_open": 54, "base_wind": 2, "moist": 4, "terrain_static": 25,
            # base.F's qc0/qi0, exact zero for this sounding.
            "base_condensate": 2,
        }
        assert hashlib.sha256(owner.carrier("terrain_open", "pi0").tobytes()).hexdigest() == (
            "7be5ea48e48b6e41ada959266dcbec2fb1c635de3fd975585645067cb29a5f93"
        )


def test_the_level_entry_reproduces_the_uniform_one_raw_bit_exactly(
    rayleigh_library: Path,
) -> None:
    """param.F:8652-8671 reads zh and zf; a uniform column is one such pair."""

    from cm1core.cm1_vertical_grid import uniform_vertical_levels
    from cm1core.gm1_rayleigh_profiles import build_rayleigh_profiles

    for nz, dz in ((237, 75.0), (75, 200.0), (40, 500.0)):
        uniform = build_uniform_rayleigh_profiles(rayleigh_library, nz=nz, dz_m=dz)
        levels = build_rayleigh_profiles(
            rayleigh_library, levels=uniform_vertical_levels(nz, dz)
        )
        for name in ("tauh", "taus", "tauf"):
            assert getattr(levels, name).tobytes() == getattr(uniform, name).tobytes()
        assert dict(levels.parameters) == dict(uniform.parameters)
        # A stretch_z=0 coordinate takes the established entry point, so even
        # the receipt's producer name is unchanged.
        assert levels.receipt["producer"] == "Rust gm1_rayleigh_profiles_fill"


def test_a_stretched_column_damps_from_its_own_top_and_levels(
    rayleigh_library: Path,
) -> None:
    from cm1core.cm1_vertical_grid import build_vertical_levels
    from cm1core.gm1_rayleigh_profiles import build_rayleigh_profiles
    from cm1core.stretch_z1_child_15km import STRETCH_Z1_CHILD_15KM_PARAMETERS as P

    levels = build_vertical_levels(
        stretch_z=1, nk=P.nk, dz_m=P.dz, ztop_m=P.ztop, str_bot_m=P.str_bot,
        str_top_m=P.str_top, dz_bot_m=P.dz_bot, dz_top_m=P.dz_top,
    )
    profiles = build_rayleigh_profiles(
        rayleigh_library, levels=levels, damping_depth_m=5000.0
    )
    assert profiles.receipt["producer"] == "Rust gm1_rayleigh_profiles_fill_levels"
    # param.F:8193 and :8656: maxz is zf(nk+1)=15000, and zd = maxz - depth.
    assert profiles.parameters["model_top_m"] == 15000.0
    assert profiles.parameters["damping_start_m"] == 10000.0
    # param.F:8170 keeps dz as the metric reference, not a layer depth.
    assert profiles.parameters["dz_m"] == 100.0
    assert profiles.tauf[0] == np.float32(0.0)
    assert profiles.tauf[P.nk] == np.float32(1.0)
    for height, tau in zip(levels.mass_heights_m, profiles.tauh):
        assert (tau > 0.0) == (height > 10000.0)
    assert np.all(np.diff(profiles.tauf) >= np.float32(0.0))
    # The 75-level uniform column with the same nz is a different atmosphere.
    uniform = build_uniform_rayleigh_profiles(
        rayleigh_library, nz=P.nk, dz_m=200.0, damping_depth_m=5000.0
    )
    assert uniform.tauh.tobytes() != profiles.tauh.tobytes()
    assert profiles.receipt["vertical_coordinate"]["levels_below_1000m"] == 22


@requires_cm1_pristine
def test_the_in_memory_initializer_takes_a_declared_coordinate(
    rayleigh_library: Path,
) -> None:
    """The orchestration seam for param.F:7814-8139, driven end to end."""

    from cm1core.cm1_vertical_grid import build_vertical_levels, uniform_vertical_levels

    sounding = (
        ROOT / "upstream/cm1-21.1/pristine/run/config_files/sea_breeze/input_sounding"
    ).read_bytes()
    # Four 20 m layers then eight of 320 m: zh(1) is 10 m, not 0.5*dz for any
    # dz, and the model top is the declared 2640 m.
    faces = [0.0, 20.0, 40.0, 60.0, 80.0]
    while len(faces) < 13:
        faces.append(faces[-1] + 320.0)
    levels = build_vertical_levels(
        stretch_z=4, nk=12, dz_m=100.0, ztop_m=faces[-1], levels_m=faces
    )
    geometry = IdealizedGeometry((12, 2, 3), (1000.0, 1000.0, 100.0), (0.0, 0.0), (1500.0, 1000.0))
    with RustIdealizedCarrierMap(
        rayleigh_library, sounding, iinit=1, geometry=geometry, levels=levels
    ) as owner:
        zh = owner.carrier("terrain_static", "zh")
        mh = owner.carrier("terrain_static", "mh")
        assert zh.reshape(12, -1)[0, 0] == np.float32(10.0)
        assert zh.reshape(12, -1)[11, 0] == np.float32(levels.mass_heights_m[11])
        # param.F:8170: dz over the layer depth, not one.
        assert mh.reshape(12, -1)[0, 0] == np.float32(5.0)
        assert mh.reshape(12, -1)[11, 0] == np.float32(0.3125)

    # The stretch_z=0 member takes the established entry point, so a caller
    # that passes it gets the state it has always got, word for word.
    uniform_geometry = IdealizedGeometry(
        (4, 2, 3), (1000.0, 1000.0, 500.0), (0.0, 0.0), (1500.0, 1000.0)
    )
    with RustIdealizedCarrierMap(
        rayleigh_library, sounding, iinit=1, geometry=uniform_geometry
    ) as plain, RustIdealizedCarrierMap(
        rayleigh_library, sounding, iinit=1, geometry=uniform_geometry,
        levels=uniform_vertical_levels(4, 500.0),
    ) as declared:
        for group, names in plain.groups.items():
            for name in names:
                assert (
                    plain.carrier(group, name).tobytes()
                    == declared.carrier(group, name).tobytes()
                ), f"{group}.{name}"

    with pytest.raises(ValueError, match="exactly the geometry's nz"):
        RustIdealizedCarrierMap(
            rayleigh_library, sounding, iinit=1, geometry=uniform_geometry, levels=levels
        )
    with pytest.raises(TypeError, match="VerticalLevels"):
        RustIdealizedCarrierMap(
            rayleigh_library, sounding, iinit=1, geometry=uniform_geometry, levels=object()
        )
