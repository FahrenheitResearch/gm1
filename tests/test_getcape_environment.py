"""CM1's CAPE/CIN/LCL/LFC, from TOML text to a changed published frame.

The capability under test is `src/getcape.F:14-594`, reached through two CM1
call sites: `src/base.F:3143-3189`, which prints the parameters of the
base-state sounding for every moist run, and `src/writeout.F:2895-3030`, which
computes the four 2-D planes per column and writes whichever the
`output_cape`/`output_cin`/`output_lcl`/`output_lfc` switches select.

Every test here runs on the host.  The device kernel is proved by compiling
the packaged kernel *text* with a C++ compiler and comparing raw words against
the Python source-order oracle; the device itself is GPU-gated elsewhere.
"""

from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from cm1core import cartesian_door  # noqa: E402
from cm1core.cartesian_door import build_runtime_config, run_cartesian  # noqa: E402
from cm1core.cartesian_environment import (  # noqa: E402
    ENVIRONMENT_GROUP,
    environment_field_names,
    environment_scratch_bytes,
)
from cm1core.cartesian_history import cartesian_history_profile  # noqa: E402
from cm1core.cm1_output_selection import OUTPUT_GROUP  # noqa: E402
from cm1core.config import (  # noqa: E402
    NO_ENVIRONMENT_FIELDS,
    load_config,
    parse_mapping,
)
from cm1core.cuda_getcape import (  # noqa: E402
    CUDA_GETCAPE_KERNEL_NAME,
    CUDA_GETCAPE_SOURCE,
    CUDA_GETCAPE_SOURCE_BYTES,
    CUDA_GETCAPE_SOURCE_SHA256,
    GETCAPE_FIELD_NAMES,
    GETCAPE_FIELD_SWITCHES,
    GetCapeFieldSelection,
    cpu_getcape,
    require_getcape_selection,
)
from cm1core.errors import CM1Refusal  # noqa: E402
from cm1core.sounding_report import format_sounding_report  # noqa: E402

EXAMPLES = ROOT / "examples"
UNIFORM = EXAMPLES / "cartesian-uniform-grid.toml"
ENVIRONMENT = EXAMPLES / "cartesian-environment-fields.toml"
ANALYTIC = EXAMPLES / "cartesian-wk-analytic-sounding-shear.toml"

#: Every configuration document this port shipped before the environment
#: fields became declarable, with its canonical sha256 at 6c6c9c4 -- the commit
#: immediately before this branch.  A document that declares no [diagnostics]
#: table projects no diagnostics key, so these must not move.
PRE_EXISTING_DECK_SHA256 = {
    "cartesian-fifty-metre-fine-step.toml":
        "39c3e8198371318b55955f26c1cf94c83d831a1945d6d132544f9cebbfa6fcef",
    "cartesian-stretched-tornado-grade-grid.toml":
        "d4f0b4a9fa9bf371a6e726067fa4ce4998925cbdd876831fbad6365212a9d811",
    "cartesian-supercell-attempt.toml":
        "f325e5e78bffd1d57052b1c1dd4e085b1eb91f619070bb83a3d85b02b22b333b",
    "cartesian-supercell-surface-drag.toml":
        "5bc41131130449a709a673f780c47e844187e06d105af339391930d7bf8f0a77",
    "cartesian-supercell-translating-frame.toml":
        "77288cd92837908d70baedebc1aa914c2dc6a5bdb1f90d3d8255cfed15a545ca",
    "cartesian-uniform-grid.toml":
        "60b552c0a1cccd9174bc839d15b1258aac7f97999f8fd206be74a51975f1db48",
    "cartesian-wk-analytic-sounding-shear.toml":
        "e6ca95ed63f9555381a4ec8d04b93cf4cfff324119b0fb0df1ed9d14e696cee7",
    "dry-psolver2-bootstrap.toml":
        "0394dea1f77722e54fe6b1b5a07fe56037cda84800ef870b9a269566e6043340",
    "official-supercell-native-context.toml":
        "33761ba3ef1e35ce8dd6fa3fa126c4ed2437bdbe49d3b9e73adb2fa2bcf92b8c",
    "official-supercell-product.toml":
        "b3dc296531782eaa166eb26cec30fa030b7405485af1cf2e3ff1af23351e4038",
    "real-atmosphere-100m-product.toml":
        "07c7dde701491ce71a70cec8c56de2cd50ed84574cb098250815d76eaa0c9c00",
}


def _deck(**diagnostics: object) -> dict:
    """The shipped uniform deck as a mapping, with a [diagnostics] table."""

    import tomllib

    raw = tomllib.loads(UNIFORM.read_text(encoding="utf-8"))
    if diagnostics:
        raw["diagnostics"] = {
            "output_cape": False,
            "output_cin": False,
            "output_lcl": False,
            "output_lfc": False,
            "parcel_source": 3,
            "adiabat": 1,
            "mixed_layer_depth_m": 500.0,
            "pressure_increment_pa": 200.0,
            "storm_motion": "bunkers-right",
            **diagnostics,
        }
    return raw


@pytest.fixture(scope="module")
def initializer_library() -> Path:
    """The built cm1-state cdylib, reused in tree rather than rebuilt."""

    supplied = os.environ.get("GM1_CM1_STATE_LIBRARY")
    if supplied:
        return Path(supplied).resolve(strict=True)
    name = "cm1_state.dll" if sys.platform == "win32" else "libcm1_state.so"
    built = ROOT / "rust/cm1-state/target/release" / name
    if not built.is_file():
        result = subprocess.run(
            [
                "cargo", "build", "--offline", "--release", "--lib",
                "--manifest-path", str(ROOT / "rust/cm1-state/Cargo.toml"),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0 or not built.is_file():
            pytest.skip(f"the cm1-state cdylib is not built here: {result.stderr[-400:]}")
    return built.resolve()


# --------------------------------------------------------------------------
# The default document must not move.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(PRE_EXISTING_DECK_SHA256))
def test_every_pre_existing_deck_is_byte_identical(name: str) -> None:
    """Adding the environment fields must not move a sealed configuration hash."""

    config = load_config(EXAMPLES / name)
    assert config.sha256 == PRE_EXISTING_DECK_SHA256[name]
    assert config.diagnostics is None
    assert config.environment_fields == NO_ENVIRONMENT_FIELDS
    assert "diagnostics" not in config.canonical_dict()


def test_the_default_history_frame_is_byte_for_byte_what_it_was() -> None:
    """writeout.F:2895: no switch, no sweep, no plane, no change in payload.

    The Cartesian lane's published payload is a sealed number -- the whole
    bit-reproducibility claim is stated in bytes -- so the carrier table of a
    document that declares no [diagnostics] table has to be the twelve
    carriers it has always been, and the payload the same integer.
    """

    runtime = build_runtime_config(load_config(UNIFORM))
    profile = cartesian_history_profile(runtime)
    names = [carrier.name for carrier in profile.carriers]
    # The twelve carriers it has always published, first and in order...
    assert names[:12] == [
        "closure.tkea", "pp3d", "prs", "q3d", "rf", "rho", "terrain_msl",
        "th3d", "u3d", "v3d", "w3d", "zf",
    ]
    # ...then CM1's default &param9 products, appended after them by the
    # int/output merge 7183ce6 (so the twelve keep their arena offsets), with
    # the rotation/swath words and output_dbz defaulting off in the tables that
    # own them since a0cee5e/f9be0a4.  No getcape plane is among them.
    assert names[12:] == [
        "rain", "prate", "snownc", "graupelnc", "sr", "th", "zhval",
        "pi0", "th0", "prs0", "qv0", "u0", "v0",
    ]
    twelve = sum(carrier.nbytes for carrier in profile.carriers[:12])
    assert twelve == 23_808_000
    assert profile.payload_nbytes == 25_468_000
    assert not runtime.environment_field_selection.any_field
    assert environment_field_names(runtime.environment_field_selection) == ()
    assert environment_scratch_bytes(runtime, runtime.environment_field_selection) == {
        "column_scratch_bytes": 0,
        "field_plane_bytes": 0,
        "getcape_scratch_peak_bytes": 0,
    }


# --------------------------------------------------------------------------
# From TOML text to a changed frame.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("field", GETCAPE_FIELD_NAMES)
def test_one_switch_in_the_toml_adds_exactly_one_plane_to_the_frame(field: str) -> None:
    """writeout.F:2937-3028: each switch publishes its own plane and no other."""

    switch = GETCAPE_FIELD_SWITCHES[field]
    off = build_runtime_config(parse_mapping(_deck(**{switch: False})))
    on = build_runtime_config(parse_mapping(_deck(**{switch: True})))
    off_profile = cartesian_history_profile(off)
    on_profile = cartesian_history_profile(on)
    added = [
        carrier
        for carrier in on_profile.carriers
        if carrier.name not in {row.name for row in off_profile.carriers}
    ]
    assert [carrier.name for carrier in added] == [field]
    plane = added[0]
    # One producer per CM1 name: the &param9 engine, whose getcape kernel is
    # differenced against CM1's own compiled getcape.F, publishes the plane.
    # The environment sweep would publish a second carrier under this name.
    assert plane.group == OUTPUT_GROUP
    assert [c.name for c in on_profile.carriers].count(field) == 1
    assert not any(c.group == ENVIRONMENT_GROUP for c in on_profile.carriers)
    assert plane.shape == on.grid.surface_shape
    assert plane.stagger == "surface"
    assert switch in plane.meaning and "mixed-layer parcel" in plane.meaning
    grid = on.grid
    assert (
        on_profile.payload_nbytes - off_profile.payload_nbytes
        == 4 * grid.nx * grid.ny
    )
    assert environment_field_names(on.environment_field_selection) == (field,)


def test_the_shipped_environment_deck_publishes_all_four_planes() -> None:
    """The document a user would actually write, followed to the carrier table."""

    config = load_config(ENVIRONMENT)
    assert config.diagnostics is not None
    assert config.sha256 != load_config(UNIFORM).sha256
    runtime = build_runtime_config(config)
    profile = cartesian_history_profile(runtime)
    names = [carrier.name for carrier in profile.carriers]
    # Each plane exactly once, in writeout.F's order, from the &param9 engine.
    assert [name for name in names if name in GETCAPE_FIELD_NAMES] == list(GETCAPE_FIELD_NAMES)
    assert len(names) == len(set(names))
    uniform = cartesian_history_profile(build_runtime_config(load_config(UNIFORM)))
    assert profile.payload_nbytes == uniform.payload_nbytes + 4 * 4 * 96 * 96
    assert cartesian_door.cartesian_history_plan(runtime)["carrier_names"] == names
    record = runtime.record()
    assert record["environment_fields"]["published_fields"] == list(GETCAPE_FIELD_NAMES)
    assert record["environment_fields"]["parcel_source_name"] == "mixed-layer"
    assert record["environment_fields"]["adiabat_name"] == "pseudoadiabatic-liquid"
    assert record["storm_motion"] == "bunkers-right"
    assert "writeout.F:2895-3030" in record["environment_field_authority"]
    priced = environment_scratch_bytes(runtime, runtime.environment_field_selection)
    assert priced["getcape_scratch_peak_bytes"] > 0


def test_the_parcel_and_adiabat_in_the_toml_reach_the_carrier_description() -> None:
    """A document that lifts a surface parcel must not be described as mixed-layer."""

    config = parse_mapping(_deck(output_cape=True, parcel_source=1, adiabat=2))
    runtime = build_runtime_config(config)
    assert runtime.parcel_source == 1
    assert runtime.adiabat == 2
    profile = cartesian_history_profile(runtime)
    plane = next(c for c in profile.carriers if c.name == "cape")
    assert "surface parcel" in plane.meaning
    record = runtime.record()["environment_fields"]
    assert record["parcel_source_name"] == "surface"
    assert record["adiabat_name"] == "reversible-liquid"


def test_the_getcape_controls_in_the_toml_reach_the_runtime_contract() -> None:
    config = parse_mapping(
        _deck(output_cin=True, mixed_layer_depth_m=1000.0, pressure_increment_pa=50.0)
    )
    runtime = build_runtime_config(config)
    assert runtime.mixed_layer_depth_m == 1000.0
    assert runtime.pressure_increment_pa == 50.0
    selection = runtime.environment_field_selection
    assert selection.mixed_layer_depth_m == 1000.0
    assert selection.pressure_increment_pa == 50.0
    # And the controls actually change the arithmetic they name.
    column = _column()
    shallow = cpu_getcape(*column, mixed_layer_depth_m=500.0)
    deep = cpu_getcape(*column, mixed_layer_depth_m=1000.0)
    assert shallow["cape"] != deep["cape"]


def test_the_storm_motion_in_the_toml_reaches_the_report(
    initializer_library: Path,
) -> None:
    """The SRH sign flips between the movers, so the name has to travel."""

    from cm1core.cartesian_door import require_initializer_selection

    reports = {}
    for motion in ("bunkers-right", "bunkers-left"):
        config = parse_mapping(_deck(storm_motion=motion))
        assert build_runtime_config(config).storm_motion == motion
        selection, _ = require_initializer_selection(config)
        reports[motion] = cartesian_door.require_sounding_report(
            config,
            library=initializer_library,
            sounding_bytes=_SOUNDING.encode("utf-8"),
            selection=selection,
        )
    right, left = reports["bunkers-right"], reports["bunkers-left"]
    assert right["storm_motion"] == "bunkers-right"
    assert left["storm_motion"] == "bunkers-left"
    assert right["srh_0_3km_m2_per_s2"] != left["srh_0_3km_m2_per_s2"]
    # Both movers share the mean wind and the shear vector; only the deviation
    # differs, so the shear the report publishes cannot depend on the choice.
    assert right["shear_0_6km_m_per_s"] == left["shear_0_6km_m_per_s"]


# --------------------------------------------------------------------------
# Refusals: by key and by value, each naming the wrong result it prevents.
# --------------------------------------------------------------------------


def test_an_unknown_diagnostics_key_is_refused_by_name() -> None:
    raw = _deck(output_cape=True)
    raw["diagnostics"]["output_srh"] = True
    with pytest.raises(CM1Refusal) as caught:
        parse_mapping(raw)
    assert caught.value.detail.code == "CM1-CFG-UNKNOWN"
    assert caught.value.detail.option == "diagnostics.output_srh"


def test_a_partial_diagnostics_table_is_refused_by_the_missing_key() -> None:
    raw = _deck(output_cape=True)
    del raw["diagnostics"]["adiabat"]
    with pytest.raises(CM1Refusal) as caught:
        parse_mapping(raw)
    # a0cee5e (the uhdiag merge) made [diagnostics] two all-or-none families,
    # so a half-declared getcape family is refused by its own code, still
    # naming the missing key.
    assert caught.value.detail.code == "CM1-CFG-DIAGNOSTICS-PARTIAL"
    assert caught.value.detail.option == "diagnostics.adiabat"
    assert "diagnostics.adiabat" in caught.value.detail.remedy


def test_a_diagnostics_table_outside_the_cartesian_lane_is_refused() -> None:
    import tomllib

    raw = tomllib.loads((EXAMPLES / "official-supercell-product.toml").read_text())
    raw["diagnostics"] = {"output_cape": True}
    with pytest.raises(CM1Refusal) as caught:
        parse_mapping(raw)
    assert caught.value.detail.code == "CM1-CFG-UNKNOWN"
    assert caught.value.detail.option == "diagnostics"


@pytest.mark.parametrize(
    "key,value,anchor",
    [
        ("parcel_source", 0, "getcape.F:242-250"),
        ("parcel_source", 4, "getcape.F:242-250"),
        ("adiabat", 0, "getcape.F:419-423"),
        ("adiabat", 5, "getcape.F:419-423"),
    ],
)
def test_a_parcel_or_adiabat_getcape_stops_on_is_refused_by_value(
    key: str, value: int, anchor: str
) -> None:
    with pytest.raises(CM1Refusal) as caught:
        parse_mapping(_deck(output_cape=True, **{key: value}))
    detail = caught.value.detail
    assert detail.option == f"diagnostics.{key}"
    assert detail.value == value
    assert anchor in detail.reason


@pytest.mark.parametrize(
    "key", ["mixed_layer_depth_m", "pressure_increment_pa"]
)
@pytest.mark.parametrize("value", [0.0, -1.0])
def test_a_nonpositive_getcape_control_is_refused_by_value(key: str, value: float) -> None:
    with pytest.raises(CM1Refusal) as caught:
        parse_mapping(_deck(output_cape=True, **{key: value}))
    assert caught.value.detail.option == f"diagnostics.{key}"
    assert caught.value.detail.value == value


def test_an_unnamed_storm_motion_is_refused_by_value() -> None:
    with pytest.raises(CM1Refusal) as caught:
        parse_mapping(_deck(storm_motion="mean-wind"))
    detail = caught.value.detail
    assert detail.option == "diagnostics.storm_motion"
    assert detail.value == "mean-wind"
    assert "Bunkers" in detail.reason
    assert "flips" in detail.reason


@pytest.mark.parametrize("switch", sorted(GETCAPE_FIELD_SWITCHES.values()))
def test_a_nonboolean_output_switch_is_refused_by_value(switch: str) -> None:
    with pytest.raises(CM1Refusal) as caught:
        parse_mapping(_deck(**{switch: 1}))
    assert caught.value.detail.option == f"diagnostics.{switch}"


def test_the_runtime_contract_refuses_a_parcel_the_door_would_have_caught() -> None:
    """A programmatic caller that bypasses the door still cannot ask for one."""

    from cm1core.grid_geometry import CartesianGridGeometry
    from cm1core.gm1_runtime import RuntimeConfig

    grid = CartesianGridGeometry(nx=16, ny=16, nz=16, dx_m=500.0, dy_m=500.0, dz_m=500.0)
    with pytest.raises(CM1Refusal) as caught:
        RuntimeConfig(grid=grid, output_cape=True, parcel_source=9)
    assert caught.value.detail.code == "CM1-GETCAPE-PARCEL"
    with pytest.raises(CM1Refusal) as caught:
        RuntimeConfig(grid=grid, adiabat=0)
    assert caught.value.detail.code == "CM1-GETCAPE-ADIABAT"
    with pytest.raises(CM1Refusal) as caught:
        RuntimeConfig(grid=grid, storm_motion="mean-wind")
    assert caught.value.detail.code == "CM1-SOUNDING-REPORT-MOTION"
    assert "flips" in caught.value.detail.reason or "sign" in caught.value.detail.reason


def test_every_refusal_names_a_concrete_wrong_result() -> None:
    """A refusal that does not say what it prevents is not a refusal."""

    for selection, code in (
        (GetCapeFieldSelection(parcel_source=7), "CM1-GETCAPE-PARCEL"),
        (GetCapeFieldSelection(adiabat=7), "CM1-GETCAPE-ADIABAT"),
        (GetCapeFieldSelection(mixed_layer_depth_m=0.0), "CM1-GETCAPE-CONTROL"),
        (GetCapeFieldSelection(pressure_increment_pa=0.0), "CM1-GETCAPE-CONTROL"),
    ):
        with pytest.raises(CM1Refusal) as caught:
            require_getcape_selection(selection)
        assert caught.value.detail.code == code
        assert len(caught.value.detail.reason) > 80
    # And the valid selection the default document is in must not refuse.
    require_getcape_selection(GetCapeFieldSelection())
    require_getcape_selection(
        GetCapeFieldSelection(output_cape=True, parcel_source=1, adiabat=4)
    )


# --------------------------------------------------------------------------
# The transcription itself.
# --------------------------------------------------------------------------


def _column() -> tuple[list[float], list[float], list[float]]:
    """One analytic convective column; no fixture, place or date in it."""

    import math

    p, t, q = [], [], []
    for level in range(41):
        z = level * 500.0
        p.append(1000.0 * math.exp(-z / 8500.0))
        t.append(30.0 - 8.0e-3 * min(z, 12000.0))
        q.append(max(0.017 * math.exp(-z / 2500.0), 1.0e-8))
    return p, t, q


def _synthetic_sounding(levels: int = 48, dz: float = 500.0) -> str:
    """One admissible ``input_sounding`` in CM1's own layout, built from nothing.

    ``base.F:495-607``: a three-word header in hPa/K/(g/kg), then rows of
    ``z theta qv u v`` with strictly increasing height.  The profile is a
    constant-stability troposphere under a stable stratosphere with a veering,
    deepening hodograph, which is enough to reach an LCL, an LFC and an EL and
    to give the two Bunkers movers different helicity.  It names no place,
    date or fixture.
    """

    import math

    rows = ["  1000.00   300.00   14.00"]
    for level in range(levels):
        z = (level + 0.5) * dz
        theta = 300.0 + (0.004 * z if z < 12000.0 else 48.0 + 0.012 * (z - 12000.0))
        qv = max(14.0 * math.exp(-z / 2600.0), 0.004)
        fraction = min(z / 6000.0, 1.0)
        angle = fraction * math.pi / 2.0
        u = 22.0 * math.sin(angle)
        v = 9.0 * math.cos(angle) - 9.0
        rows.append(f"  {z:10.4f} {theta:12.6f} {qv:12.6f} {u:10.4f} {v:10.4f}")
    return "\n".join(rows) + "\n"


_SOUNDING = _synthetic_sounding()


def test_the_python_oracle_and_the_rust_transcription_agree_word_for_word(
    initializer_library: Path,
) -> None:
    """Two independent transcriptions of getcape.F, compared raw word by word.

    The Python oracle is what the packaged CUDA text is checked against and
    the Rust one is what every run receipt reports.  If they ever diverge, one
    of them has stopped being getcape.F.
    """

    import numpy as np

    library = ctypes.CDLL(str(initializer_library))
    f32p = ctypes.POINTER(ctypes.c_float)
    library.gm1_getcape_column.restype = ctypes.c_int32
    library.gm1_getcape_column.argtypes = (
        ctypes.c_size_t, f32p, f32p, f32p, ctypes.c_int32, ctypes.c_int32,
        ctypes.c_float, ctypes.c_float, f32p,
    )
    p, t, q = (np.ascontiguousarray(values, dtype=np.float32) for values in _column())
    for source in (1, 2, 3):
        for adiabat in (1, 2, 3, 4):
            out = np.zeros(9, dtype=np.float32)
            status = library.gm1_getcape_column(
                len(p),
                p.ctypes.data_as(f32p),
                t.ctypes.data_as(f32p),
                q.ctypes.data_as(f32p),
                source,
                adiabat,
                ctypes.c_float(500.0),
                ctypes.c_float(200.0),
                out.ctypes.data_as(f32p),
            )
            assert status == 0
            oracle = cpu_getcape(p, t, q, source=source, adiabat=adiabat)
            for index, name in enumerate(
                ("cape", "cin", "lcl", "lfc", "el", "psource", "tsource", "qvsource")
            ):
                assert out[index].tobytes() == np.float32(oracle[name]).tobytes(), (
                    f"source={source} adiabat={adiabat} {name}: "
                    f"rust {out[index]!r} vs python {oracle[name]!r}"
                )
            assert bool(out[8]) == oracle["converged"]
            assert oracle["cape"] > 0.0


def test_the_packaged_kernel_text_is_pinned_and_uses_only_rounding_intrinsics() -> None:
    import hashlib

    assert hashlib.sha256(CUDA_GETCAPE_SOURCE_BYTES).hexdigest() == CUDA_GETCAPE_SOURCE_SHA256
    assert f"void {CUDA_GETCAPE_KERNEL_NAME}(" in CUDA_GETCAPE_SOURCE
    for intrinsic in ("__fadd_rn", "__fsub_rn", "__fmul_rn", "__fdiv_rn"):
        assert intrinsic in CUDA_GETCAPE_SOURCE
    # The four planes CM1 writes, and the anchors a reader follows back.
    for anchor in ("getcape.F:14-594", "writeout.F:2907-2914", "getcape.F:526-538"):
        assert anchor in CUDA_GETCAPE_SOURCE
    for name in GETCAPE_FIELD_NAMES:
        assert f"out_{name}" in CUDA_GETCAPE_SOURCE


def test_the_packaged_kernel_text_reproduces_the_oracle_raw_word_for_raw_word() -> None:
    """Compile the shipped CUDA text with g++ and compare every output word.

    This is the whole device claim that can be made without a device: the text
    is well-formed under -Wall -Wextra -Werror, it reproduces the source-order
    oracle exactly on every parcel and every adiabat, and its per-column result
    does not depend on the order or the number of threads.  NVRTC code
    generation and CUDA's own expf/logf/powf are GPU-gated and stated as such.
    """

    from oracle.getcape_host_oracle import OracleError, verify

    try:
        receipt = verify()
    except OracleError as error:
        pytest.fail(str(error))
    except FileNotFoundError:
        pytest.skip("no host C++ compiler is installed")
    assert receipt["status"] == "passed-host-raw-word-exact-all-parcels-and-adiabats"
    assert receipt["kernel_source_sha256"] == CUDA_GETCAPE_SOURCE_SHA256
    assert len(receipt["comparisons"]) == 3 * 4 * 2 * 3
    assert all(
        all(row["raw_word_exact"].values()) for row in receipt["comparisons"]
    )
    # The fixture has to be a convective one, or the comparison proves nothing.
    planes = receipt["cm1_default_selector_planes"]
    assert min(planes["cape"]) > 1000.0
    assert min(planes["lcl"]) > 0.0 and min(planes["lfc"]) > 0.0
    assert all(lfc >= lcl for lcl, lfc in zip(planes["lcl"], planes["lfc"]))


# --------------------------------------------------------------------------
# The t=0 report: every run prints it and every receipt carries it.
# --------------------------------------------------------------------------


def test_every_run_records_its_own_initial_sounding_parameters(
    initializer_library: Path, monkeypatch, tmp_path: Path, capsys
) -> None:
    """From a TOML with no [diagnostics] table to a receipt that states the air.

    The device, the estate and the execution are stood in for; what is under
    test is that ``run_cartesian`` computes base.F:3143-3189's report on the
    host, prints it, and writes it into the receipt -- for a document that asks
    for no 2-D planes at all, because CM1 gates that block on nothing either.
    """

    monkeypatch.setattr(cartesian_door, "require_runnable", lambda config, **kwargs: None)
    monkeypatch.setattr(
        cartesian_door, "resolve_initializer_library", lambda argument: initializer_library
    )
    monkeypatch.setattr(cartesian_door, "require_cuda_runtime", lambda: None)
    monkeypatch.setattr(
        cartesian_door, "_execute",
        lambda **kwargs: ({"completed_steps": 11, "status": "ok"}, None),
    )
    receipt_path = tmp_path / "receipt.json"
    assert run_cartesian(
        load_config(ANALYTIC), config_label=str(ANALYTIC), output=receipt_path
    ) == 0
    written = json.loads(receipt_path.read_text(encoding="utf-8"))
    report = written["initial_sounding_parameters"]
    assert report["schema"] == "gm1.sounding-parameters/v1"
    # base.F:3165-3167 runs all three parcels and this reports all three.
    for parcel in ("surface_based", "most_unstable", "mixed_layer"):
        assert report[parcel]["converged"] is True
        assert report[parcel]["cape_j_per_kg"] > 0.0
        assert report[parcel]["el_m_agl"] > report[parcel]["lcl_m_agl"] > 0.0
    assert report["srh_0_1km_m2_per_s2"] != 0.0
    assert report["shear_0_6km_m_per_s"] > report["shear_0_1km_m_per_s"]
    assert report["storm_motion"] == "bunkers-right"
    assert "getcape.F" in report["cm1_parcel_authority"]
    assert "Bunkers" in report["storm_motion_authority"]
    assert "1 hPa" in report["claim_limit"]
    # And a human reading the terminal sees it, not only the JSON.
    printed = capsys.readouterr().err
    assert "Thermodynamic properties of the initial sounding" in printed
    assert "0-1,0-3 km SRH" in printed
    assert format_sounding_report(report) in printed


def test_the_sustained_storm_runner_records_the_report_too(
    initializer_library: Path, monkeypatch, tmp_path: Path, capsys
) -> None:
    """The two-hour driver is a second door; a capability reachable from only
    one of them is the defect this branch exists to end.

    ``tools/run_cartesian_storm.py`` does not call ``run_cartesian``: it drives
    the runtime itself and streams its own diagnostic record.  The t=0 report
    has to reach that record's header as well, because a run that stops writes
    no end-of-run receipt and the header is then the only surviving statement
    of what atmosphere it was integrating.
    """

    from tools import run_cartesian_storm

    sounding = tmp_path / "input_sounding"
    sounding.write_text(_SOUNDING, encoding="utf-8")
    monkeypatch.setattr(run_cartesian_storm, "require_runnable", lambda config, **kwargs: None)
    monkeypatch.setattr(
        run_cartesian_storm,
        "resolve_initializer_library",
        lambda argument: initializer_library,
    )

    class _Stop(RuntimeError):
        pass

    def _no_device():
        raise _Stop("the device is not opened by this test")

    monkeypatch.setattr(run_cartesian_storm, "require_cuda_runtime", _no_device)
    # The storm run-geometry admission (d6966ec, gap2/stormdiag, merged before
    # this test's branch landed) refuses this deliberately tiny 16x16 grid
    # before the report is reached; the geometry gate is covered by
    # test_run_geometry.py, and this test is about the sounding report.
    monkeypatch.setattr(run_cartesian_storm, "require_run_geometry", lambda *a, **k: {})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_cartesian_storm.py",
            "--nx", "16", "--ny", "16", "--nz", "16",
            "--dx", "500.0", "--dy", "500.0", "--dz", "500.0",
            "--steps", "4",
            "--diagnostic-interval-steps", "2",
            "--history-interval-steps", "2",
            "--output-cape",
            "--sounding", str(sounding),
            "--library", str(initializer_library),
            "--output", str(tmp_path / "receipt.json"),
            "--diagnostics", str(tmp_path / "rows.jsonl"),
        ],
    )
    with pytest.raises(_Stop):
        run_cartesian_storm.main()
    printed = capsys.readouterr().err
    assert "Thermodynamic properties of the initial sounding" in printed
    assert "0-1,0-3 km SRH" in printed
    # And the header the streamed record carries is assembled from the same
    # record, so the two cannot drift apart.
    source = Path(run_cartesian_storm.__file__).read_text(encoding="utf-8")
    assert '"initial_sounding_parameters": sounding_report,' in source


def test_the_sustained_storm_runner_can_switch_the_planes_on(tmp_path: Path) -> None:
    """The driver that runs the long integrations has to reach the switches.

    A capability declarable only in a TOML deck, when the driver that runs the
    two-hour storms builds its own document from flags, is a capability no run
    can select.  A plain invocation must still produce exactly the document it
    always did, or every sealed config_sha256 in the existing records moves.
    """

    import argparse

    from tools.run_cartesian_storm import document

    base = dict(
        nx=96, ny=96, nz=40, dx=500.0, dy=500.0, dz=500.0, stretch_z=0,
        ztop=0.0, str_bot=0.0, str_top=0.0, dz_bot=0.0, dz_top=0.0,
        dt=0.5, nsound=6, u_move=0.0, v_move=0.0, steps=120, tile_size=64,
        memory_budget_mib=8192, capacity_profile=None,
        output_cape=False, output_cin=False, output_lcl=False, output_lfc=False,
        parcel_source=3, adiabat=1, mixed_layer_depth=500.0,
        pressure_increment=200.0, storm_motion="bunkers-right",
        # --bubble-center-x/y (927b5be, gap2/stormdiag) default to None: the
        # domain-centre bubble this driver has always built.
        bubble_center_x=None, bubble_center_y=None,
    )
    plain = parse_mapping(document(argparse.Namespace(**base)))
    # The canonical hash this driver produces for this grid.  It was
    # 34f07fec... on this branch (510f9d5); 05e6921 recomputed every sealed
    # digest in this file after int/dynamics began projecting dynamics.alph,
    # coriolis, idiff and kdiv unconditionally, and this pin (unreachable then
    # because the Namespace above lacked bubble_center_x) moves with them.  It
    # was then pinned at a4216a16... carrying the top-level
    # "tornado_diagnostics": null that canonical_dict leaked into every
    # identity from a0cee5e; with that leak removed it is the digest below.
    assert plain.sha256 == (
        "f5064071c1fec8de640c7111676cdad26b14e2c664110b732105fc95a334c486"
    )
    assert plain.diagnostics is None
    assert "diagnostics" not in plain.canonical_dict()

    for flag, check in (
        ("output_cape", lambda c: c.environment_fields.output_cape),
        ("output_cin", lambda c: c.environment_fields.output_cin),
        ("output_lcl", lambda c: c.environment_fields.output_lcl),
        ("output_lfc", lambda c: c.environment_fields.output_lfc),
    ):
        switched = parse_mapping(document(argparse.Namespace(**{**base, flag: True})))
        assert check(switched)
        assert switched.sha256 != plain.sha256
        assert build_runtime_config(switched).environment_field_selection.any_field
    for key, value, read in (
        ("parcel_source", 1, lambda c: c.environment_fields.parcel_source),
        ("adiabat", 2, lambda c: c.environment_fields.adiabat),
        ("mixed_layer_depth", 1000.0, lambda c: c.environment_fields.mixed_layer_depth_m),
        ("pressure_increment", 50.0, lambda c: c.environment_fields.pressure_increment_pa),
        ("storm_motion", "bunkers-left", lambda c: c.environment_fields.storm_motion),
    ):
        switched = parse_mapping(document(argparse.Namespace(**{**base, key: value})))
        assert read(switched) == value
        assert switched.sha256 != plain.sha256


def test_a_library_without_the_report_is_refused_before_the_device(
    monkeypatch, tmp_path: Path
) -> None:
    """A stale library must refuse by name, not integrate a mute run."""

    from cm1core.sounding_report import initial_sounding_report

    empty = tmp_path / "libcm1_state.so"
    empty.write_bytes(b"")
    with pytest.raises(CM1Refusal) as caught:
        initial_sounding_report(empty, b"", isnd=5, iwnd=2, nz=40, dz_m=500.0)
    assert caught.value.detail.code == "CM1-SOUNDING-REPORT-LIBRARY"
    assert "never states" in caught.value.detail.reason


def test_a_column_the_report_cannot_be_built_from_refuses_by_name(
    initializer_library: Path,
) -> None:
    from cm1core.sounding_report import initial_sounding_report

    with pytest.raises(CM1Refusal) as caught:
        initial_sounding_report(
            initializer_library, b"", isnd=5, iwnd=2, nz=3, dz_m=500.0
        )
    assert caught.value.detail.code == "CM1-SOUNDING-REPORT"
    assert "cannot state the" in caught.value.detail.reason


# --------------------------------------------------------------------------
# The renderer.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("product", sorted(GETCAPE_FIELD_NAMES))
def test_the_render_door_admits_each_environment_plane(product: str) -> None:
    from cm1core import doors

    source = Path(doors.__file__).read_text(encoding="utf-8")
    assert f'"{product}",' in source
    renderer = (ROOT / "rust/cm1-render/src/main.rs").read_text(encoding="utf-8")
    # Both the admission list and the palette dispatch, so a product that is
    # accepted at the door cannot fall through to the generic scalar palette
    # and be rendered with a per-frame normalization that hides a weakening
    # inflow between two frames.
    assert f'| "{product}"' in renderer or f'"{product}" | ' in renderer
    assert f'"{product}" => Palette::' in renderer


# --------------------------------------------------------------------------
# One producer per CM1 output name.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "deck", ["cartesian-environment-fields.toml", "cartesian-supercell-storm-diagnostics.toml"]
)
def test_the_shipped_getcape_decks_plan_with_one_carrier_per_name(deck: str) -> None:
    runtime = build_runtime_config(load_config(EXAMPLES / deck))
    plan = cartesian_door.cartesian_history_plan(runtime)
    names = plan["carrier_names"]
    assert len(names) == len(set(names))
    for field in GETCAPE_FIELD_NAMES:
        assert names.count(field) == 1


def test_the_document_parcel_reaches_the_single_getcape_producer() -> None:
    """The [diagnostics] parcel controls stay reachable through the &param9 engine."""

    from cm1core.cartesian_output_products import CartesianOutputProducts

    runtime = build_runtime_config(
        parse_mapping(_deck(output_cape=True, parcel_source=1, adiabat=2,
                            mixed_layer_depth_m=1000.0))
    )
    engine = CartesianOutputProducts.__new__(CartesianOutputProducts)
    engine.config = runtime
    assert engine.getcape_controls() == {
        "parcel_source": 1, "pressure_increment_pa": 200.0,
        "mixed_layer_depth_m": 1000.0, "adiabat": 2,
    }
    engine.selection = runtime.output
    markers = [
        row for row in CartesianOutputProducts.unvalidated_markers(engine)
        if "getcape" in row["not_checked"]
    ]
    assert markers and "adiabat" in markers[0]["not_checked"]
    default = build_runtime_config(parse_mapping(_deck(output_cape=True)))
    engine.config, engine.selection = default, default.output
    assert engine.getcape_controls() == {
        "parcel_source": 3, "pressure_increment_pa": 200.0,
        "mixed_layer_depth_m": 500.0, "adiabat": 1,
    }
    assert not [
        row for row in CartesianOutputProducts.unvalidated_markers(engine)
        if "getcape" in row["not_checked"]
    ]


def test_a_repeated_carrier_name_is_refused_without_a_grid_remedy(monkeypatch) -> None:
    from cm1core import cartesian_history

    runtime = build_runtime_config(load_config(ENVIRONMENT))
    original = cartesian_history._carrier_table

    def doubled(config, **kwargs):
        table = original(config, **kwargs)
        return table + tuple(row for row in table if row.name == "cape")

    monkeypatch.setattr(cartesian_history, "_carrier_table", doubled)
    with pytest.raises(cartesian_history.DuplicateHistoryCarrierError) as raised:
        cartesian_history_profile(runtime)
    assert raised.value.names == ("cape",)
    with pytest.raises(CM1Refusal) as caught:
        cartesian_door.cartesian_history_plan(runtime)
    assert caught.value.detail.code == "CM1-CARTESIAN-HISTORY-CARRIER-NAME"
    assert "grid" not in caught.value.detail.remedy
    assert "cape" in caught.value.detail.remedy


@pytest.mark.parametrize("deck", sorted(p.name for p in EXAMPLES.glob("cartesian-*.toml")))
def test_every_shipped_cartesian_deck_prices_the_frame_its_runtime_publishes(deck: str) -> None:
    """The plan must include the rotation/swath carriers and find one producer per name."""

    from cm1core.cartesian_tornado_diagnostics import tornado_carrier_table

    runtime = build_runtime_config(load_config(EXAMPLES / deck))
    published = cartesian_history_profile(
        runtime, tornado_carriers=tornado_carrier_table(runtime)
    )
    names = [carrier.name for carrier in published.carriers]
    assert len(names) == len(set(names))
    assert cartesian_door.cartesian_history_plan(runtime)["carrier_names"] == names


def test_the_sustained_storm_runner_prints_a_plan_without_a_sounding(
    monkeypatch, tmp_path, capsys
) -> None:
    """``--plan`` prices the run before a sounding or a device is resolved.

    A merge pasted the header's ``initial_sounding_parameters`` entry into the
    plan document, where the sounding report does not exist yet, and kept a
    reference to the retired ``HISTORY_CADENCE_DECLARATION``; every ``--plan``
    invocation then raised instead of printing.  The runnable check is the
    installation's (the ArWen estate) and is not what this test is about.
    """

    from tools import run_cartesian_storm as runner

    monkeypatch.setattr(runner, "require_runnable", lambda *args, **kwargs: None)
    monkeypatch.setattr(sys, "argv", [
        "run_cartesian_storm.py", "--plan",
        "--nx", "64", "--ny", "64", "--nz", "40",
        "--dx", "500", "--dy", "500", "--dz", "500", "--steps", "1200",
        "--output", str(tmp_path / "receipt.json"),
        "--diagnostics", str(tmp_path / "diagnostics.jsonl"),
    ])
    assert runner.main() == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["schema"] == "gpuwm-cm1.cartesian-sustained-plan/v1"
    assert plan["status"] == "priced-no-device-opened"
    assert "initial_sounding_parameters" not in plan
    assert plan["cadence_declaration"] == runner.CADENCE_DECLARATION
    assert not (tmp_path / "receipt.json").exists()
