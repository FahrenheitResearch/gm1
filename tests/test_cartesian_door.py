from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from cm1core.cartesian_door import (
    CARTESIAN_SOUNDING_RESOURCE,
    allocation_plan,
    build_runtime_config,
    measured_steps,
    moving_frame_record,
    require_cartesian_lane,
    require_grid_extent,
    require_level_capacity,
    require_moving_frame,
    require_new_receipt_path,
    require_sounding_extent,
    resolve_initializer_library,
    resolve_sounding,
    selected_capacity_profile,
)
import cm1core.cartesian_door as cartesian_door
from cm1core.config import Lane, load_config, parse_mapping
import cm1core.engine_pin as engine_pin
from cm1core.engine_capacity import WSM6_256_CAPACITY_PROFILE
from cm1core.errors import CM1Refusal
from cm1core.preflight import blockers
from cm1core.profiles import profile_identity
import cm1core.rust_tools as rust_tools


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "cartesian-uniform-grid.toml"
SECTIONS = ("run", "grid", "dynamics", "boundaries", "closures", "physics", "runtime")
def sounding(*heights_m: float) -> str:
    """A CM1 r21.1 input_sounding shape: one surface line, then height rows."""

    return "  1000.0  300.5  5.0\n" + "".join(
        f"  {height}  300.5  5.0  0.0  0.0\n" for height in heights_m
    )


def cartesian_mapping() -> dict[str, object]:
    payload = load_config(EXAMPLE).canonical_dict()
    return {section: copy.deepcopy(payload[section]) for section in SECTIONS}


def configured(**overrides: dict[str, object]):
    raw = cartesian_mapping()
    for section, values in overrides.items():
        raw[section].update(values)  # type: ignore[union-attr]
    return parse_mapping(raw)


def test_import_is_cuda_cold_and_gate_is_named() -> None:
    command = (
        "import sys; import cm1core.cartesian_door as module; "
        "assert module.CARTESIAN_GATE == 'cartesian-runtime'; "
        "assert module.CARTESIAN_RECEIPT_SCHEMA.startswith('gpuwm-cm1.'); "
        "assert not any(name == 'cupy' or name.startswith('cupy.') "
        "for name in sys.modules)"
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")
    subprocess.run(
        [sys.executable, "-c", command],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )


def test_admission_refuses_before_any_cupy_import(tmp_path: Path) -> None:
    """A refused run must never have touched the CUDA import surface."""

    fake = tmp_path / "fake-site" / "cupy"
    fake.mkdir(parents=True)
    (fake / "__init__.py").write_text(
        "raise AssertionError('cupy imported before admission')", encoding="utf-8"
    )
    command = (
        "import sys, json; from cm1core.cli import main; "
        "status = main(['cartesian', sys.argv[1], '--output', sys.argv[2], '--json']); "
        "print(json.dumps({'status': status, 'cupy': any("
        "name == 'cupy' or name.startswith('cupy.') for name in sys.modules)}))"
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(tmp_path / "fake-site"), str(ROOT / "src"))
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            command,
            str(ROOT / "examples" / "official-supercell-product.toml"),
            str(tmp_path / "receipt.json"),
        ],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout.strip().splitlines()[-1]) == {
        "status": 2,
        "cupy": False,
    }
    assert json.loads(completed.stderr)["code"] == "CM1-CARTESIAN-LANE"
    assert not (tmp_path / "receipt.json").exists()


def test_lane_identity_is_bound_to_the_installed_profile_manifest() -> None:
    identity = profile_identity(Lane.CARTESIAN.value)
    assert identity["lane"] == Lane.CARTESIAN.value == "cartesian"
    assert identity["lane_name"] == Lane.CARTESIAN.value
    assert identity["admitted_envelope"]["levels"] == [8, 80]
    assert identity["admitted_envelope"]["levels_with_capacity_profile"] == [8, 256]
    assert identity["admitted_envelope"]["capacity_profile"] == WSM6_256_CAPACITY_PROFILE
    assert "not native-CM1 trajectory identity" in identity["status"]


def test_example_fixture_hash_and_line_endings_are_stable() -> None:
    fixture = profile_identity(Lane.CARTESIAN.value)["semantic_config_fixture"]
    payload = (ROOT / fixture["source_distribution_path"]).read_bytes()
    assert len(payload) == fixture["bytes"]
    assert hashlib.sha256(payload).hexdigest() == fixture["sha256"]
    assert b"\r\n" not in payload


def test_example_is_the_standard_cm1_column_and_is_accepted() -> None:
    config = load_config(EXAMPLE)
    assert config.run.lane is Lane.CARTESIAN
    assert (config.grid.nz, config.grid.dz_m) == (40, 500.0)
    assert config.runtime is not None
    assert config.runtime.capacity_profile == "default"
    assert selected_capacity_profile(config) is None
    assert measured_steps(config) == 10


def test_cartesian_preflight_drops_only_the_repaired_blockers() -> None:
    codes = {item.code for item in blockers(load_config(EXAMPLE))}
    assert "CM1-CUDA-TRAJECTORY-INCOMPLETE" not in codes
    assert "CM1-INSTRUMENTS-MISSING" not in codes
    assert "CM1-RUST-LIVE-WIRING-MISSING" not in codes
    product = {
        item.code
        for item in blockers(
            load_config(ROOT / "examples" / "official-supercell-product.toml")
        )
    }
    assert {
        "CM1-CUDA-TRAJECTORY-INCOMPLETE",
        "CM1-INSTRUMENTS-MISSING",
        "CM1-RUST-LIVE-WIRING-MISSING",
    } <= product


def test_another_lane_refuses_by_name() -> None:
    config = load_config(ROOT / "examples" / "official-supercell-product.toml")
    with pytest.raises(CM1Refusal) as caught:
        require_cartesian_lane(config)
    assert caught.value.detail.code == "CM1-CARTESIAN-LANE"
    assert caught.value.detail.option == "run.lane"
    assert caught.value.detail.value == "product"


def test_levels_are_the_standard_range_unless_a_profile_is_named() -> None:
    assert require_level_capacity(load_config(EXAMPLE)) == (8, 80)
    deep = {"grid": {"nz": 237}}
    with pytest.raises(CM1Refusal) as caught:
        require_level_capacity(configured(**deep))
    assert caught.value.detail.code == "CM1-CARTESIAN-LEVEL-CAPACITY"
    assert caught.value.detail.option == "grid.nz"
    assert caught.value.detail.value == 237
    assert "runtime.capacity_profile" in caught.value.detail.remedy
    admitted = configured(
        grid={"nz": 237},
        runtime={"capacity_profile": WSM6_256_CAPACITY_PROFILE},
    )
    assert require_level_capacity(admitted) == (8, 256)
    assert selected_capacity_profile(admitted) == WSM6_256_CAPACITY_PROFILE


def test_grid_extent_refuses_a_domain_smaller_than_the_bound_stencil() -> None:
    for section, values, option in (
        ("grid", {"nx": 7, "ny": 7, "nz": 8}, None),
        ("grid", {"nx": 96, "ny": 96, "nz": 7}, "grid.nz"),
    ):
        raw = cartesian_mapping()
        raw[section].update(values)  # type: ignore[union-attr]
        if option is None:
            assert require_grid_extent(parse_mapping(raw)).nx == 7
            continue
        with pytest.raises(CM1Refusal) as caught:
            require_grid_extent(parse_mapping(raw))
        assert caught.value.detail.code == "CM1-CARTESIAN-GRID-EXTENT"
        assert caught.value.detail.option == option


def test_damping_depth_refuses_reaching_below_half_the_model_top() -> None:
    with pytest.raises(CM1Refusal) as caught:
        build_runtime_config(configured(closures={"upper_damping_depth_m": 12000.0}))
    assert caught.value.detail.code == "CM1-CARTESIAN-DAMPING-DEPTH"
    assert caught.value.detail.option == "closures.upper_damping_depth_m"


def test_runtime_config_binds_the_declared_execution_table() -> None:
    config = load_config(EXAMPLE)
    runtime_config = build_runtime_config(config)
    assert runtime_config.acoustic_substeps == 6
    assert (runtime_config.tile_nx, runtime_config.tile_ny) == (64, 64)
    assert runtime_config.memory_budget_bytes == 8192 * 1024**2
    assert runtime_config.microphysics == "wsm6-mp6"
    assert runtime_config.damping_depth_m == config.closures.upper_damping_depth_m
    assert runtime_config.grid.ztop_m == 20000.0
    plan = allocation_plan(runtime_config)
    assert plan["config"]["grid"]["nz"] == 40
    assert plan["dynamics"] and plan["closure"]


def test_the_fixed_frame_is_the_default_and_carries_no_offset() -> None:
    """``closures.moving_frame`` off means exactly the old behaviour."""

    config = load_config(EXAMPLE)
    assert not config.closures.moving_frame
    runtime_config = build_runtime_config(config)
    assert runtime_config.u_move_m_s == 0.0
    assert runtime_config.v_move_m_s == 0.0
    frame = require_moving_frame(config)
    assert not frame.translates
    assert frame.imove == 0
    assert moving_frame_record(frame)["imove"] == 0


def test_the_declared_frame_reaches_the_runtime_contract_and_the_receipt() -> None:
    """CM1's own supercell deck velocity, end to end through the door.

    ``build_runtime_config`` is what the history publisher hashes into every
    frame's ``config_sha256``, so a frame velocity that stopped at the config
    object would leave every published frame claiming a fixed domain.
    """

    config = configured(
        closures={"moving_frame": True, "u_move_m_s": 12.5, "v_move_m_s": 3.0}
    )
    runtime_config = build_runtime_config(config)
    assert runtime_config.u_move_m_s == 12.5
    assert runtime_config.v_move_m_s == 3.0
    record = runtime_config.record()
    assert record["u_move_m_s"] == 12.5
    assert record["v_move_m_s"] == 3.0
    assert "base.F:2663-2669" in str(record["moving_frame_authority"])

    frame = require_moving_frame(config)
    assert (frame.u_move_m_s, frame.v_move_m_s) == (12.5, 3.0)
    assert frame.imove == 1
    published = moving_frame_record(frame)
    assert published["u_move_m_s"] == 12.5
    assert published["v_move_m_s"] == 3.0
    assert published["imove"] == 1
    assert "base.F:2663-2669" in str(published["authority"])
    # Whoever reads a frame has to be able to get back to the ground, and the
    # domain's own ground position is carried per frame rather than left to
    # the reader to form from the constant and the model time.
    assert "u_move_m_s + u" in str(published["convention"])
    assert "domainlocx/domainlocy" in str(published["convention"])
    assert "cm1.F:3512-3513" in str(published["convention"])
    assert "cm1.F:348-349,3512-3513" in str(published["domain_location_authority"])
    # Canonical JSON, because it is written into receipts.
    json.dumps(published, allow_nan=False, sort_keys=True)


def test_a_nonzero_frame_is_refused_while_the_branch_is_off() -> None:
    """param.F:701-702 silently zeroes them; this names the key instead."""

    with pytest.raises(CM1Refusal) as caught:
        configured(closures={"u_move_m_s": 12.5})
    assert caught.value.detail.code == "CM1-CFG-MOVING-FRAME-INACTIVE"
    assert caught.value.detail.option == "closures.moving_frame"


@pytest.mark.parametrize(
    ("key", "value"),
    (("u_move_m_s", 500.0), ("v_move_m_s", -1.0e4)),
)
def test_an_absurd_frame_speed_is_refused_by_key_and_value(
    key: str, value: float
) -> None:
    with pytest.raises(CM1Refusal) as caught:
        configured(closures={"moving_frame": True, key: value})
    assert caught.value.detail.code == "CM1-CFG-MOVING-FRAME-SPEED"
    assert caught.value.detail.option == f"closures.{key}"
    assert caught.value.detail.value == value
    # The refusal has to name what it prevents, not merely that it refused.
    assert "hodograph" in caught.value.detail.reason


@pytest.mark.parametrize(
    ("key", "value"),
    (("u_move_m_s", 12.3), ("v_move_m_s", 0.1)),
)
def test_a_frame_speed_that_is_not_a_cm1_real_is_refused_by_key_and_value(
    key: str, value: float
) -> None:
    """base.F:2667 subtracts a binary32 word; cm1.F:3512 integrates one too.

    A declared speed that does not survive binary32 would be subtracted from
    the base winds as one number and integrated into domainlocx as another,
    and every published domain location would drift from the carriers it
    describes for the rest of the run.
    """

    with pytest.raises(CM1Refusal) as caught:
        configured(closures={"moving_frame": True, key: value})
    assert caught.value.detail.code == "CM1-CFG-MOVING-FRAME-PRECISION"
    assert caught.value.detail.option == f"closures.{key}"
    assert caught.value.detail.value == value
    # The refusal names what it prevents and what to write instead.
    assert "domainlocx/domainlocy" in caught.value.detail.reason
    assert "drift" in caught.value.detail.reason
    assert "binary32" in caught.value.detail.remedy
    # CM1's own supercell deck velocity is admitted untouched.
    configured(closures={"moving_frame": True, "u_move_m_s": 12.5, "v_move_m_s": 3.0})


def test_the_declared_frame_reaches_the_runtime_clocks_domain_location() -> None:
    """cm1.F:3512-3513, from the declared configuration through the door.

    ``build_runtime_config`` is what the clock integrates against, so a frame
    velocity that stopped at the configuration object would leave every
    published frame claiming a domain that never moved.
    """

    from cm1core.gm1_runtime import RuntimeClock

    class _NoOpBackend:
        def execute(self, operation, *, step):
            pass

        def synchronize(self):
            pass

    config = configured(
        closures={"moving_frame": True, "u_move_m_s": 12.5, "v_move_m_s": 3.0}
    )
    clock = RuntimeClock(build_runtime_config(config))
    backend = _NoOpBackend()
    for _ in range(120):
        clock.step(backend)
    # One history interval of CM1's idealized deck: 120 steps, 60 s.
    assert clock.model_seconds == 60.0
    assert clock.domainlocx_m == 750.0
    assert clock.domain_location_record()["domainlocy"] == 180.0

    fixed = RuntimeClock(build_runtime_config(configured()))
    for _ in range(120):
        fixed.step(backend)
    assert (fixed.domainlocx_m, fixed.domainlocy_m) == (0.0, 0.0)


def test_sounding_must_reach_the_highest_mass_level() -> None:
    """Only the top bound binds; the initializer supplies its own surface.

    This case previously also required the first data height to sit below the
    first mass level.  The initializer prepends CM1's surface record at z=0
    (idealized_initialization.rs:357) and refuses only `height <= z_m[0] ||
    height > sounding_top`, so that lower requirement was stricter than the
    thing it stood in for and refused every grid with a fine enough dz.

    What this no longer catches: a profile whose lowest data row sits far
    above the first mass level.  The initializer interpolates those levels
    between its surface record and that row, which is what CM1 itself does,
    so there is nothing here to refuse.
    """

    grid = require_grid_extent(load_config(EXAMPLE))
    assert (grid.dz_m * 0.5, grid.ztop_m - grid.dz_m * 0.5) == (250.0, 19750.0)

    with pytest.raises(CM1Refusal) as caught:
        require_sounding_extent("label", sounding(100.0, 5000.0).encode("utf-8"), grid)
    assert caught.value.detail.code == "CM1-CARTESIAN-SOUNDING-EXTENT"
    assert caught.value.detail.value["required_mass_interval_m"] == [250.0, 19750.0]

    # A first data row above the first mass level is admitted, as the
    # initializer admits it.
    assert require_sounding_extent(
        "label", sounding(300.0, 25000.0).encode("utf-8"), grid
    ) == (300.0, 25000.0)
    assert require_sounding_extent(
        "label", sounding(100.0, 5000.0, 25000.0).encode("utf-8"), grid
    ) == (100.0, 25000.0)


def test_unreadable_sounding_records_refuse_by_line() -> None:
    grid = require_grid_extent(load_config(EXAMPLE))
    for payload in (b"surface\nnot-a-height 1 2\n", b"surface\n", b"\xff\xfe\x00"):
        with pytest.raises(CM1Refusal) as caught:
            require_sounding_extent("label", payload, grid)
        assert caught.value.detail.code == "CM1-CARTESIAN-SOUNDING-FORMAT"


def test_absent_initializer_and_sounding_name_their_own_arguments(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(rust_tools, "native_library_path", lambda name: None, raising=False)
    with pytest.raises(CM1Refusal) as caught:
        resolve_initializer_library(None)
    assert caught.value.detail.code == "CM1-CARTESIAN-INITIALIZER-MISSING"
    assert "--library" in caught.value.detail.remedy

    def unavailable(name: str) -> Path:
        assert name == cartesian_door.CARTESIAN_INITIALIZER_LIBRARY
        raise rust_tools.NativeToolError("manifest hash differs")

    monkeypatch.setattr(rust_tools, "native_library_path", unavailable, raising=False)
    with pytest.raises(CM1Refusal) as caught:
        resolve_initializer_library(None)
    assert caught.value.detail.code == "CM1-CARTESIAN-INITIALIZER-MISSING"
    assert "manifest hash differs" in caught.value.detail.reason

    library = tmp_path / "libcm1_state.so"
    library.write_bytes(b"not executed by this test")
    monkeypatch.setattr(
        rust_tools, "native_library_path", lambda name: library, raising=False
    )
    assert resolve_initializer_library(None) == library.resolve()
    assert resolve_initializer_library(library) == library.resolve()
    with pytest.raises(CM1Refusal) as caught:
        resolve_initializer_library(tmp_path / "absent.so")
    assert caught.value.detail.code == "CM1-CARTESIAN-INITIALIZER-PATH"

    # No sounding is installed: CM1 runtime assets are named, not shipped.
    with pytest.raises(CM1Refusal) as caught:
        resolve_sounding(None)
    assert caught.value.detail.code == "CM1-CARTESIAN-SOUNDING-NOT-BUNDLED"
    assert "--sounding" in caught.value.detail.remedy
    assert "run/config_files/cpm_RadConvEquil/input_sounding" in caught.value.detail.remedy
    assert "295bf8d89fa4624676e8b0d90ce232cd5b053587c699f23700fd22df5feec6a5" in (
        caught.value.detail.remedy
    )
    monkeypatch.setattr(cartesian_door, "CARTESIAN_SOUNDING_RESOURCE", "no-such-resource")
    with pytest.raises(CM1Refusal) as caught:
        resolve_sounding(None)
    assert caught.value.detail.code == "CM1-CARTESIAN-SOUNDING-MISSING"
    source = tmp_path / "input_sounding"
    payload = sounding(100.0, 25000.0)
    source.write_text(payload, encoding="utf-8")
    assert resolve_sounding(source) == (str(source), payload.encode("utf-8"))


def test_receipt_path_must_be_a_new_json_file(tmp_path: Path) -> None:
    assert require_new_receipt_path(tmp_path / "run.json") == tmp_path / "run.json"
    existing = tmp_path / "already.json"
    existing.write_text("{}", encoding="utf-8")
    for candidate in (existing, tmp_path / "run.txt", tmp_path / "absent" / "run.json"):
        with pytest.raises(CM1Refusal) as caught:
            require_new_receipt_path(candidate)
        assert caught.value.detail.code == "CM1-CARTESIAN-OUTPUT"


@pytest.mark.skipif(
    not engine_pin.inspect().admitted,
    reason="the audited ArWen estate is not installed, so admission stops earlier",
)
def test_plan_prices_the_run_without_opening_a_device(tmp_path: Path, capsys) -> None:
    from cm1core.cli import main

    receipt = tmp_path / "receipt.json"
    assert main(["cartesian", str(EXAMPLE), "--output", str(receipt), "--plan"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "priced-no-device-opened"
    assert payload["schema"] == "gpuwm-cm1.cartesian-plan/v1"
    assert payload["measured_steps"] == 10
    assert payload["warmup_steps"] == 1
    assert payload["admitted_level_bounds"] == [8, 80]
    assert payload["config"]["grid"]["ztop_m"] == 20000.0
    assert not receipt.exists()


def test_installed_resource_name_is_the_one_doctor_reports() -> None:
    from cm1core import doctor

    assert CARTESIAN_SOUNDING_RESOURCE == "cm1r21.1-cpm-radconvequil-sounding"
    findings = doctor.collect()
    finding = next(
        item for item in findings if item.subject == "GM1 Cartesian runtime inputs"
    )
    # The two Cartesian inputs are arguments, not gates: naming them must not
    # add a blocking finding to a fresh install.
    assert not finding.required
    assert finding not in doctor.blocking(findings)
    if finding.status != "present":
        assert "--sounding" in finding.detail or "--library" in finding.detail


def test_cartesian_is_an_advertised_subcommand() -> None:
    from cm1core.cli import build_parser

    parser = build_parser()
    arguments = parser.parse_args(
        ["cartesian", str(EXAMPLE), "--output", "receipt.json", "--plan"]
    )
    assert arguments.handler is cartesian_door.cartesian_door
    assert arguments.library is None and arguments.sounding is None
    assert arguments.gpu_uuid is None and arguments.lock is None
    assert arguments.plan and not arguments.trace_phases


def test_installed_initializer_and_sounding_names_match_their_providers() -> None:
    """The door and the packagers must agree on the two resource names.

    These names cross a module boundary that no single test exercised: the
    door asks rust_tools and cm1core.data for a resource by name, and a wrong
    name surfaces as "no installed library", which reads like a packaging
    failure and sends the user to --library when the wheel is fine. It cost a
    real run on an RTX 5090 to find.
    """

    from cm1core.cartesian_door import (
        CARTESIAN_INITIALIZER_LIBRARY,
    )
    from cm1core.rust_tools import EXPECTED_LIBRARIES
    from cm1core.data import packaged_inputs

    assert CARTESIAN_INITIALIZER_LIBRARY in EXPECTED_LIBRARIES
    assert CARTESIAN_SOUNDING_RESOURCE in packaged_inputs()


def test_fine_vertical_spacing_is_admitted_like_the_initializer_admits_it() -> None:
    """The admitted interval is (0, top], not (first data height, top].

    rust/cm1-state/src/idealized_initialization.rs:357 pushes CM1's surface
    record at z=0 ahead of the data rows, and interpolate_column refuses only
    `height <= z_m[0] || height > sounding_top`.  With z_m[0] == 0 the lower
    bound cannot bind for a positive spacing.

    Checking the first *data* height instead refused every grid whose first
    mass level sat below it -- with the shipped 40 km profile, every dz below
    248 m, which is every 125 m configuration in the evidence tree.  Only the
    top bound is a real constraint.
    """

    from cm1core.cartesian_door import require_sounding_extent
    from cm1core.grid_geometry import CartesianGridGeometry

    payload = sounding(124.0, 40000.0).encode("utf-8")
    label = "pristine"
    first_data_height, top = require_sounding_extent(
        label, payload, CartesianGridGeometry(8, 8, 40, 500.0, 500.0, 500.0)
    )
    assert first_data_height > 0.0, "this profile's first data row is above the surface"

    for dz, nz in ((500.0, 40), (250.0, 80), (125.0, 80), (125.0, 237), (100.0, 80)):
        grid = CartesianGridGeometry(8, 8, nz, 125.0, 125.0, dz)
        assert 0.5 * dz < first_data_height or dz >= 2.0 * first_data_height
        require_sounding_extent(label, payload, grid)

    # The top bound still binds.
    too_tall = CartesianGridGeometry(8, 8, 100, 125.0, 125.0, 500.0)
    assert too_tall.ztop_m > top
    with pytest.raises(CM1Refusal) as caught:
        require_sounding_extent(label, payload, too_tall)
    assert caught.value.detail.code == "CM1-CARTESIAN-SOUNDING-EXTENT"


def test_doctor_and_the_door_agree_that_the_sounding_is_never_installed() -> None:
    """doctor must not report an input the door will refuse.

    The package records the CM1 sounding's identity and does not redistribute
    its bytes, so --sounding is always required. A doctor that resolved the
    identity and called that "present" contradicted the door.
    """

    from cm1core.doctor import collect
    from cm1core.cartesian_door import resolve_sounding

    with pytest.raises(CM1Refusal) as caught:
        resolve_sounding(None)
    assert caught.value.detail.code == "CM1-CARTESIAN-SOUNDING-NOT-BUNDLED"

    finding = next(f for f in collect() if f.subject == "GM1 Cartesian runtime inputs")
    assert "--sounding" in finding.detail
    assert finding.required is False


def test_the_shipped_example_still_binds_the_single_sounding_lane() -> None:
    """No [initialization] table means base.F isnd=7 with param.F's iwnd=0."""

    from cm1core.cartesian_door import require_initializer_selection

    selection, declared = require_initializer_selection(load_config(EXAMPLE))
    assert not declared
    assert (selection.record()["isnd"], selection.record()["iwnd"]) == (7, 0)
    assert selection.reads_input_sounding


def test_integrate_and_history_dir_are_advertised_options() -> None:
    from cm1core.cli import build_parser

    parser = build_parser()
    arguments = parser.parse_args(
        ["cartesian", str(EXAMPLE), "--output", "receipt.json"]
    )
    assert arguments.integrate is False and arguments.history_dir is None
    arguments = parser.parse_args([
        "cartesian", str(EXAMPLE), "--output", "receipt.json",
        "--history-dir", "frames", "--integrate",
    ])
    assert arguments.integrate is True


def _history_carriers_action():
    from cm1core.cli import build_parser

    parser = build_parser()
    cartesian = parser._subparsers._group_actions[0].choices["cartesian"]  # noqa: SLF001
    return next(
        action for action in cartesian._actions  # noqa: SLF001
        if "--history-carriers" in action.option_strings
    )


def test_an_unknown_history_carrier_reaches_the_door_refusal_not_argparse() -> None:
    """argparse must not preempt the refusal that names the wrong result.

    ``choices=`` would exit 2 with a bare usage line before
    ``require_cartesian_history_carriers`` ran, so the operator would never see
    which spellings are admitted, why the wrong one is refused, or a refusal
    record.  Every other option in this parser refuses through refuse(); this
    one now does too.
    """

    from cm1core.cartesian_door import require_cartesian_history_carriers
    from cm1core.cartesian_history import CARTESIAN_HISTORY_CARRIER_PROFILES
    from cm1core.cli import build_parser

    action = _history_carriers_action()
    assert action.choices is None
    # The admitted spellings are still advertised, in the metavar and the help.
    for profile in CARTESIAN_HISTORY_CARRIER_PROFILES:
        assert profile in action.metavar
        assert profile in action.help

    parser = build_parser()
    arguments = parser.parse_args([
        "cartesian", str(EXAMPLE), "--output", "receipt.json",
        "--history-carriers", "minimal",
    ])
    assert arguments.history_carriers == "minimal"

    with pytest.raises(CM1Refusal) as caught:
        require_cartesian_history_carriers(arguments.history_carriers)
    detail = caught.value.detail
    assert detail.code == "CM1-CARTESIAN-HISTORY-CARRIERS"
    assert detail.option == "history-carriers"
    assert detail.value == "minimal"
    assert "no reader in this distribution declares" in detail.reason
    for profile in CARTESIAN_HISTORY_CARRIER_PROFILES:
        assert profile in detail.remedy

    for profile in CARTESIAN_HISTORY_CARRIER_PROFILES:
        assert require_cartesian_history_carriers(profile) == profile


def test_a_carrier_inventory_without_a_history_directory_refuses(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """A receipt must not name an inventory no frame of the run carries."""

    from cm1core.cartesian_door import run_cartesian
    from cm1core.cartesian_history import (
        COMPLETE_CARRIER_PROFILE,
        RECONSTRUCTIBLE_OMITTED_CARRIER_PROFILE,
    )

    monkeypatch.setattr(cartesian_door, "require_runnable", lambda config, **kwargs: None)
    config = load_config(EXAMPLE)

    with pytest.raises(CM1Refusal) as caught:
        run_cartesian(
            config,
            config_label=str(EXAMPLE),
            output=tmp_path / "receipt.json",
            plan=True,
            history_carriers=RECONSTRUCTIBLE_OMITTED_CARRIER_PROFILE,
        )
    detail = caught.value.detail
    assert detail.code == "CM1-CARTESIAN-HISTORY-CARRIERS-UNPUBLISHED"
    assert detail.option == "history-carriers"
    assert detail.value == RECONSTRUCTIBLE_OMITTED_CARRIER_PROFILE
    assert "publishes no history" in detail.reason
    assert "--history-dir" in detail.remedy

    # The default inventory with no directory is the pre-existing behaviour and
    # stays admitted; the plan then records no inventory at all.
    assert run_cartesian(
        config,
        config_label=str(EXAMPLE),
        output=tmp_path / "receipt.json",
        plan=True,
        history_carriers=COMPLETE_CARRIER_PROFILE,
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["history_directory"] is None
    assert payload["history_carriers"] is None


def test_the_plan_records_the_carrier_inventory_each_frame_will_publish(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    from cm1core.cartesian_door import run_cartesian
    from cm1core.cartesian_history import RECONSTRUCTIBLE_OMITTED_CARRIER_PROFILE

    monkeypatch.setattr(cartesian_door, "require_runnable", lambda config, **kwargs: None)
    frames = tmp_path / "frames"
    assert run_cartesian(
        load_config(EXAMPLE),
        config_label=str(EXAMPLE),
        output=tmp_path / "receipt.json",
        plan=True,
        history_dir=frames,
        history_carriers=RECONSTRUCTIBLE_OMITTED_CARRIER_PROFILE,
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "priced-no-device-opened"
    assert payload["history_directory"] == str(frames)
    assert payload["history_carriers"] == RECONSTRUCTIBLE_OMITTED_CARRIER_PROFILE
    assert not frames.exists()
