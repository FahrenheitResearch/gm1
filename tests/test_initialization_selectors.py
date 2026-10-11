"""Config text to changed initial state, for every CM1 initialization selector.

Each test here proves one path end to end: a value written in an
``[initialization]`` table reaches the Rust initializer and changes the words
it emits.  Nothing is asserted about correctness against CM1 -- that is what
``tools/oracle/initialization_differential.py`` measures against the real
binary, and ``records/CM1-INITIALIZATION-VALIDATION.json`` records -- and
these tests are what keep the door from quietly closing again.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from authority_trees import requires_cm1_pristine

from cm1core.cartesian_door import bubble_center_m, horizontal_origin_m
from cm1core.config import load_config
from cm1core.errors import CM1Refusal
from cm1core.gm1_idealized_residency import (
    IMPLEMENTED_IINIT,
    IMPLEMENTED_IWND,
    IdealizedSource,
    InitializerSelection,
)
from cm1core.grid_geometry import CartesianGridGeometry
from cm1core.initialization_validation import (
    UNVALIDATED_MARKER,
    unvalidated_initialization,
    validated_rows,
)

ROOT = Path(__file__).resolve().parents[1]
CRATE = ROOT / "rust/cm1-state"
SOUNDING = ROOT / "upstream/cm1-21.1/pristine/run/config_files/sea_breeze/input_sounding"


@pytest.fixture(scope="module")
def library(tmp_path_factory: pytest.TempPathFactory) -> Path:
    target = tmp_path_factory.mktemp("gm1-init-selectors-cdylib")
    subprocess.run(
        ["cargo", "build", "--release", "--lib", "--target-dir", str(target)],
        cwd=CRATE,
        check=True,
    )
    names = ("cm1_state.dll", "libcm1_state.so", "libcm1_state.dylib")
    found = [target / "release" / n for n in names if (target / "release" / n).is_file()]
    assert len(found) == 1
    return found[0]


def geometry() -> CartesianGridGeometry:
    return CartesianGridGeometry(nx=16, ny=16, nz=12, dx_m=1000.0, dy_m=1000.0, dz_m=500.0)


def build(library: Path, selection: InitializerSelection, name: str, layout: str = "physical"):
    sounding = SOUNDING.read_bytes() if selection.reads_input_sounding else b""
    g = geometry()
    with IdealizedSource(library, sounding, g, selection=selection) as source:
        return np.array(source.field(name, layout=layout), dtype="<f4", copy=True)


def toml_document(table: str, *, run_extra: str = "") -> str:
    # CM1 has one iconly word and this port spells it once, in [run]; a test
    # that wants the inspection lane declares it there.
    return f"""
[run]
lane = "cartesian"
duration_s = 60.0
dt_s = 0.5
history_interval_s = 60.0
restart_interval_s = 60.0
{run_extra}

{table}

[grid]
nx = 16
ny = 16
nz = 12
dx_m = 1000.0
dy_m = 1000.0
dz_m = 500.0
terrain = false
stretch_x = false
stretch_y = false
stretch_z = false

[dynamics]
psolver = 3
adaptive_timestep = false
scalar_horizontal_order = 5
scalar_vertical_order = 5
velocity_horizontal_order = 5
velocity_vertical_order = 5
scalar_weno = 2
velocity_weno = 0
weno_order = 5
dry_mass_correction = true

[boundaries]
west = 2
east = 2
south = 2
north = 2
bottom = 1
top = 1
radiative_order = 4

[closures]
sgs_model = 1
vertical_implicit_diffusion = true
upper_damping = 1
upper_damping_depth_m = 1200.0
upper_damping_rate_s_inv = 0.0033333333333
moving_frame = false
u_move_m_s = 0.0
v_move_m_s = 0.0

[runtime]
acoustic_substeps = 6
tile_nx = 16
tile_ny = 16
warmup_steps = 0
memory_budget_mib = 1024
capacity_profile = "default"

[physics]
component = "wsm6-mp6"
equation_set = 2
"""


def write_config(tmp_path: Path, table: str, *, run_extra: str = "") -> Path:
    path = tmp_path / "gpuwm-cm1.toml"
    path.write_text(toml_document(table, run_extra=run_extra), encoding="utf-8")
    return path


# -- base.F:2286-2640: the eleven base-state wind branches -------------------


def test_every_iwnd_branch_is_configurable_and_emits_its_own_hodograph(library: Path) -> None:
    seen: dict[int, bytes] = {}
    for iwnd in sorted(IMPLEMENTED_IWND):
        selection = InitializerSelection(isnd=5, iwnd=iwnd, iinit=1)
        u0 = build(library, selection, "base_wind.u0", layout="native")
        v0 = build(library, selection, "base_wind.v0", layout="native")
        payload = u0.tobytes() + v0.tobytes()
        for other, before in seen.items():
            assert payload != before, f"iwnd={iwnd} emitted iwnd={other}'s hodograph"
        seen[iwnd] = payload
    assert len(seen) == 11


def test_iwnd_seven_is_refused_by_value_because_base_F_has_no_arm(tmp_path: Path) -> None:
    with pytest.raises(CM1Refusal) as error:
        load_config(write_config(tmp_path, "[initialization]\nisnd = 5\niwnd = 7"))
    assert "iwnd=7 does not exist upstream" in str(error.value)


def test_iwnd_six_constant_wind_follows_the_declared_testcase(library: Path) -> None:
    # base.F:2467-2510 selects uval/vval by testcase; nothing else about the
    # state changes, so a different testcase is a different constant wind.
    default = build(library, InitializerSelection(isnd=5, iwnd=6), "base_wind.u0", "native")
    stratocumulus = build(
        library,
        InitializerSelection(isnd=5, iwnd=6, testcase=4),
        "base_wind.u0",
        "native",
    )
    assert np.allclose(default, 10.0)
    assert np.allclose(stratocumulus, 7.0)


# -- init3d.F:450-1416: the perturbation arms -------------------------------


def test_every_iinit_arm_is_configurable_and_emits_its_own_field(library: Path) -> None:
    controls = {
        0: {},
        1: {},
        2: {"bptpert_k": -6.0, "zdep_m": 2500.0, "ric_m": 6000.0},
        3: {"bptpert_k": 2.0, "bhrad_m": 1200.0, "bvrad_m": 900.0,
            "rjc_m": 2500.0, "bubble_spacing_y_m": 4000.0},
        # init3d.F:579-647 on Weisman-Klemp's sounding, with CM1's qt_mb = 0 there (param.F:6336).
        4: {},
        5: {"bptpert_k": -15.0, "bhrad_m": 4000.0, "bvrad_m": 2000.0, "zc_m": 3000.0},
        8: {"bptpert_k": 2.0, "bvrad_m": 1500.0, "zc_m": 1500.0, "random_seed": 11},
        11: {"bptpert_k": 0.01, "bvrad_m": 10000.0, "bhrad_m": 5000.0},
        13: {"bptpert_k": 1.0, "bvrad_m": 1000.0},
    }
    # init3d.F:1303-1396: the forcing arms build no tha; they are applied
    # during the run (cm1core.initiation_forcing), so they emit iinit=0's field.
    forcing = {9, 10, 12}
    # init3d.F:697-1256's vortex reads its radii from xhref(nx/2+i), which are
    # radii only in the centred frame (iorigin = 2); this corner-origin grid
    # puts the vortex centre's neighbours inside the first radius, where CM1
    # would read rref(0), and the builder says so.  Its field is checked
    # word for word against CM1 by the initial-condition differential
    # (evidence/oracle/initialization/iinit-7).
    with pytest.raises(ValueError, match=r"rref\(0\)"):
        build(library, InitializerSelection(isnd=5, iwnd=2, iinit=7), "terrain_open.tha")
    forcing = forcing | {7}
    seen: dict[int, bytes] = {}
    for iinit in sorted(IMPLEMENTED_IINIT - forcing):
        selection = InitializerSelection(isnd=5, iwnd=2, iinit=iinit, **controls[iinit])
        field = build(library, selection, "terrain_open.tha")
        payload = field.tobytes()
        for other, before in seen.items():
            assert payload != before, f"iinit={iinit} emitted iinit={other}'s field"
        seen[iinit] = payload
    assert len(seen) == len(IMPLEMENTED_IINIT) - len(forcing)
    assert np.all(np.frombuffer(seen[0], dtype="<f4") == 0.0)
    for iinit in sorted(forcing - {7}):
        field = build(library, InitializerSelection(isnd=5, iwnd=2, iinit=iinit), "terrain_open.tha")
        assert field.tobytes() == seen[0], f"iinit={iinit} must leave tha at zero"


@pytest.mark.parametrize("iinit", [6, 14])
def test_unported_iinit_arms_are_refused_with_the_wrong_result_they_prevent(
    tmp_path: Path, iinit: int
) -> None:
    with pytest.raises(CM1Refusal) as error:
        load_config(
            write_config(tmp_path, f"[initialization]\nisnd = 5\niwnd = 2\niinit = {iinit}")
        )
    message = str(error.value)
    assert "init3d.F:450-1416" in message
    assert "under a receipt naming" in message


def test_bubble_placement_moves_the_perturbation(library: Path, tmp_path: Path) -> None:
    # The concrete failure this exists for: a trigger frozen at the domain
    # centre.  Declaring ric/rjc must move it, and the door must resolve the
    # undeclared case to CM1's centerx/centery (init3d.F:456-457).
    centred = build(library, InitializerSelection(isnd=5, iwnd=2, iinit=1, bhrad_m=1500.0), "terrain_open.tha")
    offset = build(
        library,
        InitializerSelection(isnd=5, iwnd=2, iinit=1, bhrad_m=1500.0, ric_m=2500.0, rjc_m=2500.0),
        "terrain_open.tha",
    )
    assert centred.shape == offset.shape
    assert not np.array_equal(centred, offset)
    assert int(np.argmax(centred)) != int(np.argmax(offset))

    config = load_config(write_config(tmp_path, "[initialization]\nisnd = 5\niwnd = 2"))
    assert horizontal_origin_m(config) == (0.0, 0.0)
    assert bubble_center_m(config) == (8000.0, 8000.0)
    centred_config = load_config(
        write_config(tmp_path, "[initialization]\nisnd = 5\niwnd = 2\niorigin = 2")
    )
    assert horizontal_origin_m(centred_config) == (-8000.0, -8000.0)
    assert bubble_center_m(centred_config) == (0.0, 0.0)


def test_bubble_amplitude_and_radii_reach_the_field(library: Path) -> None:
    weak = build(library, InitializerSelection(isnd=5, iwnd=2, iinit=1), "terrain_open.tha")
    strong = build(
        library, InitializerSelection(isnd=5, iwnd=2, iinit=1, bptpert_k=4.0), "terrain_open.tha"
    )
    assert pytest.approx(float(strong.max()), rel=1e-6) == 4.0 * float(weak.max())


# -- init3d.F:1446-1600: seeded random perturbations ------------------------


def test_an_unseeded_random_perturbation_is_refused_by_value(tmp_path: Path) -> None:
    with pytest.raises(CM1Refusal) as error:
        load_config(
            write_config(tmp_path, "[initialization]\nisnd = 5\niwnd = 2\nirandp = 1")
        )
    message = str(error.value)
    assert "CM1-CFG-RANDOM-SEED-MISSING" in message
    assert "wall clock" in message


def test_a_seeded_random_perturbation_is_reproducible_and_seed_sensitive(library: Path) -> None:
    def field(seed: int) -> np.ndarray:
        return build(
            library,
            InitializerSelection(isnd=5, iwnd=2, iinit=0, irandp=1, random_seed=seed),
            "terrain_open.tha",
        )

    a, b, c = field(20240611), field(20240611), field(20240612)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    # init3d.F:1462 is 0.25 K for testcase=0.
    assert float(np.abs(a).max()) <= 0.25
    assert float(np.abs(a).max()) > 0.2


# -- init3d.F:1735-1830 and base.F:592-609 ----------------------------------


def test_ibalance_changes_only_the_initial_pressure(library: Path) -> None:
    unbalanced = InitializerSelection(isnd=5, iwnd=2, iinit=1, ibalance=0)
    hydrostatic = InitializerSelection(isnd=5, iwnd=2, iinit=1, ibalance=1)
    zero = build(library, unbalanced, "terrain_open.ppi")
    integrated = build(library, hydrostatic, "terrain_open.ppi")
    assert np.all(zero == 0.0)
    assert np.any(integrated != 0.0)
    # The theta perturbation is the same field either way.
    assert np.array_equal(
        build(library, unbalanced, "terrain_open.tha"),
        build(library, hydrostatic, "terrain_open.tha"),
    )


@requires_cm1_pristine
def test_imoist_zero_emits_an_exactly_dry_state(library: Path) -> None:
    dry = build(
        library,
        InitializerSelection(isnd=7, iwnd=0, iinit=1, imoist=0),
        "moist.q3d",
    )
    assert np.all(dry == 0.0)
    moist = build(library, InitializerSelection(isnd=7, iwnd=0, iinit=1), "moist.q3d")
    assert np.any(moist != 0.0)


def test_a_dry_integration_is_refused_and_a_dry_inspection_is_not(tmp_path: Path) -> None:
    with pytest.raises(CM1Refusal) as error:
        load_config(
            write_config(tmp_path, "[initialization]\nisnd = 5\niwnd = 2\nimoist = 0")
        )
    assert "CM1-CFG-CARTESIAN-DRY-INTEGRATION" in str(error.value)
    document = toml_document(
        "[initialization]\nisnd = 5\niwnd = 2\nimoist = 0", run_extra="iconly = 1"
    ).replace('component = "wsm6-mp6"', 'component = "off"').replace(
        "equation_set = 2", "equation_set = 1"
    )
    path = tmp_path / "dry.toml"
    path.write_text(document, encoding="utf-8")
    assert load_config(path).initialization.imoist == 0


def test_maintain_rh_rescales_the_vapour_field(library: Path) -> None:
    plain = build(library, InitializerSelection(isnd=5, iwnd=2, iinit=1), "moist.q3d")
    held = build(
        library,
        InitializerSelection(isnd=5, iwnd=2, iinit=1, maintain_rh=True),
        "moist.q3d",
    )
    assert not np.array_equal(plain, held)


# -- the sealed identity and the unvalidated marker -------------------------


def test_an_undeclared_optional_key_keeps_the_established_canonical_bytes(
    tmp_path: Path,
) -> None:
    before = load_config(write_config(tmp_path, "[initialization]\nisnd = 5\niwnd = 2"))
    assert before.canonical_dict()["initialization"] == {"isnd": 5, "iwnd": 2}
    after = load_config(
        write_config(tmp_path, "[initialization]\nisnd = 5\niwnd = 2\niinit = 1")
    )
    projected = after.canonical_dict()["initialization"]
    assert projected["declared"] == ["iinit"]
    assert projected["ibalance"] == 0
    assert after.sha256 != before.sha256


def test_the_validation_record_covers_every_selectable_value() -> None:
    rows = {(row["selector"], row["value"]) for row in validated_rows()}
    for iwnd in IMPLEMENTED_IWND:
        assert ("iwnd", str(iwnd)) in rows
    for iinit in IMPLEMENTED_IINIT:
        assert ("iinit", str(iinit)) in rows
    for selector, values in (
        ("isnd", (1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 17, 18, 19, 20, 21, 22, 23)),
        ("ibalance", (0, 1, 2)),
        ("imoist", (0, 1)),
        ("irandp", (0, 1)),
        ("iorigin", (1, 2)),
    ):
        for value in values:
            assert (selector, str(value)) in rows


def test_the_unvalidated_marker_names_exactly_what_was_not_checked() -> None:
    clean = unvalidated_initialization(InitializerSelection(isnd=5, iwnd=2, iinit=1))
    assert clean["status"] == "validated"
    assert clean["items"] == []
    noisy = unvalidated_initialization(
        InitializerSelection(isnd=5, iwnd=2, iinit=1, irandp=1, random_seed=3)
    )
    assert noisy["status"] == UNVALIDATED_MARKER
    assert [item["selector"] for item in noisy["items"]] == ["irandp"]
    assert "gfortran" in noisy["items"][0]["not_checked"]


def test_doctor_reports_the_unvalidated_selectors_by_name() -> None:
    from cm1core.doctor import collect

    subjects = [finding.subject for finding in collect()]
    assert any(
        subject.startswith(f"{UNVALIDATED_MARKER}: initialization.irandp") for subject in subjects
    )


# -- README.namelist:233-236: iconly ---------------------------------------


def test_iconly_writes_one_model_time_zero_frame_and_opens_no_device(
    library: Path, tmp_path: Path
) -> None:
    from cm1core.cartesian_door import run_cartesian

    path = write_config(
        tmp_path, "[initialization]\nisnd = 5\niwnd = 2\niinit = 1", run_extra="iconly = 1"
    )
    receipt = tmp_path / "receipt.json"
    history = tmp_path / "ic"
    code = run_cartesian(
        load_config(path),
        config_label=str(path),
        output=receipt,
        library=library,
        history_dir=history,
    )
    assert code == 0
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload["status"] == "built-initial-condition-and-stopped"
    assert payload["device_opened"] is False
    assert payload["unvalidated"]["status"] == "validated"
    frame = history / "frame-000000"
    assert (frame / "history-carriers.f32").is_file()
    assert (frame / "history-field-table.tsv").is_file()
    rows = (frame / "history-field-table.tsv").read_text(encoding="utf-8").splitlines()
    # fccb316 made the iconly frame CM1's nstep=0 &param9 frame: the default
    # output selection's products (rain, prate, the WSM6 accumulators, sr, th,
    # zhval and the base state) follow the prognostic carriers.
    assert [row.split("\t")[0] for row in rows] == [
        "closure.tkea", "pp3d", "prs", "q3d", "rf", "rho", "terrain_msl",
        "th3d", "u3d", "v3d", "w3d", "zf",
        "rain", "prate", "snownc", "graupelnc", "sr", "th", "zhval",
        "pi0", "th0", "prs0", "qv0", "u0", "v0",
    ]
    assert payload["initialization_bubble_center_xy_m"] == [8000.0, 8000.0]


def test_the_initial_condition_carries_the_initialized_density_not_the_base(
    library: Path,
) -> None:
    # init3d.F:2394's rho divides by the TOTAL theta.  Reading base.F's rho0
    # here started every run from a density inconsistent with its own theta by
    # the perturbation's own amplitude.
    selection = InitializerSelection(isnd=5, iwnd=2, iinit=1)
    rho = build(library, selection, "terrain_open.rho")
    rho0 = build(library, selection, "terrain_open.rho0")
    tha = build(library, selection, "terrain_open.tha")
    assert np.any(tha > 0.0)
    assert not np.array_equal(rho, rho0)
    # Outside the perturbation the two agree to the last binary32 bit or two:
    # base.F:2028 writes rd*(1+qv*reps) where init3d.F:2395 writes
    # (rd+qv*rv), which is the same number and a different rounding.
    undisturbed = tha == 0.0
    disturbed = tha > 0.5
    quiet = float(np.abs(rho[undisturbed] / rho0[undisturbed] - 1.0).max())
    loud = float(np.abs(rho[disturbed] / rho0[disturbed] - 1.0).max())
    assert quiet < 1e-6, quiet
    # Inside it they differ by the perturbation's own amplitude: 1 K on a
    # ~300 K base is 0.33%, which is the buoyancy the bubble exists to make.
    assert loud > 1e-3, loud
