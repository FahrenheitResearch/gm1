"""Four step-time cuts that change no published word, and the probe that measures them.

* The CuPy pool is no longer handed back to the driver in every phase
  (``GM1_DYNAMICS_POOL_RELEASE``, default ``keep``).  Every ``cupy.empty``
  buffer then holds whatever the pool returns, so the whole door is run on the
  host with every uninitialized allocation poisoned (NaN, then -3.0e38) and
  its published history and restart words compared with the unpoisoned run's.
* The final RK stage no longer launches the plain moisture advection whose
  every word the pdef chain rewrites (``GM1_FINAL_STAGE_MOISTURE_ADVECTION``,
  default ``pdef-only``; ``tests/test_stretched_scalar_species_generic.py``
  holds the launch log to b200ad4's, and the words to b200ad4's except the
  final stage's ``q3d``, which moved to CM1's ``(s*d)*rdx`` pdef product
  association and is held to ``PDEF_ASSOCIATION_DIGESTS``).
* ``scalar_boundaries`` keeps only its ``pp3d`` fills.
* A stretched binding's ``sigmaf(nk+1)`` is read back once, not per launch.

``tests/test_acoustic_compact_dead_stores.py`` covers the acoustic dead stores.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------ switches


def test_the_pool_policy_defaults_to_keep_and_refuses_what_it_does_not_know() -> None:
    from cm1core.cuda_cartesian_dynamics import (
        DYNAMICS_POOL_RELEASE_ENVIRONMENT as NAME,
        dynamics_pool_release_mode,
    )

    assert NAME == "GM1_DYNAMICS_POOL_RELEASE"
    assert dynamics_pool_release_mode({}) == "keep"
    assert dynamics_pool_release_mode({NAME: "keep"}) == "keep"
    assert dynamics_pool_release_mode({NAME: "phase"}) == "phase"
    for value in ("", "free", "PHASE", "1"):
        with pytest.raises(ValueError, match=NAME):
            dynamics_pool_release_mode({NAME: value})


def test_the_final_stage_advection_defaults_to_the_pdef_chain_alone() -> None:
    from cm1core.cuda_scalar_moist_rk import (
        FINAL_STAGE_MOISTURE_ADVECTION_ENVIRONMENT as NAME,
        final_stage_moisture_advection_mode,
        final_stage_moisture_advection_record,
    )

    assert NAME == "GM1_FINAL_STAGE_MOISTURE_ADVECTION"
    assert final_stage_moisture_advection_mode({}) == "pdef-only"
    assert final_stage_moisture_advection_mode({NAME: "launch"}) == "launch"
    with pytest.raises(ValueError, match=NAME):
        final_stage_moisture_advection_mode({NAME: "both"})
    assert final_stage_moisture_advection_record(1, "pdef-only") == "scalar-advection5"
    assert final_stage_moisture_advection_record(2, "launch") == "scalar-advection5"
    assert final_stage_moisture_advection_record(3, "pdef-only") == "pdef-chain-only"
    assert final_stage_moisture_advection_record(3, "launch") == "scalar-advection5-then-pdef-chain"


class _Stream:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def synchronize(self) -> None:
        pass


def _shell(pool_release: str | None):
    from cm1core.cuda_cartesian_dynamics import CartesianDynamics

    frees = []
    dynamics = object.__new__(CartesianDynamics)
    dynamics.stream = _Stream()
    dynamics.cupy = SimpleNamespace(
        get_default_memory_pool=lambda: SimpleNamespace(free_all_blocks=lambda: frees.append(1))
    )
    dynamics.arrays = {}
    dynamics._scalar_workspace = False
    if pool_release is not None:
        dynamics.pool_release = pool_release
    return dynamics, frees


@pytest.mark.parametrize(("policy", "expected"), (("keep", 0), ("phase", 1), (None, 0)))
def test_a_completed_phase_returns_the_pool_only_under_phase(policy, expected) -> None:
    dynamics, frees = _shell(policy)
    for phase in ("acoustic", "scalar", "all"):
        before = len(frees)
        dynamics.release_phase_workspace(phase)
        assert len(frees) - before == expected


def test_the_dynamics_records_its_pool_policy_in_its_receipts(monkeypatch) -> None:
    source = (ROOT / "src/cm1core/cuda_cartesian_dynamics.py").read_text()
    # One read of the environment, at construction, recorded beside the other
    # dynamics receipts every run publishes.
    assert source.count("dynamics_pool_release_mode()") == 1
    assert '"schema": "gm1.cartesian-dynamics-device-pool/v1"' in source
    assert source.count("free_all_blocks()") == 1  # only inside _release_pool


# ---------------------------------------------------------- scalar boundaries


def test_scalar_boundaries_fill_only_the_pressure_halo_and_allocate_nothing() -> None:
    from cm1core.cuda_cartesian_dynamics import CartesianDynamics
    from cm1core.cuda_psolver3_terrain_open import terrain_open_sound_shapes
    from cm1core.grid_geometry import CartesianGridGeometry

    calls = []

    def kernel(index):
        return lambda grid, block, args, stream=None: calls.append((index, args[0]))

    grid = CartesianGridGeometry(9, 8, 6, 100.0, 100.0, 100.0)
    dynamics = object.__new__(CartesianDynamics)
    dynamics.stream = _Stream()
    dynamics.geometry = grid
    dynamics.cupy = SimpleNamespace(uint32=np.uint32)
    dynamics.open_compiled = SimpleNamespace(kernels=tuple(kernel(i) for i in range(8)))
    dynamics.scalar_compiled = SimpleNamespace(kernels=tuple(kernel(100 + i) for i in range(4)))
    dynamics.registry = SimpleNamespace(allocate=pytest.fail, release=pytest.fail)
    pp3d = np.zeros(terrain_open_sound_shapes(9, 8, 6)["pp3d"], dtype=np.float32)
    dynamics.arrays = {"pp3d": pp3d}
    dynamics.fields = {}
    dynamics.numq = 6
    dynamics._expected_phase = ("scalar-boundary", 2)
    dynamics.scalar_boundaries(2)
    assert [(index, value is pp3d) for index, value in calls] == [(3, True), (4, True)]
    assert dynamics._expected_phase == ("pre", 3)


# ---------------------------------------------------------- sigmaf top memo


def test_a_bindings_model_top_is_read_back_once_and_a_wrong_top_is_always_refused() -> None:
    from tools.oracle import host_cupy

    from cm1core.cuda_gm1_child_stretched_metrics import (
        Gm1ChildStretchedMetrics,
        require_generated_vertical,
    )

    cp = host_cupy.build()
    nk = 5
    faces = np.array([0, 0, 20, 50, 100, 170, 260, 260], dtype=np.float32)

    def binding(top: float) -> Gm1ChildStretchedMetrics:
        dev = lambda a: host_cupy._dev(np.ascontiguousarray(a, dtype=np.float32))  # noqa: E731
        return Gm1ChildStretchedMetrics(
            dev(np.zeros(nk + 2)), dev(faces), dev(np.ones(nk + 2)), dev(np.ones(nk + 3)),
            top, 100.0, "host",
            MappingProxyType({"schema": "fixture", "vertical_coordinate": {
                "nk": nk, "ztop_m": top, "dz_reference_m": 100.0}}),
        )

    good = binding(260.0)
    host_cupy.COUNTS.clear()
    for _ in range(4):
        require_generated_vertical(good, nk=nk, cupy=cp)
    assert host_cupy.COUNTS["device_to_host_read"] == 1
    again = binding(260.0)
    require_generated_vertical(again, nk=nk, cupy=cp)
    assert host_cupy.COUNTS["device_to_host_read"] == 2
    wrong = binding(250.0)
    for _ in range(2):
        with pytest.raises(ValueError, match="sigmaf"):
            require_generated_vertical(wrong, nk=nk, cupy=cp)
    assert host_cupy.COUNTS["device_to_host_read"] == 4


# -------------------------------------------------- the door, poisoned, on the host

DECK = textwrap.dedent("""\
    [run]
    lane = "cartesian"
    duration_s = 1.5
    dt_s = 0.5
    history_interval_s = 0.5
    restart_interval_s = 1.5

    [initialization]
    isnd = 5
    iwnd = 2

    [grid]
    nx = 16
    ny = 14
    nz = 35
    dx_m = 100.0
    dy_m = 100.0
    dz_m = 100.0
    terrain = false
    stretch_x = false
    stretch_y = false
    stretch_z = 1
    ztop_m = 2700.0
    str_bot_m = 100.0
    str_top_m = 700.0
    dz_bot_m = 20.0
    dz_top_m = 100.0

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
    bottom = 3
    top = 1
    radiative_order = 4

    [closures]
    sgs_model = 4
    t2p_avg = 1
    vertical_implicit_diffusion = true
    upper_damping = 1
    upper_damping_depth_m = 1000.0
    upper_damping_rate_s_inv = 0.0033333333333
    moving_frame = true
    u_move_m_s = 12.5
    v_move_m_s = 3.0
    bbc = 3
    sfcmodel = 1
    isfcflx = 0
    cecd = 1
    cnstcd = 0.001

    [runtime]
    acoustic_substeps = 6
    warmup_steps = 1
    memory_budget_mib = 2048
    capacity_profile = "default"

    [physics]
    component = "COMPONENT"
    equation_set = 2
    output_dbz = DBZ
""")

#: SHA-256 of every published carrier's word digest and every envelope's payload
#: and restored-arena digests, from the three-step run of DECK with WSM6 and with
#: Morrison.  Host-emulated words from g++ 13.3 and glibc 2.39 (Ubuntu 24.04, as
#: in the release's test container), so a different compiler or libm can move
#: them (g++ 15 does); the poison comparisons below do not depend on these.
#:
#: The words carry weno5's linear weights as CM1's binary32 literals
#: (adv_routines.F:6189-6191) and CM1's sign of zero on the vertical WENO flux
#: faces; the ``stretch_z = 1`` deck above on CM1's ``terrain_flag=.false.``
#: stretched statements (every vertical flux difference ``((d*rdz)*mh(1,1,k))``
#: or ``((d*rdz)*mf(1,1,k))``, the base state read at the column); every column
#: component's latent heating on CM1's constant-volume step; and CM1's step-1 TKE
#: start.
PRE_CHANGE_WORDS_SHA256 = {
    "wsm6-mp6": ("0ee0094febba6b5722743ca33bf6e035bdb891bef1dda14e905d2df942a0fe36", 132),
    "morrison-mp10": ("7bdcad2054db1aea49dc78181279040a59df90ac750663fc280d328f781e8a0c", 131),
}


def _cargo_artifact(name: str, environment: str, cargo_args: list[str]) -> Path:
    supplied = os.environ.get(environment)
    if supplied:
        return Path(supplied).resolve(strict=True)
    target = ROOT / "build/rust-native-target"
    built = target / "release" / name
    if not built.is_file():
        if shutil.which("cargo") is None:
            pytest.skip(f"cargo is not available to build {name}; set {environment}")
        result = subprocess.run(
            ["cargo", "build", "--offline", "--release", *cargo_args, "--manifest-path",
             str(ROOT / "rust/cm1-state/Cargo.toml"), "--target-dir", str(target)],
            capture_output=True, text=True,
        )
        if result.returncode != 0 or not built.is_file():
            pytest.skip(f"{name} could not be built here: {result.stderr[-400:]}")
    return built.resolve()


@pytest.fixture(scope="module")
def native(tmp_path_factory):
    if shutil.which("g++") is None:
        pytest.skip("the host kernel emulator needs g++")
    return SimpleNamespace(
        library=_cargo_artifact("libcm1_state.so", "GM1_CM1_STATE_LIBRARY", ["--lib"]),
        publisher=_cargo_artifact("cm1_forecast_publish", "GM1_FORECAST_PUBLISHER",
                                  ["--bin", "cm1_forecast_publish"]),
        root=tmp_path_factory.mktemp("door-host"),
    )


def _door(native, component: str, name: str, *extra: str) -> dict:
    deck = native.root / f"{component}.toml"
    if not deck.exists():
        deck.write_text(DECK.replace("COMPONENT", component).replace(
            "DBZ", "true" if component == "wsm6-mp6" else "false"))
    workdir = native.root / f"{component}-{name}"
    if (workdir / "manifest.json").is_file():  # one run per variant per module
        return json.loads((workdir / "manifest.json").read_text())
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    completed = subprocess.run(
        [sys.executable, str(ROOT / "tools/oracle/cartesian_door_host_run.py"), str(deck),
         "--library", str(native.library), "--publisher", str(native.publisher),
         "--workdir", str(workdir), *extra],
        capture_output=True, text=True, env=environment,
    )
    assert completed.returncode == 0, completed.stderr[-3000:]
    return json.loads((workdir / "manifest.json").read_text())


def _words_sha256(manifest: dict) -> tuple[str, int]:
    import hashlib

    document = {
        "carriers": {name: row["sha256"] for name, row in manifest["carriers"].items()},
        "envelopes": {name: {"payload": row["envelope_payload_sha256"], "restored": row.get("restored")}
                      for name, row in manifest["publications"].items()},
    }
    return hashlib.sha256(json.dumps(document, sort_keys=True).encode()).hexdigest(), len(manifest["carriers"])


@pytest.mark.parametrize("component", ("wsm6-mp6", "morrison-mp10"))
def test_every_uninitialized_buffer_poisoned_publishes_the_same_history_and_restart(native, component) -> None:
    from tools.oracle.compare_door_host_runs import compare

    clean = _door(native, component, "off")
    assert clean["door_status"] == 0
    assert _words_sha256(clean) == PRE_CHANGE_WORDS_SHA256[component]
    # Three frames of six files and one restart of four: since perf/async-publish
    # a restart keeps its words only in restart.cm1fcp, and the manifest reads
    # them from the arena restored out of it, under the same carrier names.
    assert len(clean["files"]) == 22 and len(clean["publications"]) == 4
    assert not any(name.endswith("restart-carriers.f32") for name in clean["files"])
    for poison in ("nan", "big"):
        poisoned = _door(native, component, poison, "--poison", poison)
        assert poisoned["counts"]["poisoned_allocation"] > 100
        result = compare(clean, poisoned)
        assert result["identical"], result
        assert result["nonfinite_carriers"] == []


def test_the_poison_comparison_sees_a_buffer_that_needed_its_zeros(native) -> None:
    """Negative control: the persistent padded owners' halo words are never
    written after their zeroed allocation, and the restart publishes them."""

    from tools.oracle.compare_door_host_runs import compare

    clean = _door(native, "wsm6-mp6", "off")
    control = _door(native, "wsm6-mp6", "control", "--poison", "big", "--zero-as-empty", "padded.")
    result = compare(clean, control)
    assert not result["identical"]
    assert result["differing_carriers"]
    assert all(name.startswith("restarts/") for name in result["differing_carriers"])


# ------------------------------------------------------------ the device probe


FAKE_CUPY = textwrap.dedent('''\
    import numpy as _np

    class _Pool:
        def __init__(self):
            self.freed = 0
            self.total = 1 << 30
        def free_all_blocks(self):
            self.freed += 1
            self.total = 0
        def total_bytes(self):
            return self.total
        def used_bytes(self):
            return 0

    _POOL = _Pool()

    def get_default_memory_pool():
        return _POOL

    def empty(shape, dtype=_np.float32):
        return _np.zeros(shape, dtype=dtype)

    def empty_like(a, dtype=None):
        return _np.zeros_like(a, dtype=dtype)

    class _Device:
        def synchronize(self):
            pass

    class cuda:
        Device = _Device
''')

FAKE_DYNAMICS = textwrap.dedent('''\
    import cupy

    def run(method):
        return method()

    def pre_acoustic():
        cupy.get_default_memory_pool().free_all_blocks()

    def _release_pool():
        cupy.get_default_memory_pool().free_all_blocks()

    def post_acoustic():
        run(_release_pool)
''')


def _probe(tmp_path: Path, mode: str, *, find_spec_first: bool = False) -> tuple[subprocess.CompletedProcess, dict | None]:
    tag = f"{mode}-find-spec" if find_spec_first else mode
    fake = tmp_path / f"fake-{tag}"
    (fake / "cupy").mkdir(parents=True)
    (fake / "cupy" / "__init__.py").write_text(FAKE_CUPY)
    (fake / "cuda_cartesian_dynamics.py").write_text(FAKE_DYNAMICS)
    logs = tmp_path / f"logs-{tag}"
    # cartesian_door.require_cuda_runtime asks importlib.util.find_spec("cupy")
    # before it imports CuPy; the probe has to survive that lookup.
    lookup = 'import importlib.util; assert importlib.util.find_spec("cupy") is not None\n'
    script = (lookup if find_spec_first else "") + textwrap.dedent('''\
        import json, cupy, cuda_cartesian_dynamics as d
        for _ in range(3):
            d.pre_acoustic()
            d.post_acoustic()
        cupy.get_default_memory_pool().free_all_blocks()
        print(json.dumps({
            "freed": cupy.get_default_memory_pool()._pool.freed
                     if hasattr(cupy.get_default_memory_pool(), "_pool")
                     else cupy.get_default_memory_pool().freed,
            "empty_nan": bool(__import__("numpy").isnan(cupy.empty((2,))).all()),
            "empty_int": int(cupy.empty((2,), dtype="int32").sum()),
        }))
    ''')
    environment = {
        **os.environ, "GM1_PROBE": mode, "GM1_PROBE_LOG_DIR": str(logs),
        "PYTHONPATH": os.pathsep.join([str(ROOT / "tools/perf_probe"), str(fake)]),
    }
    completed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                               env=environment, cwd=tmp_path)
    summaries = list(logs.glob("probe-*-summary.json")) if logs.exists() else []
    return completed, (json.loads(summaries[0].read_text()) if summaries else None)


def test_the_probe_announces_itself_and_times_only_the_dynamics_frees(tmp_path) -> None:
    completed, summary = _probe(tmp_path, "time")
    assert completed.returncode == 0, completed.stderr
    assert "GM1_PROBE loaded mode=time" in completed.stderr
    result = json.loads(completed.stdout)
    assert result["freed"] == 7  # six timed dynamics frees, one passed through
    assert summary["intercepted_calls"] == 6 and summary["passthrough_free_calls"] == 1
    assert set(summary["sites"]) == {"pre_acoustic", "post_acoustic"}
    assert summary["large_steps_seen"] == 1
    assert summary["freed_bytes_total"] == 1 << 30
    rows = [json.loads(line) for line in next((tmp_path / "logs-time").glob("*.jsonl")).read_text().splitlines()]
    assert len(rows) == 6 and all("sync_s" in row and "free_s" in row for row in rows)


def test_the_probe_keeps_or_poisons_as_asked_and_refuses_an_unknown_mode(tmp_path) -> None:
    completed, summary = _probe(tmp_path, "keep")
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["freed"] == 1  # only the non-dynamics call
    assert summary["intercepted_calls"] == 6
    completed, summary = _probe(tmp_path, "poison-nan")
    result = json.loads(completed.stdout)
    assert result == {"freed": 7, "empty_nan": True, "empty_int": 0}
    assert summary["poisoned_allocations"] == 1
    completed, _ = _probe(tmp_path, "off")
    assert json.loads(completed.stdout)["empty_nan"] is False
    assert "GM1_PROBE loaded mode=off" in completed.stderr
    completed, _ = _probe(tmp_path, "sometimes")
    assert completed.returncode != 0 and completed.stdout == ""


@pytest.mark.parametrize("mode", ("off", "time", "keep", "poison-nan", "poison-big"))
def test_the_probe_still_patches_cupy_after_the_doors_find_spec_lookup(tmp_path, mode) -> None:
    """The door's find_spec("cupy") used to consume the hook, so the import
    that followed ran unpatched and every free reached the pool."""

    completed, summary = _probe(tmp_path, mode, find_spec_first=True)
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    if mode == "off":
        assert summary is None and "GM1_PROBE summary" not in completed.stderr
        assert result == {"freed": 7, "empty_nan": False, "empty_int": 0}
        return
    assert summary["cupy_patched"] is True and summary["cupy_imported"] is True
    if mode in ("time", "keep"):
        assert summary["intercepted_calls"] == 6
        assert result["freed"] == (7 if mode == "time" else 1)
    else:
        assert summary["poisoned_allocations"] == 1
        assert result["empty_nan"] is (mode == "poison-nan")


def test_a_probe_that_never_reached_cupy_says_so_and_the_report_fails_it(tmp_path) -> None:
    from tools.perf_probe.m0_m4_report import probe_effect

    fake = tmp_path / "fake"
    (fake / "cupy").mkdir(parents=True)
    (fake / "cupy" / "__init__.py").write_text(FAKE_CUPY)
    # An import that bypasses sys.meta_path, as a probe bug would.
    script = textwrap.dedent('''\
        import importlib.util, sys
        spec = importlib.util.spec_from_file_location("cupy", sys.argv[1])
        module = importlib.util.module_from_spec(spec)
        sys.modules["cupy"] = module
        spec.loader.exec_module(module)
        module.get_default_memory_pool().free_all_blocks()
    ''')
    environment = {**os.environ, "GM1_PROBE": "keep", "PYTHONPATH": str(ROOT / "tools/perf_probe")}
    completed = subprocess.run([sys.executable, "-c", script, str(fake / "cupy" / "__init__.py")],
                               capture_output=True, text=True, env=environment, cwd=tmp_path)
    assert completed.returncode == 0, completed.stderr
    stderr = tmp_path / "stderr.txt"
    stderr.write_text(completed.stderr)
    effect = probe_effect(stderr, "keep")
    assert effect["summaries"] == 1 and effect["cupy_patched"] is False and effect["acted"] is False


def test_the_probe_patches_the_installed_cupy_after_find_spec(tmp_path) -> None:
    """The real CuPy import path (no device is touched: CUDA_VISIBLE_DEVICES is empty)."""

    import importlib.util

    if importlib.util.find_spec("cupy") is None:
        pytest.skip("CuPy is not installed here")
    script = textwrap.dedent('''\
        import importlib.util, json
        importlib.util.find_spec("cupy")
        import cupy
        print(json.dumps({"pool": hasattr(cupy.get_default_memory_pool, "__wrapped__"),
                          "empty": hasattr(cupy.empty, "__wrapped__")}))
    ''')
    seen = {}
    for mode in ("keep", "poison-big"):
        environment = {**os.environ, "GM1_PROBE": mode, "CUDA_VISIBLE_DEVICES": "",
                       "PYTHONPATH": str(ROOT / "tools/perf_probe")}
        completed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                                   env=environment, cwd=tmp_path)
        if completed.returncode != 0:
            pytest.skip(f"CuPy does not import here: {completed.stderr[-300:]}")
        seen[mode] = json.loads(completed.stdout.strip().splitlines()[-1])
        assert '"cupy_patched": true' in completed.stderr
    assert seen == {"keep": {"pool": True, "empty": False}, "poison-big": {"pool": False, "empty": True}}
