"""CPU profile arithmetic, staging and refusal checks for history frames.

Storage is emulated with numpy; no CUDA transfer, envelope validation or
forecast skill is claimed by anything in this file.
"""

import hashlib
import json
from pathlib import Path
import struct
import subprocess
import time
from types import SimpleNamespace

import numpy as np
import pytest

from cm1core.cartesian_history import (
    ARENA_FILENAME,
    CARTESIAN_HISTORY_CARRIER_PROFILES,
    CARTESIAN_HISTORY_PROFILE_ID,
    CARTESIAN_HISTORY_REDUCED_PROFILE_ID,
    COMPLETE_CARRIER_PROFILE,
    FIELD_TABLE_FILENAME,
    FORECAST_ADMISSION_HOST_BYTES_PER_PAYLOAD_BYTE,
    MAX_FORECAST_PAYLOAD_BYTES,
    OMITTED_CARRIERS,
    RECEIPT_FILENAME,
    SCALAR_TABLE_FILENAME,
    RECONSTRUCTIBLE_CARRIERS,
    RECONSTRUCTIBLE_OMITTED_CARRIER_PROFILE,
    STAGING_BUFFER_NAME,
    cartesian_history_profile,
    forecast_admission_host_bytes,
    publish_cartesian_history,
    require_cartesian_publication_boundary,
    require_forecast_publisher,
    resume_clock_from_publication,
    require_history_carrier_profile,
)
from cm1core import cartesian_history
from cm1core import cuda_cartesian_runtime as runtime_module
from cm1core.cuda_cartesian_runtime import CudaCartesianRuntime
from cm1core.forecast_arena import require_forecast_arena_carriers
from cm1core.gm1_idealized_residency import DeviceBufferRegistry
from cm1core.gm1_runtime import (
    CM1_COURANT_STOP,
    CM1_COURANT_STOP_AUTHORITY,
    CM1_COURANT_TARGET,
    DOMAIN_LOCATION_AUTHORITY,
    RuntimeClock,
    RuntimeConfig,
)
from cm1core.grid_geometry import CartesianGridGeometry


class Stream:
    def __init__(self):
        self.depth = 0
        self.syncs = 0

    def __enter__(self):
        self.depth += 1
        return self

    def __exit__(self, *_):
        self.depth -= 1

    def synchronize(self):
        self.syncs += 1


class CpuStorage:
    """Only array storage and the host copy are emulated, never kernels."""

    float32 = np.float32
    float64 = np.float64
    ndarray = np.ndarray
    dtype = staticmethod(np.dtype)
    asnumpy = staticmethod(np.asarray)
    copyto = staticmethod(np.copyto)

    def __init__(self, stream):
        self.stream = stream

    def empty(self, shape, *, dtype):
        assert self.stream.depth > 0
        return np.empty(shape, dtype=dtype)

    def zeros(self, shape, *, dtype):
        assert self.stream.depth > 0
        return np.zeros(shape, dtype=dtype)


class MetricMap(dict):
    """The runtime's immutable metric map: a mapping that carries its proofs.

    ``upload_immutable_source_metrics`` records, per metric, how the words were
    admitted.  ``zf`` earns ``"horizontal-profile"`` only after every
    horizontal plane of the source was checked equal.
    """

    def __init__(self, values, proofs):
        super().__init__(values)
        self.proofs = dict(proofs)


def _config(nx=9, ny=11, nz=8, spacing=500.0, output=None):
    return RuntimeConfig(
        CartesianGridGeometry(nx, ny, nz, spacing, spacing, spacing), output=output
    )


def _fake_runtime(config, *, budget_bytes=64 * 1024**2, completed_steps=2):
    g = config.grid
    stream = Stream()
    storage = CpuStorage(stream)
    registry = DeviceBufferRegistry(storage, stream, budget_bytes=budget_bytes)
    fields, owners = {}, {}
    for name, stagger in (("u3d", "u"), ("v3d", "v"), ("w3d", "w")):
        padded = g.padded_shape(stagger, halo_xy=3, halo_z=1)
        owner = np.arange(int(np.prod(padded)), dtype=np.float32).reshape(padded)
        owners[name] = owner
        fields[name] = owner[g.physical_slices(stagger, halo_xy=3, halo_z=1)]
    for name, shape in (("pp3d", g.mass_shape), ("prs", g.mass_shape), ("rho", g.mass_shape),
                        ("th3d", g.mass_shape), ("rf", g.w_shape), ("q3d", (6, *g.mass_shape))):
        fields[name] = np.full(shape, len(name), dtype=np.float32)
    metrics = MetricMap(
        {
            "terrain_msl": np.zeros(g.surface_shape, dtype=np.float32),
            "zf": np.broadcast_to(
                (np.arange(g.nz + 1, dtype=np.float32) * np.float32(g.dz_m))[:, None, None],
                g.w_shape,
            ),
        },
        {
            "terrain_msl": {"kind": "separate-source-array"},
            "zf": {"kind": "horizontal-profile"},
        },
    )
    dynamics = SimpleNamespace(
        fields=fields, metrics=metrics,
        base_profiles={
            "pi0": np.linspace(1.0, 0.4, g.nz, dtype=np.float32),
        },
        tke={"tkea": np.full(g.w_shape, 0.5, dtype=np.float32)},
        _expected_phase=("held", 1),
    )
    # CM1 &param9 products.  This fixture emulates storage only, so the
    # products are filled with numpy planes of the declared extent: the point
    # under test is the profile, the field table and the arena, not the
    # kernels, which tools/oracle/cm1_output_diagnostics_oracle.py exercises
    # against closed-form fields.
    from cm1core.cm1_output_selection import output_carrier_rows

    output_fields = {}
    for row in output_carrier_rows(
        config.output, mass=g.mass_shape, w=g.w_shape, u=g.u_shape, v=g.v_shape,
        surface=g.surface_shape, profile=(g.nz,),
    ):
        output_fields[row[0]] = np.full(row[4], float(len(row[0])), dtype=np.float32)
    runtime = SimpleNamespace(
        cp=storage, stream=stream, registry=registry, config=config, dynamics=dynamics,
        output_fields=output_fields,
        # The real clock, advanced the real way: the published
        # domainlocx/domainlocy are cm1.F:3512-3513's own running sum over
        # these steps, not a number the fixture wrote down.
        clock=_advanced_clock(config, completed_steps),
        finish=SimpleNamespace(completed_physics_step=completed_steps,
                               completed_solve3_step=completed_steps),
        runtime_source_identity={"aggregate_sha256": "11" * 32},
    )
    return runtime, owners


class _NoOpBackend:
    """A CartesianBackend that executes nothing, so only the clock moves."""

    def execute(self, operation, *, step):
        pass

    def synchronize(self):
        pass


def _advanced_clock(config, completed_steps):
    clock = RuntimeClock(config)
    backend = _NoOpBackend()
    for _ in range(completed_steps):
        clock.step(backend)
    return clock


def test_supported_grids_price_under_the_forecast_payload_ceiling():
    from cm1core.cm1_output_selection import OutputSelection

    bare = OutputSelection(values={
        key: 0 if value else 0 for key, value in OutputSelection().values.items()
    } | {"output_format": 1, "output_filetype": 2, "output_thpert": 1,
         "output_prs": 1, "output_pipert": 1, "output_rho": 1, "output_tke": 1,
         "output_qv": 1, "output_q": 1, "output_u": 1, "output_v": 1,
         "output_w": 1, "output_th": 0})
    for nz, expected in ((40, 412_928_000), (80, 822_656_000)):
        profile = cartesian_history_profile(_config(400, 400, nz, output=bare))
        assert profile.payload_nbytes == expected
        assert profile.payload_nbytes < MAX_FORECAST_PAYLOAD_BYTES
        assert sum(carrier.nbytes for carrier in profile.carriers) == expected
        assert profile.record()["maximum_forecast_payload_bytes"] == 16 * 1024**3
        assert profile.record()["admission_host_bytes"] == 2 * expected
    profile = cartesian_history_profile(_config(400, 400, 80))
    # Only the strided wind views and the broadcast zf profile are staged.
    assert profile.staged_names() == ("u3d", "v3d", "w3d", "zf")
    assert profile.staging_bytes == 51_328_000 + 51_328_000 + 51_840_000 + 51_840_000


def test_profile_omits_geopotential_by_name_and_keeps_the_remaining_carriers():
    profile = cartesian_history_profile(_config())
    names = tuple(carrier.name for carrier in profile.carriers)
    # The prognostic-state carriers come first, in the order they always had;
    # CM1 &param9 products follow, selected by cm1_output_selection.
    assert names[:12] == ("closure.tkea", "pp3d", "prs", "q3d", "rf", "rho",
                          "terrain_msl", "th3d", "u3d", "v3d", "w3d", "zf")
    assert set(names[12:]) == set(profile.record()["selected_output_names"])
    assert "total_geopotential_zf" not in names
    # th3d is CM1's tha.  The meaning string is what rust/gm1-view labels the
    # field with, and calling the perturbation "current potential temperature"
    # made a +/-10 K field read as a 300 K one.
    th3d = next(carrier for carrier in profile.carriers if carrier.name == "th3d")
    assert "perturbation" in th3d.meaning and "th0+tha" in th3d.meaning
    assert "invented field" in OMITTED_CARRIERS["total_geopotential_zf"]
    assert profile.record()["omitted_carriers"] == dict(OMITTED_CARRIERS)
    for carrier in profile.carriers:
        assert len(carrier.bounds()) == 2 * len(carrier.shape)
        assert carrier.bounds()[1::2] == carrier.shape


def test_grids_the_old_two_gibibyte_cap_refused_now_price_and_publish():
    """The tornado-scale grids the 2 GiB number refused, priced and admitted.

    None of these ever exceeded what CM1FHS1 can address: the container's
    header length is a 64-bit field and every payload offset is an exact
    decimal JSON integer.  The old ceiling was host-resource policy, and it,
    not the device, is what capped the publishable grid at 33.4M gridpoints.
    """

    for nx, ny, nz, expected in (
        (600, 600, 120, 3_136_899_360),
        (600, 600, 180, 4_692_389_040),
        (700, 700, 75, 2_681_702_100),
        (800, 800, 100, 4_654_722_800),
        (1000, 1000, 80, 5_832_642_240),
    ):
        profile = cartesian_history_profile(_config(nx, ny, nz))
        assert profile.payload_nbytes == expected
        assert expected > 2 * 1024**3
        assert profile.payload_nbytes <= MAX_FORECAST_PAYLOAD_BYTES
        assert profile.admission_host_bytes == 2 * expected
    # 600x600x180 is 64.8M gridpoints, 1.94x the old cap's 33.4M ceiling.
    assert 600 * 600 * 180 == 64_800_000 > 33_400_000


def test_a_grid_over_the_payload_ceiling_refuses_with_its_computed_host_cost():
    with pytest.raises(ValueError, match="19899491920") as error:
        cartesian_history_profile(_config(1400, 1400, 140))
    message = str(error.value)
    assert "nx=1400, ny=1400, nz=140" in message
    assert str(MAX_FORECAST_PAYLOAD_BYTES) in message
    # The doubled host cost -- the artifact plus one float32 copy of every
    # carrier -- is priced on the Rust read side now
    # (forecast.rs:forecast_read_admission_host_bytes).  This message prices
    # the payload against the contract and names the carriers that make it up.
    assert "will not publish a frame that silently drops a carrier" in message
    # The refusal names the wrong result it prevents and the way out.  The
    # "aborts on allocation" clause belongs to the Rust read gate, which is
    # where the host cost is priced; here the wrong result is a frame that
    # silently drops a carrier, and the way out is the reduced profile.
    assert RECONSTRUCTIBLE_OMITTED_CARRIER_PROFILE in message


def test_admission_host_cost_mirrors_the_rust_helper():
    """rust/cm1-state/src/forecast.rs:830 is the authority for this number."""

    assert FORECAST_ADMISSION_HOST_BYTES_PER_PAYLOAD_BYTE == 2
    assert forecast_admission_host_bytes(0) == 0
    assert forecast_admission_host_bytes(4_155_264_000) == 8_310_528_000
    with pytest.raises(ValueError, match="non-negative integer"):
        forecast_admission_host_bytes(-1)


def test_the_reduced_carrier_profile_is_opt_in_and_names_what_it_omits():
    complete = cartesian_history_profile(_config(400, 400, 80))
    reduced = cartesian_history_profile(
        _config(400, 400, 80), carrier_profile=RECONSTRUCTIBLE_OMITTED_CARRIER_PROFILE
    )
    assert complete.carrier_profile == COMPLETE_CARRIER_PROFILE
    assert complete.profile_id == CARTESIAN_HISTORY_PROFILE_ID
    assert reduced.profile_id == CARTESIAN_HISTORY_REDUCED_PROFILE_ID
    names = tuple(carrier.name for carrier in reduced.carriers)
    assert "prs" not in names and "zf" not in names
    # rf stays: CM1 rebuilds it from c1/c2 and the edge extrapolations at
    # solve3.F:666 and solve3.F:678-679, and this frame carries none of them.
    assert "rf" in names
    assert "base.pi0" in names and "metrics.zf_profile" in names
    assert reduced.payload_nbytes < complete.payload_nbytes
    assert reduced.payload_nbytes == (
        complete.payload_nbytes - (51_200_000 + 51_840_000) + 4 * 80 + 4 * 81
    )
    record = reduced.record()
    assert record["carrier_profile"] == RECONSTRUCTIBLE_OMITTED_CARRIER_PROFILE
    assert set(record["reconstructible_carriers"]) == {"prs", "zf"}
    assert "solve2.F:1588" in record["reconstructible_carriers"]["prs"]
    cost = record["reduced_profile_reader_cost"]
    # The cost names the reader that applies the formulas, not a disclaimer
    # that none does.
    assert "reduced_history.rs" in cost and "cm1-render" in cost
    assert complete.record()["reconstructible_carriers"] == {}
    assert complete.record()["reduced_profile_reader_cost"] is None


def test_an_unknown_carrier_profile_is_refused_by_value():
    assert set(CARTESIAN_HISTORY_CARRIER_PROFILES) == {
        COMPLETE_CARRIER_PROFILE, RECONSTRUCTIBLE_OMITTED_CARRIER_PROFILE,
    }
    for value in ("", "minimal", "COMPLETE", "reconstructible_omitted"):
        with pytest.raises(ValueError, match="history carrier profile") as error:
            require_history_carrier_profile(value)
        assert "no reader in this distribution declares" in str(error.value)
        with pytest.raises(ValueError, match="history carrier profile"):
            cartesian_history_profile(_config(), carrier_profile=value)
def test_a_grid_over_the_payload_cap_refuses_with_its_computed_size():
    # The size in the refusal is this run's own &param9 payload, not a fixed
    # number: a selection that asks for more fields refuses a smaller grid,
    # and the message must say which grid and which size.
    # 1000x1000x80 used to be over the cap; gap/payloadcap raised the ceiling
    # to 16 GiB and it publishes now, so the grid that exercises the refusal
    # has to be one this tree actually refuses.
    with pytest.raises(ValueError, match="19899491920") as error:
        cartesian_history_profile(_config(1400, 1400, 140))
    assert "nx=1400, ny=1400, nz=140" in str(error.value)
    assert str(MAX_FORECAST_PAYLOAD_BYTES) in str(error.value)


def test_publication_stages_strided_carriers_and_releases_the_staging_buffer(tmp_path):
    config = _config()
    runtime, owners = _fake_runtime(config)
    g = config.grid
    profile = cartesian_history_profile(config)
    raw = {carrier.name: (runtime.dynamics.fields if carrier.group == "fields"
                          else runtime.dynamics.metrics if carrier.group == "metrics"
                          else runtime.output_fields if carrier.group == "output"
                          else runtime.dynamics.tke)[carrier.key]
           for carrier in profile.carriers}
    # Every staged carrier is exactly the kind the serializer refuses raw.
    for name in profile.staged_names():
        with pytest.raises(ValueError, match="strided device view"):
            require_forecast_arena_carriers({name: raw[name]}, {name: raw[name].shape}, cupy=runtime.cp)
    before = runtime.registry.registered_allocation_bytes
    receipt = publish_cartesian_history(runtime, tmp_path / "frame")
    assert runtime.registry.registered_allocation_bytes == before
    assert STAGING_BUFFER_NAME not in runtime.registry.buffers
    expected = b"".join(np.ascontiguousarray(raw[carrier.name]).tobytes() for carrier in profile.carriers)
    arena = (tmp_path / "frame" / ARENA_FILENAME).read_bytes()
    assert arena == expected
    assert len(arena) == profile.payload_nbytes
    assert receipt["arena"]["nbytes"] == profile.payload_nbytes
    assert receipt["step"] == 2 and receipt["model_time_s"] == 1.0
    assert receipt["forecast_envelope"]["published"] is False
    assert receipt["restore_supported"] is False
    stored = json.loads((tmp_path / "frame" / RECEIPT_FILENAME).read_text())
    assert stored == receipt
    rows = (tmp_path / "frame" / FIELD_TABLE_FILENAME).read_text().splitlines()
    assert len(rows) == len(profile.carriers)
    first = rows[0].split("\t")
    assert first[0] == "closure.tkea" and first[4] == ",".join(str(e) for e in g.w_shape)
    # The arena's name, relative to the frame: a published frame is renamed
    # out of its staging directory and may be copied elsewhere, so an absolute
    # path written at staging time would name a file that no longer exists.
    assert first[6] == ARENA_FILENAME
    assert [int(row.split("\t")[7]) for row in rows] == [
        receipt["arena"]["carrier_offset_bytes"][carrier.name] for carrier in profile.carriers
    ]
    # The staged winds carry the padded owner's physical rectangle, not halos.
    for name in ("u3d", "v3d", "w3d"):
        start = receipt["arena"]["carrier_offset_bytes"][name]
        stop = start + int(np.prod(raw[name].shape)) * 4
        published = np.frombuffer(arena[start:stop], dtype="<f4").reshape(raw[name].shape)
        assert np.array_equal(published, np.ascontiguousarray(raw[name]))
        assert not np.array_equal(published.reshape(-1), owners[name].reshape(-1)[:published.size])


def test_publication_refuses_outside_the_post_solve3_quiescent_boundary(tmp_path):
    config = _config()
    runtime, _ = _fake_runtime(config)
    assert require_cartesian_publication_boundary(runtime) == 2
    runtime.dynamics._expected_phase = ("acoustic", 2)
    with pytest.raises(RuntimeError, match="quiescent boundary"):
        publish_cartesian_history(runtime, tmp_path / "mid-stage")
    runtime.dynamics._expected_phase = ("held", 1)
    runtime.finish.completed_solve3_step = 1
    with pytest.raises(RuntimeError, match="pre- and post-physics"):
        publish_cartesian_history(runtime, tmp_path / "pre-solve3")
    runtime.finish.completed_solve3_step = 2
    runtime.clock.completed_steps = 0
    with pytest.raises(RuntimeError, match="no large step has completed"):
        publish_cartesian_history(runtime, tmp_path / "initialized")
    runtime.clock.completed_steps = 2
    runtime.clock.failed = True
    with pytest.raises(RuntimeError, match="failed CUDA step"):
        publish_cartesian_history(runtime, tmp_path / "failed")
    assert list(tmp_path.iterdir()) == []


def test_existing_directory_or_foreign_geometry_refuses_before_any_transfer(tmp_path):
    config = _config()
    runtime, _ = _fake_runtime(config)
    existing = tmp_path / "frame"
    existing.mkdir()
    with pytest.raises(ValueError, match="already exists"):
        publish_cartesian_history(runtime, existing)
    assert list(existing.iterdir()) == []
    foreign = cartesian_history_profile(_config(nz=12))
    with pytest.raises(ValueError, match="prices another geometry"):
        publish_cartesian_history(runtime, tmp_path / "foreign", profile=foreign)
    assert not (tmp_path / "foreign").exists()


def test_forecast_publisher_is_admitted_only_at_the_exact_field_table_abi(tmp_path):
    script = tmp_path / "fake-publish"
    script.write_text("#!/bin/sh\necho gpuwm-cm1-forecast-other/v1\n")
    script.chmod(0o755)
    with pytest.raises(ValueError, match="reports ABI"):
        require_forecast_publisher(script)
    script.write_text("#!/bin/sh\necho gpuwm-cm1-forecast-publish/v3\n")
    assert require_forecast_publisher(script) == script.resolve()
    plain = tmp_path / "not-executable"
    plain.write_text("#!/bin/sh\n")
    with pytest.raises(ValueError, match="not executable"):
        require_forecast_publisher(plain)
    with pytest.raises(FileNotFoundError):
        require_forecast_publisher(tmp_path / "absent")


def test_publication_hands_the_exact_field_table_to_the_forecast_publisher(tmp_path):
    script = tmp_path / "fake-publish"
    script.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "--abi" ]; then echo gpuwm-cm1-forecast-publish/v3; exit 0; fi\n'
        'printf "%s\\n" "$1 $3 $6 $7 $8" > "$2"\n'
        'echo "abi=gpuwm-cm1-forecast-publish/v3"\n'
    )
    script.chmod(0o755)
    runtime, _ = _fake_runtime(_config())
    receipt = publish_cartesian_history(runtime, tmp_path / "frame", publisher=script)
    envelope = receipt["forecast_envelope"]
    assert envelope["published"] is True
    written = (tmp_path / "frame" / "history.cm1fhs").read_text().split()
    assert written[0] == "history"
    assert written[1] == "gm1.cartesian-history/v1"
    assert written[2:] == ["2", "1.0", "0.5"]
    assert "abi=gpuwm-cm1-forecast-publish/v3" in envelope["rust_output"]


def test_a_missing_runtime_carrier_refuses_by_name_and_frees_the_staging_budget(tmp_path):
    config = _config()
    runtime, _ = _fake_runtime(config)
    del runtime.dynamics.metrics["zf"]
    before = runtime.registry.registered_allocation_bytes
    with pytest.raises(ValueError, match="metrics\\['zf'\\]"):
        publish_cartesian_history(runtime, tmp_path / "frame")
    assert runtime.registry.registered_allocation_bytes == before
    assert STAGING_BUFFER_NAME not in runtime.registry.buffers


def _benchmark_runtime(config, *, sleep_s=0.0):
    """A CudaCartesianRuntime shell: only the receipt inputs are emulated."""

    runtime, _ = _fake_runtime(config, completed_steps=0)
    shell = object.__new__(CudaCartesianRuntime)
    for name, value in vars(runtime).items():
        setattr(shell, name, value)
    shell.clock = RuntimeClock(config)
    shell.finish = SimpleNamespace(completed_physics_step=0, completed_solve3_step=0, step_receipts={})
    shell.dynamics.last_receipts = {}
    shell.closure = SimpleNamespace(last_receipts={})
    shell.source_identity = {"kind": "test-shell"}
    shell.warm_component = {"levels": config.grid.nz}
    shell.component_scratch_reserve = {"levels": config.grid.nz}
    shell.phase_wall_seconds = {}
    shell.memory_samples = [{"device_used_bytes": 1}]
    shell.steps = 0
    # The receipt now names the &param9 selection this run resolved; the shell
    # emulates the record, not the kernels.
    shell._reflectivity_due = False
    shell.output = SimpleNamespace(
        reflectivity_wanted=False,
        record=lambda: {"selection": config.output.record(), "unvalidated": []},
        materialize=lambda: setattr(shell, "output_fields", shell.output_fields),
    )

    backend = _NoOpBackend()

    def step():
        time.sleep(sleep_s)
        shell.steps += 1
        shell.clock.step(backend)
        shell.finish.completed_physics_step = shell.clock.completed_steps
        shell.finish.completed_solve3_step = shell.clock.completed_steps

    shell.step = step
    shell.synchronize = lambda: None
    # misclibs.F:2714 is measured in every field scan now and the harness acts
    # on it, so the shell answers with a state the run may continue from.
    shell.field_checks = lambda: {
        "th3d": {"finite": True},
        "advective_courant": {
            "number": 0.25,
            "stop": CM1_COURANT_STOP,
            "target": CM1_COURANT_TARGET,
            "above_adaptive_target": False,
            "stop_reached": False,
            "stop_reason": None,
            "authority": CM1_COURANT_STOP_AUTHORITY,
        },
    }
    return shell


def test_benchmark_publishes_frames_only_on_request_and_outside_the_timed_section(monkeypatch, tmp_path):
    config = _config()
    shell = _benchmark_runtime(config)
    plain = shell.benchmark(warmup_steps=1, measured_steps=2)
    assert plain["history_publication"] is None
    assert plain["history_publication_in_timed_section"] is False
    assert "history staging/publication" in plain["timing_scope"]

    published_at = []

    def slow_publish(runtime, directory, **keywords):
        published_at.append(time.perf_counter())
        time.sleep(0.25)
        return cartesian_history.publish_cartesian_history(runtime, directory, **keywords)

    monkeypatch.setattr(runtime_module, "publish_cartesian_history", slow_publish)
    shell = _benchmark_runtime(config, sleep_s=0.002)
    directory = tmp_path / "frames"
    started = time.perf_counter()
    receipt = shell.benchmark(warmup_steps=0, measured_steps=2, history_directory=directory)
    assert time.perf_counter() - started > 0.25
    assert shell.steps == 2 and len(published_at) == 1
    published = receipt["history_publication"]
    assert published["step"] == 2 and published["model_time_s"] == 1.0
    assert (directory / ARENA_FILENAME).stat().st_size == published["arena"]["nbytes"]
    # The timed section holds only the stepping loop, never the publication.
    assert receipt["wall_seconds_in_timed_section"] < 0.1
    assert receipt["wall_seconds_in_timed_section"] >= sum(receipt["individual_step_wall_seconds"])


def test_the_timing_harness_publishes_its_frame_before_cm1s_courant_stop(tmp_path):
    """``cm1.F:3517-3520`` then ``cm1.F:4385-4392``, in the harness too.

    The stop is CM1's whatever entry point is driving; what must not happen is
    the frame being discarded because the stop was applied first.
    """

    from cm1core.cuda_cartesian_runtime import (
        INTEGRATION_COURANT_STOP_STATUS,
        CartesianCourantStop,
    )

    shell = _benchmark_runtime(_config(), sleep_s=0.0)
    record = shell.field_checks()
    record["advective_courant"].update({
        "number": 1.6,
        "above_adaptive_target": True,
        "stop_reached": True,
        "stop_reason": (
            "the advective Courant number 1.600000 has reached CM1's own stop "
            "value 1.5 (misclibs.F:2714)"
        ),
    })
    shell.field_checks = lambda: record

    directory = tmp_path / "frames"
    with pytest.raises(CartesianCourantStop) as caught:
        shell.benchmark(
            warmup_steps=0, measured_steps=2, history_directory=directory
        )
    # The frame is on disk, and the receipt that names the step came with it.
    assert (directory / ARENA_FILENAME).exists()
    receipt = caught.value.receipt
    assert receipt["status"] == INTEGRATION_COURANT_STOP_STATUS
    assert receipt["courant_stop"]["step"] == 2
    assert receipt["courant_stop"]["frames_published"] == 1
    assert receipt["courant_stop"]["advective_courant"] == 1.6
    assert "misclibs.F:2714" in receipt["courant_stop"]["reason"]
    assert "no part of this record is a qualified run" in receipt["claim_limit"]


def test_benchmark_refuses_an_unpublishable_frame_before_taking_any_step(tmp_path):
    shell = _benchmark_runtime(_config(1400, 1400, 140), sleep_s=0.0)
    with pytest.raises(ValueError, match="bytes"):
        shell.benchmark(warmup_steps=1, measured_steps=1, history_directory=tmp_path / "frames")
    assert shell.steps == 0
    existing = tmp_path / "existing"
    existing.mkdir()
    shell = _benchmark_runtime(_config())
    with pytest.raises(ValueError, match="already exists"):
        shell.benchmark(warmup_steps=1, measured_steps=1, history_directory=existing)
    assert shell.steps == 0
    with pytest.raises(ValueError, match="publishes nothing"):
        shell.benchmark(warmup_steps=1, measured_steps=1, history_publisher="/bin/true")
    assert shell.steps == 0


def test_envelope_suffix_is_what_the_installed_render_door_accepts() -> None:
    """The producer must not need --format to reach its own consumer.

    doors.py maps ".cm1fhs" and ".cm1fcp" to the envelope format and refuses
    anything else.  A file named after the CM1FHS1 magic is refused by the
    shipped renderer, which was caught only by rendering a real published
    frame; this pins the two ends together.
    """

    from cm1core.cartesian_history import ENVELOPE_FILENAME

    doors = (Path(__file__).resolve().parents[1] / "src/cm1core/doors.py").read_text(
        encoding="utf-8"
    )
    suffix = Path(ENVELOPE_FILENAME).suffix.lower()
    assert f'"{suffix}": "cm1fhs"' in doors, (
        f"the published envelope suffix {suffix!r} is not one the render door maps"
    )


def test_a_frame_without_a_publisher_does_not_claim_a_rust_finite_check() -> None:
    """No publisher means no Rust ran, so nothing was finite-checked.

    The claim limit was written unconditionally, so an arena-only frame
    carried a receipt asserting a check that never executed.
    """

    from cm1core.cartesian_history import (
        CARTESIAN_HISTORY_ARENA_ONLY_CLAIM_LIMIT,
        CARTESIAN_HISTORY_CLAIM_LIMIT,
    )

    assert "not finite-checked" in CARTESIAN_HISTORY_ARENA_ONLY_CLAIM_LIMIT
    assert "no Rust ran" in CARTESIAN_HISTORY_ARENA_ONLY_CLAIM_LIMIT
    assert "finite-checked by Rust" in CARTESIAN_HISTORY_CLAIM_LIMIT
    assert CARTESIAN_HISTORY_ARENA_ONLY_CLAIM_LIMIT != CARTESIAN_HISTORY_CLAIM_LIMIT


def test_staged_carriers_are_gated_on_their_own_dtype_and_extent() -> None:
    """The arena gate runs on the staged view, not the source.

    u3d, v3d, w3d and zf are copied into a pre-shaped float32 slice, so a
    wrong-dtype source would be silently cast and a wrong-extent source
    silently broadcast, and the frame would then pass every downstream check.
    The source itself has to be checked.
    """

    from cm1core.cartesian_history import _stage_carriers

    source = (
        Path(__file__).resolve().parents[1] / "src/cm1core/cartesian_history.py"
    ).read_text(encoding="utf-8")
    body = source[source.index("def _stage_carriers("): source.index("def ", source.index("def _stage_carriers(") + 10)]
    assert "would silently cast" in body or "silently cast" in body or "cast it" in body
    assert 'value.dtype != cupy.dtype("float32")' in body
    assert "tuple(value.shape) != tuple(expected)" in body
    # Both refusals name the concrete wrong result they prevent.
    assert "publish a frame the source never held" in body
    assert "publish a frame of the wrong grid" in body


def test_the_published_frame_carries_the_translating_frame_it_was_taken_in(tmp_path):
    """CM1 base.F:2663-2669 and cm1.F:3512-3513, in the frame's own metadata.

    Every wind carrier in the arena is frame-relative and every coordinate in
    the field table is a frame coordinate.  ``umove``/``vmove`` map the winds
    back (init3d.F:32-33); ``domainlocx``/``domainlocy`` say where the domain
    itself is, in ground metres, at this frame's model time.  A reader without
    them maps frame positions straight onto the ground and is wrong by
    ``umove`` times the model time at every frame.
    """

    still = _config()
    moving = RuntimeConfig(still.grid, u_move_m_s=12.5, v_move_m_s=3.0)

    fixed_receipt = publish_cartesian_history(
        _fake_runtime(still)[0], tmp_path / "fixed"
    )
    fixed_frame = fixed_receipt["moving_frame"]
    assert fixed_frame["imove"] == 0
    assert fixed_frame["umove"] == 0.0
    assert fixed_frame["vmove"] == 0.0
    # cm1.F:348-349.  A fixed frame never leaves CM1's initial value.
    assert fixed_frame["domainlocx"] == 0.0
    assert fixed_frame["domainlocy"] == 0.0

    receipt = publish_cartesian_history(
        _fake_runtime(moving)[0], tmp_path / "moving"
    )
    frame = receipt["moving_frame"]
    assert frame["imove"] == 1
    assert frame["umove"] == 12.5
    assert frame["vmove"] == 3.0
    assert frame["authority"] == DOMAIN_LOCATION_AUTHORITY
    assert frame["cadence"] == "post-large-step-after-model-time"
    # model_time_s is 2 completed half-second steps, and the integral CM1
    # accumulates over them is exactly umove*t.
    assert receipt["model_time_s"] == 1.0
    assert frame["completed_steps"] == 2
    assert frame["domainlocx"] == 12.5
    assert frame["domainlocy"] == 3.0
    # CM1's own units, so a consumer that knows CM1 needs no translation:
    # writeout.F:9634-9648 and domaindiag.F:464-483 for the locations,
    # writeout_nc.F:283-289 for the speeds.
    assert frame["units"] == {
        "umove": "m/s", "vmove": "m/s", "domainlocx": "m", "domainlocy": "m",
    }
    assert frame["descriptions"] == {
        "domainlocx": "x location of (center of) domain",
        "domainlocy": "y location of (center of) domain",
    }
    assert frame["envelope_scalars"] == ["domainlocx", "domainlocy", "umove", "vmove"]

    # The config block the envelope's config_sha256 is taken over carries the
    # frame too, so an unlabelled frame cannot share a digest with a labelled
    # one.
    assert receipt["config"]["u_move_m_s"] == 12.5
    assert receipt["config_sha256"] != fixed_receipt["config_sha256"]
    stored = json.loads((tmp_path / "moving" / RECEIPT_FILENAME).read_text())
    assert stored["moving_frame"] == frame


def test_the_scalar_table_carries_the_cm1_pair_as_exact_binary64_bits(tmp_path):
    """cm1.F:266 declares domainlocx double precision; it travels as bits.

    A decimal round trip through the command line would hand Rust a number
    the runtime never accumulated, and a frame published at 0.1 m/s would be
    placed a little further along the ground than the carriers actually are.
    """

    config = RuntimeConfig(_config().grid, u_move_m_s=0.25, v_move_m_s=-0.125)
    runtime, _ = _fake_runtime(config, completed_steps=7)
    publish_cartesian_history(runtime, tmp_path / "frame")
    rows = [
        line.split("\t")
        for line in (tmp_path / "frame" / SCALAR_TABLE_FILENAME)
        .read_text(encoding="ascii").splitlines()
    ]
    assert [row[0] for row in rows] == ["domainlocx", "domainlocy", "umove", "vmove"]
    assert {row[1] for row in rows} == {"f64-bits"}
    decoded = {
        row[0]: struct.unpack("<d", struct.pack("<Q", int(row[2], 16)))[0] for row in rows
    }
    assert decoded == {
        "domainlocx": 7 * 0.5 * 0.25,
        "domainlocy": 7 * 0.5 * -0.125,
        "umove": 0.25,
        "vmove": -0.125,
    }
    assert all(len(row[2]) == 16 and row[2] == row[2].lower() for row in rows)
    receipt = json.loads((tmp_path / "frame" / RECEIPT_FILENAME).read_text())
    assert receipt["scalar_table_sha256"] == hashlib.sha256(
        (tmp_path / "frame" / SCALAR_TABLE_FILENAME).read_bytes()
    ).hexdigest()


def test_a_published_frame_resumes_the_integral_rather_than_restarting_it(tmp_path):
    """restart_write.F:399-400 then restart_read.F:419-420.

    CM1 reads the persisted pair back into an ``intent(inout)`` pair, so a
    resumed run continues the integral.  A run that stopped at 7,200 steps and
    resumed must land exactly where an uninterrupted 14,400-step run lands; an
    accumulator that reset would be exactly half the distance out.
    """

    config = RuntimeConfig(_config().grid, u_move_m_s=12.5, v_move_m_s=3.0)
    backend = _NoOpBackend()

    whole = RuntimeClock(config)
    for _ in range(14_400):
        whole.step(backend)

    runtime, _ = _fake_runtime(config, completed_steps=7_200)
    receipt = publish_cartesian_history(runtime, tmp_path / "frame")
    resumed = resume_clock_from_publication(config, receipt)
    assert resumed.completed_steps == 7_200
    assert resumed.domainlocx_m == runtime.clock.domainlocx_m
    for _ in range(7_200):
        resumed.step(backend)
    assert resumed.domainlocx_m == whole.domainlocx_m
    assert resumed.domainlocy_m == whole.domainlocy_m
    assert resumed.completed_steps == whole.completed_steps
    # The defect this closes: a reset accumulator is half the distance out.
    assert resumed.domainlocx_m != whole.domainlocx_m / 2.0

    # The same receipt read back off disk resumes identically.
    from_disk = resume_clock_from_publication(config, tmp_path / "frame" / RECEIPT_FILENAME)
    assert from_disk.domainlocx_m == runtime.clock.domainlocx_m


def test_resuming_under_another_frame_or_without_the_pair_is_refused(tmp_path):
    """One constant frame velocity per integral; a mismatch is not continued."""

    config = RuntimeConfig(_config().grid, u_move_m_s=12.5, v_move_m_s=3.0)
    runtime, _ = _fake_runtime(config, completed_steps=64)
    receipt = publish_cartesian_history(runtime, tmp_path / "frame")

    other = RuntimeConfig(config.grid, u_move_m_s=3.0, v_move_m_s=3.0)
    with pytest.raises(ValueError, match="ground the carriers never covered"):
        resume_clock_from_publication(other, receipt)

    stripped = dict(receipt)
    stripped["moving_frame"] = {
        key: value for key, value in receipt["moving_frame"].items()
        if key != "domainlocx"
    }
    with pytest.raises(ValueError, match="carries no domainlocx"):
        resume_clock_from_publication(config, stripped)

    without = dict(receipt)
    del without["moving_frame"]
    with pytest.raises(ValueError, match="predates the cm1.F:3512-3513"):
        resume_clock_from_publication(config, without)

    foreign = dict(receipt, schema="gm1.something-else/v1")
    with pytest.raises(ValueError, match="is the only artifact"):
        resume_clock_from_publication(config, foreign)


# ------------------- the real Rust publisher, end to end on this machine


@pytest.fixture(scope="session")
def forecast_publisher(tmp_path_factory):
    """The actual cm1-forecast-publish binary, built from this tree."""

    target = tmp_path_factory.mktemp("gm1-forecast-publish")
    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [
            "cargo", "build", "--release", "--bin", "cm1_forecast_publish",
            "--manifest-path", str(root / "rust/cm1-state/Cargo.toml"),
            "--target-dir", str(target),
        ],
        check=True, capture_output=True,
    )
    return (target / "release" / "cm1_forecast_publish").resolve(strict=True)


def test_the_real_publisher_writes_the_pair_into_the_cm1fhs1_envelope(
    forecast_publisher, tmp_path
):
    """cm1.F:3512-3513 all the way into the artifact's own bytes.

    The envelope stores an f64 scheduler scalar as sixteen hexadecimal
    characters (forecast.rs f64_bits), so the accumulated domainlocx is
    findable in the published file verbatim -- not a value rounded on the way
    through a command line.
    """

    config = RuntimeConfig(_config().grid, u_move_m_s=12.5, v_move_m_s=3.0)
    runtime, _ = _fake_runtime(config, completed_steps=120)
    receipt = publish_cartesian_history(
        runtime, tmp_path / "frame", publisher=forecast_publisher
    )
    envelope = receipt["forecast_envelope"]
    assert envelope["published"] is True
    assert envelope["abi"] == "gpuwm-cm1-forecast-publish/v3"

    payload = Path(envelope["path"]).read_bytes()
    assert payload.startswith(b"CM1FHS1\0")
    for name, value in (
        ("domainlocx", 750.0), ("domainlocy", 180.0), ("umove", 12.5), ("vmove", 3.0),
    ):
        bits = f"{struct.unpack('<Q', struct.pack('<d', value))[0]:016x}"
        assert name.encode("ascii") in payload, name
        assert bits.encode("ascii") in payload, name
    assert receipt["moving_frame"]["domainlocx"] == 750.0


def test_the_real_publisher_refuses_a_domain_location_that_is_not_the_integral(
    forecast_publisher, tmp_path
):
    """The Rust boundary re-derives cm1.F:3512-3513 before writing a byte.

    A host accumulator that drifted, reset or was wired to the wrong clock
    would otherwise stamp a wrong ground position onto an artifact that
    outlives the run.
    """

    directory = tmp_path / "hand-written"
    directory.mkdir()
    arena = directory / "carriers.f32"
    arena.write_bytes(struct.pack("<4f", 1.0, 2.0, 3.0, 4.0))
    table = directory / "fields.tsv"
    table.write_text(
        "th3d\tcurrent potential-temperature perturbation\tK\tscalar\t2,2\t1,2,1,2\t"
        f"{arena}\t0\n",
        encoding="ascii",
    )

    def publish(values, output):
        scalars = directory / f"{output}.tsv"
        scalars.write_text(
            "".join(
                f"{name}\tf64-bits\t{struct.unpack('<Q', struct.pack('<d', value))[0]:016x}\n"
                for name, value in values
            ),
            encoding="ascii",
        )
        return subprocess.run(
            [
                str(forecast_publisher), "history", str(directory / output),
                "gm1.cartesian-history/v1", "11" * 32, "22" * 32,
                "120", "60.0", "0.5", str(table), str(scalars),
            ],
            check=False, capture_output=True, text=True,
        )

    exact = [("domainlocx", 750.0), ("domainlocy", 180.0), ("umove", 12.5), ("vmove", 3.0)]
    assert publish(exact, "good.cm1fhs").returncode == 0

    drifted = [("domainlocx", 750.000000001)] + exact[1:]
    failed = publish(drifted, "drifted.cm1fhs")
    assert failed.returncode != 0
    assert "domainlocx" in failed.stderr
    assert "wrong patch of ground" in failed.stderr
    assert not (directory / "drifted.cm1fhs").exists()

    partial = publish(exact[:1], "partial.cm1fhs")
    assert partial.returncode != 0
    assert "1 of the 4" in partial.stderr

    # A frame-speed that is not a CM1 `real` cannot describe the carriers.
    widened = [("umove", 12.500000000000002)] + [
        row for row in exact if row[0] != "umove"
    ]
    refused = publish(widened, "widened.cm1fhs")
    assert refused.returncode != 0
    assert "not a CM1 `real`" in refused.stderr
def test_the_reduced_profile_publishes_the_columns_its_formulas_read(tmp_path):
    """Opt-in, exact, and named: prs and zf leave, their inputs arrive.

    CM1 r21.1 solve2.F:1588 rebuilds prs from pi0 and pp3d, so the frame keeps
    pp3d whole and carries pi0 as the column this runtime broadcasts it from.
    zf is that same broadcast, proved plane by plane before upload, so the
    column is the field and nothing is lost.
    """

    config = _config()
    runtime, _ = _fake_runtime(config)
    profile = cartesian_history_profile(
        config, carrier_profile=RECONSTRUCTIBLE_OMITTED_CARRIER_PROFILE
    )
    receipt = publish_cartesian_history(
        runtime, tmp_path / "frame",
        carrier_profile=RECONSTRUCTIBLE_OMITTED_CARRIER_PROFILE,
    )
    assert receipt["profile"]["carrier_profile"] == RECONSTRUCTIBLE_OMITTED_CARRIER_PROFILE
    assert receipt["profile"]["profile_id"] == CARTESIAN_HISTORY_REDUCED_PROFILE_ID
    assert "omits ['prs', 'zf']" in receipt["claim_limit"]
    assert STAGING_BUFFER_NAME not in runtime.registry.buffers

    arena = (tmp_path / "frame" / ARENA_FILENAME).read_bytes()
    assert len(arena) == profile.payload_nbytes
    offsets = receipt["arena"]["carrier_offset_bytes"]
    assert set(offsets) == {carrier.name for carrier in profile.carriers}

    zf_source = runtime.dynamics.metrics["zf"]
    start = offsets["metrics.zf_profile"]
    column = np.frombuffer(arena[start:start + 4 * (config.grid.nz + 1)], dtype="<f4")
    assert np.array_equal(column, np.asarray(zf_source[:, 0, 0]))
    # The column is the field: every horizontal plane of the source is it.
    assert np.array_equal(
        np.broadcast_to(column[:, None, None], config.grid.w_shape), np.asarray(zf_source)
    )
    start = offsets["base.pi0"]
    pi0 = np.frombuffer(arena[start:start + 4 * config.grid.nz], dtype="<f4")
    assert np.array_equal(pi0, runtime.dynamics.base_profiles["pi0"])

    rows = (tmp_path / "frame" / FIELD_TABLE_FILENAME).read_text().splitlines()
    # The invariant is that the field table and the carrier list agree;
    # the count itself follows the &param9 selection and is not fixed.
    assert len(rows) == len(profile.carriers)
    names = [row.split("\t")[0] for row in rows]
    assert "prs" not in names and "zf" not in names
    assert names == sorted(names)


def test_a_broadcast_column_without_the_runtime_proof_refuses_by_name(tmp_path):
    config = _config()
    runtime, _ = _fake_runtime(config)
    runtime.dynamics.metrics.proofs["zf"] = {"kind": "separate-source-array"}
    before = runtime.registry.registered_allocation_bytes
    with pytest.raises(ValueError, match="misplace every height"):
        publish_cartesian_history(
            runtime, tmp_path / "frame",
            carrier_profile=RECONSTRUCTIBLE_OMITTED_CARRIER_PROFILE,
        )
    assert runtime.registry.registered_allocation_bytes == before
    assert STAGING_BUFFER_NAME not in runtime.registry.buffers
    assert not (tmp_path / "frame").exists()


def test_a_profile_and_a_requested_carrier_inventory_that_disagree_refuse(tmp_path):
    config = _config()
    runtime, _ = _fake_runtime(config)
    reduced = cartesian_history_profile(
        config, carrier_profile=RECONSTRUCTIBLE_OMITTED_CARRIER_PROFILE
    )
    with pytest.raises(ValueError, match="carrier inventory"):
        publish_cartesian_history(runtime, tmp_path / "frame", profile=reduced)
    assert not (tmp_path / "frame").exists()


def test_the_python_payload_ceiling_mirrors_the_rust_constant_exactly():
    """Two copies of one number drift; this is the seam that catches it.

    rust/cm1-state/src/forecast.rs owns the ceiling and the admission cost.
    The Python door prices every frame against its mirror before the device
    opens, so a raised Rust ceiling with a stale Python mirror would refuse a
    grid Rust admits, and a lowered one would let a frame through to fail in
    the publisher after the whole device-to-host transfer.
    """

    source = (
        Path(__file__).resolve().parents[1] / "rust/cm1-state/src/forecast.rs"
    ).read_text(encoding="utf-8")
    for declaration, expected in (
        ("pub const MAX_FORECAST_PAYLOAD_BYTES: usize = ", MAX_FORECAST_PAYLOAD_BYTES),
        (
            "pub const FORECAST_ADMISSION_HOST_BYTES_PER_PAYLOAD_BYTE: usize = ",
            FORECAST_ADMISSION_HOST_BYTES_PER_PAYLOAD_BYTE,
        ),
    ):
        assert source.count(declaration) == 1, declaration
        spelling = source.split(declaration, 1)[1].split(";", 1)[0]
        assert eval(spelling.replace("_", "")) == expected, spelling  # noqa: S307


def test_the_reconstruction_formula_is_the_one_this_runtime_evaluates():
    """A recorded formula that drifts from the kernel is a false receipt.

    The reduced profile tells a reader to rebuild prs as CM1 r21.1
    solve2.F:1588 ``p00*((pi0+pp3d)**(cp/rd))``.  Both GM1 pressure kernels
    evaluate exactly that: p00 is the float32 constant 0x47c35000 (100000) and
    the exponent is applied to pi0+pp3d.  This binds the claim to the source.
    """

    root = Path(__file__).resolve().parents[1] / "src/cm1core"
    for module in ("cuda_cartesian_startup.py",
                   "cuda_gm1_tilestream_rk3_eqtset2_remove_eos.py"):
        source = (root / module).read_text(encoding="utf-8")
        assert "pow_cpdrd_positive_normal" in source, module
        assert "0x47c35000u" in source, module
    startup = (root / "cuda_cartesian_startup.py").read_text(encoding="utf-8")
    assert "__fadd_rn(pi0[index], pp3d[index])" in startup
    formula = RECONSTRUCTIBLE_CARRIERS["prs"]
    assert "solve2.F:1588" in formula
    assert "p00*((pi0+pp3d)**(cp/rd))" in formula
    # The exponent's own authority line, not the range around it: constants.F
    # defines rovcp/rddcp/rddcv at :132-134 and cpdrd, the exponent this
    # formula uses, at :137.
    assert "constants.F:137" in formula
    assert "constants.F:17" in formula and "constants.F:113-114" in formula
    # The claim does not overreach: bit identity needs the pinned pow too.
    assert "bit for bit" in formula and "pow" in formula


def _rust_string_constants(source: str) -> dict[str, str]:
    """Every ``const NAME: &str = "value";`` in one Rust source."""

    import re

    return {
        match.group(1): match.group(2)
        for match in re.finditer(
            r'^(?:pub )?const ([A-Z0-9_]+): &str =\s*"((?:[^"\\]|\\.)*)";',
            source,
            re.MULTILINE,
        )
    }


def test_the_rust_reader_rebuilds_exactly_what_this_profile_omits():
    """The write half and the read half must name the same carriers.

    A reduced frame is only readable because rust/cm1-state/src/reduced_history.rs
    knows which carriers were dropped and which columns replaced them.  If this
    module renamed a column, or dropped a third carrier, that reader would
    rebuild nothing and the frame would again be a format nothing can open.
    This is the seam that catches it.
    """

    source = (
        Path(__file__).resolve().parents[1] / "rust/cm1-state/src/reduced_history.rs"
    ).read_text(encoding="utf-8")
    constants = _rust_string_constants(source)
    assert constants["REDUCED_HISTORY_PROFILE_ID"] == CARTESIAN_HISTORY_REDUCED_PROFILE_ID
    assert constants["COMPLETE_HISTORY_PROFILE_ID"] == CARTESIAN_HISTORY_PROFILE_ID

    config = _config()
    reduced = cartesian_history_profile(
        config, carrier_profile=RECONSTRUCTIBLE_OMITTED_CARRIER_PROFILE
    )
    names = {carrier.name for carrier in reduced.carriers}
    for key in ("PI0_COLUMN_CARRIER", "ZF_PROFILE_COLUMN_CARRIER", "PP3D_CARRIER"):
        assert constants[key] in names, key

    # The two names the reader rebuilds are exactly the two this profile drops.
    rebuilt = source.split("pub const RECONSTRUCTED_CARRIERS: [&str; 2] = [", 1)[1]
    rebuilt = [word.strip().strip('"') for word in rebuilt.split("]", 1)[0].split(",") if word.strip()]
    assert rebuilt == sorted(RECONSTRUCTIBLE_CARRIERS)
    assert not (names & set(rebuilt))

    # And it rebuilds them under the complete profile's own declarations, so a
    # renderer sees one field whichever profile produced the frame.
    complete = {
        carrier.name: carrier
        for carrier in cartesian_history_profile(config).carriers
    }
    for name, prefix in (("prs", "PRS"), ("zf", "ZF")):
        assert constants[f"{prefix}_MEANING"] == complete[name].meaning
        assert constants[f"{prefix}_UNITS"] == complete[name].units
        assert constants[f"{prefix}_STAGGER"] == complete[name].stagger


def test_the_reduced_field_table_carries_every_input_the_reader_reads(tmp_path):
    """The published field table is what the Rust boundary parses.

    The reader reads pp3d at full extent plus two columns.  This drives the
    real writer and checks the rows it emitted, so a shape or a name that the
    reconstruction could not use is caught here rather than in a renderer.
    """

    config = _config()
    runtime, _ = _fake_runtime(config)
    publish_cartesian_history(
        runtime, tmp_path / "frame",
        carrier_profile=RECONSTRUCTIBLE_OMITTED_CARRIER_PROFILE,
    )
    rows = {
        row.split("\t")[0]: row.split("\t")
        for row in (tmp_path / "frame" / FIELD_TABLE_FILENAME).read_text().splitlines()
    }
    g = config.grid
    assert rows["pp3d"][4] == ",".join(str(extent) for extent in g.mass_shape)
    assert rows["base.pi0"][4] == str(g.nz)
    assert rows["metrics.zf_profile"][4] == str(g.nz + 1)
    assert "prs" not in rows and "zf" not in rows


def test_a_carrier_inventory_without_a_history_directory_refuses_before_any_step():
    """The adjacent publisher guard's rule, applied to the inventory too."""

    config = _config()
    shell = _benchmark_runtime(config)
    with pytest.raises(ValueError, match="publishes nothing"):
        shell.benchmark(
            warmup_steps=0, measured_steps=1,
            history_carrier_profile=RECONSTRUCTIBLE_OMITTED_CARRIER_PROFILE,
        )
    assert shell.clock.completed_steps == 0
    # The default inventory with no directory keeps working exactly as before.
    benchmark = shell.benchmark(warmup_steps=0, measured_steps=1)
    assert benchmark["history_publication"] is None
    assert shell.clock.completed_steps == 1


# The complete-profile frame this module published at the branch point
# (49319e5, before the payload ceiling was raised and --history-carriers
# existed), measured for the 9x11x8 geometry `_config()` builds.  Reproduced by
# running the branch-point module, exported with `git archive`, against this
# file's own fake runtime.
# The default frame grew when int/output made CM1's &param9 selection part of
# the lane: run/namelist.input ships output_rain, the seven solve3.F swaths,
# output_th, output_zh and output_basestate on, and this port now publishes
# them.  The first twelve offsets below are byte-for-byte what they were --
# the &param9 carriers are appended AFTER the carriers a frame has always
# published, so no reader of the original twelve has to move.
DEFAULT_FRAME_ARENA_NBYTES = 65_016
DEFAULT_FRAME_ARENA_SHA256 = (
    "f1d06855ba83359d81ed1f84365ad490447bd8c42504210be59bff37ee6ff605"
)
DEFAULT_FRAME_FIELD_TABLE_SHA256 = (
    "31ad2ba6e2bfc70ba0134e79068686f15c6efaa49791a0a8d236128d779f0a42"
)
DEFAULT_FRAME_CARRIER_OFFSETS = {
    # Unchanged, and that is the point of pinning them:
    "closure.tkea": 0, "pp3d": 3564, "prs": 6732, "q3d": 9900, "rf": 28908,
    "rho": 32472, "terrain_msl": 35640, "th3d": 36036, "u3d": 39204,
    "v3d": 42724, "w3d": 46180, "zf": 49744,
    # ...and CM1's &param9 products, appended after them:
    "rain": 53308, "prate": 53704, "snownc": 54100, "graupelnc": 54496,
    "sr": 54892, "sws": 55288, "svs": 55684, "sps": 56080, "srs": 56476,
    "sgs": 56872, "sus": 57268, "shs": 57664, "dbz": 58060, "cref": 61228,
    "th": 61624, "zhval": 64792, "pi0": 64824, "th0": 64856, "prs0": 64888,
    "qv0": 64920, "u0": 64952, "v0": 64984,
}


def test_the_default_frame_is_byte_identical_to_the_pre_change_publication(tmp_path):
    """Raising the ceiling and adding a profile changed no default byte.

    Every number below was produced by the branch-point module, not by this
    one: the arena digest, its length, each carrier's offset, and the field
    table with only its absolute arena path column normalized.  A frame
    published with no ``carrier_profile`` argument must reproduce all of them,
    or the default configuration moved.
    """

    import hashlib

    config = _config()
    runtime, _ = _fake_runtime(config)
    receipt = publish_cartesian_history(runtime, tmp_path / "frame")

    arena = (tmp_path / "frame" / ARENA_FILENAME).read_bytes()
    assert len(arena) == DEFAULT_FRAME_ARENA_NBYTES
    assert hashlib.sha256(arena).hexdigest() == DEFAULT_FRAME_ARENA_SHA256
    assert receipt["arena"]["carrier_offset_bytes"] == DEFAULT_FRAME_CARRIER_OFFSETS
    assert receipt["profile"]["payload_nbytes"] == DEFAULT_FRAME_ARENA_NBYTES
    assert receipt["profile"]["profile_id"] == CARTESIAN_HISTORY_PROFILE_ID
    assert receipt["profile"]["carrier_profile"] == COMPLETE_CARRIER_PROFILE

    rows = [
        row.split("\t")
        for row in (tmp_path / "frame" / FIELD_TABLE_FILENAME).read_text().splitlines()
    ]
    for row in rows:
        row[6] = "<ARENA>"
    normalized = ("\n".join("\t".join(row) for row in rows) + "\n").encode("ascii")
    assert hashlib.sha256(normalized).hexdigest() == DEFAULT_FRAME_FIELD_TABLE_SHA256
