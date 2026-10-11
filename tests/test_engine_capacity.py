from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from cm1core import engine_pin
from cm1core.arwen_wsm6_component import (
    WSM6_COMPONENT_PERSISTENCE,
    WSM6_SURFACE_SCRATCH,
    component_metadata,
    component_metadata_sha256,
    create_arwen_wsm6_component,
)
from cm1core.cuda_gm1_tilestream_wsm6 import Gm1TiledWsm6SourceBinding
from cm1core.engine_capacity import (
    AUDITED_DIST_INFO_DIRNAME,
    CAPACITY_PATCHED_SURFACE,
    WSM6_256_CAPACITY_PROFILE,
    build_wsm6_capacity_overlay,
    capacity_level_tier,
    capacity_profile_sha256,
    capacity_surface,
    patch_capacity_source,
)
from cm1core.errors import CM1Refusal


ROOT = Path(__file__).resolve().parents[1]
#: The preserved audited estate for the version ``engine_pin`` admits.  It is
#: not distributed, so a checkout without it skips rather than fails:
#: these tests are about the overlay mechanism, and the mechanism cannot be
#: exercised without the real bytes it is defined against.
AUDITED = ROOT / f"tmp/arwen-estate-gpuwm-{engine_pin.GPUWM_MEASURED_VERSION}"

requires_audited_estate = pytest.mark.skipif(
    not AUDITED.is_dir(),
    reason=f"no preserved audited ArWen estate at {AUDITED}",
)


#: The audited surface plus the minimum import closure the overlay's own
#: verification subprocess reaches.  ``physics_vertical_contract`` imports
#: WDM6's ceiling from its constants module rather than transcribing it, so a
#: fixture that copies only the pinned surface cannot import the patched
#: contract at all.
BASE_COPY_EXTRA = (
    "gpuwm/__init__.py",
    # gpuwm 2.8.0's package __init__ imports it before anything else.
    "gpuwm/cupy_windows_warning.py",
    "gpuwm/core/__init__.py",
    "gpuwm/core/wdm6_constants.py",
)


@pytest.fixture
def base_copy(tmp_path: Path) -> Path:
    if not AUDITED.is_dir():
        pytest.skip(f"no preserved audited ArWen estate at {AUDITED}")
    source = tmp_path / "base"
    for name in (*engine_pin.AUDITED_WSM6_SURFACE, *BASE_COPY_EXTRA):
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((AUDITED / name).read_bytes())
    metadata = source / f"{AUDITED_DIST_INFO_DIRNAME}/METADATA"
    metadata.parent.mkdir()
    metadata.write_bytes(
        "Metadata-Version: 2.1\nName: gpuwm\n"
        f"Version: {engine_pin.GPUWM_MEASURED_VERSION}\n".encode("utf-8")
    )
    (source / "gpuwm/support.py").write_text("value = 'unchanged'\n", encoding="utf-8")
    stale = source / "gpuwm/__pycache__/support.cpython-313.pyc"
    stale.parent.mkdir()
    stale.write_bytes(b"stale bytecode must not enter the overlay")
    return source


def test_capacity_metadata_is_explicit_and_the_default_is_byte_unchanged() -> None:
    # The document grew a dense_ice block (CM1 ihail, param.F:3697-3706), the
    # transcribed q3d axis and the CM1-to-ArWen carrier map when the component
    # adapter was generalized over CM1's ptype, and then moved again when the
    # undeclared ihail moved from 0 to CM1's own shipped 1
    # (run/namelist.input:56, src/param.F:3697-3706).  That move changes the
    # five constants of src/mp_wsm6.F:88-99 and therefore the words of every
    # WSM6 run; it is deliberate and re-recorded here.  Earlier documents
    # hashed to 08c9eb9790f5fc453ee95886fad47e1bdf093a5774bc3b71096047dbb1697390
    # and a8cb435d5662d43075620fc48c16528a4a0baf85f1ca95ff61cfdc46ddd5e848.
    # c0375ad then carried int/micro's document (which hashed to
    # 0bb5610a98259475ce37706279f537f0e29cd7f494d55b65a9dc5a04a9081a17) onto
    # sota/physics' re-pinned 2.7.4 seam (8b6f232): only the "engine" block
    # moved -- measured_version/requirement 2.5.8 -> 2.7.4, audited_commit ->
    # audited_source_identity, and the nine re-audited surface digests that
    # engine_pin.AUDITED_WSM6_SURFACE records, asserted equal just below.
    # The 2.8.0 re-pin moved the "engine" block again and nothing else:
    # measured_version/requirement 2.7.4 -> 2.8.0, audited_source_identity ->
    # the 2.8.0 wheel, six re-pinned surface digests plus
    # gpuwm/kernel_compile_notice.py, and so audited_surface_sha256.  The
    # 2.7.4 document hashed to
    # c84914d44291994efb15c8e54f7aa636d6859a9ae684f8980d1d6c65f92292e2.
    assert component_metadata_sha256() == (
        "c3830541de52263e8d3f80faabad5d940b1efc8dccbe6799890987c7cad2418a"
    )
    default = component_metadata()
    assert default["execution"]["mp_physics"] == 6
    assert default["execution"]["wsm6_hail_opt"] == 1
    # One integer away is the branch every recorded WSM6 run before this move
    # executed.
    assert component_metadata(ihail=0)["execution"]["wsm6_hail_opt"] == 0
    assert default["carriers"]["species"] == ["qv", "qc", "qr", "qi", "qs", "qg"]
    assert default["persistence"]["component_owned"] == [
        "h_diabatic", "effc", "effi", "effs",
        "mp_rainnc", "mp_rainncv", "mp_snownc", "mp_snowncv",
        "mp_graupelnc", "mp_graupelncv", "mp_sr",
    ]
    assert default["engine"]["audited_surface"] == engine_pin.AUDITED_WSM6_SURFACE
    assert default["carriers"]["vertical_level_bounds"] == [2, 80]
    assert "capacity_profile" not in default
    extended = component_metadata(capacity_profile=WSM6_256_CAPACITY_PROFILE)
    assert extended["carriers"]["vertical_level_bounds"] == [2, 256]
    assert extended["capacity_profile_sha256"] == capacity_profile_sha256()
    assert extended["engine"]["audited_surface"] == capacity_surface(WSM6_256_CAPACITY_PROFILE)
    for name, digest in engine_pin.AUDITED_WSM6_SURFACE.items():
        assert (extended["engine"]["audited_surface"][name] != digest) == (
            name in CAPACITY_PATCHED_SURFACE
        )


@pytest.mark.parametrize("nz,tier", [(2, 64), (64, 64), (65, 80), (80, 80),
                                      (81, 128), (128, 128), (129, 256),
                                      (237, 256), (256, 256)])
def test_capacity_tier_boundaries(nz: int, tier: int) -> None:
    assert capacity_level_tier(nz) == tier


@pytest.mark.parametrize("nz", [True, 1, 257, 237.0])
def test_invalid_capacity_dimensions_refuse(nz: object) -> None:
    with pytest.raises(ValueError, match="integer 2 <= nz <= 256"):
        capacity_level_tier(nz)


@requires_audited_estate
def test_patch_preserves_every_coefficient_and_dispatch_function() -> None:
    name = "gpuwm/core/wsm6_constants.py"
    original = (AUDITED / name).read_bytes()
    changed = patch_capacity_source(name, original)
    before = ast.parse(original)
    after = ast.parse(changed)
    functions_before = {node.name: ast.dump(node) for node in before.body
                        if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    functions_after = {node.name: ast.dump(node) for node in after.body
                       if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    assert functions_before == functions_after
    with pytest.raises(ValueError, match="base SHA-256"):
        patch_capacity_source(name, original + b"\n")
    with pytest.raises(ValueError, match="no WSM6 capacity patch"):
        patch_capacity_source("gpuwm/core/kernels/wsm6.cu", b"unchanged kernel")


def test_overlay_build_is_deterministic_and_never_changes_the_source(
    base_copy: Path, tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    receipt1 = build_wsm6_capacity_overlay(base_copy, first)
    receipt2 = build_wsm6_capacity_overlay(base_copy, second)
    assert receipt1 == receipt2
    assert receipt1["changed_files"] == sorted(CAPACITY_PATCHED_SURFACE)
    assert not list(first.rglob("*.pyc"))
    assert (first / "gpuwm/support.py").read_bytes() == (base_copy / "gpuwm/support.py").read_bytes()
    for name, digest in engine_pin.AUDITED_WSM6_SURFACE.items():
        assert hashlib.sha256((base_copy / name).read_bytes()).hexdigest() == digest
    for name, digest in capacity_surface(WSM6_256_CAPACITY_PROFILE).items():
        assert hashlib.sha256((first / name).read_bytes()).hexdigest() == digest
    with pytest.raises(FileExistsError):
        build_wsm6_capacity_overlay(base_copy, first)
    with pytest.raises(ValueError, match="outside the audited source"):
        build_wsm6_capacity_overlay(base_copy, base_copy / "overlay")


def test_drift_refuses_before_an_overlay_directory_is_created(
    base_copy: Path, tmp_path: Path,
) -> None:
    path = base_copy / "gpuwm/core/microphysics.py"
    path.write_bytes(path.read_bytes() + b"\n")
    destination = tmp_path / "must-not-exist"
    with pytest.raises(ValueError, match="microphysics.py differs"):
        build_wsm6_capacity_overlay(base_copy, destination)
    assert not destination.exists()


def test_engine_cross_profile_and_changed_math_refuse(
    base_copy: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "overlay"
    build_wsm6_capacity_overlay(base_copy, target)
    monkeypatch.setattr(engine_pin, "installed_version",
                        lambda: engine_pin.GPUWM_MEASURED_VERSION)
    monkeypatch.setattr(engine_pin, "installed_root", lambda: base_copy)
    assert engine_pin.inspect().admitted
    assert not engine_pin.inspect(capacity_profile=WSM6_256_CAPACITY_PROFILE).admitted
    monkeypatch.setattr(engine_pin, "installed_root", lambda: target)
    default = engine_pin.inspect()
    assert not default.admitted
    assert default.moved == tuple(sorted(CAPACITY_PATCHED_SURFACE))
    extended = engine_pin.inspect(capacity_profile=WSM6_256_CAPACITY_PROFILE)
    assert extended.admitted
    assert extended.as_dict()["capacity_profile_sha256"] == capacity_profile_sha256()
    changed = target / "gpuwm/core/kernels/wsm6.cu"
    changed.write_bytes(changed.read_bytes() + b"\n")
    assert engine_pin.inspect(capacity_profile=WSM6_256_CAPACITY_PROFILE).moved == (
        "gpuwm/core/kernels/wsm6.cu",
    )
    with pytest.raises(ValueError, match="unknown WSM6 capacity_profile"):
        engine_pin.inspect(capacity_profile="invented")


def test_actual_dependency_tier_helper_and_contract_match_without_cuda(
    base_copy: Path, tmp_path: Path,
) -> None:
    target = tmp_path / "overlay"
    build_wsm6_capacity_overlay(base_copy, target)
    script = (
        "import sys,json; "
        "from importlib.metadata import version; "
        "from gpuwm.core.wsm6_constants import wsm6_level_tier,WSM6_VERTICAL_LEVEL_BOUNDS; "
        "from gpuwm.physics_vertical_contract import WSM6_VERTICAL_LEVEL_BOUNDS as public; "
        "assert public==WSM6_VERTICAL_LEVEL_BOUNDS==(2,256); "
        f"assert version('gpuwm')=='{engine_pin.GPUWM_MEASURED_VERSION}'; "
        "assert [wsm6_level_tier(n) for n in (64,80,81,128,129,237,256)]==[64,80,128,128,256,256,256]; "
        "assert 'cupy' not in sys.modules; print(json.dumps({'passed':True}))"
    )
    env = {**os.environ, "PYTHONPATH": str(target)}
    result = subprocess.run([sys.executable, "-c", script], env=env,
                            check=True, capture_output=True, text=True)
    assert json.loads(result.stdout) == {"passed": True}


def test_component_extended_levels_require_explicit_profile_before_runtime_import() -> None:
    kwargs = dict(component_id="wsm6-mp6", nx=3, ny=2, nz=237,
                  dx_m=75.0, dy_m=75.0, dz_m=75.0, dt_s=0.5)
    with pytest.raises(CM1Refusal, match="per-column storage"):
        create_arwen_wsm6_component(**kwargs)
    component = create_arwen_wsm6_component(**kwargs, capacity_profile=WSM6_256_CAPACITY_PROFILE)
    assert component.nz == 237
    assert component.state is None
    with pytest.raises(CM1Refusal, match="per-column storage"):
        create_arwen_wsm6_component(**{**kwargs, "nz": 257},
                                   capacity_profile=WSM6_256_CAPACITY_PROFILE)


def test_compact_source_binding_carries_the_capacity_profile() -> None:
    shape = (237, 2, 3)
    interface = np.broadcast_to(np.arange(238, dtype=np.float32)[:, None, None],
                               (238, 2, 3)).copy()
    kwargs = dict(
        q3d=np.zeros((6, *shape), dtype=np.float32),
        pi0=np.ones(shape, dtype=np.float32),
        th0=np.full(shape, 300.0, dtype=np.float32),
        pp3d=np.zeros(shape, dtype=np.float32),
        th3d=np.zeros(shape, dtype=np.float32),
        p=np.full(shape, 100000.0, dtype=np.float32),
        rho=np.ones(shape, dtype=np.float32), phb=interface,
        persistence={name: np.zeros((2, 3) if name in WSM6_SURFACE_SCRATCH else shape,
                                    dtype=np.float32)
                     for name in WSM6_COMPONENT_PERSISTENCE},
        source_identity_sha256="1" * 64, pdefq_receipt_sha256="2" * 64,
        completed_calls=0, allow_compact_validation=True,
    )
    with pytest.raises(ValueError, match="2 <= nk <= 80"):
        Gm1TiledWsm6SourceBinding.create(**kwargs)
    binding = Gm1TiledWsm6SourceBinding.create(**kwargs, capacity_profile=WSM6_256_CAPACITY_PROFILE)
    assert binding.nk == 237
    assert binding.capacity_profile == WSM6_256_CAPACITY_PROFILE


def test_pin_inspection_is_memoized_without_ceasing_to_check() -> None:
    """The tiled WSM6 launcher calls inspect() once per horizontal tile.

    Uncached that is 0.71 ms x 16 tiles per step, 2.7 minutes of rehashing
    across the 14,400 steps of a two-hour run. The result is memoized on a
    (path, size, mtime_ns) stamp of every audited file, so a file that changes
    under a running process is still re-read and re-hashed. Memoizing the
    answer must never become not asking the question.
    """

    from cm1core import engine_pin

    root = engine_pin.installed_root()
    if root is None:
        pytest.skip("no installed ArWen estate to inspect")

    first = engine_pin.inspect()
    assert engine_pin.inspect() is first, "repeat inspection should be served from cache"

    target = root / sorted(engine_pin.audited_surface(None))[0]
    original = target.read_bytes()
    try:
        target.write_bytes(original + b"\n")
        tampered = engine_pin.inspect()
        assert tampered is not first
        assert tampered.admitted != first.admitted or not first.admitted
    finally:
        target.write_bytes(original)

    restored = engine_pin.inspect()
    assert restored.admitted == first.admitted


EXAMPLE = ROOT / "examples" / "cartesian-uniform-grid.toml"


def _wsm6_nvrtc_source(files: dict[str, bytes], tier: int | None) -> str:
    """Reassemble exactly what ``cp.RawModule`` is handed for the WSM6 module.

    ``gpuwm.core.kernels.module_source`` is the ``CUDA_DEFINES`` preamble plus
    ``common.cuh`` plus the ``.cu`` text, and ``module_source_int_defines``
    inserts ``#define WSM6_KMAX <n>`` between them.  ``files`` carries the exact
    bytes a configured lane would have on its import path, so this function is
    how a config choice is followed all the way to the string NVRTC compiles.
    """
    namespace: dict[str, object] = {}
    exec(compile(files["gpuwm/core/constants.py"].decode("utf-8"),
                 "constants.py", "exec"), namespace)
    defines = namespace["CUDA_DEFINES"]
    lines = [f"#define {key} {float(value)!r}f" for key, value in defines.items()]  # type: ignore[union-attr]
    lines.append(files["gpuwm/core/kernels/common.cuh"].decode("utf-8"))
    preamble = "\n".join(lines) + "\n"
    prefix = "" if tier is None else f"#define WSM6_KMAX {tier}\n"
    return preamble + prefix + files["gpuwm/core/kernels/wsm6.cu"].decode("utf-8")


def _estate_files(patched: bool) -> dict[str, bytes]:
    files = {
        name: (AUDITED / name).read_bytes()
        for name in ("gpuwm/core/constants.py", "gpuwm/core/kernels/common.cuh",
                     "gpuwm/core/kernels/wsm6.cu", "gpuwm/core/wsm6_constants.py")
    }
    if patched:
        name = "gpuwm/core/wsm6_constants.py"
        files[name] = patch_capacity_source(name, files[name])
    return files


def _wsm6_level_tier(files: dict[str, bytes], nz: int) -> int:
    """Ask the estate's own tier ladder, not a transcription of it."""
    namespace: dict[str, object] = {}
    exec(compile(files["gpuwm/core/wsm6_constants.py"].decode("utf-8"),
                 "wsm6_constants.py", "exec"), namespace)
    return namespace["wsm6_level_tier"](nz)  # type: ignore[operator,index]


def test_config_text_chooses_the_admitted_surface_and_the_default_is_unmoved() -> None:
    """From TOML text to the bytes the seam admits, both ways.

    The default lane must be bit-identical across the 2.5.8 -> 2.7.4 move, and
    "bit-identical" here means the thing that decides an answer: the admitted
    source surface and the vertical bound the adapter reports.  Selecting the
    capacity profile in the config text -- one string, one integer -- must move
    both, or the profile is unreachable.
    """
    from cm1core.cartesian_door import selected_capacity_profile
    from cm1core.config import parse_mapping
    import tomllib

    text = EXAMPLE.read_text(encoding="utf-8")
    assert 'capacity_profile = "default"' in text

    default = parse_mapping(tomllib.loads(text))
    assert default.runtime is not None
    assert default.runtime.capacity_profile == "default"
    assert selected_capacity_profile(default) is None

    metadata = component_metadata(capacity_profile=selected_capacity_profile(default))
    assert metadata["carriers"]["vertical_level_bounds"] == [2, 80]
    assert metadata["engine"]["audited_surface"] == dict(
        sorted(engine_pin.AUDITED_WSM6_SURFACE.items()))
    assert "capacity_profile" not in metadata
    # The stock surface, byte for byte: the default lane admits the estate as
    # published, with no overlay anywhere in it.
    assert metadata["engine"]["audited_surface"]["gpuwm/core/wsm6_constants.py"] == (
        engine_pin.AUDITED_WSM6_SURFACE["gpuwm/core/wsm6_constants.py"])

    deep = text.replace('capacity_profile = "default"',
                        f'capacity_profile = "{WSM6_256_CAPACITY_PROFILE}"')
    deep = deep.replace("nz = 40", "nz = 237")
    admitted = parse_mapping(tomllib.loads(deep))
    assert selected_capacity_profile(admitted) == WSM6_256_CAPACITY_PROFILE

    extended = component_metadata(capacity_profile=selected_capacity_profile(admitted))
    assert extended["carriers"]["vertical_level_bounds"] == [2, 256]
    assert extended["engine"]["audited_surface"] == capacity_surface(
        WSM6_256_CAPACITY_PROFILE)
    assert extended["capacity_profile"]["profile"] == WSM6_256_CAPACITY_PROFILE
    for name, digest in CAPACITY_PATCHED_SURFACE.items():
        assert extended["engine"]["audited_surface"][name] == digest
        assert engine_pin.AUDITED_WSM6_SURFACE[name] != digest


@requires_audited_estate
def test_the_default_lane_compiles_the_bytes_the_wrf_oracle_graded() -> None:
    """The one number that says the version bump did not move microphysics.

    Assembled here from the pinned estate's own bytes rather than read out of a
    record, so this is a measurement and not a restatement.  The default lane
    selects no capacity profile, so it compiles the module with no
    ``WSM6_KMAX`` define at all -- the exact string
    ``gpuwm.certify.kernel_manifest.record_module`` digests.
    """
    stock = _estate_files(patched=False)
    default_source = _wsm6_nvrtc_source(stock, None)
    assert hashlib.sha256(default_source.encode("utf-8")).hexdigest() == (
        engine_pin.AUDITED_WSM6_NVRTC_SOURCE_SHA256
    )

    # The default 40-level example column lands on the 64 tier under stock
    # 2.7.4, and the capacity build must not move it: a config that does not
    # ask for the profile compiles the same source it always did.
    assert _wsm6_level_tier(stock, 40) == 64
    assert _wsm6_level_tier(stock, 80) == 80
    patched = _estate_files(patched=True)
    assert _wsm6_level_tier(patched, 40) == 64
    assert _wsm6_level_tier(patched, 80) == 80
    for tier in (64, 80):
        assert _wsm6_nvrtc_source(stock, tier) == _wsm6_nvrtc_source(patched, tier)

    # A 237-level column has no tier at all under stock 2.7.4 -- which is the
    # defect the overlay exists for -- and lands on 256 under the capacity
    # build, compiling the same kernel text with one different define.
    with pytest.raises(ValueError):
        _wsm6_level_tier(stock, 237)
    assert _wsm6_level_tier(patched, 237) == 256
    deep_source = _wsm6_nvrtc_source(patched, 256)
    assert deep_source != default_source
    assert deep_source.replace("#define WSM6_KMAX 256\n", "") == default_source


CAPACITY_AUTHORITY = ROOT / "docs" / "ARWEN-WSM6-CAPACITY-256-AUTHORITY.md"


def test_the_capacity_authority_states_the_digests_the_code_applies() -> None:
    """The overlay's written decision and its executable delta must agree.

    The authority record carries a baseline/capacity-build digest pair per
    patched file. Those two digests are what a rebuilt overlay is verified
    against, so a record that quotes a stale pair documents a build nobody can
    reproduce.
    """
    prose = CAPACITY_AUTHORITY.read_text(encoding="utf-8")
    assert engine_pin.GPUWM_MEASURED_VERSION in prose
    for name, patched in CAPACITY_PATCHED_SURFACE.items():
        assert name in prose
        assert engine_pin.AUDITED_WSM6_SURFACE[name] in prose, name
        assert patched in prose, name
    # The retirement condition, written where the decision lives.  "Fixed means
    # default": the day stock ArWen admits a deeper WSM6 column, this profile
    # goes away instead of being carried.
    assert "the profile is not retired" in prose.lower()
    assert "retired that day" in prose


@requires_audited_estate
def test_the_capacity_authority_digests_are_reproducible_from_the_pinned_estate() -> None:
    for name, patched in CAPACITY_PATCHED_SURFACE.items():
        raw = (AUDITED / name).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == engine_pin.AUDITED_WSM6_SURFACE[name]
        assert hashlib.sha256(patch_capacity_source(name, raw)).hexdigest() == patched


#: The overlay tree built on the 2.8.0 base, as the authority record
#: names it.  Not distributed, so this skips rather than fails.
BUILT_OVERLAY = ROOT / "work/arwen-wsm6-capacity-256-v3/overlay"


@pytest.mark.skipif(not BUILT_OVERLAY.is_dir(),
                    reason=f"no built capacity overlay at {BUILT_OVERLAY}")
def test_the_built_overlay_is_admitted_and_answers_the_deep_column() -> None:
    """The rebuilt overlay, not a fixture copy of it.

    A fixture proves the patch function; only the published tree proves the
    thing a deep run would actually put on ``PYTHONPATH``.  It is checked twice:
    by the seam's own surface admission, and by importing it in a fresh process
    with no CUDA anywhere.
    """
    receipt = json.loads(
        (BUILT_OVERLAY / "gm1-wsm6-capacity-overlay.json").read_text(encoding="utf-8"))
    assert receipt["capacity_profile"]["base_version"] == engine_pin.GPUWM_MEASURED_VERSION
    assert receipt["capacity_profile"]["base_source_identity"] == (
        engine_pin.GPUWM_AUDITED_SOURCE_IDENTITY)
    assert receipt["gpu_used"] is False
    assert receipt["runtime_monkeypatch"] is False
    assert receipt["source_tree_modified"] is False
    assert receipt["changed_files"] == sorted(CAPACITY_PATCHED_SURFACE)
    assert receipt["capacity_profile_sha256"] == capacity_profile_sha256(
        WSM6_256_CAPACITY_PROFILE)

    for name, digest in capacity_surface(WSM6_256_CAPACITY_PROFILE).items():
        payload = (BUILT_OVERLAY / name).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == digest, name

    script = (
        "import sys,json; "
        "from importlib.metadata import version; "
        "from gpuwm.core.wsm6_constants import wsm6_level_tier,WSM6_KERNEL_LEVEL_TIERS; "
        "from gpuwm.physics_vertical_contract import WSM6_VERTICAL_LEVEL_BOUNDS as public; "
        "print(json.dumps({'version':version('gpuwm'),'public':list(public),"
        "'tiers':list(WSM6_KERNEL_LEVEL_TIERS),"
        "'tier_map':[wsm6_level_tier(n) for n in (40,64,80,237,256)],"
        "'cupy':'cupy' in sys.modules}))"
    )
    env = {**os.environ, "PYTHONPATH": str(BUILT_OVERLAY)}
    result = subprocess.run([sys.executable, "-c", script], env=env,
                            check=True, capture_output=True, text=True)
    assert json.loads(result.stdout) == {
        "version": engine_pin.GPUWM_MEASURED_VERSION,
        "public": [2, 256],
        "tiers": [64, 80, 128, 256],
        # The 40-level default column still lands on 64: the overlay adds
        # tiers, it never moves an existing one.
        "tier_map": [64, 64, 80, 256, 256],
        "cupy": False,
    }
