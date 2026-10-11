"""Column capacity changes must alter storage declarations only, never math."""

from __future__ import annotations

import hashlib
import re
from dataclasses import replace
from types import SimpleNamespace

import pytest

from cm1core import cuda_gm1_child_stretched_sgs_tke as stretched
from cm1core import cuda_sgs_stress_tendency as stress
from cm1core import cuda_tke_evolution as tke
from cm1core.grid_geometry import CartesianGridGeometry


def _sha(source: str) -> str:
    return hashlib.sha256(source.encode()).hexdigest()


def test_default_sources_and_compiled_metadata_keep_the_sealed_75_level_identity() -> None:
    assert stress.build_cuda_sgs_stress_source() == stress.CUDA_SGS_STRESS_SOURCE
    assert tke.build_cuda_tke_evolution_source() == tke.CUDA_TKE_EVOLUTION_SOURCE
    assert stretched.build_cuda_gm1_child_stretched_sources() == (
        stretched.CUDA_GM1_CHILD_STRETCHED_SGS_SOURCE,
        stretched.CUDA_GM1_CHILD_STRETCHED_TKE_SOURCE,
    )
    assert _sha(stretched.build_cuda_gm1_child_stretched_sources()[0]) == stretched.EXPECTED_STRETCHED_SGS_SOURCE_SHA256
    assert _sha(stretched.build_cuda_gm1_child_stretched_sources()[1]) == stretched.EXPECTED_STRETCHED_TKE_SOURCE_SHA256
    assert stress.CompiledSgsStressKernels(None, (), b"", b"", "86", {}).max_nk == 75
    assert tke.CompiledTkeEvolutionKernels(None, (), b"", b"", "86", {}).max_nk == 75
    assert stretched.CompiledChildStretchedSgsTke(None, (), None, (), "86", {}).max_nk == 75


@pytest.mark.parametrize("capacity", (8, 74, 76, 237))
def test_expanding_column_storage_leaves_every_arithmetic_character_unchanged(capacity: int) -> None:
    generated = (
        stress.build_cuda_sgs_stress_source(max_nk=capacity),
        tke.build_cuda_tke_evolution_source(max_nk=capacity),
        *stretched.build_cuda_gm1_child_stretched_sources(max_nk=capacity),
    )
    originals = (
        stress.CUDA_SGS_STRESS_SOURCE, tke.CUDA_TKE_EVOLUTION_SOURCE,
        stretched.CUDA_GM1_CHILD_STRETCHED_SGS_SOURCE,
        stretched.CUDA_GM1_CHILD_STRETCHED_TKE_SOURCE,
    )
    for source, original in zip(generated, originals):
        replacements = {}
        for size, legacy in ((capacity, 75), (capacity - 1, 74)):
            old = f"float hx[{size}],hy[{size}],cp[{size}],dp[{size}],sol[{size}]"
            new = f"float hx[{legacy}],hy[{legacy}],cp[{legacy}],dp[{legacy}],sol[{legacy}]"
            replacements[old] = new
        restored = re.sub("|".join(re.escape(old) for old in replacements), lambda match: replacements[match.group()], source)
        restored = restored.replace(f",cr[{capacity + 1}];", ",cr[76];")
        assert restored == original
    assert generated[0].count(f"cr[{capacity + 1}]") == 2
    assert generated[0].count(f"sol[{capacity - 1}]") == 1
    assert generated[1].count(f"sol[{capacity - 1}]") == 2  # includes disabled historical kernel
    assert generated[3].count(f"sol[{capacity - 1}]") == 1


def test_shape_helpers_cover_237_levels_and_source_faces_without_a_kernel_capacity_guess() -> None:
    shapes = stress.sgs_stress_shapes(9, 11, 237)
    assert shapes["uten"] == (237, 11, 10)
    assert shapes["vten"] == (237, 12, 9)
    assert shapes["wten"] == (238, 11, 9)
    shapes = tke.tke_evolution_shapes(9, 11, 237)
    assert shapes["tkea"] == (238, 11, 9)
    assert shapes["rru"] == (239, 17, 16)
    assert shapes["rrv"] == (239, 18, 15)
    assert shapes["rrw"] == (240, 17, 15)
    for function in (stress.sgs_stress_shapes, tke.tke_evolution_shapes):
        with pytest.raises(ValueError, match="indexing"):
            function(1 << 31, 11, 237)


@pytest.mark.parametrize("capacity", (True, 0, 2, 7.5, 1 << 31))
def test_malformed_capacity_refuses_before_cuda(capacity: object) -> None:
    for function in (stress.compile_cuda_sgs_stress_tendency, tke.compile_cuda_tke_evolution, stretched.compile_cuda_gm1_child_stretched_sgs_tke):
        with pytest.raises(ValueError, match="max_nk"):
            function(max_nk=capacity)


def test_source_extension_refuses_unrecognized_or_changed_declarations(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(RuntimeError, match="declaration inventory"):
        stress._resize_sgs_column_workspace(stress.CUDA_SGS_STRESS_SOURCE.replace("cr[76]", "cr[77]"), max_nk=237)
    with pytest.raises(RuntimeError, match="declaration inventory"):
        tke._resize_tke_column_workspace(tke.CUDA_TKE_EVOLUTION_SOURCE.replace("sol[74]", "sol[75]"), max_nk=237, declaration_count=2)
    monkeypatch.setattr(stretched, "CUDA_GM1_CHILD_STRETCHED_SGS_SOURCE", stretched.CUDA_GM1_CHILD_STRETCHED_SGS_SOURCE + "\n")
    with pytest.raises(RuntimeError, match="sources differ"):
        stretched.build_cuda_gm1_child_stretched_sources(max_nk=237)


_ELF = b"\x7fELF\x02\x01\x01" + bytes(11) + (190).to_bytes(2, "little")
_COMPILES: list[tuple[str, str, tuple[str, ...]]] = []


class _Image:
    """``cupy.cuda.function.Module``: records the image it is given."""

    def load(self, image: bytes) -> None:
        self.image = bytes(image)
        assert self.image.startswith(_ELF)
        self.code = self.image[len(_ELF):].decode()

    def get_function(self, name: str):
        return SimpleNamespace(name=name)


class _Program:
    def __init__(self, source: str, name: str, *, method: str, name_expressions=None):
        self.source, self.name, self.method = source, name, method

    def compile(self, options):
        _COMPILES.append((self.name, self.method, tuple(options)))
        body = self.source.encode()
        return (body if self.method == "ptx" else _ELF + body), {}


def _fake_compiler(monkeypatch: pytest.MonkeyPatch) -> None:
    compiler = SimpleNamespace(_get_nvrtc_version=lambda: (13, 0), _get_arch=lambda: "86", _NVRTCProgram=_Program)
    function = SimpleNamespace(Module=_Image)
    # No RawModule: CuPy builds one with -ftz=true appended, so a compile
    # function that reached for it would fail here.
    fake_cupy = SimpleNamespace(
        __name__="cupy",
        cuda=SimpleNamespace(
            Device=lambda: SimpleNamespace(id=0),
            compiler=compiler,
            function=function,
            runtime=SimpleNamespace(
                getDeviceProperties=lambda _id: {"major": 8, "minor": 6, "name": b"CPU compile-contract spy"},
                runtimeGetVersion=lambda: 13000,
                driverGetVersion=lambda: 13000,
            ),
        ),
    )
    modules = {"cupy": fake_cupy, "cupy.cuda.compiler": compiler, "cupy.cuda.function": function}
    real_import = stress.importlib.import_module
    monkeypatch.setattr(stress.importlib, "import_module", lambda name: modules.get(name) or real_import(name))
    _COMPILES.clear()


def test_each_compile_uses_the_capacity_source_for_the_executed_image_and_receipt(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_compiler(monkeypatch)
    for module, function, builder in (
        (stress, stress.compile_cuda_sgs_stress_tendency, stress.build_cuda_sgs_stress_source),
        (tke, tke.compile_cuda_tke_evolution, tke.build_cuda_tke_evolution_source),
    ):
        compiled = function(max_nk=237)
        source = builder(max_nk=237)
        assert compiled.max_nk == 237
        assert compiled.source_sha256 == _sha(source)
        # The image that runs is the recorded cubin, built from the capacity source.
        assert compiled.module_owner.code == source
        assert compiled.ptx == source.encode()
        assert compiled.cubin == compiled.module_owner.image == _ELF + source.encode()
        module._require_compiled_capacity(compiled, 237)
        with pytest.raises(ValueError, match="workspace 237"):
            module._require_compiled_capacity(compiled, 238)
        with pytest.raises(ValueError, match="source hash"):
            module._require_compiled_capacity(replace(compiled, source_sha256="0" * 64), 237)
    compiled = stretched.compile_cuda_gm1_child_stretched_sgs_tke(max_nk=237)
    sources = stretched.build_cuda_gm1_child_stretched_sources(max_nk=237)
    assert compiled.sgs_module.image.code == sources[0]
    assert compiled.tke_module.image.code == sources[1]
    assert compiled.tke_module.options[-2:] == ("--device-as-default-execution-space", "--gpu-architecture=sm_86")
    assert (compiled.sgs_source_sha256, compiled.tke_source_sha256) == tuple(map(_sha, sources))
    assert compiled.max_nk == 237
    stretched._require_compiled_capacity(compiled, 237)
    with pytest.raises(ValueError, match="workspace 237"):
        stretched._require_compiled_capacity(compiled, 238)
    with pytest.raises(ValueError, match="source hashes"):
        stretched._require_compiled_capacity(replace(compiled, tke_source_sha256="0" * 64), 237)
    # Every image the four compiles built keeps binary32 sub-normals.
    cubins = [options for _, method, options in _COMPILES if method == "cubin"]
    assert len(cubins) == 4
    for options in cubins:
        assert options.count("--ftz=false") == 1 and not any("ftz=true" in option for option in options)
        assert options[-1] == "--gpu-architecture=sm_86"


def test_all_launchers_refuse_a_too_small_compiled_column_before_cuda() -> None:
    common = dict(ni=9, nj=11, nk=237)
    spacings = dict(dx=75, dy=75, dz=75, dt=0.5)
    calls = (
        lambda: stress.launch_cuda_sgs_stress_tendency({}, **common, **spacings),
        lambda: tke.launch_cuda_tke_diffusion({}, **common, **spacings),
        lambda: tke.launch_cuda_tke_rk_stage({}, workspace=None, nrk=1, **common, **spacings),
        lambda: tke.launch_cuda_tke_commit({}, completed_nrk=3, **common),
        lambda: tke.launch_cuda_tke_evolution_frozen_flux({}, **common, **spacings),
        lambda: stretched.launch_cuda_gm1_child_stretched_sgs_stress({}, metrics=None, ztop=17775, **common, **spacings),
        lambda: stretched.launch_cuda_gm1_child_stretched_tke_diffusion({}, metrics=None, ztop=17775, **common, **spacings),
        lambda: stretched.launch_cuda_gm1_child_stretched_tke_rk_stage({}, workspace=None, metrics=None, nrk=1, ztop=17775, **common, **spacings),
        lambda: stretched.launch_cuda_gm1_child_stretched_tke_commit({}, completed_nrk=3, **common),
    )
    for call in calls:
        with pytest.raises(ValueError, match="workspace 75"):
            call()


def test_generic_uniform_geometry_requires_the_exact_grid_and_source_metric_view() -> None:
    grid = CartesianGridGeometry(400, 400, 237, 75, 75, 75)
    args = dict(ni=96, nj=102, nk=237, dx=75, dy=75, dz=75, ztop=17775, enabled=True)
    stretched._require_uniform_geometry(grid, **args)
    for change in ({"nk": 75}, {"dz": 200}, {"ztop": 15000}, {"ni": 401}, {"enabled": False}):
        with pytest.raises(ValueError):
            stretched._require_uniform_geometry(grid, **(args | change))
    with pytest.raises(ValueError, match="source-bound"):
        stretched.bind_gm1_child_stretched_sgs_tke_metrics({}, geometry=grid)
