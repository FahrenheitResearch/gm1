"""Exact operator-source manifest for the real stretched 50 m GM1 child.

The durable GM1 checkpoint contract owns a historical 22-key baseline.  The
child does not relabel those keys when it selects terrain-aware or stretched-z
operators; every child replacement and admission layer is added under a new
namespaced key.  Inline same-stream lifecycle operations are bound by their
semantic source seals from the child composer.

This module is CUDA-free.  It hashes regular source files and returns the
compact canonical JSON bytes consumed both by the post-final wrapper and by
the Rust paired post-step publisher.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType


GM1_BASELINE_OPERATOR_SOURCE_PATHS: Mapping[str, str] = MappingProxyType(
    {
        "closure.coordinator": "src/cm1core/cuda_gm1_closure.py",
        "closure.rayleigh-held": "src/cm1core/cuda_rayleigh_held.py",
        "closure.scalar-turbulence": "src/cm1core/cuda_gm1_scalar_turbulence.py",
        "closure.sgs-local": "src/cm1core/cuda_sgs_tke.py",
        "closure.sgs-stress": "src/cm1core/cuda_sgs_stress_tendency.py",
        "closure.terrain-deformation": "src/cm1core/cuda_terrain_deformation.py",
        "closure.tke-evolution": "src/cm1core/cuda_tke_evolution.py",
        "dynamics.apmasscon": "src/cm1core/cuda_apmasscon.py",
        "dynamics.lateral-closure": "src/cm1core/cuda_lateral_closure.py",
        "dynamics.moist-stage-bridge": "src/cm1core/cuda_moist_stage_bridge.py",
        "dynamics.momentum-advection5": "src/cm1core/cuda_momentum_advection5.py",
        "dynamics.momentum-rk": "src/cm1core/cuda_momentum_rk.py",
        "dynamics.psolver3-terrain-open": "src/cm1core/cuda_psolver3_terrain_open.py",
        "dynamics.scalar-advection5": "src/cm1core/cuda_advection5.py",
        "dynamics.scalar-moist-rk": "src/cm1core/cuda_scalar_moist_rk.py",
        "dynamics.sound3-open": "src/cm1core/cuda_sound3_open_boundary.py",
        "dynamics.terrain-open-boundary": "src/cm1core/cuda_terrain_open_boundary.py",
        "forecast.coordinator": "src/cm1core/cuda_gm1_forecast.py",
        "physics.arwen-wsm6-estate": "src/cm1core/arwen_wsm6_component.py",
        "physics.transition": "src/cm1core/initialized_product_wsm6_transition.py",
        "physics.wsm6-pdefq": "src/cm1core/cuda_wsm6_pdefq.py",
        "state.initialized-product-terrain-open": (
            "src/cm1core/cuda_initialized_product_terrain_open.py"
        ),
    }
)


GM1_CHILD_OPERATOR_SOURCE_PATHS: Mapping[str, str] = MappingProxyType(
    {
        # The component-generic column adapter.  Since the WSM6 binding was
        # generalized over CM1's ptype (src/param.F builds one q3d table per
        # ptype), the bytes that execute a WSM6 column live in BOTH
        # arwen_wsm6_component.py -- which the durable 22-key baseline already
        # measures as physics.arwen-wsm6-estate -- and this file.  It joins the
        # child inventory rather than the baseline because that order is
        # durable and sealed; what matters is that no executing byte is
        # unmeasured.
        "physics.arwen-column-adapter": (
            "src/cm1core/arwen_column_component.py"
        ),
        "child.closure.fresh-turbsmag-init": (
            "src/cm1core/cuda_gm1_child_turbsmag_init.py"
        ),
        "child.closure.live-nm-def": (
            "src/cm1core/cuda_gm1_child_stretched_nm_def.py"
        ),
        "child.closure.moist-nm": "src/cm1core/cuda_moist_nm.py",
        "child.closure.scalar-turbulence": (
            "src/cm1core/cuda_gm1_child_stretched_scalar_turbulence.py"
        ),
        "child.closure.stretched-sgs-tke": (
            "src/cm1core/cuda_gm1_child_stretched_sgs_tke.py"
        ),
        "child.composer.large-step": (
            "src/cm1core/cuda_gm1_child_step_composer.py"
        ),
        "child.dynamics.psolver3-open": (
            "src/cm1core/cuda_gm1_child_psolver3_open.py"
        ),
        "child.dynamics.sound3-column": "src/cm1core/cuda_sound3_column.py",
        "child.dynamics.stretched-metrics": (
            "src/cm1core/cuda_gm1_child_stretched_metrics.py"
        ),
        "child.dynamics.stretched-moist": (
            "src/cm1core/cuda_gm1_child_stretched_moist.py"
        ),
        "child.dynamics.stretched-momentum": (
            "src/cm1core/cuda_gm1_child_stretched_momentum.py"
        ),
        "child.dynamics.stretched-scalar": (
            "src/cm1core/cuda_gm1_child_stretched_scalar.py"
        ),
        "child.grid.stretch-z1-core": "src/cm1core/cuda_stretch_z1.py",
        "child.grid.stretch-z1-profile": "src/cm1core/stretch_z1_child_15km.py",
        "child.grid.stretch-z1-terrain": "src/cm1core/cuda_stretch_z1_terrain.py",
        "child.physics.eqtset2-wsm6-bridge": (
            "src/cm1core/eqtset2_wsm6_bridge.py"
        ),
        "child.physics.post-final-rk": (
            "src/cm1core/cuda_gm1_child_post_final_rk.py"
        ),
        "child.state.atmosphere-admission": (
            "src/cm1core/gm1_child_atmosphere_admission.py"
        ),
        "child.state.auxiliary-admission": "src/cm1core/cuda_gm1_child_nm_def.py",
        "child.state.auxiliary-interfaces": (
            "src/cm1core/gm1_child_auxiliary_interfaces.py"
        ),
        # The superseded uniform-z closure reader is retained because the
        # historical operator-replay gates still reproduce those exact bytes.
        # The production child is initialized from the corrected stretched-z
        # closure below, so that admission module is part of the operator set
        # whose digests every published receipt carries.
        "child.state.closure-v2-admission": "src/cm1core/gm1_child_closure_v2.py",
        "child.state.stretched-closure-admission": (
            "src/cm1core/gm1_child_stretched_closure_v3.py"
        ),
        "child.state.full-envelope": "src/cm1core/cuda_gm1_child_large_step.py",
        "child.state.initialization-interface": (
            "src/cm1core/gm1_child_initialization.py"
        ),
        "child.state.operator-manifest-builder": (
            "src/cm1core/gm1_child_operator_manifest.py"
        ),
        "child.state.static-admission": "src/cm1core/gm1_child_static_admission.py",
        "child.state.terrain-admission": "src/cm1core/cuda_gm1_child_terrain.py",
    }
)


@dataclass(frozen=True, slots=True)
class Gm1ChildOperatorManifest:
    """One exact no-newline operator manifest and its independent seal."""

    canonical_bytes: bytes
    sha256: str
    operators: Mapping[str, str]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _regular_source_sha256(root: Path, relative: str) -> str:
    path = root / relative
    metadata = path.lstat()
    if path.is_symlink() or not path.is_file() or metadata.st_size <= 0:
        raise ValueError(f"operator source must be one nonempty regular file: {path}")
    return _sha256_file(path)


def _semantic_child_pins() -> Mapping[str, str]:
    composer = importlib.import_module("cm1core.cuda_gm1_child_step_composer")
    return MappingProxyType(
        {
            composer.GM1_CHILD_COMPLETED_CLOSURE_REFRESH_OPERATOR_KEY: (
                composer.GM1_CHILD_COMPLETED_CLOSURE_REFRESH_SOURCE_SHA256
            ),
            composer.GM1_CHILD_FINAL_DYNAMICS_SYNC_OPERATOR_KEY: (
                composer.GM1_CHILD_FINAL_DYNAMICS_SYNC_SOURCE_SHA256
            ),
            composer.GM1_CHILD_SOLVE3_RHO_BOUNDARY_OPERATOR_KEY: (
                composer.GM1_CHILD_SOLVE3_RHO_BOUNDARY_SOURCE_SHA256
            ),
        }
    )


def build_gm1_child_operator_manifest(
    root: Path | None = None,
) -> Gm1ChildOperatorManifest:
    """Hash the installed baseline-plus-child source inventory fail-closed."""

    root = Path(__file__).resolve().parents[2] if root is None else root.resolve()
    post_final = importlib.import_module("cm1core.cuda_gm1_child_post_final_rk")
    baseline_names = tuple(GM1_BASELINE_OPERATOR_SOURCE_PATHS)
    if baseline_names != tuple(post_final.GM1_BASELINE_OPERATOR_KEYS):
        raise RuntimeError("child manifest baseline differs from the durable 22-key order")
    if set(GM1_BASELINE_OPERATOR_SOURCE_PATHS) & set(GM1_CHILD_OPERATOR_SOURCE_PATHS):
        raise RuntimeError("child source inventory collides with a durable baseline key")

    sources: dict[str, str] = {}
    for name, relative in (
        *GM1_BASELINE_OPERATOR_SOURCE_PATHS.items(),
        *GM1_CHILD_OPERATOR_SOURCE_PATHS.items(),
    ):
        sources[name] = _regular_source_sha256(root, relative)
    for name, value in _semantic_child_pins().items():
        if name in sources:
            raise RuntimeError(f"semantic child source key collides with file key {name}")
        sources[name] = value

    required = dict(post_final.child_post_final_rk_operator_source_pins())
    for name, expected in required.items():
        actual = sources.get(name)
        if actual != expected:
            raise RuntimeError(
                f"child post-final source pin {name} differs: {actual!r} != {expected!r}"
            )
    if not (len(post_final.GM1_BASELINE_OPERATOR_KEYS) <= len(sources) <= 128):
        raise RuntimeError("child operator inventory is outside the admitted v1 cap")

    ordered = dict(sorted(sources.items()))
    canonical = json.dumps(
        {
            "model": "GM1",
            "operators": ordered,
            "schema": "gm1.operator-source-manifest/v1",
        },
        allow_nan=False,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    if canonical.endswith(b"\n"):
        raise RuntimeError("canonical child operator manifest must not contain a newline")
    post_final_seal = post_final.require_child_post_final_rk_operator_sources(ordered)
    seal = hashlib.sha256(canonical).hexdigest()
    if seal != post_final_seal:
        raise RuntimeError("post-final and Rust-ready operator manifest seals differ")
    return Gm1ChildOperatorManifest(
        canonical,
        seal,
        MappingProxyType(ordered),
    )


__all__ = (
    "GM1_BASELINE_OPERATOR_SOURCE_PATHS",
    "GM1_CHILD_OPERATOR_SOURCE_PATHS",
    "Gm1ChildOperatorManifest",
    "build_gm1_child_operator_manifest",
)
