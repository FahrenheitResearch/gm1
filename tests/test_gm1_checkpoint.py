from __future__ import annotations

import importlib
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest


def _module():
    return importlib.import_module("cm1core.gm1_checkpoint")


def _small_layout() -> dict[str, object]:
    module = _module()
    carriers = []
    offset = 0
    for index, name in enumerate(module.GM1_CHECKPOINT_FIELDS):
        shape = [1, 1, index + 1]
        nbytes = (index + 1) * 4
        carriers.append(
            {
                "name": name,
                "nbytes": nbytes,
                "offset_nbytes": offset,
                "shape": shape,
            }
        )
        offset += nbytes
    return {"carriers": carriers, "payload_nbytes": offset}


def _small_views():
    module = _module()
    carriers = {
        name: np.arange(index + 1, dtype=np.float32).reshape(1, 1, index + 1)
        for index, name in enumerate(module.GM1_CHECKPOINT_FIELDS)
    }
    aliases = {
        alias: carriers[canonical]
        for alias, canonical in module.GM1_WSM6_ALIASES.items()
    }
    return carriers, aliases


def test_import_is_cuda_cold() -> None:
    before = set(sys.modules)
    module = _module()
    assert module.GM1_CHECKPOINT_CARRIER_ABI.endswith("/v1")
    introduced = set(sys.modules) - before
    assert "cupy" not in introduced


def test_exact_inventory_aliases_and_overlap_are_fail_closed() -> None:
    module = _module()
    layout = _small_layout()
    carriers, aliases = _small_views()
    ordered = module._validate_views(carriers, aliases, layout)
    assert tuple(name for name, _, _ in ordered) == tuple(
        item["name"] for item in layout["carriers"]
    )

    missing = dict(carriers)
    missing.pop("ua")
    with pytest.raises(ValueError, match="missing"):
        module._validate_views(missing, aliases, layout)

    wrong_aliases = dict(aliases)
    wrong_aliases["state.pp3d"] = aliases["state.pp3d"].copy()
    with pytest.raises(ValueError, match="exact pp3d"):
        module._validate_views(carriers, wrong_aliases, layout)

    aliased = dict(carriers)
    overlapping = np.arange(2, dtype=np.float32)
    aliased["ua"] = overlapping[:1].reshape(1, 1, 1)
    aliased["u3d"] = overlapping.reshape(1, 1, 2)
    with pytest.raises(ValueError, match="alias"):
        module._validate_views(aliased, aliases, layout)


def test_compose_requires_safe_boundary_and_exact_live_aliases() -> None:
    module = _module()
    carriers, _ = _small_views()
    product_fields = {
        name: carriers[name] for name in module.GM1_PHYSICAL_CHECKPOINT_FIELDS
    }
    restart_arrays = {
        **{name: carriers[name] for name in module.GM1_WSM6_UNIQUE_FIELDS},
        **{
            alias: carriers[canonical]
            for alias, canonical in module.GM1_WSM6_ALIASES.items()
        },
    }
    synchronized = []
    state = SimpleNamespace(
        ready_event=SimpleNamespace(synchronize=lambda: synchronized.append(True)),
        scalar_rk=SimpleNamespace(next_nrk=1, completed=False),
        moist_bridge=SimpleNamespace(
            terms_ready=False, buoyancy_applied=False, post_acoustic_ready=False
        ),
        closure=SimpleNamespace(
            completed_stage=0,
            refresh_index=1,
            tke={"tke3d": carriers["closure.tkea"].copy()},
            checkpoint_arrays=lambda: {"tkea": carriers["closure.tkea"]},
        ),
        transition=SimpleNamespace(
            completed_transitions=1,
            capture_restart=lambda: SimpleNamespace(arrays=restart_arrays),
        ),
        product=SimpleNamespace(field=lambda name: product_fields[name]),
        prepared=SimpleNamespace(
            arrays={name: carriers[name] for name in module.GM1_RADIATION_HISTORY_FIELDS}
        ),
        step_index=1,
        next_stage=1,
        solve1_ready=False,
        mass1_f64=1.0,
        time_seconds=0.5,
    )
    actual, aliases, cadence = module.compose_live_gm1_checkpoint_views(state)
    assert synchronized == [True]
    assert set(actual) == set(module.GM1_CHECKPOINT_FIELDS)
    assert set(aliases) == set(module.GM1_WSM6_ALIASES)
    assert cadence["model_step"] == 1
    assert cadence["scheduler_entries"]["closure.completed_stage"] == 0

    state.moist_bridge.terms_ready = True
    with pytest.raises(ValueError, match="safe post-large-step"):
        module.compose_live_gm1_checkpoint_views(state)

    state.moist_bridge.terms_ready = False
    state.closure.tke["tke3d"].view(np.uint32).reshape(-1)[0] ^= np.uint32(1)
    with pytest.raises(ValueError, match="raw-word identical"):
        module.compose_live_gm1_checkpoint_views(state)


def test_canonical_scheduler_json_rejects_nan() -> None:
    module = _module()
    assert json.loads(module._canonical_json({"z": 1, "a": False})) == {
        "a": False,
        "z": 1,
    }
    with pytest.raises(ValueError):
        module._canonical_json({"bad": float("nan")})
