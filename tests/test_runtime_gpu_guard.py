"""GM1's runtime GPU guard stays exclusive across the gpuwm 2.8.0 move.

gpuwm 2.8.0's ``preflight_exclusive_gpu`` no longer raises on a CUDA
co-tenant; it returns an admission receipt.  These tests stand a fake
``gpuwm.supervisor`` in for the real one, so they need no GPU, no
nvidia-smi and no installed gpuwm.
"""

from __future__ import annotations

import sys
import types
from contextlib import contextmanager

import pytest

from cm1core import runtime_gpu_guard


class _Refused(RuntimeError):
    pass


def _fake_supervisor(receipts: list[object]) -> types.ModuleType:
    module = types.ModuleType("gpuwm.supervisor")
    calls: list[int] = []

    @contextmanager
    def lock(gpu_uuid: str, *, run_id: str):
        yield

    def preflight(gpu_uuid: str, *, approved_pids: set[int]) -> object:
        calls.append(1)
        return receipts[min(len(calls), len(receipts)) - 1]

    module.GPUFileLock = lock
    module.preflight_exclusive_gpu = preflight
    module.GPUPreflightError = _Refused
    module.calls = calls
    return module


@pytest.fixture
def supervisor(monkeypatch: pytest.MonkeyPatch):
    def install(receipts: list[object]) -> types.ModuleType:
        module = _fake_supervisor(receipts)
        package = types.ModuleType("gpuwm")
        package.supervisor = module
        monkeypatch.setitem(sys.modules, "gpuwm", package)
        monkeypatch.setitem(sys.modules, "gpuwm.supervisor", module)
        monkeypatch.setattr(runtime_gpu_guard.time, "sleep", lambda seconds: None)
        return module

    return install


@pytest.mark.parametrize("receipt", [None, {"verdict": "exclusive", "cotenants": ""}])
def test_an_exclusive_card_is_admitted_under_both_engine_versions(supervisor, receipt) -> None:
    module = supervisor([receipt])
    with runtime_gpu_guard.exclusive_runtime_gpu("GPU-test") as token:
        assert token.startswith("gm1-runtime|")
    assert len(module.calls) == 3


@pytest.mark.parametrize(
    "verdict", ["admitted", "admitted-unpriced-run", "admitted-unpriced-device"])
def test_a_shared_card_the_engine_would_admit_is_refused(supervisor, verdict) -> None:
    supervisor([{"verdict": verdict, "cotenants": "pid=4242 name='other'"}])
    with pytest.raises(_Refused, match="pid=4242"):
        with runtime_gpu_guard.exclusive_runtime_gpu("GPU-test"):
            pytest.fail("the guarded block must not run on a shared card")


def test_a_co_tenant_that_arrives_mid_run_is_refused_at_the_end(supervisor) -> None:
    supervisor([None, None, {"verdict": "admitted-unpriced-run", "cotenants": "pid=7"}])
    with pytest.raises(_Refused, match="pid=7"):
        with runtime_gpu_guard.exclusive_runtime_gpu("GPU-test"):
            pass


def test_a_receipt_with_no_verdict_is_refused() -> None:
    # A future engine whose receipt does not say "exclusive" is not trusted.
    class Refused(RuntimeError):
        pass

    import types as _types
    module = _types.ModuleType("gpuwm.supervisor")
    module.GPUPreflightError = Refused
    with pytest.MonkeyPatch.context() as patch:
        package = _types.ModuleType("gpuwm")
        package.supervisor = module
        patch.setitem(sys.modules, "gpuwm", package)
        patch.setitem(sys.modules, "gpuwm.supervisor", module)
        for receipt in ({}, {"cotenants": "pid=9"}, "admitted", 0):
            with pytest.raises(Refused):
                runtime_gpu_guard.require_exclusive(receipt, "GPU-test")
        runtime_gpu_guard.require_exclusive({"verdict": "exclusive"}, "GPU-test")
        runtime_gpu_guard.require_exclusive(None, "GPU-test")


def test_every_gm1_preflight_call_passes_through_the_verdict_check() -> None:
    """No shipped module calls the engine's preflight and ignores its receipt."""
    from pathlib import Path
    import re

    root = Path(runtime_gpu_guard.__file__).resolve().parent
    calls = {
        path.relative_to(root).as_posix(): text
        for path in root.rglob("*.py")
        if "preflight_exclusive_gpu(" in (text := path.read_text(encoding="utf-8"))
    }
    assert set(calls) == {"runtime_gpu_guard.py"}
    text = calls["runtime_gpu_guard.py"]
    wrapped = re.findall(r"require_exclusive\(\s*preflight_exclusive_gpu\(", text)
    assert len(wrapped) == text.count("preflight_exclusive_gpu(") >= 1
