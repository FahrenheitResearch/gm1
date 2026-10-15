"""The flat acoustic stage handed the run's lateral step, derived from the sealed stage.

:mod:`cm1core.cuda_gm1_flat_lateral_acoustic` runs the sealed launcher of
:mod:`cm1core.cuda_gm1_child_psolver3_open` with its small-step loop replaced,
for one launch, by the sealed loop plus sound.F's two lateral calls.  The
words it gives are compared with CM1's compiled sound in
``evidence/oracle/walls-operators`` (the ``flat`` variant of ``sound``); here
the derivation and the plumbing are held: the sealed module is untouched, the
derived loop is the sealed loop with exactly the listed insertions, the
sealed loop is put back after every launch, and a run without a lateral step
goes to the sealed launcher as it always has.
"""

from __future__ import annotations

import inspect
import textwrap
from types import SimpleNamespace
from typing import Any

import pytest

from cm1core import cuda_gm1_child_psolver3_open as sealed
from cm1core import cuda_gm1_flat_lateral_acoustic as lateral_stage
from cm1core.cuda_boundary_families import LateralBoundarySelection


def test_the_derived_loop_is_the_sealed_loop_with_the_two_lateral_calls() -> None:
    sealed_text = textwrap.dedent(inspect.getsource(sealed._enqueue_flat_stage))
    derived = lateral_stage.lateral_stage_source()
    # Taking the insertions back out gives the sealed text exactly.
    restored = derived
    for old, new in reversed(lateral_stage._LATERAL_EDITS):
        assert restored.count(new) == 1
        restored = restored.replace(new, old)
    assert restored == sealed_text
    assert "enqueue_radbc(" in derived and "enqueue_restrict_outward_flux(" in derived
    assert "apply_compiled=lateral.roflux_cm1_order" in derived
    assert inspect.signature(lateral_stage._enqueue_flat_stage_lateral).parameters["lateral"].default is (
        inspect.Parameter.empty
    )


def _step(sides: tuple[int, int, int, int] = (2, 2, 2, 2), irbc: int = 2, roflux: int = 0) -> Any:
    west, east, south, north = sides
    selection = LateralBoundarySelection(west=west, east=east, south=south, north=north, irbc=irbc,
                                         roflux=roflux)
    return SimpleNamespace(selection=selection, record=lambda: {"selection": selection.record()})


def test_no_lateral_step_is_the_sealed_launcher(monkeypatch) -> None:
    calls = []

    def launcher(arrays, metrics, **kwargs):
        calls.append((arrays, metrics, kwargs, sealed._enqueue_flat_stage))
        return "sealed launch"

    monkeypatch.setattr(sealed, "launch_cuda_gm1_child_psolver3_open_stage", launcher)
    loop = sealed._enqueue_flat_stage
    assert lateral_stage.launch_cuda_gm1_flat_acoustic_stage("a", "m", nrk=1, lateral=None) == "sealed launch"
    assert calls == [("a", "m", {"nrk": 1}, loop)]


@pytest.mark.parametrize(("irbc", "production"), [(1, False), (2, False), (4, True)])
def test_a_lateral_step_runs_the_derived_loop_once_and_puts_the_sealed_one_back(
    monkeypatch, irbc: int, production: bool
) -> None:
    seen = {}

    def launcher(arrays, metrics, **kwargs):
        seen["loop"] = sealed._enqueue_flat_stage
        seen["ablations"] = kwargs["ablations"]
        return SimpleNamespace(receipt={})

    monkeypatch.setattr(sealed, "launch_cuda_gm1_child_psolver3_open_stage", launcher)
    loop = sealed._enqueue_flat_stage
    launch = lateral_stage.launch_cuda_gm1_flat_acoustic_stage(
        "a", "m", vertical=SimpleNamespace(flat_column=(1, 2, 3)), lateral_sides=(2, 2, 2, 2),
        ablations=sealed.TerrainOpenSoundAblations(stage_family_fills=False), lateral=_step(irbc=irbc),
    )
    assert seen["loop"].func is lateral_stage._enqueue_flat_stage_lateral
    assert seen["ablations"].radiation_history_production is production
    assert seen["ablations"].stage_family_fills is False
    assert sealed._enqueue_flat_stage is loop
    assert launch.receipt["lateral_step"]["small_step_radbc"] is (irbc == 2)
    assert launch.receipt["lateral_step"]["durran_klemp_production"] is production


def test_the_sealed_loop_comes_back_when_the_launch_fails(monkeypatch) -> None:
    def launcher(arrays, metrics, **kwargs):
        raise RuntimeError("no device")

    monkeypatch.setattr(sealed, "launch_cuda_gm1_child_psolver3_open_stage", launcher)
    loop = sealed._enqueue_flat_stage
    with pytest.raises(RuntimeError, match="no device"):
        lateral_stage.launch_cuda_gm1_flat_acoustic_stage(
            "a", "m", vertical=SimpleNamespace(flat_column=(1, 2, 3)), lateral=_step())
    assert sealed._enqueue_flat_stage is loop


def test_only_the_flat_stage_takes_a_lateral_step_and_only_its_own_sides() -> None:
    with pytest.raises(ValueError, match="only the flat stage"):
        lateral_stage.launch_cuda_gm1_flat_acoustic_stage(
            "a", "m", vertical=SimpleNamespace(flat_column=None), lateral=_step())
    with pytest.raises(ValueError, match="one run has one family"):
        lateral_stage.launch_cuda_gm1_flat_acoustic_stage(
            "a", "m", vertical=SimpleNamespace(flat_column=(1, 2, 3)), lateral_sides=(2, 2, 2, 2),
            lateral=_step(sides=(3, 3, 2, 2)))


def test_the_sealed_module_is_release_0_1_7s() -> None:
    """The real child's operator manifest pins these bytes; the lateral step lives beside them."""

    assert "lateral" not in inspect.signature(sealed.launch_cuda_gm1_child_psolver3_open_stage).parameters
    assert "lateral" not in inspect.signature(sealed._enqueue_flat_stage).parameters
