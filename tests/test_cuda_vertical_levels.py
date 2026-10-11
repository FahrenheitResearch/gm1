"""The device tail for a declared level set: CPU-checkable contract.

The kernel launch itself needs a CUDA device and is exercised by
``launch_cuda_vertical_levels``, which holds the device words to the
``cm1_vertical_grid`` host transcription raw word for raw word.  Everything
below runs without a device.
"""

from __future__ import annotations

import hashlib
import re

import pytest

from cm1core.cm1_vertical_grid import build_vertical_levels, uniform_vertical_levels
from cm1core.cuda_vertical_levels import (
    CUDA_VERTICAL_LEVELS_ABI,
    CUDA_VERTICAL_LEVELS_SOURCE,
    CUDA_VERTICAL_LEVELS_SOURCE_BYTES,
    CUDA_VERTICAL_LEVELS_SOURCE_SHA256,
    SOURCE_ANCHORS,
    launch_cuda_vertical_levels,
)


def test_the_source_is_self_pinned_and_reconstructs_no_level_from_dz() -> None:
    assert (
        hashlib.sha256(CUDA_VERTICAL_LEVELS_SOURCE_BYTES).hexdigest()
        == CUDA_VERTICAL_LEVELS_SOURCE_SHA256
    )
    assert CUDA_VERTICAL_LEVELS_ABI in CUDA_VERTICAL_LEVELS_SOURCE
    # dz appears only as param.F:8170,8176's metric reference, never as a
    # multiplier that would rebuild a level.
    assert "m32(dz" not in CUDA_VERTICAL_LEVELS_SOURCE
    assert "(float)(k-1u),dz" not in CUDA_VERTICAL_LEVELS_SOURCE
    assert CUDA_VERTICAL_LEVELS_SOURCE.count("d32(dz,") == 2  # mh and mf only
    # param.F:8165-8166, :8172-8173, :8178-8179 ghosts and :8212-8224 check.
    for fragment in (
        "zf[0]=-zf[2]",
        "zh[0]=-zh[1]",
        "mf[0]=mf[1]",
        "bad[0]=flag",
    ):
        assert fragment in CUDA_VERTICAL_LEVELS_SOURCE, fragment
    assert re.search(r"extern \"C\" __global__ void cm1_vertical_tail_v1\(", CUDA_VERTICAL_LEVELS_SOURCE)


def test_the_authority_anchors_name_the_shared_tail() -> None:
    assert SOURCE_ANCHORS["shared_tail"] == "src/param.F:8162-8182"
    assert SOURCE_ANCHORS["inverse_metrics"] == "src/param.F:8189-8209"
    assert SOURCE_ANCHORS["user_w_levels"] == "src/param.F:8072-8132"


def test_the_launcher_refuses_anything_but_a_built_coordinate() -> None:
    with pytest.raises(TypeError, match="VerticalLevels"):
        launch_cuda_vertical_levels(object())
    with pytest.raises(TypeError, match="VerticalLevels"):
        launch_cuda_vertical_levels({"zf": [0.0, 1.0]})


def test_the_host_transcription_the_device_is_held_to_is_the_shared_tail() -> None:
    """The device kernel is compared against these words, so they must be
    exactly what ``param.F:8162-8209`` produces from the same faces."""

    uniform = uniform_vertical_levels(12, 250.0)
    rebuilt = build_vertical_levels(
        stretch_z=4, nk=12, dz_m=250.0, ztop_m=3000.0,
        levels_m=uniform.face_heights_m,
    )
    for name in ("zf", "zh", "sigmaf", "sigma", "mh", "mf", "rmh", "rmf"):
        assert getattr(rebuilt, name).tobytes() == getattr(uniform, name).tobytes(), name
    assert rebuilt.sha256 == uniform.sha256
