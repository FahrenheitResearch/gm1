"""The stretched acoustic stage binds CM1's alph and kdiv, not constants.

CM1 r21.1 ``sound.F`` is one routine for every vertical coordinate: it forms
``beta=1.0-alph`` and ``asq=alph*alph`` (``sound.F:81-82``) and scales the
divergence damper by ``kdiv`` (``sound.F:756,782``), both read from
``param3``.  ``config.py`` admits any value in [0, 1] and the uniform stage has
always bound them, but ``cuda_gm1_child_psolver3_open`` handed its kernels the
0.60/0.10 constants, so a ``stretch_z=1`` run that declared anything else
integrated CM1's defaults under a receipt that named the declared words.

The words are checked on the REAL stage launcher through
:mod:`tools.oracle.host_cupy` (the packaged CUDA text compiled by g++
``-ffp-contract=off``), over all three RK stages of one large step and every
acoustic carrier.
"""

from __future__ import annotations

import shutil

import numpy as np
import pytest

from cm1core.cuda_gm1_child_psolver3_open import launch_cuda_gm1_child_psolver3_open_stage

needs_gxx = pytest.mark.skipif(shutil.which("g++") is None, reason="the host kernel emulator needs g++")


@pytest.fixture(scope="module")
def harness():
    from tools.oracle import stretched_acoustic_reals_identity

    return stretched_acoustic_reals_identity


@pytest.fixture(scope="module")
def constants(harness):
    return harness.run(alph=None, kdiv=None)


@needs_gxx
def test_a_document_declaring_cm1_s_defaults_integrates_the_constant_words(harness, constants) -> None:
    """Binary64 0.60/0.10 from the runtime configuration round to the same binary32 words."""

    from tools.oracle import host_cupy

    host_cupy.COUNTS.clear()
    document = harness.run(alph=0.60, kdiv=0.10)
    assert host_cupy.COUNTS["raw_kernel_launch"] > 100
    result = harness.compare(constants, document)
    assert result["raw_word_identical"] is True, result["differing"]
    assert result["carriers"] == 135
    # Not vacuous: the stage moved the prognostic carriers and stayed finite.
    words = document["words"]
    assert all(np.isfinite(value).all() for value in words.values())
    for name in ("pp3d", "ppx", "u3d", "v3d", "w3d", "rrw"):
        assert np.count_nonzero(words[f"rk3.{name}"]) > words[f"rk3.{name}"].size // 2, name
    assert document["receipts"][3]["acoustic_reals"] == {
        "alph": float(np.float32(0.60)), "kdiv": float(np.float32(0.10)),
    }


@needs_gxx
def test_declared_reals_reach_the_kernels(harness, constants) -> None:
    """A declared alph/kdiv moves the words to exactly what the kernels compute with them."""

    declared = harness.run(alph=0.55, kdiv=0.35)
    moved = harness.compare(constants, declared)
    assert not moved["raw_word_identical"]
    assert {"rk3.pp3d", "rk3.ppx", "rk3.w3d", "rk3.rrw"} <= set(moved["differing"])
    # The same kernels with their CM1 constants replaced by the declared words:
    # the binding is exactly "the kernel reads this real", nothing else moved.
    constant_bound = harness.run(alph=None, kdiv=None, constant_alph=0.55, constant_kdiv=0.35)
    same = harness.compare(declared, constant_bound)
    assert same["raw_word_identical"] is True, same["differing"]
    assert declared["receipts"][1]["acoustic_reals"] == {
        "alph": float(np.float32(0.55)), "kdiv": float(np.float32(0.35)),
    }


@needs_gxx
def test_kdiv_alone_is_bound(harness, constants) -> None:
    """kdiv enters only the divergence damper; it must move pressure on its own."""

    damped = harness.run(alph=0.60, kdiv=0.0)
    moved = harness.compare(constants, damped)
    assert {"rk1.ppd", "rk3.pp3d", "rk3.ppx"} <= set(moved["differing"])
    # sound.F:756,782 with kdiv=0: the final-stage ppx is exactly zero.
    assert not np.any(damped["words"]["rk3.ppx"][1:-1, 1:-1, 1:-1])


@pytest.mark.parametrize(
    ("reals", "error", "match"),
    (
        ({"alph": 1.5}, ValueError, r"alph=1.5 is outside CM1's admitted \[0, 1\]"),
        ({"kdiv": -0.1}, ValueError, r"kdiv=-0.1 is outside CM1's admitted \[0, 1\]"),
        ({"kdiv": "0.1"}, TypeError, "kdiv must be numeric or None"),
        ({"alph": True}, TypeError, "alph must be numeric or None"),
    ),
)
def test_reals_outside_cm1_s_interval_are_refused_before_any_device_work(reals, error, match) -> None:
    with pytest.raises(error, match=match):
        launch_cuda_gm1_child_psolver3_open_stage(
            {}, {}, ni=8, nj=8, nk=8, nrk=1, dt=1.0, nsound=6, dx=250.0, dy=250.0,
            vertical=object(), **reals,
        )
