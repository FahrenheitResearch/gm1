"""CM1 r21.1 lower boundary and surface drag: host-side proofs.

Every arithmetic expectation in this file is transcribed here from the CM1
source by hand, in the source's own statement order, so a passing test means
the operator agrees with CM1 and not merely with itself.  The device tests
are marked GPU-gated; everything else runs on a CPU-only machine.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import hashlib
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import cm1_marks

from cm1core import cuda_surface_drag as drag
from cm1core.config import FREE_SLIP_LOWER_BOUNDARY, load_config, parse_mapping
from cm1core.cuda_gm1_child_stretched_sgs_tke import _require_surface_binding
from cm1core.cuda_sgs_stress_tendency import SgsStressSelection
from cm1core.errors import CM1Refusal
from cm1core.eqtset2_surface_dissipation import (
    SurfaceDissipationCarriers,
    cpu_surface_dissipative_heating_inplace,
)
from cm1core.surface_exchange_coefficients import (
    SurfaceExchangeSelection,
    cpu_surface_exchange_coefficients_inplace,
    surface_exchange_shapes,
)


ROOT = Path(__file__).resolve().parents[1]
AUTHORITY = ROOT / "upstream" / "cm1-21.1" / "pristine"
NI, NJ, NK = 6, 5, 7
DZ = 250.0
UMOVE, VMOVE = 12.5, 3.0
CNSTCD = 0.0035


def _libm():
    name = ctypes.util.find_library("m")
    library = ctypes.CDLL(name) if name else ctypes.CDLL(None)
    bound = {}
    for symbol in ("logf", "expf", "sqrtf"):
        function = getattr(library, symbol)
        function.argtypes = [ctypes.c_float]
        function.restype = ctypes.c_float
        bound[symbol] = function
    return bound


_LIBM = _libm()


def f32(value) -> np.float32:
    return np.float32(value)


def logf(value) -> np.float32:
    return np.float32(_LIBM["logf"](ctypes.c_float(float(np.float32(value)))))


def expf(value) -> np.float32:
    return np.float32(_LIBM["expf"](ctypes.c_float(float(np.float32(value)))))


def sqrtf(value) -> np.float32:
    return np.float32(_LIBM["sqrtf"](ctypes.c_float(float(np.float32(value)))))


def _fixture(seed: int = 11) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    arrays: dict[str, np.ndarray] = {}
    for name, shape in drag.surface_drag_shapes(NI, NJ, NK).items():
        arrays[name] = np.zeros(shape, np.float32)
    arrays["u3d"][:] = rng.uniform(-14.0, 20.0, arrays["u3d"].shape).astype(np.float32)
    arrays["v3d"][:] = rng.uniform(-12.0, 16.0, arrays["v3d"].shape).astype(np.float32)
    arrays["rf"][:] = rng.uniform(0.8, 1.25, arrays["rf"].shape).astype(np.float32)
    arrays["mf"][:] = rng.uniform(0.9, 1.1, arrays["mf"].shape).astype(np.float32)
    for name in ("kmh", "kmv", "khh", "khv"):
        arrays[name][:] = rng.uniform(0.5, 40.0, arrays[name].shape).astype(np.float32)
    arrays["tkea"][:] = rng.uniform(0.01, 3.0, arrays["tkea"].shape).astype(np.float32)
    arrays["zh"][:] = (
        np.arange(NK, dtype=np.float32)[:, None, None] * np.float32(DZ)
        + np.float32(0.5 * DZ)
    )
    arrays["zf"][:] = np.arange(NK + 1, dtype=np.float32)[:, None, None] * np.float32(DZ)
    arrays["t13"][:] = rng.uniform(-3.0, 3.0, arrays["t13"].shape).astype(np.float32)
    arrays["t23"][:] = rng.uniform(-3.0, 3.0, arrays["t23"].shape).astype(np.float32)
    return arrays


SEMI_SLIP = drag.SurfaceDragSelection(
    bbc=3, sfcmodel=1, isfcflx=0, cecd=1, cnstcd=CNSTCD,
    u_move_m_s=UMOVE, v_move_m_s=VMOVE,
)
NO_SLIP = drag.SurfaceDragSelection(
    bbc=2, sfcmodel=0, isfcflx=0, cecd=1, cnstcd=CNSTCD,
    u_move_m_s=UMOVE, v_move_m_s=VMOVE,
)


# ---------------------------------------------------------------- provenance


def test_import_does_not_import_cupy_or_open_cuda() -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, cm1core.cuda_surface_drag, cm1core.eqtset2_surface_dissipation; "
            "assert not any(n == 'cupy' or n.startswith('cupy.') for n in sys.modules)",
        ],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_cuda_source_matches_its_pin_and_declares_every_kernel() -> None:
    source = drag.build_cuda_surface_drag_source()
    assert (
        hashlib.sha256(source.encode()).hexdigest()
        == drag.CUDA_SURFACE_DRAG_SOURCE_SHA256
    )
    for name in drag.SURFACE_DRAG_KERNEL_NAMES:
        assert f'extern "C" __global__ void {name}(' in source
    # Nothing else may claim to be a kernel of this ABI.
    declared = set(re.findall(r'extern "C" __global__ void (\w+)\(', source))
    assert declared == set(drag.SURFACE_DRAG_KERNEL_NAMES)


@pytest.mark.skipif(
    not AUTHORITY.is_dir(), reason="the extracted CM1 r21.1 release tree is absent"
)
def test_every_authority_pin_matches_the_release_bytes() -> None:
    for relative, expected in drag.AUTHORITY_SOURCE_SHA256.items():
        actual = hashlib.sha256((AUTHORITY / relative).read_bytes()).hexdigest()
        assert actual == expected, relative


# Each anchor names a source range and the mark (tests/cm1_marks.py) of the
# exact CM1 statement this port transcribed from it.  A moved line makes the citation wrong, and a wrong
# citation is the thing that makes a ported formula unverifiable.
ANCHOR_EVIDENCE = {
    "ground_relative_wind": ("src/turb.F", 296, 307, "9ce357e2348669a6/26"),
    "lowest_level_wind_and_height": ("src/turb.F", 532, 555, "5c767082d167892e/35"),
    "getcecd_dispatch": ("src/turb.F", 614, 616, "b892532fe719660d/12"),
    "getcecd_constant_drag": ("src/sfcphys.F", 120, 136, "c1c5970a76d23022/27"),
    "tkebc_lower_face": ("src/turb.F", 3244, 3306, "0cfe911603503bba/37"),
    "sfcstress_free_slip": ("src/turb.F", 4536, 4546, "f505fef21434722e/21"),
    "sfcstress_no_slip": ("src/turb.F", 4548, 4564, "c7c8876b630a90f9/29"),
    "sfcstress_semi_slip": ("src/turb.F", 4583, 4605, "3d7c1010b9ed26b4/47"),
    "surface_stress_magnitude": ("src/turb.F", 4692, 4700, "b01073123fcd448a/15"),
    "momentum_lower_flux_consumer": ("src/turbtend.F", 1442, 1460, "dbe2dc53b70e296e/45"),
    "momentum_lower_flux_consumer_v": ("src/turbtend.F", 2212, 2230, "66f24afcdddc0af5/45"),
    "momentum_lower_flux_explicit": ("src/turbtend.F", 1411, 1421, "55e92a19f3c22da1/42"),
    "diff2_no_slip_ground_relative": ("src/diff2.F", 940, 951, "bca54f18d33809f5/51"),
    "diff2_semi_slip": ("src/diff2.F", 952, 965, "46795ad2c6862456/31"),
    "surface_dissipative_heating": ("src/solve1.F", 917, 988, "56c66500e63ae73b/65"),
    "bbc_range_check": ("src/param.F", 1518, 1532, "e48c81a92d337105/16"),
    "bbc3_requires_sfcmodel": ("src/param.F", 2073, 2088, "c684ffbe2a348f83/32"),
    "sfcmodel_forces_bbc3": ("src/param.F", 873, 889, "80baa28048a91fc7/5"),
    "sfcmodel1_cecd_range": ("src/param.F", 1349, 1365, "001b065b986e1f20/15"),
    "isfcflx_requires_sfcmodel": ("src/param.F", 2090, 2106, "244f5fe2138b015b/27"),
    "surface_enthalpy_flux": ("src/sfcphys.F", 275, 453, "c62d489ff5b78d6d/18"),
    "surface_enthalpy_flux_consumer": ("src/turbtend.F", 517, 519, "66525a30bd70f9fc/31"),
    "karman": ("src/constants.F", 23, 23, "c5fa7b4dcf07033c/6"),
}


@pytest.mark.skipif(
    not AUTHORITY.is_dir(), reason="the extracted CM1 r21.1 release tree is absent"
)
@pytest.mark.parametrize("key", sorted(ANCHOR_EVIDENCE))
def test_each_source_anchor_still_points_at_its_statement(key: str) -> None:
    relative, first, last, statement = ANCHOR_EVIDENCE[key]
    assert drag.SOURCE_ANCHORS[key].startswith(f"{relative}:")
    lines = (AUTHORITY / relative).read_text(encoding="utf-8", errors="replace").splitlines()
    window = "\n".join(lines[first - 1 : last])
    assert cm1_marks.has(window, statement), f"{key}: {drag.SOURCE_ANCHORS[key]}"


@pytest.mark.skipif(
    not AUTHORITY.is_dir(), reason="the extracted CM1 r21.1 release tree is absent"
)
def test_the_enthalpy_flux_anchors_are_the_routine_and_its_computed_branch() -> None:
    """One citation for ``sfcflux``, and it is the routine's real extent.

    The anchor a reader follows has to land on the subroutine, not on a
    comment banner above it or on lines past its ``end subroutine``.  The
    computed branch this port actually runs is a sub-range of it and is
    anchored separately, so a reader cannot mistake the whole routine --
    which also carries set_flx=1 and the testcase overrides -- for what runs.
    """

    lines = (AUTHORITY / "src/sfcphys.F").read_text(
        encoding="utf-8", errors="replace"
    ).splitlines()
    first, last = 275, 453
    assert lines[first - 1].strip().startswith("subroutine sfcflux(")
    assert lines[last - 1].strip() == "end subroutine sfcflux"
    assert not lines[first - 2].strip().startswith("subroutine")
    assert "sfcphys.F:275-453" in drag.SOURCE_ANCHORS["surface_enthalpy_flux"]
    inner_first, inner_last = 341, 361
    assert (
        drag.SOURCE_ANCHORS["surface_enthalpy_flux_normal"]
        == f"src/sfcphys.F:{inner_first}-{inner_last}"
    )
    window = "\n".join(lines[inner_first - 1 : inner_last])
    # The heat and moisture fluxes from the exchange coefficients and the 10 m wind.
    assert cm1_marks.has(window, "f08e85b82f006682/29")
    assert cm1_marks.has(window, "299bbad0649afb31/29")
    assert first < inner_first and inner_last < last


def _refusal_reason(selection) -> str:
    with pytest.raises(CM1Refusal) as caught:
        drag.validate_surface_drag_selection(selection)
    return caught.value.detail.reason


def test_karman_matches_the_release_constant() -> None:
    if not AUTHORITY.is_dir():
        pytest.skip("the extracted CM1 r21.1 release tree is absent")
    text = (AUTHORITY / "src/constants.F").read_text(encoding="utf-8", errors="replace")
    match = re.search(r"karman\s*=\s*([0-9.eEdD+-]+)", text)
    assert match is not None
    assert np.float32(float(match.group(1).replace("d", "e").replace("D", "e"))) == np.float32(
        drag.CM1_KARMAN_F32
    )


# ------------------------------------------------------------------ refusals


@pytest.mark.parametrize(
    ("selection", "code"),
    (
        (drag.SurfaceDragSelection(bbc=0), "CM1-SFC-BBC-RANGE"),
        (drag.SurfaceDragSelection(bbc=4), "CM1-SFC-BBC-RANGE"),
        (drag.SurfaceDragSelection(bbc=3, sfcmodel=0), "CM1-SFC-SEMI-SLIP-WITHOUT-SURFACE-LAYER"),
        (drag.SurfaceDragSelection(bbc=1, sfcmodel=1), "CM1-SFC-SURFACE-LAYER-WITHOUT-SEMI-SLIP"),
        (drag.SurfaceDragSelection(bbc=3, sfcmodel=5), "CM1-SFC-SFCMODEL-UNSUPPORTED"),
        (drag.SurfaceDragSelection(bbc=3, sfcmodel=1, cecd=0), "CM1-SFC-CECD-RANGE"),
        (drag.SurfaceDragSelection(bbc=3, sfcmodel=1, cecd=4), "CM1-SFC-CECD-RANGE"),
        (drag.SurfaceDragSelection(bbc=3, sfcmodel=1, isfcflx=2), "CM1-SFC-ISFCFLX-RANGE"),
        (
            drag.SurfaceDragSelection(bbc=2, sfcmodel=0, isfcflx=1),
            "CM1-SFC-ENTHALPY-FLUX-WITHOUT-SURFACE-LAYER",
        ),
        (
            drag.SurfaceDragSelection(bbc=3, sfcmodel=1, set_znt=2),
            "CM1-SFC-SET-SWITCH-RANGE",
        ),
        (
            drag.SurfaceDragSelection(bbc=3, sfcmodel=1, set_ust=-1),
            "CM1-SFC-SET-SWITCH-RANGE",
        ),
        (
            drag.SurfaceDragSelection(bbc=3, sfcmodel=1, set_znt=1, set_ust=1),
            "CM1-SFC-SET-ZNT-AND-UST",
        ),
        (
            drag.SurfaceDragSelection(bbc=2, sfcmodel=0, set_znt=1),
            "CM1-SFC-SET-SWITCH-WITHOUT-SURFACE-LAYER",
        ),
        (
            drag.SurfaceDragSelection(bbc=3, sfcmodel=1, set_znt=1, cnst_znt=0.0),
            "CM1-SFC-CNST-ZNT-RANGE",
        ),
        (
            drag.SurfaceDragSelection(bbc=3, sfcmodel=1, set_ust=1, cnst_ust=-0.1),
            "CM1-SFC-CNST-UST-RANGE",
        ),
        (
            drag.SurfaceDragSelection(bbc=3, sfcmodel=1, isfcflx=1, cnstce=-1e-3),
            "CM1-SFC-CNSTCE-RANGE",
        ),
        (drag.SurfaceDragSelection(bbc=3, sfcmodel=1, cnstcd=0.0), "CM1-SFC-CNSTCD-RANGE"),
        (drag.SurfaceDragSelection(bbc=3, sfcmodel=1, cnstcd=-1e-3), "CM1-SFC-CNSTCD-RANGE"),
        (
            drag.SurfaceDragSelection(
                bbc=3,
                sfcmodel=1,
                cnstcd=drag.CNSTCD_BINARY32_ROUGHNESS_CEILING,
            ),
            "CM1-SFC-CNSTCD-RANGE",
        ),
        (
            drag.SurfaceDragSelection(bbc=3, sfcmodel=1, cnstcd=1e-6),
            "CM1-SFC-CNSTCD-RANGE",
        ),
        (drag.SurfaceDragSelection(bbc=3, sfcmodel=1, u_move_m_s=math.nan), "CM1-SFC-FRAME-VELOCITY"),
        (drag.SurfaceDragSelection(bbc=3, sfcmodel=1, v_move_m_s=math.inf), "CM1-SFC-FRAME-VELOCITY"),
    ),
)
def test_every_inadmissible_lower_boundary_is_refused_by_name(selection, code) -> None:
    with pytest.raises(CM1Refusal) as caught:
        drag.validate_surface_drag_selection(selection)
    assert caught.value.detail.code == code
    assert caught.value.detail.option.startswith("closures.")
    # A refusal has to name the wrong result, not the rule.
    assert len(caught.value.detail.reason) > 60


@pytest.mark.parametrize("cnstcd", (1e-4, 1e-3, 0.01, 1.0, 2.0, 100.0, 1e10))
def test_cm1_puts_no_upper_bound_on_the_drag_coefficient(cnstcd: float) -> None:
    """``src/param.F`` bounds ``cnstcd`` nowhere, so neither does this port.

    The retired refusal claimed a ``Cd`` above one would make
    ``src/sfcphys.F:130``'s log-layer wind ratio change sign.  It cannot:
    for any ``znt > 0`` both logarithms take arguments strictly above one,
    so ``var`` is strictly positive.  This runs the port's own getcecd
    transcription and checks that, rather than asserting it.
    """

    selection = drag.SurfaceDragSelection(bbc=3, sfcmodel=1, cnstcd=cnstcd)
    assert drag.validate_surface_drag_selection(selection).cnstcd == cnstcd
    arrays = _fixture()
    drag.cpu_surface_drag_inplace(
        arrays, ni=NI, nj=NJ, nk=NK, dz=DZ, selection=selection
    )
    assert np.all(np.isfinite(arrays["znt"])) and np.all(arrays["znt"] > 0.0)
    assert np.all(np.isfinite(arrays["ust"])) and np.all(arrays["ust"] > 0.0)
    # var is not stored, so read it back out of the 10 m wind it scales.
    var = arrays["s10"] / arrays["s1"]
    assert np.all(np.isfinite(var)) and np.all(var > 0.0)


def _roughness_f32(cd):
    """``src/sfcphys.F:128`` -- ``znt = 10/(exp(karman/sqrt(cd)) - 1)``."""

    _logf, expf, sqrtf = drag._host_libm_f32()
    karman = np.float32(drag.CM1_KARMAN_F32)
    exponent = np.float32(karman / sqrtf(np, np.float32(cd)))
    return np.float32(
        np.float32(10.0) / np.float32(expf(np, exponent) - np.float32(1.0))
    )


def test_the_drag_coefficient_bounds_are_exactly_where_the_roughness_stops() -> None:
    """Each bound is the first binary32 Cd with no roughness length at all.

    ``src/sfcphys.F:128`` is ``znt = 10/(exp(karman/sqrt(cd)) - 1)``.  The
    bounds are not taste judgements about physical drag coefficients: below
    the floor the exponential overflows and ``znt`` is exactly zero, and at
    the ceiling ``karman/sqrt(cd)`` reaches ``2**-24``, the exponential
    rounds to exactly ``1.0`` and the subtraction gives zero.  One unit in
    the last place inside each bound, CM1's arithmetic still works.
    """

    floor = np.float32(drag.CNSTCD_BINARY32_ROUGHNESS_FLOOR)
    ceiling = np.float32(drag.CNSTCD_BINARY32_ROUGHNESS_CEILING)
    under = np.nextafter(floor, np.float32(0.0), dtype=np.float32)
    below = np.nextafter(ceiling, np.float32(0.0), dtype=np.float32)

    assert _roughness_f32(floor) > 0.0 and np.isfinite(_roughness_f32(floor))
    assert _roughness_f32(under) == np.float32(0.0)
    assert np.isfinite(_roughness_f32(below)) and _roughness_f32(below) > 0.0
    assert np.isinf(_roughness_f32(ceiling))

    for admitted in (floor, below):
        drag.validate_surface_drag_selection(
            drag.SurfaceDragSelection(bbc=3, sfcmodel=1, cnstcd=float(admitted))
        )
    for refused, fragment in ((under, "overflows"), (ceiling, "2**-24")):
        with pytest.raises(CM1Refusal) as caught:
            drag.validate_surface_drag_selection(
                drag.SurfaceDragSelection(bbc=3, sfcmodel=1, cnstcd=float(refused))
            )
        assert caught.value.detail.code == "CM1-SFC-CNSTCD-RANGE"
        assert fragment in caught.value.detail.reason
        # A remedy that names a number has to name one this operator accepts.
        for text in re.findall(r"[0-9]+\.[0-9]+e[+-][0-9]+", caught.value.detail.remedy):
            drag.validate_surface_drag_selection(
                drag.SurfaceDragSelection(bbc=3, sfcmodel=1, cnstcd=float(text))
            )
            assert np.isfinite(_roughness_f32(np.float32(float(text))))
            assert _roughness_f32(np.float32(float(text))) > 0.0


def test_a_non_positive_drag_coefficient_names_cm1s_actual_clamp() -> None:
    """``src/sfcphys.F:127`` clamps before the sqrt, so that is the mechanism.

    The refused path is not a not-a-number sqrt: CM1 replaces the declared
    value with ``1e-10`` first, and only then does ``exp(40000)`` overflow
    and take the roughness to zero.
    """

    with pytest.raises(CM1Refusal) as caught:
        drag.validate_surface_drag_selection(
            drag.SurfaceDragSelection(bbc=3, sfcmodel=1, cnstcd=-1e-3)
        )
    reason = caught.value.detail.reason
    assert caught.value.detail.code == "CM1-SFC-CNSTCD-RANGE"
    assert "max(1.0e-10,cnstcd)" in reason
    # the remedy it offers is admitted
    drag.validate_surface_drag_selection(
        drag.SurfaceDragSelection(bbc=3, sfcmodel=1, cnstcd=0.001)
    )
    # and the clamped value really does destroy the roughness length
    znt = _roughness_f32(np.float32(1.0e-10))
    assert znt == np.float32(0.0)
    assert not np.isfinite(np.float32(np.float32(1.0) / znt))


def test_the_free_slip_default_is_admitted_and_inactive() -> None:
    selection = drag.validate_surface_drag_selection(drag.FREE_SLIP_SELECTION)
    assert selection.bbc == drag.FREE_SLIP == 1
    assert not selection.active
    assert not selection.uses_surface_layer


def test_a_binding_cannot_disagree_with_the_stress_selection() -> None:
    with pytest.raises(CM1Refusal) as caught:
        _require_surface_binding(SgsStressSelection(bbc=3), None)
    assert caught.value.detail.code == "CM1-SGS-STRESS-SURFACE-BINDING"
    _require_surface_binding(SgsStressSelection(bbc=1), None)


def test_a_free_slip_binding_is_refused_before_it_can_launch_anything() -> None:
    with pytest.raises(ValueError, match="free-slip"):
        drag.SurfaceDragBinding(
            selection=drag.FREE_SLIP_SELECTION,
            compiled=object(),
            arrays={name: None for name in drag.SURFACE_STATE_NAMES},
            dz=DZ,
        )


def test_a_binding_without_its_surface_carriers_is_refused() -> None:
    with pytest.raises(ValueError, match="missing surface carriers"):
        drag.SurfaceDragBinding(
            selection=SEMI_SLIP, compiled=object(), arrays={"u1": None}, dz=DZ
        )


# ------------------------------------------------------ free slip is untouched


def test_free_slip_writes_nothing_at_all() -> None:
    arrays = _fixture()
    before = {name: value.copy() for name, value in arrays.items()}
    receipt = drag.cpu_surface_drag_inplace(
        arrays, ni=NI, nj=NJ, nk=NK, dz=DZ, selection=drag.FREE_SLIP_SELECTION
    )
    assert receipt == {"applied": False, "bbc": 1}
    for name, value in arrays.items():
        assert np.array_equal(
            value.view(np.uint32), before[name].view(np.uint32)
        ), name


# ------------------------------------------------------- the surface layer


def test_the_surface_wind_is_ground_relative_not_frame_relative() -> None:
    arrays = _fixture()
    drag.cpu_surface_drag_inplace(
        arrays, ni=NI, nj=NJ, nk=NK, dz=DZ, selection=SEMI_SLIP
    )
    # src/turb.F:303-304 then src/turb.F:539-541.
    for j in range(NJ):
        for i in range(NI):
            left = f32(arrays["u3d"][0, j, i] + f32(UMOVE))
            right = f32(arrays["u3d"][0, j, i + 1] + f32(UMOVE))
            south = f32(arrays["v3d"][0, j, i] + f32(VMOVE))
            north = f32(arrays["v3d"][0, j + 1, i] + f32(VMOVE))
            u1 = f32(f32(0.5) * f32(left + right))
            v1 = f32(f32(0.5) * f32(south + north))
            assert arrays["u1"][j, i] == u1
            assert arrays["v1"][j, i] == v1
            assert arrays["s1"][j, i] == sqrtf(f32(f32(u1 * u1) + f32(v1 * v1)))
            assert arrays["za"][j, i] == f32(
                arrays["zh"][0, j, i] - arrays["terrain_msl"][j, i]
            )


def test_a_frame_wind_and_a_ground_wind_give_different_drag() -> None:
    """The distinction the refusal exists to protect, measured."""

    moving = _fixture()
    fixed = _fixture()
    drag.cpu_surface_drag_inplace(moving, ni=NI, nj=NJ, nk=NK, dz=DZ, selection=SEMI_SLIP)
    drag.cpu_surface_drag_inplace(
        fixed,
        ni=NI,
        nj=NJ,
        nk=NK,
        dz=DZ,
        selection=drag.SurfaceDragSelection(
            bbc=3, sfcmodel=1, isfcflx=0, cecd=1, cnstcd=CNSTCD
        ),
    )
    assert not np.array_equal(moving["ust"], fixed["ust"])
    assert not np.array_equal(moving["t13"][0], fixed["t13"][0])


def test_getcecd_agrees_word_for_word_with_the_native_verified_operator() -> None:
    """The constant-Cd branch is cross-checked against the sealed operator.

    ``cm1core.surface_exchange_coefficients`` is the already source-exact
    ``getcecd`` transcription.  This operator re-implements only the
    ``cecd=1`` branch of it, on the device, so the two must produce the same
    binary32 words from the same surface wind.
    """

    arrays = _fixture()
    drag.cpu_surface_drag_inplace(arrays, ni=NI, nj=NJ, nk=NK, dz=DZ, selection=SEMI_SLIP)
    reference = {
        name: np.zeros(shape, np.float32)
        for name, shape in surface_exchange_shapes(NI, NJ).items()
    }
    for name in ("u1", "v1", "s1", "za"):
        reference[name][:] = arrays[name]
    reference["xland"][:] = np.float32(1.0)
    # cecd=1 overwrites znt from cnstcd (src/sfcphys.F:128); the sealed
    # operator still requires a positive incoming roughness because its
    # iterating branches read it as a first guess.
    reference["znt"][:] = np.float32(0.01)
    cpu_surface_exchange_coefficients_inplace(
        reference,
        ni=NI,
        nj=NJ,
        selection=SurfaceExchangeSelection(
            set_znt=0,
            set_ust=0,
            cecd=1,
            isfcflx=0,
            use_avg_sfc=False,
            cnst_ust_m_s=0.0,
            cnstcd=CNSTCD,
            cnstce=0.001,
            avgsfcu_m_s=0.0,
            avgsfcv_m_s=0.0,
            avgsfcs_m_s=0.0,
        ),
    )
    for name in ("cd", "znt", "ust", "u10", "v10", "s10"):
        assert np.array_equal(
            arrays[name].view(np.uint32), reference[name].view(np.uint32)
        ), name


def test_getcecd_matches_a_hand_transcription_of_sfcphys_120_136() -> None:
    arrays = _fixture()
    drag.cpu_surface_drag_inplace(arrays, ni=NI, nj=NJ, nk=NK, dz=DZ, selection=SEMI_SLIP)
    karman = f32(drag.CM1_KARMAN_F32)
    for j in range(NJ):
        for i in range(NI):
            cd = max(f32(1.0e-10), f32(CNSTCD))
            znt = f32(f32(10.0) / f32(expf(f32(karman / sqrtf(cd))) - f32(1.0)))
            rznt = f32(f32(1.0) / znt)
            reference = logf(f32(f32(f32(10.0) + znt) * rznt))
            level = logf(f32(f32(arrays["za"][j, i] + znt) * rznt))
            var = f32(reference / level)
            assert arrays["cd"][j, i] == cd
            assert arrays["znt"][j, i] == znt
            assert arrays["u10"][j, i] == f32(arrays["u1"][j, i] * var)
            assert arrays["ust"][j, i] == max(
                f32(f32(arrays["s1"][j, i] * karman) / level), f32(1.0e-6)
            )


def test_the_tke_lower_face_becomes_karman_znt_ust() -> None:
    arrays = _fixture()
    khv1 = arrays["khv"][1].copy()
    kmv1 = arrays["kmv"][1].copy()
    khh1 = arrays["khh"][1].copy()
    kmh1 = arrays["kmh"][1].copy()
    drag.cpu_surface_drag_inplace(arrays, ni=NI, nj=NJ, nk=NK, dz=DZ, selection=SEMI_SLIP)
    karman = f32(drag.CM1_KARMAN_F32)
    for j in range(NJ):
        for i in range(NI):
            kmh = f32(f32(karman * arrays["znt"][j, i]) * arrays["ust"][j, i])
            assert arrays["tkea"][0, j, i] == f32(0.0)
            assert arrays["kmh"][0, j, i] == kmh
            assert arrays["kmv"][0, j, i] == kmh
            assert arrays["khv"][0, j, i] == f32(
                kmh * f32(khv1[j, i] / f32(f32(1.0e-10) + kmv1[j, i]))
            )
            assert arrays["khh"][0, j, i] == f32(
                kmh * f32(khh1[j, i] / f32(f32(1.0e-10) + kmh1[j, i]))
            )
    assert np.all(arrays["kmh"][0] > 0.0)


# --------------------------------------------------------- the surface stress


def test_semi_slip_stress_matches_a_hand_transcription_of_turb_4596_4605() -> None:
    arrays = _fixture()
    drag.cpu_surface_drag_inplace(arrays, ni=NI, nj=NJ, nk=NK, dz=DZ, selection=SEMI_SLIP)
    ust, u1, v1, s1, rf = (arrays[name] for name in ("ust", "u1", "v1", "s1", "rf"))

    def clamp(index: int, extent: int) -> int:
        return 0 if index < 0 else (extent - 1 if index >= extent else index)

    for j in range(NJ):
        for i in range(NI + 1):
            l, r = clamp(i - 1, NI), clamp(i, NI)
            west = f32(f32(ust[j, l] * ust[j, l]) * f32(u1[j, l] / max(s1[j, l], f32(0.01))))
            east = f32(f32(ust[j, r] * ust[j, r]) * f32(u1[j, r] / max(s1[j, r], f32(0.01))))
            expected = f32(f32(f32(0.25) * f32(west + east)) * f32(rf[0, j, l] + rf[0, j, r]))
            assert arrays["t13"][0, j, i] == expected
    for j in range(NJ + 1):
        s, n = clamp(j - 1, NJ), clamp(j, NJ)
        for i in range(NI):
            lo = f32(f32(ust[s, i] * ust[s, i]) * f32(v1[s, i] / max(s1[s, i], f32(0.01))))
            hi = f32(f32(ust[n, i] * ust[n, i]) * f32(v1[n, i] / max(s1[n, i], f32(0.01))))
            expected = f32(f32(f32(0.25) * f32(lo + hi)) * f32(rf[0, s, i] + rf[0, n, i]))
            assert arrays["t23"][0, j, i] == expected


def test_no_slip_stress_matches_a_hand_transcription_of_turb_4555_4564() -> None:
    arrays = _fixture()
    # bbc=2 never runs getcecd, so the eddy viscosity at k=2 is what carries
    # the stress; leave the fixture's kmv in place and read it back.
    kmv = arrays["kmv"].copy()
    drag.cpu_surface_drag_inplace(arrays, ni=NI, nj=NJ, nk=NK, dz=DZ, selection=NO_SLIP)
    rdz = f32(f32(1.0) / f32(DZ))
    mf, rf = arrays["mf"], arrays["rf"]

    def clamp(index: int, extent: int) -> int:
        return 0 if index < 0 else (extent - 1 if index >= extent else index)

    for j in range(NJ):
        for i in range(NI + 1):
            l, r = clamp(i - 1, NI), clamp(i, NI)
            term = f32(f32(2.0) * f32(arrays["u3d"][0, j, i] + f32(UMOVE)))
            term = f32(term * rdz)
            term = f32(term * f32(0.5))
            term = f32(term * f32(mf[0, j, l] + mf[0, j, r]))
            term = f32(term * f32(0.5))
            term = f32(term * f32(rf[0, j, l] + rf[0, j, r]))
            term = f32(term * f32(0.5))
            term = f32(term * f32(kmv[1, j, l] + kmv[1, j, r]))
            assert arrays["t13"][0, j, i] == term


def test_no_slip_leaves_the_surface_layer_alone() -> None:
    """``getcecd`` runs only for ``sfcmodel>=1`` (``src/turb.F:614-616``)."""

    arrays = _fixture()
    drag.cpu_surface_drag_inplace(arrays, ni=NI, nj=NJ, nk=NK, dz=DZ, selection=NO_SLIP)
    for name in ("cd", "znt", "ust", "u10", "v10", "s10"):
        assert np.all(arrays[name] == 0.0), name
    # With znt and ust zero, tkebc's lower face is zero: that is CM1's own
    # arithmetic for bbc=2 without a surface layer, not a special case.
    assert np.all(arrays["kmh"][0] == 0.0)


def test_the_stress_opposes_the_surface_wind_and_free_slip_exerts_none() -> None:
    """The reason this operator exists, stated as a measurement.

    ``src/turbtend.F:1417`` makes the explicit lowest-level stress
    divergence ``(t13(k=2) - t13(k=1)) * rdz * ...``.  With free slip the two
    are equal and the surface exerts no force.  With semi slip ``t13(k=1)``
    carries the sign of the surface wind, so the divergence opposes it.
    """

    arrays = _fixture()
    free = _fixture()
    free["t13"][0] = free["t13"][1]
    free["t23"][0] = free["t23"][1]
    assert np.array_equal(free["t13"][1] - free["t13"][0], np.zeros_like(free["t13"][0]))

    drag.cpu_surface_drag_inplace(arrays, ni=NI, nj=NJ, nk=NK, dz=DZ, selection=SEMI_SLIP)
    interior = 1
    signed = 0
    for j in range(NJ):
        for i in range(interior, NI + 1 - interior):
            u1 = 0.5 * (arrays["u1"][j, min(i, NI - 1)] + arrays["u1"][j, max(i - 1, 0)])
            if abs(u1) < 0.5:
                continue
            signed += 1
            # t13(k=1) has the sign of the surface wind, so
            # turbz = (t13(2) - t13(1)) * positive carries the opposite sign.
            assert math.copysign(1.0, float(arrays["t13"][0, j, i])) == math.copysign(
                1.0, float(u1)
            )
    assert signed > 0


def test_stau_matches_turb_4692_4700() -> None:
    arrays = _fixture()
    drag.cpu_surface_drag_inplace(arrays, ni=NI, nj=NJ, nk=NK, dz=DZ, selection=SEMI_SLIP)
    for j in range(NJ):
        for i in range(NI):
            x = f32(f32(0.5) * f32(arrays["t13"][0, j, i] + arrays["t13"][0, j, i + 1]))
            y = f32(f32(0.5) * f32(arrays["t23"][0, j, i] + arrays["t23"][0, j + 1, i]))
            assert arrays["stau"][j, i] == f32(
                sqrtf(f32(f32(x * x) + f32(y * y))) / arrays["rf"][0, j, i]
            )
    assert np.all(arrays["stau"] >= 0.0)


def test_the_surface_dissipation_rate_matches_solve1_956_957() -> None:
    arrays = _fixture()
    drag.cpu_surface_drag_inplace(arrays, ni=NI, nj=NJ, nk=NK, dz=DZ, selection=SEMI_SLIP)
    karman = f32(drag.CM1_KARMAN_F32)
    for j in range(NJ):
        for i in range(NI):
            dz1 = f32(arrays["zf"][1, j, i] - arrays["zf"][0, j, i])
            znt = arrays["znt"][j, i]
            ust = arrays["ust"][j, i]
            cubed = f32(f32(ust * ust) * ust)
            expected = f32(
                f32(cubed * logf(f32(f32(dz1 + znt) / znt))) / f32(karman * dz1)
            )
            assert arrays["epsd_sfc"][j, i] == expected
    assert np.all(arrays["epsd_sfc"] > 0.0)


def test_no_slip_publishes_no_surface_dissipation_rate() -> None:
    """``src/solve1.F:932-935``: only ``bbc=3`` moves ``k1``."""

    arrays = _fixture()
    drag.cpu_surface_drag_inplace(arrays, ni=NI, nj=NJ, nk=NK, dz=DZ, selection=NO_SLIP)
    assert np.all(arrays["epsd_sfc"] == 0.0)


# ------------------------------------------------- surface dissipative heating


def test_surface_heating_matches_a_hand_transcription_of_solve1_950_962() -> None:
    rng = np.random.default_rng(3)
    shape = (NK, NJ, NI)

    def volume(low: float, high: float) -> np.ndarray:
        return rng.uniform(low, high, shape).astype(np.float32)

    carriers = SurfaceDissipationCarriers(
        thten1=np.zeros(shape, np.float32),
        epsd_surface=rng.uniform(1e-4, 2.0, (NJ, NI)).astype(np.float32),
        pi0=volume(0.7, 1.0),
        ppi=volume(-0.02, 0.02),
        qv=volume(0.0, 0.02),
        qc=volume(0.0, 0.002),
        qr=volume(0.0, 0.002),
        qi=volume(0.0, 0.001),
        qs=volume(0.0, 0.001),
        qg=volume(0.0, 0.001),
    )
    seed = rng.uniform(-1e-3, 1e-3, shape).astype(np.float32)
    carriers.thten1[:] = seed
    cpu_surface_dissipative_heating_inplace(carriers, nx=NI, ny=NJ, nz=NK)

    # src/constants.F:127-130, in CM1's own default-REAL statement order:
    # cv = cp-rd, cvv = cpv-rv, cpdcv = cp/cv, each rounded to binary32.
    cv = f32(f32(1005.7) - f32(287.04))
    cvv = f32(f32(1870.0) - f32(461.5))
    cpl = f32(4190.0)
    cpi = f32(2106.0)
    cpdcv = f32(f32(1005.7) / cv)
    for j in range(NJ):
        for i in range(NI):
            ql = f32(carriers.qc[0, j, i] + carriers.qr[0, j, i])
            qice = f32(f32(carriers.qi[0, j, i] + carriers.qs[0, j, i]) + carriers.qg[0, j, i])
            cvm = f32(carriers.qv[0, j, i] * cvv)
            cvm = f32(cvm + cv)
            cvm = f32(cvm + f32(ql * cpl))
            cvm = f32(cvm + f32(qice * cpi))
            work = f32(carriers.pi0[0, j, i] + carriers.ppi[0, j, i])
            cvm = f32(cvm * cpdcv)
            cvm = f32(cvm * work)
            dheat = f32(carriers.epsd_surface[j, i] / cvm)
            assert carriers.thten1[0, j, i] == f32(seed[0, j, i] + dheat)
    # Only the lowest level is touched; k=2..nk stay with the interior
    # consumer (src/solve1.F:932-935).
    assert np.array_equal(carriers.thten1[1:], seed[1:])
    assert np.all(carriers.thten1[0] > seed[0])


def test_surface_heating_refuses_an_unported_equation_set() -> None:
    shape = (NK, NJ, NI)
    zeros = lambda: np.zeros(shape, np.float32)
    carriers = SurfaceDissipationCarriers(
        thten1=zeros(),
        epsd_surface=np.zeros((NJ, NI), np.float32),
        pi0=zeros(), ppi=zeros(), qv=zeros(), qc=zeros(),
        qr=zeros(), qi=zeros(), qs=zeros(), qg=zeros(),
    )
    with pytest.raises(CM1Refusal) as caught:
        cpu_surface_dissipative_heating_inplace(
            carriers, nx=NI, ny=NJ, nz=NK, equation_set=3
        )
    assert caught.value.detail.code == "CM1-SFC-DISS-EQTSET"


def test_surface_heating_refuses_a_mis_shaped_surface_rate() -> None:
    shape = (NK, NJ, NI)
    zeros = lambda: np.zeros(shape, np.float32)
    carriers = SurfaceDissipationCarriers(
        thten1=zeros(),
        epsd_surface=np.zeros((NJ + 1, NI), np.float32),
        pi0=zeros(), ppi=zeros(), qv=zeros(), qc=zeros(),
        qr=zeros(), qi=zeros(), qs=zeros(), qg=zeros(),
    )
    with pytest.raises(CM1Refusal) as caught:
        cpu_surface_dissipative_heating_inplace(carriers, nx=NI, ny=NJ, nz=NK)
    assert caught.value.detail.code == "CM1-SFC-DISS-SHAPE"


# ------------------------------------------------------------ config surface


def test_free_slip_remains_the_default_and_the_canonical_document_is_unchanged() -> None:
    config = load_config(ROOT / "examples" / "cartesian-supercell-translating-frame.toml")
    assert config.closures.lower_boundary is None
    assert config.closures.bbc == 1
    assert config.closures.sfcmodel == 0
    assert config.closures.isfcflx == 0
    assert FREE_SLIP_LOWER_BOUNDARY.bbc == 1
    closures = config.canonical_dict()["closures"]
    assert set(closures) == {
        "sgs_model",
        "vertical_implicit_diffusion",
        "upper_damping",
        "upper_damping_depth_m",
        "upper_damping_rate_s_inv",
        "moving_frame",
        "u_move_m_s",
        "v_move_m_s",
    }


# Every configuration document this port shipped before the lower boundary
# became declarable, with the canonical-document sha256 it had at 49319e5 --
# the commit immediately before the surface-drag branch.  A free-slip
# document projects no lower-boundary key, so these are unchanged, and any
# future change to the grammar that touched them would move one of these.
PRE_EXISTING_DECK_SHA256 = {
    "cartesian-supercell-attempt.toml": "f325e5e78bffd1d57052b1c1dd4e085b1eb91f619070bb83a3d85b02b22b333b",
    "cartesian-supercell-translating-frame.toml": "77288cd92837908d70baedebc1aa914c2dc6a5bdb1f90d3d8255cfed15a545ca",
    "cartesian-uniform-grid.toml": "60b552c0a1cccd9174bc839d15b1258aac7f97999f8fd206be74a51975f1db48",
    "cartesian-wk-analytic-sounding-shear.toml": "e6ca95ed63f9555381a4ec8d04b93cf4cfff324119b0fb0df1ed9d14e696cee7",
    "dry-psolver2-bootstrap.toml": "0394dea1f77722e54fe6b1b5a07fe56037cda84800ef870b9a269566e6043340",
    "official-supercell-native-context.toml": "33761ba3ef1e35ce8dd6fa3fa126c4ed2437bdbe49d3b9e73adb2fa2bcf92b8c",
    "official-supercell-product.toml": "b3dc296531782eaa166eb26cec30fa030b7405485af1cf2e3ff1af23351e4038",
    "real-atmosphere-100m-product.toml": "07c7dde701491ce71a70cec8c56de2cd50ed84574cb098250815d76eaa0c9c00",
}


@pytest.mark.parametrize("name", sorted(PRE_EXISTING_DECK_SHA256))
def test_every_pre_existing_deck_is_byte_identical(name: str) -> None:
    """Adding the lower boundary must not move a sealed configuration hash."""

    config = load_config(ROOT / "examples" / name)
    assert config.sha256 == PRE_EXISTING_DECK_SHA256[name]
    # and none of them declares the table, so none of them can drift into it
    assert config.closures.lower_boundary is None
    assert config.closures.bbc == 1
    assert set(config.canonical_dict()["closures"]).isdisjoint(
        {"bbc", "sfcmodel", "isfcflx", "cecd", "cnstcd"}
    )


# The free-slip closure allocation plan, canonical JSON, at 49319e5.  The
# surface group, the k=2..nk idiss slice and every byte of workspace pricing
# are conditional on bbc, so a plan that moved would mean the free-slip run
# stopped being the run it was.
# Moved once, by four bytes, when the closure phase began folding CM1's ksmax
# (misclibs.F:2846-2866) into a device scalar beside the invalid-operand
# counter.  The eddy coefficients exist only inside this phase's tile
# workspace, so it is the one place the diffusion number can be measured.
FREE_SLIP_ALLOCATION_PLAN_SHA256 = {
    (31, 23, 12, 500.0, 500.0, 250.0, 11, 8): (
        "e0d0aec92db1cd876e75be677df1bdc44ee164d221dcbe5ca3203f9808b02c3c"
    ),
    (400, 400, 237, 75.0, 75.0, 75.0, 64, 64): (
        "825ca38ea88ac176443e8b85ba71932bad603b53d9c1977454effe22891ebfaf"
    ),
}


@pytest.mark.parametrize("geometry", sorted(FREE_SLIP_ALLOCATION_PLAN_SHA256))
def test_the_free_slip_allocation_plan_is_byte_identical(geometry) -> None:
    import json

    from cm1core.cuda_cartesian_closure import cartesian_closure_allocation_plan
    from cm1core.grid_geometry import CartesianGridGeometry
    from cm1core.gm1_runtime import RuntimeConfig

    nx, ny, nz, dx, dy, dz, tile_nx, tile_ny = geometry
    grid = CartesianGridGeometry(nx=nx, ny=ny, nz=nz, dx_m=dx, dy_m=dy, dz_m=dz)
    plan = cartesian_closure_allocation_plan(
        RuntimeConfig(grid, tile_nx=tile_nx, tile_ny=tile_ny)
    )
    blob = json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()
    assert (
        hashlib.sha256(blob).hexdigest()
        == FREE_SLIP_ALLOCATION_PLAN_SHA256[geometry]
    )
    assert "surface" not in plan["held_groups"]
    assert plan["held_groups"]["idiss"]["epsd"][0] == nz


def test_the_shipped_semi_slip_deck_binds_cm1s_surface_layer() -> None:
    from cm1core.cartesian_door import build_runtime_config

    config = load_config(ROOT / "examples" / "cartesian-supercell-surface-drag.toml")
    assert (config.closures.bbc, config.closures.sfcmodel) == (3, 1)
    assert (config.closures.isfcflx, config.closures.cecd) == (0, 1)
    assert config.boundaries.bottom == config.closures.bbc
    closures = config.canonical_dict()["closures"]
    for name in ("bbc", "sfcmodel", "isfcflx", "cecd", "cnstcd"):
        assert name in closures
    runtime = build_runtime_config(config)
    assert runtime.bbc == 3
    selection = runtime.surface_drag_selection
    assert selection.active and selection.uses_surface_layer
    assert selection.u_move_m_s == config.closures.u_move_m_s
    record = runtime.record()
    assert record["bbc"] == 3 and record["cnstcd"] == config.closures.cnstcd


def test_the_closure_allocates_and_launches_nothing_for_a_free_slip_bottom() -> None:
    from cm1core.cuda_cartesian_closure import cartesian_closure_allocation_plan
    from cm1core.grid_geometry import CartesianGridGeometry
    from cm1core.gm1_runtime import RuntimeConfig

    grid = CartesianGridGeometry(nx=31, ny=23, nz=12, dx_m=500.0, dy_m=500.0, dz_m=250.0)
    free = cartesian_closure_allocation_plan(
        RuntimeConfig(grid, tile_nx=11, tile_ny=8)
    )
    assert "surface" not in free["held_groups"]
    drag_plan = cartesian_closure_allocation_plan(
        RuntimeConfig(grid, tile_nx=11, tile_ny=8, bbc=3, sfcmodel=1, cnstcd=CNSTCD)
    )
    assert set(drag_plan["held_groups"]["surface"]) == set(
        drag.SURFACE_STATE_NAMES
    ) | set(drag.SURFACE_PERSISTENT_NAMES)
    flux_plan = cartesian_closure_allocation_plan(
        RuntimeConfig(
            grid,
            tile_nx=11,
            tile_ny=8,
            bbc=3,
            sfcmodel=1,
            isfcflx=1,
            cnstcd=CNSTCD,
        )
    )
    assert set(flux_plan["held_groups"]["surface"]) == set(
        drag.SURFACE_STATE_NAMES
    ) | set(drag.SURFACE_PERSISTENT_NAMES) | set(drag.SURFACE_FLUX_STATE_NAMES)
    assert (
        drag_plan["held_phase_payload_bytes"] > free["held_phase_payload_bytes"]
    )
    # src/solve1.F:932-935: bbc=3 starts the interior heating at k=2.
    assert drag_plan["held_groups"]["idiss"]["epsd"][0] == grid.nz - 1
    assert free["held_groups"]["idiss"]["epsd"][0] == grid.nz


# ------------------------------------------------------------------ GPU gated


# ------------------------------------------------------- the launch sequence


class _RecordingKernel:
    def __init__(self, index: int, log: list) -> None:
        self.index = index
        self.log = log

    def __call__(self, grid, block, args, *, stream=None) -> None:
        self.log.append((drag.SURFACE_DRAG_KERNEL_NAMES[self.index], grid, block, args))


class _Compiled:
    source_sha256 = drag.CUDA_SURFACE_DRAG_SOURCE_SHA256
    target_arch = "86"

    def __init__(self, log: list) -> None:
        self.kernels = tuple(
            _RecordingKernel(index, log)
            for index in range(len(drag.SURFACE_DRAG_KERNEL_NAMES))
        )


class _Stream:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


@pytest.fixture
def cupy_shim(monkeypatch):
    """A binary32/uint32 stand-in so the launch order can be read on a CPU."""

    shim = type(
        "cupy",
        (),
        {"uint32": np.uint32, "float32": np.float32, "ndarray": np.ndarray},
    )
    monkeypatch.setitem(sys.modules, "cupy", shim)
    return shim


def _binding(selection, arrays, log) -> drag.SurfaceDragBinding:
    return drag.SurfaceDragBinding(
        selection=selection,
        compiled=_Compiled(log),
        arrays={name: arrays[name] for name in drag.SURFACE_STATE_NAMES},
        dz=DZ,
    )


def test_the_surface_layer_enqueues_getsfc_then_getcecd(cupy_shim) -> None:
    """``src/turb.F:539-543``, then ``:614-616``.

    ``tkebc`` is not part of this call: ``src/turb.F:1378`` runs it after
    ``tkekm`` (``src/turb.F:1351``), which is what writes the eddy
    coefficients ``tkebc`` reads at ``k=2``.
    """

    arrays = _fixture()
    log: list = []
    drag.enqueue_surface_layer(
        _binding(SEMI_SLIP, arrays, log),
        arrays,
        ni=NI,
        nj=NJ,
        nk=NK,
        stream=_Stream(),
    )
    assert [entry[0] for entry in log] == [
        "cm1_sfc_ground_relative_wind_v1",
        "cm1_sfc_getcecd_cnstcd_v1",
    ]
    # The frame velocity reaches the wind kernel by value.
    assert log[0][3][-2:] == (np.float32(UMOVE), np.float32(VMOVE))
    assert log[1][3][-1] == np.float32(CNSTCD)


def test_no_slip_skips_getcecd_entirely(cupy_shim) -> None:
    """``src/turb.F:614``: ``getcecd`` runs only for ``sfcmodel>=1``."""

    arrays = _fixture()
    log: list = []
    drag.enqueue_surface_layer(
        _binding(NO_SLIP, arrays, log), arrays, ni=NI, nj=NJ, nk=NK, stream=_Stream()
    )
    assert [entry[0] for entry in log] == ["cm1_sfc_ground_relative_wind_v1"]


@pytest.mark.parametrize("selection", (SEMI_SLIP, NO_SLIP), ids=("semi-slip", "no-slip"))
def test_the_tke_lower_face_is_its_own_single_launch(cupy_shim, selection) -> None:
    """``src/turb.F:1378`` is one ``tkebc`` call per step, after ``tkekm``."""

    arrays = _fixture()
    log: list = []
    drag.enqueue_surface_tke_face(
        _binding(selection, arrays, log),
        arrays,
        ni=NI,
        nj=NJ,
        nk=NK,
        stream=_Stream(),
    )
    assert [entry[0] for entry in log] == ["cm1_sfc_tkebc_lower_v1"]


@pytest.mark.parametrize(
    ("selection", "expected"),
    (
        (SEMI_SLIP, ["cm1_sfc_stress_semi_slip_v1", "cm1_sfc_stau_v1"]),
        (NO_SLIP, ["cm1_sfc_stress_no_slip_v1", "cm1_sfc_stau_v1"]),
    ),
    ids=("semi-slip", "no-slip"),
)
def test_the_stress_enqueue_follows_cm1s_own_order(cupy_shim, selection, expected) -> None:
    """``sfcstress`` (``src/turb.F:4417``) then ``stau``, and nothing else.

    ``sfcstress`` computes no dissipation rate; CM1 forms that inside
    ``solve1`` at the point of use.
    """

    arrays = _fixture()
    log: list = []
    drag.enqueue_surface_stress(
        _binding(selection, arrays, log),
        arrays,
        ni=NI,
        nj=NJ,
        nk=NK,
        stream=_Stream(),
    )
    assert [entry[0] for entry in log] == expected


@pytest.mark.parametrize(
    ("selection", "expected"),
    (
        (SEMI_SLIP, ["cm1_sfc_dissipation_v1"]),
        (NO_SLIP, []),
    ),
    ids=("semi-slip", "no-slip"),
)
def test_the_surface_dissipation_rate_is_a_solve1_launch(
    cupy_shim, selection, expected
) -> None:
    """``src/solve1.F:932-935``: only ``bbc=3`` replaces the k=1 average."""

    arrays = _fixture()
    log: list = []
    binding = drag.SurfaceDragBinding(
        selection=selection,
        compiled=_Compiled(log),
        arrays={
            **{name: arrays[name] for name in drag.SURFACE_STATE_NAMES},
            "zf": arrays["zf"],
        },
        dz=DZ,
    )
    drag.enqueue_surface_dissipation(
        binding, {}, ni=NI, nj=NJ, nk=NK, stream=_Stream()
    )
    assert [entry[0] for entry in log] == expected


@pytest.mark.parametrize("selection", (SEMI_SLIP, NO_SLIP), ids=("semi-slip", "no-slip"))
def test_the_closure_supplies_every_carrier_the_stress_enqueue_indexes(
    cupy_shim, selection
) -> None:
    """The exact carrier split the Cartesian closure uses, launched for real.

    The closure owns twelve surface words plus ``zf`` and hands the momentum,
    density, metric and eddy-coefficient carriers straight through from the
    tile it already holds.  If that split were short one array the kernels
    would only find out on a device.
    """

    arrays = _fixture()
    log: list = []
    owned = {name: arrays[name] for name in drag.SURFACE_STATE_NAMES}
    owned["zf"] = arrays["zf"]
    binding = drag.SurfaceDragBinding(
        selection=selection, compiled=_Compiled(log), arrays=owned, dz=DZ
    )
    passed_through = {
        name: arrays[name]
        for name in ("t13", "t23", "rf", "mf", "kmv", "u3d", "v3d")
    }
    drag.enqueue_surface_stress(
        binding, passed_through, ni=NI, nj=NJ, nk=NK, stream=_Stream()
    )
    assert log
    layer = {
        name: arrays[name]
        for name in ("u3d", "v3d", "zh", "terrain_msl", "tkea", "kmh", "kmv", "khh", "khv", "zf")
    }
    log.clear()
    drag.enqueue_surface_layer(
        binding, layer, ni=NI, nj=NJ, nk=NK, stream=_Stream()
    )
    assert log


def test_a_mis_shaped_carrier_is_refused_before_any_kernel_launches(cupy_shim) -> None:
    arrays = _fixture()
    log: list = []
    binding = _binding(SEMI_SLIP, arrays, log)
    broken = dict(arrays)
    broken["rf"] = np.zeros((NK + 1, NJ, NI + 1), np.float32)
    with pytest.raises(ValueError, match="rf must have shape"):
        drag.enqueue_surface_stress(
            binding, broken, ni=NI, nj=NJ, nk=NK, stream=_Stream()
        )
    assert log == []
    broken = dict(arrays)
    broken["zh"] = arrays["zh"].astype(np.float64)
    with pytest.raises(TypeError, match="zh must be binary32"):
        drag.enqueue_surface_layer(
            binding, broken, ni=NI, nj=NJ, nk=NK, stream=_Stream()
        )
    assert log == []


# ------------------------------------------------- the compiled image, no GPU

NVCC = next(
    (
        path
        for path in (
            Path("/usr/local/cuda/bin/nvcc"),
            Path("/usr/local/cuda-12/bin/nvcc"),
        )
        if path.is_file()
    ),
    None,
)


@pytest.mark.skipif(NVCC is None, reason="no local CUDA toolkit to compile with")
def test_the_kernel_image_compiles_and_keeps_every_rounding_mode(tmp_path) -> None:
    """Compiling needs a toolkit, not a device, so this runs without a GPU.

    It also proves the arithmetic the host reference was written from
    survives into PTX: every add, subtract, multiply, divide and square root
    this operator writes is a separate round-to-nearest instruction, never a
    contracted fused multiply-add, which is what makes a statement-for-
    statement CM1 transcription meaningful.
    """

    source = tmp_path / "surface_drag.cu"
    source.write_text(drag.build_cuda_surface_drag_source(), encoding="utf-8")
    ptx = tmp_path / "surface_drag.ptx"
    result = subprocess.run(
        [
            str(NVCC),
            "-std=c++14",
            "--fmad=false",
            "--ftz=false",
            "--prec-div=true",
            "--prec-sqrt=true",
            "-arch=compute_86",
            "-ptx",
            str(source),
            "-o",
            str(ptx),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    text = ptx.read_text(encoding="utf-8")
    entries = set(re.findall(r"\.entry\s+(\w+)", text))
    assert entries == set(drag.SURFACE_DRAG_KERNEL_NAMES)
    for opcode in ("add.rn.f32", "mul.rn.f32", "div.rn.f32", "sqrt.rn.f32"):
        assert opcode in text, opcode
    # The only fused multiply-adds admitted are the ones inside CUDA's own
    # expf/logf implementations, which are whole functions, not contractions
    # of this operator's statements.  Outside those two kernels there are
    # none at all.
    body = {}
    current = None
    for line in text.splitlines():
        match = re.search(r"\.entry\s+(\w+)", line)
        if match:
            current = match.group(1)
            body[current] = []
        elif current is not None:
            body[current].append(line)
    transcendental = {
        "cm1_sfc_getcecd_cnstcd_v1",
        "cm1_sfc_dissipation_v1",
        "cm1_sfc_getcecd_set_znt_v1",
        "cm1_sfc_getcecd_set_ust_v1",
        "cm1_sfc_getcecd_water_v1",
        "cm1_sfc_flux_v1",
    }
    for name, lines in body.items():
        if name in transcendental:
            continue
        assert not any("fma.rn.f32" in line for line in lines), name


# ------------------------------------------------------------------ GPU gated


@pytest.mark.skipif(
    os.environ.get("CM1_CUDA_TESTS") != "1",
    reason="requires an explicitly coordinated GPU-mutex run",
)
def test_cuda_no_slip_matches_the_host_reference_raw_bits() -> None:
    """``bbc=2`` evaluates no transcendental, so device and host agree exactly."""

    import cupy

    expected = _fixture()
    drag.cpu_surface_drag_inplace(
        expected, ni=NI, nj=NJ, nk=NK, dz=DZ, selection=NO_SLIP
    )
    device = {name: cupy.asarray(value) for name, value in _fixture().items()}
    compiled = drag.compile_cuda_surface_drag()
    drag.launch_cuda_surface_layer(
        device, ni=NI, nj=NJ, nk=NK, selection=NO_SLIP, compiled=compiled
    ).ready_event.synchronize()
    drag.launch_cuda_surface_stress(
        device, ni=NI, nj=NJ, nk=NK, dz=DZ, selection=NO_SLIP, compiled=compiled
    ).ready_event.synchronize()
    for name in drag.surface_drag_shapes(NI, NJ, NK):
        actual = cupy.asnumpy(device[name])
        assert np.array_equal(
            actual.view(np.uint32), expected[name].view(np.uint32)
        ), name


@pytest.mark.skipif(
    os.environ.get("CM1_CUDA_TESTS") != "1",
    reason="requires an explicitly coordinated GPU-mutex run",
)
def test_cuda_semi_slip_is_exact_where_no_transcendental_is_evaluated() -> None:
    """``bbc=3`` runs ``getcecd``, whose roughness length is an exponential.

    ``src/sfcphys.F:128`` evaluates ``exp`` and ``src/sfcphys.F:130-134``
    evaluates ``alog``.  CUDA's ``expf``/``logf`` and the platform libm's are
    separately rounded implementations of the same functions, so the words
    downstream of them agree to within a few units in the last place rather
    than bit for bit.  Everything upstream of them -- the ground-relative
    wind, the lowest-level wind speed, its height and the drag coefficient
    itself -- evaluates only round-to-nearest primitives and ``sqrt``, and is
    compared bit for bit.
    """

    import cupy

    expected = _fixture()
    drag.cpu_surface_drag_inplace(
        expected, ni=NI, nj=NJ, nk=NK, dz=DZ, selection=SEMI_SLIP
    )
    device = {name: cupy.asarray(value) for name, value in _fixture().items()}
    compiled = drag.compile_cuda_surface_drag()
    drag.launch_cuda_surface_layer(
        device, ni=NI, nj=NJ, nk=NK, selection=SEMI_SLIP, compiled=compiled
    ).ready_event.synchronize()
    drag.launch_cuda_surface_stress(
        device, ni=NI, nj=NJ, nk=NK, dz=DZ, selection=SEMI_SLIP, compiled=compiled
    ).ready_event.synchronize()
    # src/solve1.F:955-957: the surface dissipation rate belongs to solve1,
    # not to sfcstress, and is launched where the heating reads it.
    drag.launch_cuda_surface_dissipation(
        device, ni=NI, nj=NJ, nk=NK, selection=SEMI_SLIP, compiled=compiled
    ).ready_event.synchronize()
    exact = ("u1", "v1", "s1", "za", "cd")
    for name in exact:
        actual = cupy.asnumpy(device[name])
        assert np.array_equal(
            actual.view(np.uint32), expected[name].view(np.uint32)
        ), name
    for name in ("znt", "ust", "u10", "v10", "s10", "t13", "t23", "stau", "epsd_sfc"):
        actual = cupy.asnumpy(device[name])
        reference = expected[name]
        scale = np.maximum(np.abs(reference), np.float32(1.0e-30))
        assert np.all(np.abs(actual - reference) <= 1.0e-6 * scale), name
    # The TKE lower face is karman*znt*ust, so it inherits the same bound.
    for name in ("kmh", "kmv", "khh", "khv"):
        actual = cupy.asnumpy(device[name])[0]
        reference = expected[name][0]
        scale = np.maximum(np.abs(reference), np.float32(1.0e-30))
        assert np.all(np.abs(actual - reference) <= 1.0e-6 * scale), name
        assert np.array_equal(
            cupy.asnumpy(device[name])[1:].view(np.uint32),
            expected[name][1:].view(np.uint32),
        ), name


@pytest.mark.skipif(
    os.environ.get("CM1_CUDA_TESTS") != "1",
    reason="requires an explicitly coordinated GPU-mutex run",
)
def test_cuda_image_declares_every_entry_point() -> None:
    compiled = drag.compile_cuda_surface_drag()
    text = compiled.ptx.decode("utf-8", errors="replace")
    for name in drag.SURFACE_DRAG_KERNEL_NAMES:
        assert re.search(rf"\.entry\s+{name}(?:\s|\()", text), name
    assert compiled.cubin[:7] == b"\x7fELF\x02\x01\x01"
    assert compiled.source_sha256 == drag.CUDA_SURFACE_DRAG_SOURCE_SHA256


@pytest.mark.skipif(
    os.environ.get("CM1_CUDA_TESTS") != "1",
    reason="requires an explicitly coordinated GPU-mutex run",
)
def test_cuda_surface_heating_matches_the_host_reference_raw_bits() -> None:
    import cupy

    from cm1core.eqtset2_surface_dissipation import apply_surface_dissipative_heating

    rng = np.random.default_rng(19)
    shape = (NK, NJ, NI)
    host = {
        name: rng.uniform(0.0, 0.02, shape).astype(np.float32)
        for name in ("qv", "qc", "qr", "qi", "qs", "qg")
    }
    host["pi0"] = rng.uniform(0.7, 1.0, shape).astype(np.float32)
    host["ppi"] = rng.uniform(-0.02, 0.02, shape).astype(np.float32)
    host["thten1"] = rng.uniform(-1e-3, 1e-3, shape).astype(np.float32)
    host["epsd_surface"] = rng.uniform(1e-4, 2.0, (NJ, NI)).astype(np.float32)
    expected = SurfaceDissipationCarriers(**{k: v.copy() for k, v in host.items()})
    cpu_surface_dissipative_heating_inplace(expected, nx=NI, ny=NJ, nz=NK)
    device = SurfaceDissipationCarriers(
        **{k: cupy.asarray(v.copy()) for k, v in host.items()}
    )
    apply_surface_dissipative_heating(device, nx=NI, ny=NJ, nz=NK)["ready_event"].synchronize()
    assert np.array_equal(
        cupy.asnumpy(device.thten1).view(np.uint32), expected.thten1.view(np.uint32)
    )
