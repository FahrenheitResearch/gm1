"""From configuration text to a changed surface, for every CM1 branch.

Every test here starts at a TOML document and ends at a number that is
different because of it.  ``getcecd`` has four branches
(``src/sfcphys.F:81-195``), ``init_surface`` has four ``initsfc`` patterns
(``src/init_surface.F:67-145``) and the surface enthalpy flux is a fifth
subsystem (``src/sfcphys.F:341-361`` into ``src/turbtend.F:595``); before
this work none of them could be selected at all.

The arithmetic is checked two ways.  The device transcription in
:mod:`cm1core.cuda_surface_drag` is compared raw-word against
:mod:`cm1core.surface_exchange_coefficients`, which is a separate, earlier
and independently written CPU port of the same ``getcecd`` source -- so a
mistake has to be made twice, the same way, to pass.  The enthalpy flux is
checked against its closed form, which is short enough to state exactly.
"""

from __future__ import annotations

import copy
import hashlib
import math
import sys
import tomllib
from pathlib import Path

import numpy as np
import pytest

from cm1core import cuda_surface_drag as drag
from cm1core.config import parse_mapping
from cm1core.errors import CM1Refusal
from cm1core.surface_fields import SurfaceFieldSelection, build_surface_fields
from sealed_evidence import assert_source_pin_or_owed
from cm1core.surface_landuse import (
    CM1_LANDUSE_TBL_SHA256,
    cpu_getlanduse_inplace,
    landuse_table,
)


ROOT = Path(__file__).resolve().parents[1]
AUTHORITY = ROOT / "upstream" / "cm1-21.1" / "pristine"
DECK = ROOT / "examples" / "cartesian-supercell-surface-drag.toml"


def _document(**closures) -> dict:
    raw = tomllib.loads(DECK.read_text(encoding="utf-8"))
    document = copy.deepcopy(raw)
    surface = document.pop("_surface", None)
    document["closures"].update(closures)
    if surface is not None:
        document["surface"] = surface
    return document


def _with_surface(document: dict, **surface) -> dict:
    document = copy.deepcopy(document)
    document["surface"] = {
        "initsfc": 1,
        "tsk0_k": 299.28,
        "tmn0_k": 297.28,
        "xland0": 2.0,
        "lu0": 16,
        "season": 1,
        "oceanmodel": 1,
        **surface,
    }
    return document


def _surface_layer_group(**values) -> dict:
    return {
        "cnstce": 0.001,
        "set_znt": 0,
        "cnst_znt": 0.16,
        "set_ust": 0,
        "cnst_ust": 0.25,
        **values,
    }


# ------------------------------------------------- configuration reaches it


@pytest.mark.parametrize(
    ("overrides", "branch"),
    (
        ({}, drag.GETCECD_CONSTANT_CD),
        (_surface_layer_group(set_znt=1), drag.GETCECD_SET_ZNT),
        (_surface_layer_group(set_ust=1), drag.GETCECD_SET_UST),
        ({"cecd": 2, **_surface_layer_group()}, drag.GETCECD_WIND_DEPENDENT),
        ({"cecd": 3, **_surface_layer_group()}, drag.GETCECD_WIND_DEPENDENT),
    ),
)
def test_every_getcecd_branch_is_reachable_from_a_document(overrides, branch) -> None:
    """``src/sfcphys.F:81-195``: four branches, all selectable by key."""

    config = parse_mapping(_document(**overrides))
    selection = drag.surface_drag_selection_from_config(config)
    assert selection.getcecd_branch == branch


def test_the_surface_enthalpy_flux_is_reachable_from_a_document() -> None:
    """``src/param.F:2090-2106``: isfcflx=1 with a surface layer is legal."""

    config = parse_mapping(_document(isfcflx=1))
    selection = drag.surface_drag_selection_from_config(config)
    assert selection.isfcflx == 1
    assert selection.uses_surface_enthalpy_flux


@pytest.mark.parametrize("initsfc", (1, 2, 3, 4))
def test_every_initsfc_pattern_is_reachable_from_a_document(initsfc) -> None:
    """``src/init_surface.F:67-145``: four patterns, all selectable by key."""

    config = parse_mapping(_with_surface(_document(), initsfc=initsfc))
    assert config.surface is not None
    assert config.surface.initsfc == initsfc


def test_a_document_that_declares_no_surface_keeps_its_sealed_identity() -> None:
    """The shipped semi-slip deck's canonical bytes do not move."""

    config = parse_mapping(_document())
    assert config.surface is None
    assert "surface" not in config.canonical_dict()
    projected = config.canonical_dict()["closures"]
    for name in drag.SURFACE_DRAG_KERNEL_NAMES[:0] or ():  # pragma: no cover
        pass
    for name in ("cnstce", "set_znt", "cnst_znt", "set_ust", "cnst_ust"):
        assert name not in projected


@pytest.mark.parametrize(
    ("overrides", "code"),
    (
        ({"cecd": 4, **_surface_layer_group()}, "CM1-CFG-CECD-RANGE"),
        (_surface_layer_group(set_znt=1, set_ust=1), "CM1-CFG-SET-ZNT-AND-UST"),
        (_surface_layer_group(set_znt=1, cnst_znt=0.0), "CM1-CFG-CNST-ZNT-RANGE"),
        (_surface_layer_group(set_ust=1, cnst_ust=0.0), "CM1-CFG-CNST-UST-RANGE"),
        ({"isfcflx": 1, **_surface_layer_group(cnstce=-1.0)}, "CM1-CFG-CNSTCE-RANGE"),
    ),
)
def test_the_door_refuses_each_inadmissible_surface_by_name(overrides, code) -> None:
    with pytest.raises(CM1Refusal) as caught:
        parse_mapping(_document(**overrides))
    assert caught.value.detail.code == code


@pytest.mark.parametrize(
    ("surface", "code"),
    (
        ({"initsfc": 5}, "CM1-CFG-INITSFC-RANGE"),
        ({"xland0": 0.0}, "CM1-CFG-XLAND0-RANGE"),
        ({"lu0": 34}, "CM1-CFG-LU0-RANGE"),
        ({"season": 3}, "CM1-CFG-SEASON-RANGE"),
        ({"tsk0_k": 50.0}, "CM1-CFG-TSK0-RANGE"),
        ({"oceanmodel": 2}, "CM1-CFG-OCEANMODEL-UNSUPPORTED"),
    ),
)
def test_the_door_refuses_each_inadmissible_surface_table_by_name(surface, code) -> None:
    with pytest.raises(CM1Refusal) as caught:
        parse_mapping(_with_surface(_document(), **surface))
    assert caught.value.detail.code == code


def test_a_surface_table_without_a_surface_layer_is_refused() -> None:
    document = tomllib.loads(DECK.read_text(encoding="utf-8"))
    document["closures"].update(bbc=1, sfcmodel=0)
    document["boundaries"]["bottom"] = 1
    document = _with_surface(document)
    with pytest.raises(CM1Refusal) as caught:
        parse_mapping(document)
    assert caught.value.detail.code == "CM1-CFG-SURFACE-WITHOUT-SURFACE-LAYER"


# ------------------------------------------------------ the arithmetic moves


def _carriers(ni: int, nj: int, nk: int, *, seed: int = 11) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    f32 = np.float32
    shapes = drag.surface_drag_shapes(ni, nj, nk)
    arrays = {name: np.zeros(shape, f32) for name, shape in shapes.items()}
    arrays["u3d"][:] = rng.uniform(-14.0, 14.0, shapes["u3d"]).astype(f32)
    arrays["v3d"][:] = rng.uniform(-9.0, 9.0, shapes["v3d"]).astype(f32)
    arrays["zh"][:] = (
        (np.arange(nk, dtype=np.float32)[:, None, None] + f32(0.5)) * f32(250.0)
    )
    arrays["zf"][:] = np.arange(nk + 1, dtype=np.float32)[:, None, None] * f32(250.0)
    arrays["rf"][:] = rng.uniform(0.9, 1.2, shapes["rf"]).astype(f32)
    arrays["mf"][:] = f32(1.0)
    for name in ("kmh", "kmv", "khh", "khv"):
        arrays[name][:] = rng.uniform(0.5, 30.0, shapes[name]).astype(f32)
    arrays["znt"][:] = f32(0.0321)
    arrays["xland"][:] = f32(2.0)
    arrays["mavail"][:] = f32(1.0)
    arrays["tsk"][:] = f32(300.5)
    arrays["prs"][:] = f32(1.0e5) - (
        np.arange(nk, dtype=np.float32)[:, None, None] * f32(3000.0)
    )
    arrays["th0"][:] = f32(298.0)
    arrays["tha"][:] = rng.uniform(-0.6, 0.6, shapes["tha"]).astype(f32)
    arrays["qva"][:] = rng.uniform(0.008, 0.014, shapes["qva"]).astype(f32)
    return arrays


def _exchange_oracle(arrays, selection, ni, nj):
    """The independent earlier CPU port of the same ``getcecd`` source."""

    from cm1core.surface_exchange_coefficients import (
        SurfaceExchangeSelection,
        cpu_surface_exchange_coefficients_inplace,
    )

    names = ("u1", "v1", "s1", "za", "xland", "znt", "ust", "cd", "u10", "v10", "s10")
    oracle = {name: np.array(arrays[name], copy=True) for name in names}
    oracle["ch"] = np.zeros((nj, ni), np.float32)
    oracle["cq"] = np.zeros((nj, ni), np.float32)
    cpu_surface_exchange_coefficients_inplace(
        oracle,
        ni=ni,
        nj=nj,
        selection=SurfaceExchangeSelection(
            set_znt=selection.set_znt,
            set_ust=selection.set_ust,
            cecd=selection.cecd,
            isfcflx=selection.isfcflx,
            use_avg_sfc=False,
            cnstcd=selection.cnstcd,
            cnstce=selection.cnstce,
            cnst_ust_m_s=selection.cnst_ust,
            avgsfcu_m_s=0.0,
            avgsfcv_m_s=0.0,
            avgsfcs_m_s=0.0,
        ),
    )
    return oracle


@pytest.mark.parametrize(
    "selection",
    (
        drag.SurfaceDragSelection(bbc=3, sfcmodel=1, cnstcd=0.0035),
        drag.SurfaceDragSelection(bbc=3, sfcmodel=1, set_znt=1, cnst_znt=0.16),
        drag.SurfaceDragSelection(bbc=3, sfcmodel=1, set_ust=1, cnst_ust=0.42),
        drag.SurfaceDragSelection(bbc=3, sfcmodel=1, cecd=2),
        drag.SurfaceDragSelection(bbc=3, sfcmodel=1, cecd=3),
        drag.SurfaceDragSelection(bbc=3, sfcmodel=1, cecd=2, isfcflx=1, cnstce=0.0012),
        drag.SurfaceDragSelection(bbc=3, sfcmodel=1, isfcflx=1, cnstce=0.0012),
    ),
)
def test_each_getcecd_branch_matches_the_independent_port_raw_word(selection) -> None:
    """Two separately written transcriptions of ``getcecd``, word for word.

    Both evaluate the same platform ``logf``/``expf``/``sqrtf`` through
    ctypes, so any difference is a difference in the transcription and not in
    a library.
    """

    ni, nj, nk = 7, 5, 9
    arrays = _carriers(ni, nj, nk)
    if selection.set_znt == 1:
        # src/init_surface.F:246-254 writes cnst_znt into the carrier before
        # the first step; src/sfcphys.F:85 reads the carrier, not the key.
        arrays["znt"][:] = np.float32(selection.cnst_znt)
    # getcecd reads u1/v1/s1/za, which src/turb.F:539-555 has just written,
    # and znt as it stood before the call (src/sfcphys.F:155 uses the
    # previous step's value as the water iteration's first guess).
    entry_znt = np.array(arrays["znt"], copy=True)
    drag.cpu_surface_drag_inplace(
        arrays, ni=ni, nj=nj, nk=nk, dz=250.0, selection=selection
    )
    oracle_entry = dict(arrays)
    oracle_entry["znt"] = entry_znt
    oracle = _exchange_oracle(oracle_entry, selection, ni, nj)
    for name in ("cd", "znt", "ust", "u10", "v10", "s10"):
        np.testing.assert_array_equal(
            arrays[name].view(np.uint32), oracle[name].view(np.uint32), err_msg=name
        )
    if selection.isfcflx == 1:
        for name in ("ch", "cq"):
            np.testing.assert_array_equal(
                arrays[name].view(np.uint32),
                oracle[name].view(np.uint32),
                err_msg=name,
            )


def test_the_four_branches_do_not_produce_the_same_surface() -> None:
    """A selectable key that changes nothing would be a decoration."""

    ni, nj, nk = 7, 5, 9
    results = {}
    for name, selection in (
        ("cnstcd", drag.SurfaceDragSelection(bbc=3, sfcmodel=1, cnstcd=0.0035)),
        (
            "set_znt",
            drag.SurfaceDragSelection(bbc=3, sfcmodel=1, set_znt=1, cnst_znt=0.16),
        ),
        (
            "set_ust",
            drag.SurfaceDragSelection(bbc=3, sfcmodel=1, set_ust=1, cnst_ust=0.42),
        ),
        ("cecd2", drag.SurfaceDragSelection(bbc=3, sfcmodel=1, cecd=2)),
        ("cecd3", drag.SurfaceDragSelection(bbc=3, sfcmodel=1, cecd=3)),
    ):
        arrays = _carriers(ni, nj, nk)
        drag.cpu_surface_drag_inplace(
            arrays, ni=ni, nj=nj, nk=nk, dz=250.0, selection=selection
        )
        results[name] = (
            np.array(arrays["t13"], copy=True),
            np.array(arrays["ust"], copy=True),
        )
    names = sorted(results)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            assert not np.array_equal(results[left][0], results[right][0]), (
                left,
                right,
            )
            assert not np.array_equal(results[left][1], results[right][1]), (
                left,
                right,
            )


def test_drag_acts_on_the_ground_relative_wind_not_the_frame_wind() -> None:
    """``src/turb.F:296-307``: the surface sees ``ua+umove``, not ``ua``.

    A translating frame is exactly where this goes wrong silently: the run
    still finishes, and the storm is dragged by the wrong wind.  Two runs
    that differ only in the frame velocity must differ in the surface
    stress, and a frame-relative run whose carried wind is shifted by
    ``-umove`` must reproduce the translating run word for word.
    """

    ni, nj, nk = 7, 5, 9
    umove, vmove = 12.5, 3.0
    moving = _carriers(ni, nj, nk)
    drag.cpu_surface_drag_inplace(
        moving,
        ni=ni,
        nj=nj,
        nk=nk,
        dz=250.0,
        selection=drag.SurfaceDragSelection(
            bbc=3, sfcmodel=1, cnstcd=0.0035, u_move_m_s=umove, v_move_m_s=vmove
        ),
    )
    still = _carriers(ni, nj, nk)
    drag.cpu_surface_drag_inplace(
        still,
        ni=ni,
        nj=nj,
        nk=nk,
        dz=250.0,
        selection=drag.SurfaceDragSelection(bbc=3, sfcmodel=1, cnstcd=0.0035),
    )
    assert not np.array_equal(moving["t13"], still["t13"])
    assert not np.array_equal(moving["ust"], still["ust"])

    shifted = _carriers(ni, nj, nk)
    shifted["u3d"] += np.float32(umove)
    shifted["v3d"] += np.float32(vmove)
    drag.cpu_surface_drag_inplace(
        shifted,
        ni=ni,
        nj=nj,
        nk=nk,
        dz=250.0,
        selection=drag.SurfaceDragSelection(bbc=3, sfcmodel=1, cnstcd=0.0035),
    )
    for name in ("u1", "v1", "s1", "ust", "cd", "znt"):
        np.testing.assert_array_equal(
            moving[name].view(np.uint32),
            shifted[name].view(np.uint32),
            err_msg=name,
        )


# ----------------------------------------------------- the enthalpy flux


def test_the_surface_enthalpy_flux_matches_its_closed_form() -> None:
    """``src/turb.F:292`` then ``src/sfcphys.F:349-358``, stated exactly."""

    ni, nj, nk = 7, 5, 9
    arrays = _carriers(ni, nj, nk)
    selection = drag.SurfaceDragSelection(
        bbc=3, sfcmodel=1, isfcflx=1, cnstce=0.0012, cnstcd=0.0035
    )
    drag.cpu_surface_drag_inplace(
        arrays, ni=ni, nj=nj, nk=nk, dz=250.0, selection=selection
    )
    weights = (1.875, -1.25, 0.375)
    drag.cpu_surface_enthalpy_flux_inplace(
        arrays, ni=ni, nj=nj, moisture=True, surface_pressure_weights=weights
    )
    expected_psfc = (
        weights[0] * arrays["prs"][0].astype(np.float64)
        + weights[1] * arrays["prs"][1].astype(np.float64)
        + weights[2] * arrays["prs"][2].astype(np.float64)
    )
    np.testing.assert_allclose(arrays["psfc"], expected_psfc, rtol=1e-6)
    pisfc = (arrays["psfc"].astype(np.float64) / 1.0e5) ** (287.04 / 1005.7)
    expected_th = (
        arrays["ch"].astype(np.float64)
        * arrays["s10"].astype(np.float64)
        * (
            arrays["tsk"].astype(np.float64) / pisfc
            - arrays["th0"][0].astype(np.float64)
            - arrays["tha"][0].astype(np.float64)
        )
    )
    np.testing.assert_allclose(arrays["thflux"], expected_th, rtol=2e-5)
    esl = np.minimum(
        611.2
        * np.exp(
            17.67
            * (arrays["tsk"].astype(np.float64) - 273.15)
            / (arrays["tsk"].astype(np.float64) - 29.65)
        ),
        arrays["psfc"].astype(np.float64) * 0.5,
    )
    qsat = (287.04 / 461.5) * esl / (arrays["psfc"].astype(np.float64) - esl)
    np.testing.assert_allclose(arrays["qsfc"], qsat, rtol=2e-6)
    expected_qv = (
        arrays["cq"].astype(np.float64)
        * arrays["s10"].astype(np.float64)
        * (qsat - arrays["qva"][0].astype(np.float64))
        * arrays["mavail"].astype(np.float64)
    )
    np.testing.assert_allclose(arrays["qvflux"], expected_qv, rtol=2e-5)
    # A warmer, wetter surface than the air above it heats and moistens it.
    assert float(arrays["thflux"].mean()) > 0.0
    assert float(arrays["qvflux"].mean()) > 0.0


def test_the_scalar_kernel_carries_the_lower_flux_only_for_theta_and_qv() -> None:
    """``src/solve1.F:1208,1217`` pass iflux=1; ``:1222,1233`` pass 0."""

    from cm1core.cuda_gm1_child_stretched_scalar_turbulence import (
        CUDA_GM1_CHILD_STRETCHED_SCALAR_TURBULENCE_SOURCE as source,
    )

    assert "const float*sflux" in source
    assert "unsigned iflux" in source
    assert "(iflux==1u)?M(sflux[col],rf[col]):0.0f" in source


# ------------------------------------------------------------- initsfc


def _fields(**overrides):
    values = {
        "initsfc": 1,
        "sfcmodel": 1,
        "bbc": 3,
        "isfcflx": 1,
        "set_znt": 0,
        "cnst_znt": 0.16,
        "tsk0_k": 299.28,
        "tmn0_k": 297.28,
        "xland0": 2.0,
        "lu0": 16,
        "season": 1,
        "oceanmodel": 1,
    }
    values.update(overrides)
    selection = SurfaceFieldSelection(**values)
    return build_surface_fields(
        selection=selection, nx=16, ny=4, dx_m=1000.0, dy_m=1000.0
    )


def test_initsfc_two_builds_the_wrf_sea_breeze_land_strip() -> None:
    """``src/init_surface.F:70-95``: land between 3/8 and 5/8 of the domain."""

    fields = _fields(initsfc=2)
    xland = np.asarray(fields.arrays["xland"])[0]
    tsk = np.asarray(fields.arrays["tsk"])[0]
    land = xland == np.float32(1.0)
    assert land.any() and (~land).any()
    assert np.all(tsk[land] == np.float32(280.0))
    assert np.all(tsk[~land] == np.float32(287.0))
    # The strip is contiguous and centred.
    indices = np.flatnonzero(land)
    assert np.array_equal(indices, np.arange(indices[0], indices[-1] + 1))
    assert abs((indices[0] + indices[-1]) / 2 - (len(xland) - 1) / 2) <= 0.5


def test_initsfc_four_builds_a_coastline_at_the_domain_centre() -> None:
    """``src/init_surface.F:111-124``: land west, water east."""

    fields = _fields(initsfc=4)
    xland = np.asarray(fields.arrays["xland"])[0]
    half = len(xland) // 2
    assert np.all(xland[:half] == np.float32(1.0))
    assert np.all(xland[half:] == np.float32(2.0))


def test_initsfc_three_roughness_is_overwritten_by_the_land_use_table() -> None:
    """``src/init_surface.F:97-107`` then ``:441-466`` -- and the table wins.

    This is CM1's own behaviour, and it is the opposite of what the option's
    name promises: ``getlanduse`` runs after the ``initsfc`` branches and
    assigns the roughness length of ``lu0`` to every column, so the
    rough-west/smooth-east contrast never reaches the run.  The receipt says
    so per run rather than leaving a reader to assume the name.
    """

    fields = _fields(initsfc=3)
    znt = np.asarray(fields.arrays["znt"])
    assert np.all(znt == znt.flat[0])
    assert znt.flat[0] not in (np.float32(0.010), np.float32(0.001))
    assert fields.receipt["initsfc3_roughness_overwritten_by_landuse"] is True
    # With set_znt=1 the declared roughness is re-imposed after the table
    # (src/init_surface.F:246-254), so that one does survive.
    imposed = _fields(initsfc=3, set_znt=1, cnst_znt=0.0123)
    assert np.all(np.asarray(imposed.arrays["znt"]) == np.float32(0.0123))


def test_the_land_use_table_record_is_the_release_file() -> None:
    record = landuse_table()
    assert record["authority"]["sha256"] == CM1_LANDUSE_TBL_SHA256
    if AUTHORITY.is_dir():
        text = (AUTHORITY / "run" / "LANDUSE.TBL").read_text(encoding="utf-8")
        assert hashlib.sha256(text.encode()).hexdigest() == CM1_LANDUSE_TBL_SHA256


def test_the_land_use_table_reproduces_cm1s_own_conversions() -> None:
    """``src/init_surface.F:455-464``: per-cent and centimetres, and water."""

    ni = nj = 3
    arrays = {
        name: np.zeros((nj, ni), np.float32)
        for name in ("xland", "znt", "mavail", "emiss", "thc", "albedo")
    }
    arrays["lu_index"] = np.full((nj, ni), 16, np.int32)
    arrays["lu_index"][0, 0] = 1  # urban
    cpu_getlanduse_inplace(arrays, ni=ni, nj=nj, season=1)
    rows = landuse_table()["seasons"][0]
    urban, water = rows[0], rows[15]
    assert arrays["znt"][0, 0] == np.float32(np.float32(urban["sfz0"]) / np.float32(100.0))
    assert arrays["albedo"][0, 0] == np.float32(
        np.float32(urban["albd"]) / np.float32(100.0)
    )
    assert arrays["mavail"][0, 0] == np.float32(urban["slmo"])
    assert arrays["xland"][0, 0] == np.float32(1.0)
    assert arrays["xland"][1, 1] == np.float32(2.0)
    assert arrays["znt"][1, 1] == np.float32(
        np.float32(water["sfz0"]) / np.float32(100.0)
    )


def test_a_water_surface_gets_moisture_availability_one() -> None:
    """``src/init_surface.F:277``, before the reality checks."""

    fields = _fields(xland0=2.0, lu0=16)
    assert np.all(np.asarray(fields.arrays["mavail"]) == np.float32(1.0))


def test_a_land_surface_gets_the_tabulated_moisture_availability() -> None:
    fields = _fields(xland0=1.0, lu0=2)
    rows = landuse_table()["seasons"][0]
    assert np.all(
        np.asarray(fields.arrays["mavail"]) == np.float32(rows[1]["slmo"])
    )
    assert float(np.asarray(fields.arrays["mavail"]).flat[0]) < 1.0


# ------------------------------------- the native CM1 r21.1 Fortran oracle


def _native_table():
    """Build and run the gfortran transcription of CM1's own statements."""

    import tempfile

    sys.path.insert(0, str(ROOT / "tools"))
    from oracle import surface_layer_oracle as native

    if native.gfortran() is None:
        pytest.skip("gfortran is not installed")
    with tempfile.TemporaryDirectory() as temporary:
        outputs = native.build_and_run(Path(temporary))
    assert outputs["o0"] == outputs["o2"], "the native oracle is optimization dependent"
    return native, native.parse(outputs["o0"])


def _carriers_from_native(native, table, nk: int = 9):
    """Load the harness's own input words into this port's carriers.

    The wind is placed on both u faces of a column so ``src/turb.F:539-541``
    averages it back to exactly the word the Fortran used, and the terrain is
    flat so ``src/turb.F:547-555`` returns the height it used.
    """

    f32 = np.float32
    shapes = drag.surface_drag_shapes(native.NI, native.NJ, nk)
    arrays = {name: np.zeros(shape, f32) for name, shape in shapes.items()}
    za = native.words(table, "input", "fixture", "za")
    # The harness's lowest-level wind is linear in the index, so a staggered
    # face field that is the same line shifted half a cell averages back to
    # it exactly at src/turb.F:539-541 -- which the test then asserts word
    # for word before comparing anything downstream of it.
    faces_i = np.arange(native.NI + 1, dtype=np.float32)
    faces_j = np.arange(native.NJ + 1, dtype=np.float32)
    rows = np.arange(native.NJ, dtype=np.float32)
    columns = np.arange(native.NI, dtype=np.float32)
    arrays["u3d"][0] = (
        f32(3.625) + f32(0.75) * faces_i[None, :] - f32(0.5) * rows[:, None]
    )
    arrays["v3d"][0] = (
        f32(-3.25) + f32(0.5) * faces_j[:, None] + f32(0.25) * columns[None, :]
    )
    arrays["zh"][0] = za
    arrays["zf"][:] = np.arange(nk + 1, dtype=np.float32)[:, None, None] * f32(250.0)
    arrays["rf"][:] = f32(1.0)
    arrays["mf"][:] = f32(1.0)
    for name in ("kmh", "kmv", "khh", "khv"):
        arrays[name][:] = f32(1.0)
    for name, field in (
        ("xland", "xland"),
        ("znt", "znt0"),
        ("tsk", "tsk"),
        ("mavail", "mavail"),
    ):
        arrays[name][:] = native.words(table, "input", "fixture", field)
    for level, field in ((0, "prs1"), (1, "prs2"), (2, "prs3")):
        arrays["prs"][level] = native.words(table, "input", "fixture", field)
    arrays["th0"][0] = native.words(table, "input", "fixture", "th0")
    arrays["tha"][0] = native.words(table, "input", "fixture", "tha")
    arrays["qva"][0] = native.words(table, "input", "fixture", "qva")
    return arrays


@pytest.mark.parametrize(
    ("branch", "selection"),
    (
        ("set_znt", drag.SurfaceDragSelection(bbc=3, sfcmodel=1, set_znt=1, cnst_znt=0.16)),
        ("set_ust", drag.SurfaceDragSelection(bbc=3, sfcmodel=1, set_ust=1, cnst_ust=0.42)),
        ("cnstcd", drag.SurfaceDragSelection(bbc=3, sfcmodel=1, cnstcd=0.0035)),
        ("cecd2", drag.SurfaceDragSelection(bbc=3, sfcmodel=1, cecd=2)),
        ("cecd3", drag.SurfaceDragSelection(bbc=3, sfcmodel=1, cecd=3)),
    ),
)
def test_every_getcecd_branch_is_raw_word_exact_against_gfortran(branch, selection) -> None:
    """CM1's own statements, compiled by gfortran, word for word.

    This is the strongest check available without running the whole model:
    the comparison target is CM1 r21.1 source text compiled by a Fortran
    compiler with the release's arithmetic discipline, at both ``-O0`` and
    ``-O2``, and the port is handed the Fortran's own input words.
    """

    native, table = _native_table()
    arrays = _carriers_from_native(native, table)
    np.testing.assert_array_equal(
        arrays["znt"].view(np.uint32),
        native.words(table, "input", "fixture", "znt0").view(np.uint32),
    )
    drag.cpu_surface_drag_inplace(
        arrays,
        ni=native.NI,
        nj=native.NJ,
        nk=9,
        dz=250.0,
        selection=selection,
    )
    # src/turb.F:539-543 has to reproduce the Fortran's own lowest-level wind
    # before any coefficient can be compared to it.
    for name, field in (("u1", "u1"), ("v1", "v1"), ("s1", "s1"), ("za", "za")):
        np.testing.assert_array_equal(
            arrays[name].view(np.uint32),
            native.words(table, "input", "fixture", field).view(np.uint32),
            err_msg=name,
        )
    for name in ("cd", "znt", "ust", "u10", "v10", "s10"):
        np.testing.assert_array_equal(
            arrays[name].view(np.uint32),
            native.words(table, "output", branch, name).view(np.uint32),
            err_msg=f"{branch}:{name}",
        )


@pytest.mark.parametrize(
    ("branch", "cecd"),
    (("cnstcd", 1), ("cecd2", 2), ("cecd3", 3)),
)
def test_the_enthalpy_exchange_coefficients_match_gfortran(branch, cecd) -> None:
    """``src/sfcphys.F:197-236``, raw word, over land and over water."""

    native, table = _native_table()
    arrays = _carriers_from_native(native, table)
    selection = drag.SurfaceDragSelection(
        bbc=3,
        sfcmodel=1,
        isfcflx=1,
        cecd=cecd,
        cnstcd=0.0035,
        cnstce=0.0012,
    )
    drag.cpu_surface_drag_inplace(
        arrays, ni=native.NI, nj=native.NJ, nk=9, dz=250.0, selection=selection
    )
    for name in ("ch", "cq"):
        np.testing.assert_array_equal(
            arrays[name].view(np.uint32),
            native.words(table, "output", branch, name).view(np.uint32),
            err_msg=f"{branch}:{name}",
        )


def test_the_surface_pressure_and_enthalpy_flux_match_gfortran() -> None:
    """``src/turb.F:292`` and ``src/sfcphys.F:341-361``, raw word.

    The Exner ratio and the saturation vapour pressure go through the
    platform ``powf`` and ``expf`` here and through gfortran's in the
    harness; both are the same libm, so the comparison is exact rather than
    within a tolerance.
    """

    native, table = _native_table()
    arrays = _carriers_from_native(native, table)
    selection = drag.SurfaceDragSelection(
        bbc=3, sfcmodel=1, isfcflx=1, cecd=1, cnstcd=0.0035, cnstce=0.0012
    )
    drag.cpu_surface_drag_inplace(
        arrays, ni=native.NI, nj=native.NJ, nk=9, dz=250.0, selection=selection
    )
    drag.cpu_surface_enthalpy_flux_inplace(
        arrays,
        ni=native.NI,
        nj=native.NJ,
        moisture=True,
        surface_pressure_weights=(1.875, -1.25, 0.375),
    )
    for name in ("psfc", "qsfc", "thflux", "qvflux"):
        np.testing.assert_array_equal(
            arrays[name].view(np.uint32),
            native.words(table, "output", "cnstcd", name).view(np.uint32),
            err_msg=name,
        )


# ------------------------- the packaged CUDA text, compiled for the host


def _kernel_oracle():
    sys.path.insert(0, str(ROOT / "tools"))
    from oracle import surface_layer_kernel_oracle as kernels

    if kernels.compiler() is None:
        pytest.skip("g++ is not installed")
    return kernels


@pytest.mark.parametrize(
    ("branch", "cecd"),
    (
        ("set_znt", 1),
        ("set_ust", 1),
        ("cnstcd", 1),
        ("water", 2),
        ("water", 3),
    ),
)
def test_the_packaged_cuda_text_reproduces_gfortran_raw_word(branch, cecd) -> None:
    """The same source string nvrtc compiles, compiled by g++, against CM1.

    ``__fadd_rn``/``__fsub_rn``/``__fmul_rn``/``__fdiv_rn`` have one IEEE
    binary32 meaning, so the packaged kernel text has a host meaning too, and
    it is the same one.  This runs that text -- unmodified, taken straight
    from ``build_cuda_surface_drag_source`` -- on the words the gfortran
    transcription of CM1's own statements used, and compares every word.
    """

    import tempfile

    native, table = _native_table()
    kernels = _kernel_oracle()
    inputs = {
        name: native.words(table, "input", "fixture", "znt0" if name == "znt" else name)
        for name in kernels.INPUT_ORDER
    }
    with tempfile.TemporaryDirectory() as temporary:
        work = Path(temporary)
        executable = kernels.build(work)
        outputs, nonconvergent = kernels.run(
            executable,
            work,
            ni=native.NI,
            nj=native.NJ,
            inputs=inputs,
            branch=branch,
            cecd=cecd,
            cnstcd=0.0035,
            cnstce=0.0012,
            cnst_ust=0.42,
        )
    assert nonconvergent == 0
    expected_branch = {"water": f"cecd{cecd}"}.get(branch, branch)
    for name in ("cd", "znt", "ust", "u10", "v10", "s10"):
        np.testing.assert_array_equal(
            outputs[name].view(np.uint32),
            native.words(table, "output", expected_branch, name).view(np.uint32),
            err_msg=f"{expected_branch}:{name}",
        )


def test_the_packaged_cuda_text_reproduces_the_enthalpy_flux_raw_word() -> None:
    """``getcecd`` ch/cq, ``turb.F:292`` psfc and ``sfcflux``, word for word."""

    import tempfile

    native, table = _native_table()
    kernels = _kernel_oracle()
    inputs = {
        name: native.words(table, "input", "fixture", "znt0" if name == "znt" else name)
        for name in kernels.INPUT_ORDER
    }
    with tempfile.TemporaryDirectory() as temporary:
        work = Path(temporary)
        executable = kernels.build(work)
        outputs, _ = kernels.run(
            executable,
            work,
            ni=native.NI,
            nj=native.NJ,
            inputs=inputs,
            branch="cnstcd",
            cecd=1,
            cnstcd=0.0035,
            cnstce=0.0012,
            cnst_ust=0.42,
        )
    for name in ("ch", "cq", "psfc", "qsfc", "thflux", "qvflux"):
        np.testing.assert_array_equal(
            outputs[name].view(np.uint32),
            native.words(table, "output", "cnstcd", name).view(np.uint32),
            err_msg=name,
        )


# --------------------------- the scalar lower flux, against gfortran turbsz


def _turbsz_native():
    """gfortran running CM1's own implicit vertical diffusion statements."""

    import struct
    import subprocess
    import tempfile

    sys.path.insert(0, str(ROOT / "tools"))
    from oracle import surface_layer_oracle as native

    if native.gfortran() is None:
        pytest.skip("gfortran is not installed")
    harness = ROOT / "tools" / "oracle" / "turbsz_lower_flux_harness.f90"
    outputs = {}
    with tempfile.TemporaryDirectory() as temporary:
        work = Path(temporary)
        for name, flags in native.VARIANTS.items():
            executable = work / f"turbsz-{name}"
            subprocess.run(
                [
                    native.gfortran(),
                    *flags,
                    *native.COMMON,
                    str(harness),
                    "-o",
                    str(executable),
                ],
                check=True,
                capture_output=True,
            )
            outputs[name] = subprocess.run(
                [str(executable)], check=True, capture_output=True
            ).stdout
    assert outputs["o0"] == outputs["o2"], "the native oracle is optimization dependent"
    ni, nj, nk = 5, 5, 6
    table: dict[tuple[str, str, str], np.ndarray] = {}
    for line in outputs["o0"].decode("ascii").splitlines():
        if not line or line.startswith("#"):
            continue
        _sequence, role, branch, field, i, j, k, word = line.split("\t")
        key = (role, branch, field)
        if key not in table:
            levels = nk + 1 if field in ("khv", "mf", "rf") else nk
            if field == "sflux":
                levels = 1
            table[key] = np.zeros((levels, nj, ni), np.float32)
        table[key][int(k), int(j), int(i)] = struct.unpack(
            "<f", struct.pack("<I", int(word, 16))
        )[0]
    return (ni, nj, nk), table


def test_the_scalar_lower_flux_matches_gfortran_turbsz_raw_word() -> None:
    """``src/turbtend.F:588-596,625-690``, with and without the surface flux.

    With ``iflux=0`` the kernel must reproduce the zero-flux solve it always
    produced -- that is what keeps a free-slip run bit-identical -- and with
    ``iflux=1`` it must reproduce CM1's ``dumz(i,j,1)=sflux*rf(i,j,1)``
    right-hand side through the whole Thomas sweep, at every level, not just
    the lowest.
    """

    import tempfile

    (ni, nj, nk), table = _turbsz_native()
    sys.path.insert(0, str(ROOT / "tools"))
    from oracle import turbsz_lower_flux_kernel_oracle as kernels

    if kernels.compiler() is None:
        pytest.skip("g++ is not installed")
    arrays = {
        "s": table[("input", "fixture", "s")],
        "rho": table[("input", "fixture", "rho")],
        "rr": table[("input", "fixture", "rr")],
        "mh": table[("input", "fixture", "mh")],
        "khv": table[("input", "fixture", "khv")],
        "mf": table[("input", "fixture", "mf")],
        "rf": table[("input", "fixture", "rf")],
        "sflux": table[("input", "fixture", "sflux")][0],
    }
    with tempfile.TemporaryDirectory() as temporary:
        work = Path(temporary)
        executable = kernels.build(work)
        results = {
            iflux: kernels.run(
                executable,
                work,
                ni=ni,
                nj=nj,
                nk=nk,
                dz=250.0,
                dt=2.0,
                arrays=arrays,
                iflux=iflux,
            )
            for iflux in (0, 1)
        }
    for iflux, branch in ((0, "iflux0"), (1, "iflux1")):
        np.testing.assert_array_equal(
            results[iflux].view(np.uint32),
            table[("output", branch, "turbzrr")].view(np.uint32),
            err_msg=branch,
        )
    # The surface flux has to change the answer, and at the lowest level it
    # has to warm: a positive sensible heat flux into the column.
    assert not np.array_equal(results[0], results[1])
    assert np.all(results[1][0] > results[0][0])


def test_the_zero_flux_path_is_bit_identical_to_the_kernel_before_this_work() -> None:
    """``iflux=0`` reaches ``M(temc,0.0f)``, exactly as it always did.

    The free-slip lane never sets a surface flux, so the one line added to
    this kernel must not move a single word there.  This drives the packaged
    text with the surface flux switched off against the same text with the
    flux array filled with values it must then ignore.
    """

    import tempfile

    (ni, nj, nk), table = _turbsz_native()
    sys.path.insert(0, str(ROOT / "tools"))
    from oracle import turbsz_lower_flux_kernel_oracle as kernels

    if kernels.compiler() is None:
        pytest.skip("g++ is not installed")
    base = {
        "s": table[("input", "fixture", "s")],
        "rho": table[("input", "fixture", "rho")],
        "rr": table[("input", "fixture", "rr")],
        "mh": table[("input", "fixture", "mh")],
        "khv": table[("input", "fixture", "khv")],
        "mf": table[("input", "fixture", "mf")],
        "rf": table[("input", "fixture", "rf")],
        "sflux": table[("input", "fixture", "sflux")][0],
    }
    poisoned = dict(base)
    poisoned["sflux"] = np.full_like(base["sflux"], np.float32(12345.0))
    with tempfile.TemporaryDirectory() as temporary:
        work = Path(temporary)
        executable = kernels.build(work)
        quiet = kernels.run(
            executable, work, ni=ni, nj=nj, nk=nk, dz=250.0, dt=2.0,
            arrays=base, iflux=0,
        )
        loud = kernels.run(
            executable, work, ni=ni, nj=nj, nk=nk, dz=250.0, dt=2.0,
            arrays=poisoned, iflux=0,
        )
    np.testing.assert_array_equal(quiet.view(np.uint32), loud.view(np.uint32))


# --------------------------------------------- the device evidence receipts


def test_the_device_receipts_name_the_live_operator_and_their_findings() -> None:
    """Both GPU comparisons are receipts, not prose, and both are readable.

    The first says how far the device words are from CM1's own; the second
    says the free-slip lane did not move.  A reader who wants to know whether
    a surface run is measured reads these, and `gpuwm-cm1 doctor` quotes the
    same numbers.
    """

    import json

    device = json.loads(
        (
            ROOT
            / "evidence/validation/gm1-surface-layer-device-v1/receipt.json"
        ).read_text(encoding="utf-8")
    )
    assert device["status"] == "PASS"
    assert device["execution"]["gpu_used"] is True
    assert device["comparison"]["worst_ulp"] <= 3
    # The load-bearing pin is the kernel source string -- that is what nvrtc
    # compiled and the device ran -- and it is asserted strictly.  The whole
    # module file moved after the run only because this receipt's findings
    # were written into the module's validation table, which no kernel reads.
    assert device["operator"]["cuda_source_sha256"] == (
        drag.CUDA_SURFACE_DRAG_SOURCE_SHA256
    )
    assert len(device["operator"]["module_sha256_at_run"]) == 64
    assert "cuda_source_sha256" in device["operator"]["binding"]
    # The three branches that came out bit-identical are the ones the
    # validation table may call device-exact.
    exact = {
        branch
        for branch, row in device["comparison"]["per_branch"].items()
        if row["max_ulp"] == 0
    }
    assert {"cnstcd", "set_znt", "isfcflx"} <= exact
    assert "cecd3" not in exact

    identity = json.loads(
        (
            ROOT
            / "evidence/validation/gm1-scalar-turbulence-free-slip-identity-v1"
            / "receipt.json"
        ).read_text(encoding="utf-8")
    )
    assert identity["status"] == "PASS"
    assert identity["comparison"]["raw_word_identical"] is True
    assert identity["comparison"]["surface_flux_supplied_but_switched_off"] is True
    assert identity["comparison"]["nonzero_words"] > 0
    assert identity["comparison"]["new_source_sha256"] == (
        __import__("cm1core.cuda_gm1_child_stretched_scalar_turbulence", fromlist=["x"])
        .CUDA_GM1_CHILD_STRETCHED_SCALAR_TURBULENCE_SOURCE_SHA256
    )


def test_the_validation_markers_name_the_device_evidence() -> None:
    """A selectable capability with no evidence marker reads as measured."""

    from cm1core.cuda_surface_drag import surface_validation_markers

    exact = surface_validation_markers(
        drag.SurfaceDragSelection(bbc=3, sfcmodel=1, cnstcd=0.0035)
    )
    assert "device_last_place_difference" not in exact["unvalidated"]
    assert "gm1-surface-layer-device-v1" in exact["validated"]["cnstcd"]["evidence"]

    inexact = surface_validation_markers(
        drag.SurfaceDragSelection(bbc=3, sfcmodel=1, cecd=3)
    )
    assert "device_last_place_difference" in inexact["unvalidated"]
    assert "integrated_surface_trajectory" in inexact["unvalidated"]

    free = surface_validation_markers(drag.FREE_SLIP_SELECTION)
    assert free["surface"] is False
    assert free["unvalidated"] == {}


def test_doctor_reports_the_surface_layer_as_unvalidated_not_ok() -> None:
    """``doctor`` must not let "selectable" read as "measured"."""

    import json

    from cm1core.doctor import UNVALIDATED, UNVALIDATED_MARKER, collect

    finding = next(
        item for item in collect() if item.subject == "CM1 surface layer and land surface"
    )
    assert finding.status == UNVALIDATED
    # f9be0a4: Finding prefixes every UNVALIDATED detail that lacks it with
    # doctor.UNVALIDATED_MARKER; the JSON payload follows that prefix.
    prefix = f"{UNVALIDATED_MARKER}: "
    assert finding.detail.startswith(prefix)
    detail = json.loads(finding.detail[len(prefix):])
    assert "closures.set_znt/cnst_znt" in detail["selectable"]
    assert "surface.initsfc=1/2/3/4" in detail["selectable"]
    assert detail["unvalidated"]
    assert "cecd=1 constant drag" in detail["validated_on_a_device"]["bit_identical"]
    # Not blocking: the capability works, the evidence is simply bounded.
    assert finding.required is False
