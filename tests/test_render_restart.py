"""``gpuwm-cm1 render RUNDIR``: the native renderer reached from a run folder's restart checkpoint.

The restart here is a small synthetic one written in the envelope layout GM1 publishes (CM1FCP1 magic, JSON header
with each array's offset, size and SHA-256, the payload), with the receipt beside it.  The tests walk what a user
reaches: finding the restart, reading and checking its arrays, cutting the padded winds to the model's cells,
deriving what a restart does not carry, the words each picture says about itself, the refusals, and the exact
command handed to the Rust renderer.  The real renderer is run when this checkout or install has it.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import sys

import numpy as np
import pytest

from cm1core import render_restart as RR
from cm1core import rust_tools
from cm1core.cli import main
from cm1core.render_products import PRODUCTS

NX, NY, NZ, H = 8, 6, 12, 3
DX, DZ = 1000.0, 500.0
MASS = ["qv", "qc", "qr", "qi", "qs", "qg"]
NUMBER = ["nci", "ncs", "ncr", "ncg"]


def _species_record() -> dict:
    scalars = [{"name": n, "unit": "kg/kg"} for n in MASS] + [{"name": n, "unit": "#/kg"} for n in NUMBER]
    return {"scalars": scalars, "cm1_index_ranges": {"nqv": 1}, "numq": 10}


def _arrays(time_s: float, *, moving: bool, stretched: bool, sideways_halo: bool = True) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(7)
    k = np.arange(NZ, dtype=np.float32)[:, None, None]
    out: dict[str, np.ndarray] = {}
    # Padded winds: a halo of 3 cells sideways, holding what CM1's open lateral boundaries put there (the edge
    # value copied outward), and 1 level up and down, filled with a marker the cut must drop.
    h = H if sideways_halo else 0
    for name, shape in (("u3d", (NZ, NY, NX + 1)), ("v3d", (NZ, NY + 1, NX)), ("w3d", (NZ + 1, NY, NX))):
        padded = np.full((shape[0] + 2, shape[1] + 2 * h, shape[2] + 2 * h), -999.0, np.float32)
        inner = rng.normal(0.0, 3.0, shape).astype(np.float32)
        if name == "w3d":
            inner[0] = 0.0
            inner[-1] = 0.0
            inner[6, 2, 3] = 25.0
        padded[1:-1] = np.pad(inner, ((0, 0), (h, h), (h, h)), mode="edge")
        out[f"padded.{name}"] = padded
    out["fields.th3d"] = rng.normal(-1.0, 1.0, (NZ, NY, NX)).astype(np.float32)
    out["fields.pp3d"] = np.zeros((NZ, NY, NX), np.float32)
    prs = (100000.0 * np.exp(-(k + 0.5) * DZ / 8000.0) * np.ones((NZ, NY, NX))).astype(np.float32)
    out["fields.prs"] = prs
    out["fields.rho"] = (prs / (287.04 * 280.0)).astype(np.float32)
    q = np.zeros((10, NZ, NY, NX), np.float32)
    q[0] = 0.01
    q[2, :4, 2:4, 2:5] = 2.0e-3    # rain in a few columns
    q[8, :4, 2:4, 2:5] = 1.0e4     # its number
    out["fields.q3d"] = q
    sws = rng.uniform(0.0, 30.0, (NY, NX)).astype(np.float32)
    out["swath.sws"] = sws
    out["swath.sps"] = rng.uniform(-300.0, 0.0, (NY, NX)).astype(np.float32)
    if not stretched:
        out["swath.svs"] = rng.uniform(0.0, 0.02, (NY, NX)).astype(np.float32)
        out["swath.shs"] = rng.uniform(0.0, 400.0, (NY, NX)).astype(np.float32)
    if moving:
        canvas = (9, 13)
        out["swath.sws2"] = np.zeros(canvas, np.float32)
        out["swath.sws2"][2:8, 3:11] = sws
        out["swath.sps2"] = np.full(canvas, 200000.0, np.float32)   # CM1's initial value where never reached
        out["swath.sps2"][2:8, 3:11] = -150.0
        if not stretched:
            out["swath.svs2"] = np.full(canvas, -1000.0, np.float32)
            out["swath.svs2"][2:8, 3:11] = 0.01
            out["swath.shs2"] = np.zeros(canvas, np.float32)
    return out


def write_restart(run: Path, *, number: int = 1, segment: int = 1, time_s: float = 3600.0, moving: bool = True,
                  stretched: bool = False, microphysics: str = "morrison-mp10", contract: str = "c" * 64,
                  sideways_halo: bool = True) -> Path:
    directory = run / "segments" / f"{segment:03d}" / "restarts" / f"restart-{number:06d}"
    directory.mkdir(parents=True, exist_ok=True)
    arrays = _arrays(time_s, moving=moving, stretched=stretched, sideways_halo=sideways_halo)
    specs, rows, payload = [], [], b""
    meanings = {"fields.q3d": "current qv,qc,qr,qi,qs,qg mixing ratios then nci,ncs,ncr,ncg number concentrations"}
    for name in sorted(arrays):
        data = np.ascontiguousarray(arrays[name], dtype="<f4").tobytes()
        rows.append({"name": name, "offset": len(payload), "nbytes": len(data),
                     "sha256": hashlib.sha256(data).hexdigest()})
        specs.append({"name": name, "shape": list(arrays[name].shape), "meaning": meanings.get(name, name),
                      "units": "Pa" if "sps" in name or name == "fields.prs" else "m s-1"})
        payload += data
    header = json.dumps({"arrays": rows, "contract": {"array_specs": specs}, "contract_sha256": contract,
                         "payload_nbytes": len(payload)}).encode("utf-8")
    (directory / RR.RESTART_FILE).write_bytes(RR.MAGIC + len(header).to_bytes(8, "little") + header + payload)
    grid = {"nx": NX, "ny": NY, "nz": NZ, "dx_m": DX, "dy_m": DX, "dz_m": DZ, "ztop_m": NZ * DZ}
    if stretched:
        from cm1core.cm1_vertical_grid import build_vertical_levels

        # 2 levels of 250 m, 8 stretched from 250 to 750 m, 2 of 750 m: 6 km, as CM1's stretch_z = 1 builds it.
        levels = build_vertical_levels(stretch_z=1, nk=NZ, dz_m=DZ, ztop_m=NZ * DZ, str_bot_m=500.0,
                                       str_top_m=4500.0, dz_bot_m=250.0, dz_top_m=750.0)
        grid["vertical"] = levels.record()
    receipt = {
        "model_time_s": time_s, "step": int(time_s / 6), "dt_s": 6.0,
        "forecast_envelope": {"contract_sha256": contract, "path": str(directory / RR.RESTART_FILE)},
        "config": {"grid": grid, "microphysics": microphysics, "moisture_species": _species_record(),
                   "dense_ice": {"ihail": 1, "category": "hail"},
                   "u_move_m_s": 12.5 if moving else 0.0, "v_move_m_s": 3.0 if moving else 0.0},
        "scheduler_scalars": {"domainlocx": struct.pack(">d", 4000.0).hex(),
                              "domainlocy": struct.pack(">d", 2000.0).hex()},
        "swaths": {"first_step": {"sws": 1}, "layout": {"ground_canvas_shape": [9, 13]},
                   "carried": sorted(name.split(".")[1] for name in arrays
                                     if name.startswith("swath.") and not name.endswith("2"))},
    }
    (directory / RR.PUBLICATION).write_text(json.dumps(receipt), encoding="utf-8")
    return directory


@pytest.fixture
def run(tmp_path: Path) -> Path:
    root = tmp_path / "storm-run"
    write_restart(root)
    return root


# ----------------------------------------------------------------------------------------------- finding it
def test_a_run_folder_finds_its_newest_restart_and_any_one_by_number(tmp_path, capsys):
    root = tmp_path / "r"
    write_restart(root, number=1, segment=1, time_s=1800.0)
    write_restart(root, number=2, segment=1, time_s=3600.0)
    write_restart(root, number=3, segment=2, time_s=5400.0)
    (root / "segments" / "002" / "restarts" / ".incomplete").mkdir()
    _, newest, rows = RR.choose(root, None)
    assert (newest.number, newest.segment, newest.model_time_s) == (3, 2, 5400.0)
    assert [r.number for r in rows] == [1, 2, 3]
    assert RR.choose(root, "2")[1].number == 2
    assert RR.choose(root, "1:1")[1].segment == 1
    # A restart folder or its file names one restart directly.
    one = root / "segments" / "001" / "restarts" / "restart-000001"
    assert RR.choose(one / RR.RESTART_FILE, None)[1].number == 1
    with pytest.raises(RR.RestartRenderError) as missing:
        RR.choose(root, "9")
    assert "has no restart 9" in missing.value.reason
    assert main(["render", str(root), "--list", "--json"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert listed["default"]["number"] == 3 and len(listed["restarts"]) == 3


def test_a_run_without_a_restart_is_told_where_its_pictures_are(tmp_path, capsys):
    root = tmp_path / "early"
    (root / "segments" / "001" / "frames" / "frame-000000").mkdir(parents=True)
    assert main(["render", str(root), "--product", "updraft", "--json"]) == 2
    refusal = json.loads(capsys.readouterr().err)
    assert refusal["code"] == "CM1-RENDER-NO-RESTART"
    assert "quicklook" in refusal["remedy"]


# ----------------------------------------------------------------------------------------------- reading it
def test_the_padded_winds_are_cut_to_the_model_cells_and_every_array_is_checked(run):
    _, choice, _ = RR.choose(run, None)
    frame = RR.RestartFrame(choice)
    padded = frame.raw("padded.u3d")
    u = frame.array("u3d")
    assert u.shape == (NZ, NY, NX + 1)
    assert np.array_equal(u, padded[1:-1, H:-H, H:-H])
    assert not np.any(frame.array("w3d") == -999.0)
    assert frame.array("q3d").shape == (6, NZ, NY, NX) and frame.array("nc3d").shape == (4, NZ, NY, NX)
    assert np.array_equal(frame.array("qr"), frame.raw("fields.q3d")[2])
    assert set(frame.checked) >= {"padded.u3d", "padded.w3d", "fields.q3d"}


def test_a_damaged_array_or_another_receipt_is_refused(run):
    directory = run / "segments" / "001" / "restarts" / "restart-000001"
    envelope = directory / RR.RESTART_FILE
    data = bytearray(envelope.read_bytes())
    data[-5] ^= 0xFF
    envelope.write_bytes(bytes(data))
    frame = RR.RestartFrame(RR.choose(run, None)[1])
    with pytest.raises(RR.RestartRenderError) as damaged:
        for name in sorted(frame._arrays):   # noqa: SLF001 - every array, so the damaged one is reached
            frame.raw(name)
    assert damaged.value.code == "CM1-HISTORY-ENVELOPE" and "SHA-256" in damaged.value.reason
    receipt = json.loads((directory / RR.PUBLICATION).read_text())
    receipt["forecast_envelope"]["contract_sha256"] = "d" * 64
    (directory / RR.PUBLICATION).write_text(json.dumps(receipt))
    with pytest.raises(RR.RestartRenderError) as other:
        RR.RestartFrame(RR.choose(run, None)[1])
    assert other.value.code == "CM1-RENDER-CONTRACT-MISMATCH"


def test_the_temperature_comes_from_the_equation_of_state():
    # th = T / (p/p00)^(rd/cp) with T = p / (rd rho (1 + qv reps)): the restart's 280 K air at qv = 0.01
    # (the synthetic rho left out qv, so T is 280 / (1 + 0.01 reps)).
    import tempfile

    with tempfile.TemporaryDirectory() as scratch:
        root = Path(scratch) / "r"
        write_restart(root)
        frame = RR.RestartFrame(RR.choose(root, None)[1])
        th = frame.array("th").astype(np.float64)
        prs = frame.raw("fields.prs").astype(np.float64)
        temperature = th * (prs / RR.P00) ** (RR.RD / RR.CP)
        assert np.allclose(temperature, 280.0 / (1.0 + 0.01 * RR.REPS), rtol=1e-5)


# ----------------------------------------------------------------------------------------------- the products
def _picture(run: Path, product: str, **request) -> RR.Picture:
    _, choice, _ = RR.choose(run, None)
    planes = RR.Planes(RR.RestartFrame(choice), False)
    return RR.build(planes, RR.Request(product, **request))


@pytest.mark.parametrize("product", PRODUCTS)
def test_every_product_draws_from_a_restart_and_says_how(run, product):
    request = {"field": "th3d"} if product == "scalar" else {}
    picture = _picture(run, product, **request)
    assert picture.planes and all(np.isfinite(p).all() for p in picture.planes)
    shapes = {p.shape for p in picture.planes}
    assert len(shapes) == 1
    assert picture.method, product
    words = " ".join(picture.method + [picture.scale_words, picture.level_words, picture.title or "",
                                       picture.panel_title or ""])
    assert "tornad" not in words.lower()
    if product in ("supercell", "vortex-composite", "storm-fusion"):
        assert len(picture.planes) == 3 and picture.zeta_max == RR.spin_range(DX)
    if product.startswith("swath-"):
        # The domain moves, so the swath is the ground-fixed canvas, with its own shape.
        assert shapes == {(9, 13)} and picture.extent_m == (13 * DX, 9 * DX)
    else:
        assert picture.extent_m == (NX * DX, NY * DX)


def test_the_derived_fields_are_the_quick_looks_own(run):
    _, choice, _ = RR.choose(run, None)
    planes = RR.Planes(RR.RestartFrame(choice), False)
    w = planes.frame.array("w3d")
    assert np.array_equal(_picture(run, "updraft").planes[0], w.max(axis=0))
    assert float(_picture(run, "updraft").planes[0].max()) == 25.0
    zeta = _picture(run, "vorticity", height_m=0.0).planes[0]
    assert np.array_equal(zeta, planes.qf.zeta_level(0)[0])
    refl = _picture(run, "reflectivity")
    assert refl.planes[0].max() > 5.0 and "calc_refl10cm" in " ".join(refl.method)
    level = _picture(run, "reflectivity-level")
    assert "linear in Z" in level.level_words and level.title == "GM1 REFLECTIVITY AT 1 KM"
    third = _picture(run, "cold-pool", level=3)
    assert np.array_equal(third.planes[0], planes.frame.array("th3d")[2]) and "level 3 of 12" in third.level_words


def test_vorticity_is_cm1s_calcvort_with_the_halo_at_the_domains_edge(run):
    # CM1 r21.1 misclibs.F:2037-2049: every corner, the edge ones included, differences into the halo CM1's
    # boundary conditions filled; a cell is the quarter-sum of its corners.  Checked against a float64 sum of
    # the same words, with the halo spelled out here as the edge copied outward (an open boundary).
    _, choice, _ = RR.choose(run, None)
    frame = RR.RestartFrame(choice)
    assert frame.has("zvort") and "halo" in frame.carriers["zvort"].meaning
    zeta = frame.array("zvort")
    u = np.pad(frame.array("u3d").astype(np.float64), ((0, 0), (1, 1), (0, 0)), mode="edge")
    v = np.pad(frame.array("v3d").astype(np.float64), ((0, 0), (0, 0), (1, 1)), mode="edge")
    corners = (v[:, :, 1:] - v[:, :, :-1]) / DX - (u[:, 1:, :] - u[:, :-1, :]) / DX
    expected = 0.25 * (corners[:, :-1, :-1] + corners[:, :-1, 1:] + corners[:, 1:, :-1] + corners[:, 1:, 1:])
    assert zeta.shape == (NZ, NY, NX) and zeta.dtype == np.float32
    assert np.allclose(zeta, expected, rtol=1e-5, atol=1e-8)
    # The ring is the halo's, not a copy of the inner neighbour: on a field with real edge gradients they differ.
    assert not np.array_equal(zeta[:, 0, :], zeta[:, 1, :])
    # The quick-look code reads it (vorticity, the three panels and updraft helicity all go through it).
    planes = RR.Planes(frame, False)
    assert np.array_equal(planes.zeta(4), zeta[4])
    words = " ".join(_picture(run, "vorticity").method)
    assert "boundary halo" in words and "calcvort" in words


def test_a_restart_without_a_sideways_halo_falls_back_to_the_quick_looks_spin(tmp_path):
    root = tmp_path / "nohalo"
    write_restart(root, sideways_halo=False)
    frame = RR.RestartFrame(RR.choose(root, None)[1])
    assert not frame.has("zvort")
    planes = RR.Planes(frame, False)
    zeta, source = planes.qf.zeta_level(0)
    assert source.startswith("dv/dx") and zeta.shape == (NY, NX) and np.isfinite(zeta).all()


def test_a_hurricane_takes_the_quick_looks_updraft_and_wind_ranges(run):
    from cm1core.quicklook.products import HURRICANE_W_RANGE, HURRICANE_WIND_RANGE, SWATH_WIND_RANGE

    _, choice, _ = RR.choose(run, None)
    storm = RR.Planes(RR.RestartFrame(choice), False)
    hurricane = RR.Planes(RR.RestartFrame(choice), True)
    assert RR.build(storm, RR.Request("swath-wind")).bounds == (0.0, SWATH_WIND_RANGE) == (0.0, 40.0)
    wind = RR.build(hurricane, RR.Request("swath-wind"))
    assert wind.bounds == (0.0, HURRICANE_WIND_RANGE) and "29" not in wind.scale_words
    assert RR.build(storm, RR.Request("updraft")).bounds is None   # the renderer's own 0-60
    assert RR.build(hurricane, RR.Request("updraft")).bounds == (0.0, HURRICANE_W_RANGE) == (0.0, 10.0)
    assert RR.build(hurricane, RR.Request("supercell")).w_max == HURRICANE_W_RANGE
    assert RR.build(storm, RR.Request("supercell")).w_max == 60.0


def test_updraft_helicity_takes_the_quick_looks_range_on_a_fine_grid(run):
    from cm1core.quicklook.products import UH_RANGE

    assert RR.uh_range(1000.0, False) == UH_RANGE and RR.uh_range(500.0, False) == UH_RANGE
    assert RR.uh_range(3000.0, False) is None and RR.uh_range(1000.0, True) is None
    for product in ("updraft-helicity", "swath-helicity"):
        picture = _picture(run, product)
        assert picture.bounds == (0.0, UH_RANGE) and "quick-looks' range" in picture.scale_words
    assert "thresholds" in RR.uh_scale_words(None) and "et al" not in RR.uh_scale_words(None)


def test_the_three_panels_name_the_height_of_their_spin(run, tmp_path, monkeypatch, capsys):
    picture = _picture(run, "supercell", level=3)
    assert picture.spin_title == "VERTICAL VORTICITY AT 1.25 KM"
    assert _picture(run, "supercell").spin_title == "VERTICAL VORTICITY AT 250 M"
    assert _picture(run, "vortex-composite", level=3).spin_title is None
    seen = _fake_renderer(tmp_path, monkeypatch)
    assert main(["render", str(run), "--product", "supercell", "--height", "1300", "--json"]) == 0
    assert _flags(seen["command"], "--spin-title") == ["VERTICAL VORTICITY AT 1.25 KM"]
    capsys.readouterr()
    assert main(["render", str(run), "--product", "vortex-composite", "--json"]) == 0
    assert not _flags(seen["command"], "--spin-title")


def test_a_height_above_the_model_is_refused_not_snapped_to_its_top(run, capsys):
    for words in (["--product", "vorticity", "--height", "1e9"], ["--product", "cold-pool", "--height", "50000"],
                  ["--product", "scalar", "--field", "w3d", "--height", "6001"]):
        assert main(["render", str(run), *words, "--json"]) == 2, words
        refusal = json.loads(capsys.readouterr().err)
        assert refusal["code"] == "CM1-RENDER-REQUEST" and "above this run's highest" in refusal["reason"], words
    # The top itself is a height: the highest w level is 6 km.
    assert _picture(run, "scalar", field="w3d", height_m=6000.0).level_words.startswith("w level 13 of 13")


def test_whole_is_refused_on_a_run_that_is_not_a_hurricane(run, capsys):
    assert main(["render", str(run), "--product", "updraft", "--whole", "--json"]) == 2
    refusal = json.loads(capsys.readouterr().err)
    assert refusal["code"] == "CM1-RENDER-REQUEST" and "hurricane" in refusal["reason"]
    assert not (run / "quicklook").exists()


def test_a_mistyped_run_name_or_a_run_file_is_told_so(run, tmp_path, capsys):
    typo = run.parent / (run.name + "x")
    assert main(["render", str(typo), "--product", "updraft", "--json"]) == 2
    refusal = json.loads(capsys.readouterr().err)
    assert refusal["code"] == "CM1-HISTORY-PATH" and "does not exist" in refusal["reason"]
    assert run.name in refusal["remedy"]
    (run / "run.json").write_text("{}")
    assert main(["render", str(run / "run.json"), "--product", "updraft", "--json"]) == 2
    refusal = json.loads(capsys.readouterr().err)
    assert refusal["code"] == "CM1-RENDER-RUN" and "not a restart checkpoint" in refusal["reason"]
    assert str(run.resolve()) in refusal["remedy"]
    # The envelope route keeps its own words for a missing envelope.
    missing = tmp_path / "gone.cm1fhs"
    assert main(["render", str(missing), "--field", "w", "--product", "updraft",
                 "--output", str(tmp_path / "x.png"), "--json"]) == 2
    assert json.loads(capsys.readouterr().err)["code"] == "CM1-HISTORY-PATH"


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root writes anywhere")
def test_a_run_folder_the_user_cannot_write_is_refused_with_a_way_out(run, tmp_path, capsys):
    os.chmod(run, 0o555)
    try:
        assert main(["render", str(run), "--product", "updraft", "--json"]) == 2
        refusal = json.loads(capsys.readouterr().err)
        assert refusal["code"] == "CM1-RENDER-OUTPUT" and "permission denied" in refusal["reason"]
        assert "--output" in refusal["remedy"]
    finally:
        os.chmod(run, 0o755)


def test_units_are_written_as_the_pictures_write_them(run):
    assert RR.plain_units("m s-1") == "m/s"
    assert RR.plain_units("kg kg-1") == "kg/kg"
    assert RR.plain_units("s-1") == "1/s"
    assert RR.plain_units("m2 s-2") == "m2/s2"
    assert RR.plain_units("kg m-3") == "kg/m3"
    assert RR.plain_units("kg kg-1 s-1") == "kg/kg/s"
    assert RR.plain_units("#/kg") == "1/kg"
    assert RR.plain_units("K") == "K" and RR.plain_units("Pa") == "Pa"
    scalar = _picture(run, "scalar", field="w3d", level=4)
    assert scalar.units == "m/s" and scalar.legend_units == "M/S"
    assert "m s-1" not in " ".join(scalar.method)
    assert _picture(run, "scalar", field="th3d").panel_title.startswith("TH3D, POTENTIAL TEMPERATURE PERTURBATION")


def test_the_gui_learns_what_a_restart_cannot_draw_from_its_receipt(tmp_path):
    storm = tmp_path / "storm"
    assert RR.cannot_draw(write_restart(storm)) == {}
    wsm6 = RR.cannot_draw(write_restart(tmp_path / "wsm6", microphysics="wsm6-mp6"))
    assert set(wsm6) == set(RR.REFLECTIVITY_PRODUCTS)
    assert all(reason.startswith("WSM6 writes its reflectivity to the history frames only") for reason in wsm6.values())
    stretched = RR.cannot_draw(write_restart(tmp_path / "s", stretched=True))
    assert set(stretched) == {"swath-vorticity", "swath-helicity"}
    assert "stretched levels" in stretched["swath-vorticity"]
    assert RR.cannot_draw(tmp_path / "nowhere") == {}


def test_what_the_gui_server_asks_needs_no_numpy_and_matches_the_quick_looks(tmp_path):
    # The GUI's server never imports numpy (test_gui_no_cupy); cannot_draw lives where the help's words do.
    from cm1core import render_products as P
    from cm1core.quicklook import reflectivity as R

    assert P.CPU_REFLECTIVITY == R.MORRISON and P.MODEL_REFLECTIVITY == R.OWN_REFLECTIVITY
    assert P.SCHEME_WORDS == R.PLAIN_SCHEME
    assert {product: name for product, (name, _u, _w) in RR.SWATHS.items()} == P.SWATH_NAMES
    directory = write_restart(tmp_path / "w", microphysics="wsm6-mp6")
    script = ("import json, sys\nfrom cm1core.render_products import cannot_draw\n"
              f"out = cannot_draw({str(directory)!r})\n"
              "print(json.dumps({'numpy': 'numpy' in sys.modules, 'n': len(out)}))\n")
    src = Path(RR.__file__).resolve().parents[1]
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                          env={**os.environ, "PYTHONPATH": str(src)}, check=True)
    assert json.loads(done.stdout) == {"numpy": False, "n": len(RR.REFLECTIVITY_PRODUCTS)}


def test_a_swath_cell_the_domain_never_reached_is_drawn_as_no_swath(run):
    pressure = _picture(run, "swath-pressure")
    assert pressure.planes[0].max() == 0.0 and pressure.planes[0].min() == -150.0
    spin = _picture(run, "swath-vorticity")
    assert spin.planes[0].min() == 0.0
    assert "never reached" in " ".join(spin.method)


def test_a_hurricane_is_drawn_over_its_inner_area_unless_the_whole_is_asked_for(run):
    # The 8 km hurricane's 128 cells: the inner 500 km is 62 cells around the middle; the 4 km one's, 125.
    assert RR.inner_window(128, 8000.0, 500_000.0) == slice(33, 95)
    assert RR.inner_window(256, 4000.0, 500_000.0) == slice(65, 190)
    assert RR.inner_window(8, 1000.0, 500_000.0) == slice(0, 8)
    _, choice, _ = RR.choose(run, None)
    planes = RR.Planes(RR.RestartFrame(choice), True)
    picture = RR.Picture("updraft", [np.zeros((600, 600), np.float32)], ["w"], "m/s", extent_m=(6.0e5, 6.0e5))
    planes.qf.geometry = planes.qf.geometry.__class__(**{**planes.qf.geometry.__dict__, "nx": 600, "ny": 600})
    RR._inner(planes, picture)   # noqa: SLF001
    assert picture.planes[0].shape == (500, 500) and picture.extent_m == (5.0e5, 5.0e5)
    assert "inner 500 x 500 km" in picture.method[-1] and "--whole" in picture.method[-1]


def test_stretched_levels_draw_at_their_heights_and_refuse_the_swaths_gm1_does_not_keep(tmp_path):
    root = tmp_path / "s"
    write_restart(root, stretched=True)
    _, choice, _ = RR.choose(root, None)
    planes = RR.Planes(RR.RestartFrame(choice), False)
    assert planes.levels_note == "stretched levels" and planes.qf.stretched
    assert float(planes.qf.zh[0]) < 250.0
    for product in ("swath-vorticity", "swath-helicity"):
        with pytest.raises(RR.RestartRenderError) as refused:
            RR.build(planes, RR.Request(product))
        assert refused.value.code == "CM1-RENDER-NO-SWATH" and "stretched" in refused.value.reason
    assert RR.build(planes, RR.Request("updraft-helicity")).planes[0].shape == (NY, NX)


def test_reflectivity_of_a_wsm6_restart_is_refused_with_the_reason(tmp_path):
    root = tmp_path / "w"
    write_restart(root, microphysics="wsm6-mp6")
    for product in ("reflectivity", "supercell"):
        with pytest.raises(RR.RestartRenderError) as refused:
            _picture(root, product)
        assert refused.value.code == "CM1-RENDER-NO-REFLECTIVITY"
        assert "history frames only" in refused.value.reason
    assert _picture(root, "updraft").planes[0].max() == 25.0


def test_requests_a_restart_cannot_answer_are_refused_by_name(run, capsys):
    for words, code in ((["--product", "cape"], "CM1-RENDER-REQUEST"),
                        (["--product", "radar"], "CM1-RENDER-REQUEST"),
                        (["--product", "updraft", "--height", "500"], "CM1-RENDER-REQUEST"),
                        (["--product", "vorticity", "--level", "99"], "CM1-RENDER-REQUEST"),
                        (["--product", "scalar"], "CM1-RENDER-REQUEST"),
                        (["--product", "scalar", "--field", "nosuch"], "CM1-RENDER-FIELD")):
        assert main(["render", str(run), *words, "--json"]) == 2, words
        assert json.loads(capsys.readouterr().err)["code"] == code, words


# ----------------------------------------------------------------------------------------------- the door
def _fake_renderer(tmp_path: Path, monkeypatch) -> dict:
    native = tmp_path / "cm1-render"
    native.write_bytes(b"verified")
    monkeypatch.setattr(rust_tools, "verify_native_tools", lambda: {"cm1-render": native})
    seen: dict = {}
    real_run = subprocess.run

    def fake_run(command, **kwargs):
        if not command or str(command[0]) != str(native):
            return real_run(command, **kwargs)
        arguments = [str(v) for v in command]
        seen["command"] = arguments
        control = Path(arguments[arguments.index("--input") + 1])
        seen["control"] = control.read_text()
        seen["data"] = np.fromfile(control.with_suffix(".dat"), dtype="<f4")
        Path(arguments[arguments.index("--output") + 1]).write_bytes(b"\x89PNG\r\n\x1a\n" + b"\0" * 24)
        return subprocess.CompletedProcess(arguments, 0, "rendered\n", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    return seen


def _flags(command: list[str], flag: str) -> list[str]:
    return [command[i + 1] for i, word in enumerate(command) if word == flag]


def test_the_door_hands_the_renderer_the_planes_and_the_words(run, tmp_path, monkeypatch, capsys):
    seen = _fake_renderer(tmp_path, monkeypatch)
    assert main(["render", str(run), "--product", "updraft", "--json"]) == 0
    receipt = json.loads(capsys.readouterr().out)
    command = seen["command"]
    assert _flags(command, "--format") == ["grads"] and _flags(command, "--nz") == ["1"]
    assert _flags(command, "--nx") == [str(NX)] and _flags(command, "--ny") == [str(NY)]
    # The plane handed over is the column maximum of w, exactly.
    frame = RR.RestartFrame(RR.choose(run, None)[1])
    assert np.array_equal(seen["data"].reshape(NY, NX), frame.array("w3d").max(axis=0))
    assert "xdef 8 linear" in seen["control"] and "w_column_max 0 99" in seen["control"]
    note = _flags(command, "--note")[0]
    assert note == "Restart checkpoint at T+1:00:00, not a history frame"
    footers = _flags(command, "--footer")
    low = float(frame.array("w3d").max(axis=0).min())
    assert footers[0].startswith(f"Values in this picture: {RR.fmt_value(low)} to 25 m/s. Colour scale: 0 to 60 m/s")
    assert any("domain moves at 12.5/3 m/s ground-relative" in line for line in footers)
    assert any(line.startswith("simulated") for line in footers)
    assert _flags(command, "--time-words") == ["T+1:00:00"]
    assert _flags(command, "--aspect") == ["8000.0:6000.0"]
    assert all(ord(ch) < 128 for word in command for ch in word)
    # The picture goes to the run's quicklook/native/ folder, and the receipt says what it is.
    out = run / "quicklook" / "native" / "updraft-s001-r000001.png"
    assert out.is_file() and receipt["file"] == "native/updraft-s001-r000001.png"
    assert receipt["source"] == "restart" and receipt["restart"]["number"] == 1
    assert receipt["value_range"] == [[low, 25.0]] and receipt["caveat"].endswith("not a history frame")
    assert "padded.w3d" in receipt["arrays_checked"]
    # Drawn again in place; an explicit --output is never overwritten.
    assert main(["render", str(run), "--product", "updraft", "--json"]) == 0
    capsys.readouterr()
    assert main(["render", str(run), "--product", "updraft", "--output", str(out), "--json"]) == 2
    assert json.loads(capsys.readouterr().err)["code"] == "CM1-RENDER-OUTPUT"


def test_the_three_field_products_take_the_spin_scale_of_the_grid(run, tmp_path, monkeypatch, capsys):
    seen = _fake_renderer(tmp_path, monkeypatch)
    assert main(["render", str(run), "--product", "vortex-composite", "--json"]) == 0
    command = seen["command"]
    assert _flags(command, "--w-max") == ["30.0"] and _flags(command, "--zeta-max") == [repr(RR.spin_range(DX))]
    assert _flags(command, "--classification")[0].endswith("NOT A HISTORY FRAME")
    assert not _flags(command, "--note")
    assert len(seen["data"]) == 3 * NX * NY
    capsys.readouterr()
    assert main(["render", str(run), "--product", "swath-wind", "--json"]) == 0
    command = seen["command"]
    assert _flags(command, "--nx") == ["13"] and _flags(command, "--aspect") == ["13000.0:9000.0"]
    assert any("ground-fixed canvas of 13 x 9 km" in line for line in _flags(command, "--footer"))


def test_an_envelope_with_a_field_still_takes_the_envelope_route(tmp_path, run):
    from argparse import Namespace

    from cm1core import doors

    def request(history, field):
        return Namespace(history=history, field=field, restart=None, list=False, list_fields=False, height=None)

    history = tmp_path / "frame.cm1fhs"
    history.write_bytes(b"x")
    assert doors._restart_route(request(history, ["w"])) is False
    assert doors._restart_route(request(tmp_path, [])) is True
    envelope = run / "segments" / "001" / "restarts" / "restart-000001" / RR.RESTART_FILE
    assert doors._restart_route(request(envelope, [])) is True
    assert doors._restart_route(request(envelope, ["padded.w3d"])) is False


def test_render_help_names_every_run_folder_product(capsys):
    with pytest.raises(SystemExit):
        main(["render", "--help"])
    text = capsys.readouterr().out
    for product in PRODUCTS:
        assert product in text
    assert "not a history frame" in text and "--restart" in text


def test_the_help_names_the_products_an_envelope_takes(tmp_path, monkeypatch, capsys):
    # The envelope route refuses an unknown product with its admitted set; the help names that set, no more.
    from cm1core import doors
    from cm1core.cli import ENVELOPE_RENDER_PRODUCTS

    envelope = tmp_path / "frame.cm1fhs"
    envelope.write_bytes(b"x")
    monkeypatch.setattr(doors, "_forecast_envelope_header", lambda path, fmt: {"contract_sha256": "c" * 64})
    code = main(["render", str(envelope), "--field", "w", "--product", "nosuch", "--contract-sha256", "c" * 64,
                 "--output", str(tmp_path / "x.png"), "--json"])
    refusal = json.loads(capsys.readouterr().err)
    assert code == 2 and refusal["code"] == "CM1-RENDER-REQUEST" and "admitted values" in refusal["reason"]
    assert set(re.findall(r"'([a-z-]+)'", refusal["reason"])) == set(ENVELOPE_RENDER_PRODUCTS)
    with pytest.raises(SystemExit):
        main(["render", "--help"])
    assert "An envelope with --field takes" in " ".join(capsys.readouterr().out.split())


# ----------------------------------------------------------------------------------------------- the real renderer
def _real_renderer() -> Path | None:
    explicit = os.environ.get("GM1_TEST_CM1_RENDER")
    if explicit and Path(explicit).is_file():
        return Path(explicit)
    try:
        return rust_tools.verify_native_tools()["cm1-render"]
    except Exception:  # noqa: BLE001 - a source checkout without the wheel's sealed tools
        return None


def _png_size_and_text(path: Path) -> tuple[tuple[int, int], dict[str, str]]:
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    offset, size, text = 8, (0, 0), {}
    while offset < len(data):
        length = int.from_bytes(data[offset:offset + 4], "big")
        kind = data[offset + 4:offset + 8]
        body = data[offset + 8:offset + 8 + length]
        if kind == b"IHDR":
            size = (int.from_bytes(body[:4], "big"), int.from_bytes(body[4:8], "big"))
        elif kind == b"tEXt":
            key, _, value = body.partition(b"\0")
            text[key.decode()] = value.decode("latin-1")
        offset += 12 + length
    return size, text


@pytest.mark.skipif(_real_renderer() is None, reason="no built cm1-render here (set GM1_TEST_CM1_RENDER)")
@pytest.mark.parametrize("product", ["updraft", "supercell", "swath-pressure"])
def test_the_built_renderer_draws_the_footer_under_the_picture(run, monkeypatch, capsys, product):
    binary = _real_renderer()
    monkeypatch.setattr(rust_tools, "verify_native_tools", lambda: {"cm1-render": binary})
    assert main(["render", str(run), "--product", product, "--json"]) == 0
    receipt = json.loads(capsys.readouterr().out)
    (width, height), text = _png_size_and_text(Path(receipt["output"]))
    assert text["Note" if product != "vortex-composite" else "Footer"].endswith("not a history frame")
    assert "Values in this picture" in text["Footer"]
    if product == "supercell":
        assert width == 1620
    else:
        # A 13 x 9 km ground canvas (the swath) or an 8 x 6 km domain keeps its shape: less tall than square.
        assert width == 1020 and height < 1100 + 400
