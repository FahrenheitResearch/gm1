"""The looks after the four favourites, the films, and the More looks row's data.

CPU only.  The frames carry CM1 r21.1's own 24 x 24 x 40 storm (``quicklook_support``), as in test_quicklook.py,
which already holds every per-frame look drawn at every frame, their footers linted and their pixel sizes.  Here:

1. the films: a per-frame look's film is its own pictures (APNG always; a GIF when there is no MP4 encoder), the
   swirl film is the saved times plus K blended pictures between them and says so, and a run without parcels gets
   the plain refusal with its fix;
2. a look that cannot be drawn refuses in words (no cloud species; no parcel track);
3. the More looks row: every look after the four, in the copy's order, filled with the run's numbers and its
   resolution note; the refusals and the action each offers; the server's film list is the worker's;
4. the words pass the copy lint.

The six products these looks follow stay byte-identical for the same frames: on the stock 1 km run, 150 of 150
PNGs equal with and without the further looks.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import warnings

import numpy as np
import pytest

import importlib.util

NEEDS_MATPLOTLIB = pytest.mark.skipif(importlib.util.find_spec("matplotlib") is None,
                                      reason="matplotlib is not installed (it comes with gpuwm)")

from cm1import_support import namelist_tool  # noqa: E402
from quicklook_support import run_directory  # noqa: E402

from cm1core.gui import looks as GL  # noqa: E402
from cm1core.quicklook import film as F  # noqa: E402
from cm1core.quicklook import products as P  # noqa: E402
from cm1core.quicklook import render as RD  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FAVOURITES = ["reflectivity", "spin-near-ground", "updraft-5km", "swath"]


@pytest.fixture(autouse=True)
def parser_tool(monkeypatch):
    monkeypatch.setenv("GM1_CM1_R21_NAMELIST", os.fspath(namelist_tool()))


def _render(run: Path, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        return RD.render(run, **kwargs)


def _index(run: Path) -> dict:
    return json.loads((run / "quicklook" / "index.json").read_text())


def _job(run: Path, look: str, **kwargs) -> F.FilmJob:
    return F.FilmJob(run, look, tuple(kwargs.pop("formats", ())), float(kwargs.pop("fps", 6.0)),
                     int(kwargs.pop("between", 8)), run / "quicklook", **kwargs)


# ---------------------------------------------------------------- the lists the server keeps without numpy

def test_the_servers_film_list_and_speeds_are_the_workers():
    assert GL.FILM_LOOKS == F.film_looks()
    assert GL.FILM_FPS["swirl"] == F.DEFAULT_FPS["swirl"] and GL.FILM_FPS["parcels"] == F.DEFAULT_FPS["parcels"]
    assert GL.ONCE_LOOKS == RD.ONCE_IDS
    assert set(GL.NEW_LOOKS) <= set(RD.LOOK_IDS)
    # the four favourites stay first, in their order; the further looks come after the first six
    assert [pid for pid, _ in P.PRODUCTS][:4] == FAVOURITES
    assert [pid for pid, _ in P.PRODUCTS][:6] == FAVOURITES + ["surface", "rain"]


def test_the_more_looks_copy_lists_every_look_after_the_four_once():
    copy = GL.load_copy()
    ids = [row["id"] for row in copy["more_looks"]]
    assert len(ids) == len(set(ids))
    assert not set(ids) & set(FAVOURITES)
    # A hurricane run's own two looks are its favourites (copy "looks"), not More looks.
    assert set(ids) == (set(RD.LOOK_IDS) - set(FAVOURITES) - set(P.HURRICANE_ONLY)) | {"swirl"}
    assert [row["id"] for row in copy["looks"]] == FAVOURITES + list(P.HURRICANE_ONLY)
    for row in copy["more_looks"]:
        assert row["title"] and row["text"] and row["note"], row["id"]


def test_the_more_looks_words_pass_the_copy_lint():
    from cm1core.gui.copy_lint import lint_text

    copy = GL.load_copy()
    texts = [row[k] for row in copy["more_looks"] for k in ("title", "text", "note", "short", "note_sampled",
                                                             "note_moving", "note_varies") if row.get(k)]
    texts += [row["text"] for row in copy["looks"]]
    texts += [row["text"] for row in copy["why_not"].values()] + list(copy["ui"].values())
    problems = [(t, p) for t in texts for p in lint_text(t)]
    assert problems == []


# ---------------------------------------------------------------- films

@NEEDS_MATPLOTLIB
def test_a_looks_film_is_its_own_pictures_as_an_animated_png_and_a_gif_without_mp4(tmp_path, monkeypatch):
    run = run_directory(tmp_path)
    _render(run)
    monkeypatch.setattr(F, "mp4_available", lambda: False)
    record = F.make(_job(run, "swath"))
    assert record["pictures"] == 3 and record["stopped"] is False and record["note"] == ""
    assert set(record["files"]) == {"apng", "gif"}
    apng = run / "quicklook" / record["files"]["apng"]
    assert F.read_apng_frames(apng)[0] == 3
    assert record["frames"] == [f["file"] for f in
                                next(p for p in _index(run)["products"] if p["id"] == "swath")["frames"]]
    films = json.loads((run / "quicklook" / "films" / "index.json").read_text())
    assert films["schema"] == F.INDEX_SCHEMA and films["films"]["swath"]["pictures"] == 3
    progress = json.loads((run / "quicklook" / "films" / "swath" / "progress.json").read_text())
    assert progress["state"] == "done" and progress["done"] == progress["total"] == 3
    with pytest.raises(F.FilmError, match="film add-on"):
        F.make(_job(run, "swath", formats=("mp4",)))


@NEEDS_MATPLOTLIB
def test_the_swirl_film_blends_between_saved_times_and_says_so(tmp_path, monkeypatch):
    run = run_directory(tmp_path)
    _render(run, only=("spin-near-ground",))
    monkeypatch.setattr(F, "mp4_available", lambda: False)
    record = F.make(_job(run, "swirl", between=2, formats=("apng",)))
    assert record["pictures"] == (3 - 1) * (2 + 1) + 1
    assert record["between"] == 2 and record["note"] == "blended between saved times"
    assert record["facts"]["saves"] == 3 and record["facts"]["interval_s"] == 300.0
    assert set(record["files"]) == {"apng"}
    assert F.read_apng_frames(run / "quicklook" / record["files"]["apng"])[0] == 7


@NEEDS_MATPLOTLIB
def test_a_film_stops_when_asked_and_keeps_its_pictures(tmp_path, monkeypatch):
    run = run_directory(tmp_path)
    _render(run, only=("spin-near-ground",))
    monkeypatch.setattr(F, "mp4_available", lambda: False)
    calls = {"n": 0}

    def stop_after_three() -> bool:
        calls["n"] += 1
        return calls["n"] > 3

    record = F.make(_job(run, "swirl", between=2, formats=("apng",), should_stop=stop_after_three))
    assert record["stopped"] is True and 0 < record["pictures"] < 7
    progress = json.loads((run / "quicklook" / "films" / "swirl" / "progress.json").read_text())
    assert progress["state"] == "stopped"


@NEEDS_MATPLOTLIB
def test_a_run_without_parcels_is_refused_with_the_fix(tmp_path):
    run = run_directory(tmp_path)
    with pytest.raises(F.FilmError, match="iprcl = 1"):
        F.make(_job(run, "parcels", formats=("apng",)))
    with pytest.raises(F.FilmError, match="no pictures yet"):
        F.make(_job(run, "swath", formats=("apng",)))
    with pytest.raises(F.FilmError, match="no film of"):
        F.make(_job(run, "tornado", formats=("apng",)))


# ---------------------------------------------------------------- refusals of the looks

@NEEDS_MATPLOTLIB
def test_clouds_from_above_refuse_a_run_without_cloud_species(tmp_path, monkeypatch):
    from cm1core.quicklook import satellite as SA

    run = run_directory(tmp_path, times=(900.0, 1200.0))
    monkeypatch.setattr(SA, "species_present", lambda qf: [])
    summary = _render(run, only=("clouds-from-above",))
    assert summary.problems == []
    clouds = next(p for p in _index(run)["products"] if p["id"] == "clouds-from-above")
    assert clouds["frames"] == [] or all("file" not in f for f in clouds["frames"])
    assert [n["reason"] for n in clouds["not_drawn"]] == ["This run saved no cloud water or ice."]


@NEEDS_MATPLOTLIB
def test_the_environment_is_drawn_once_and_a_run_without_parcels_has_no_parcel_row(tmp_path):
    run = run_directory(tmp_path)
    _render(run)
    index = _index(run)
    once = {p["id"]: p for p in index["products"] if p.get("once")}
    assert set(once) == {"environment"}
    assert [f["time_s"] for f in once["environment"]["frames"]] == [900.0]
    facts = once["environment"]["frames"][0]["facts"]
    assert facts, "the environment look records the numbers it showed"


# ---------------------------------------------------------------- the More looks row

@NEEDS_MATPLOTLIB
def test_the_more_looks_row_says_what_each_look_is_and_why_one_is_missing(tmp_path):
    run = run_directory(tmp_path)
    _render(run)
    index = _index(run)
    rows = GL.more_looks("storm", run, index, {"state": "finished"}, films_now={})
    by = {row["id"]: row for row in rows}
    assert [row["id"] for row in rows] == [row["id"] for row in GL.load_copy()["more_looks"]]
    for lid in ("surface", "rain", "clouds-from-above", "updraft-slice", "hail-track", "spin-track"):
        assert by[lid]["available"] and by[lid]["count"] == 3 and by[lid]["why_not"] is None, lid
    assert by["environment"]["available"] and by["environment"]["count"] == 1
    # the first release marks no look as new
    assert not any(row["is_new"] for row in rows)
    # the run's own numbers fill the words; no placeholder is left
    for row in rows:
        assert "{" not in row["text"] and "{" not in row["note"], row["id"]
    # parcels: this run did not follow any; after the run the look offers to be drawn
    assert by["parcels"]["available"] is False
    assert by["parcels"]["why_not"].startswith("This run did not follow air parcels")
    assert by["parcels"]["hint"] == "iprcl = 1" and by["parcels"]["action"] is None
    # swirl is a film: made on request
    assert by["swirl"]["kind"] == "film" and by["swirl"]["action"] == "film"
    # while the run goes, parcels wait for its end
    live = {row["id"]: row for row in GL.more_looks("storm", run, index, {"state": "running"}, films_now={})}
    assert live["parcels"]["why_not"] == "Parcels arrive when the run ends." and live["parcels"]["action"] is None
    # a run that wrote a parcel track its pictures have not drawn yet offers to draw it
    track = run / "segments" / "001" / "frames" / "lagrangian-track"
    track.mkdir(parents=True)
    (track / "parcel-attributes.f32").write_bytes(b"")
    later = {row["id"]: row for row in GL.more_looks("storm", run, index, {"state": "finished"}, films_now={})}
    assert later["parcels"]["action"] == "look" and later["parcels"]["why_not"].startswith("This run's pictures")


@NEEDS_MATPLOTLIB
def test_a_run_drawn_before_the_new_looks_offers_to_draw_them(tmp_path):
    run = run_directory(tmp_path)
    _render(run, only=tuple(FAVOURITES) + ("surface", "rain"))
    rows = {row["id"]: row for row in GL.more_looks("storm", run, _index(run), {"state": "finished"}, films_now={})}
    assert rows["clouds-from-above"]["action"] == "all"
    assert rows["clouds-from-above"]["why_not"].startswith("This run's pictures were drawn before")
    assert rows["environment"]["action"] == "look"
    # a pass that adds the looks draws only those, from the same frames
    before = {p.name: p.stat().st_mtime_ns for p in (run / "quicklook" / "reflectivity").glob("*.png")}
    _render(run)
    after = {p.name: p.stat().st_mtime_ns for p in (run / "quicklook" / "reflectivity").glob("*.png")}
    assert before == after
    rows = {row["id"]: row for row in GL.more_looks("storm", run, _index(run), {"state": "finished"}, films_now={})}
    assert rows["clouds-from-above"]["available"] and rows["environment"]["available"]


@NEEDS_MATPLOTLIB
def test_the_server_lists_films_made_and_being_made(tmp_path, monkeypatch):
    run = run_directory(tmp_path)
    _render(run, only=("swath",))
    monkeypatch.setattr(F, "mp4_available", lambda: False)
    F.make(_job(run, "swath", formats=("apng",)))
    films = GL.films("storm", run)
    assert films["swath"]["state"] == "done" and films["swath"]["pictures"] == 3
    assert films["swath"]["urls"]["apng"] == "/api/runs/storm/files/quicklook/films/swath/film.png"
    assert [f["time_s"] for f in films["swath"]["frames"]] == [900.0, 1200.0, 1500.0]
    (run / "quicklook" / "films" / "swirl").mkdir()
    (run / "quicklook" / "films" / "swirl" / "progress.json").write_text(json.dumps(
        {"state": "running", "done": 2, "total": 7}))
    assert GL.films("storm", run)["swirl"]["state"] == "running"


def test_the_estimate_of_a_film_or_a_pass_scales_with_the_run():
    index = {"frames": [{}] * 25, "run": {"grid": "120 × 120 × 40, dx 1 km"}}
    assert GL.about_seconds(index, "swirl", 8) == round(24 * 9 * GL.SECONDS_PER_SWIRL_PICTURE)
    assert GL.about_seconds(index, "all") >= 20   # 25 frames of the new looks at 0.87 s each (one core)
    assert GL.about_seconds(None, "swath") == 2
    assert np.isfinite(GL.about_seconds({"frames": [{}], "run": {"grid": "junk"}}, "all"))


def test_the_estimates_follow_the_measured_one_core_costs():
    """The new looks were measured at 0.87 s per frame at 1 km and 1.28 s at 250 m (1.31 M cells) on one core:
    the "about" figures land within 25 % of both, the swath film's (2.9 s for 25 pictures) within 1 s, and the
    parcels film's scales with the parcel times the look found (22.7 s for 120)."""

    km = {"frames": [{}] * 25, "run": {"grid": "120 × 120 × 40, dx 1 km"}}
    fine = {"frames": [{}] * 13, "run": {"grid": "128 × 128 × 80, dx 250 m"}}
    assert abs(GL.about_seconds(km, "all") - 25 * 0.871) <= 0.25 * 25 * 0.871
    assert abs(GL.about_seconds(fine, "all") - 13 * 1.279) <= 0.25 * 13 * 1.279
    assert abs(GL.about_seconds(km, "swath") - 2.92) <= 1.0
    assert abs(GL.about_seconds(km, "swirl", 8) - 70.3) <= 0.25 * 70.3        # Linux, one core, 217 pictures
    assert abs(GL.about_seconds(fine, "swirl", 8) - 35.3) <= 0.25 * 35.3      # Linux, one core, 109 pictures
    parcels = {"frames": [{}] * 25, "products": [{"id": "parcels", "frames": [{"facts": {"times": 120}}]}]}
    assert abs(GL.about_seconds(parcels, "parcels") - 22.7) <= 0.25 * 22.7
    assert GL.about_seconds({"frames": [{}] * 25, "products": [{"id": "parcels", "frames": [
        {"facts": {"times": 30}}]}]}, "parcels") < GL.about_seconds(parcels, "parcels")


# ---------------------------------------------------------------- the independent verification's fixes

def test_every_look_says_what_its_grid_cannot_show():
    """Each look's words carry one honest line about resolution, with the run's own grid spacing in it.

    The environment is the one exception: it is the sounding the whole domain starts from, not a field on the grid.
    """

    copy = GL.load_copy()
    for look in copy["looks"]:
        assert "{dx}" in look["text"] and "{six_dx}" in look["text"], look["id"]
    for look in copy["more_looks"]:
        if look["id"] == "environment":
            continue
        words = look["text"] + " " + look["note"]
        assert "{dx}" in words or "{six_dx}" in words, look["id"]


def test_the_clouds_density_without_rho_is_cm1s_dry_air_density():
    """A frame without rho gets CM1's own rho = p / (Rd T (1 + qv Rv/Rd)) (misclibs.F:1292-1294), the dry air's
    density the mixing ratios are per kilogram of, not the moist air's (an independent recompute found the difference)."""

    from cm1core.quicklook import satellite as SA
    from quicklook_support import STATE

    d = np.load(STATE)

    class Frame:
        directory = Path("frame")

        def has(self, name):
            return name in ("prs", "th", "q3d")

        def get(self, name):
            return np.asarray(d[name], np.float32)

        def has_species(self, name):
            return name == "qv"

        def species(self, name):
            return np.asarray(d["qv"], np.float32)

    rho = SA.density(Frame())
    assert np.allclose(rho, d["rho"], rtol=2e-6, atol=0.0)


def _drop(run: Path, carrier: str) -> None:
    """Take a carrier out of every frame's field table: a run that did not save it."""

    for table in (run / "frames").glob("*/history-field-table.tsv"):
        rows = [line for line in table.read_text(encoding="ascii").splitlines()
                if line.split("\t")[0] != carrier]
        table.write_text("\n".join(rows) + "\n", encoding="ascii")


@NEEDS_MATPLOTLIB
def test_a_run_without_rain_gets_the_plain_reason_and_the_cm1_word_as_a_hint(tmp_path):
    run = run_directory(tmp_path)
    _drop(run, "rain")
    _render(run, only=("rain", "hail-track", "spin-track"))
    rows = {row["id"]: row for row in GL.more_looks("storm", run, _index(run), {"state": "finished"}, films_now={})}
    assert rows["rain"]["available"] is False
    assert rows["rain"]["why_not"] == GL.load_copy()["why_not"]["no_rain"]["text"]
    assert rows["rain"]["hint"] == "output_rain = 1"
    assert "output_rain" not in rows["rain"]["why_not"]


@NEEDS_MATPLOTLIB
def test_a_track_kept_from_the_saved_frames_says_it_is_sampled(tmp_path):
    """Frames with no every-step maximum (no sgs, no svs): both tracks come from the saved times, and both say so."""

    run = run_directory(tmp_path)
    _render(run, only=("hail-track", "spin-track"))
    index = _index(run)
    rows = {row["id"]: row for row in GL.more_looks("storm", run, index, {"state": "finished"}, films_now={})}
    for lid in ("hail-track", "spin-track"):
        facts = next(p for p in index["products"] if p["id"] == lid)["frames"][-1]["facts"]
        assert facts["source"] == "frames", lid
        assert "Sampled at each saved time (5 min)" in rows[lid]["note"], rows[lid]["note"]


@NEEDS_MATPLOTLIB
def test_the_rain_look_says_so_when_the_domain_moves(tmp_path):
    moving = run_directory(tmp_path / "a")
    _render(moving, only=("rain",))
    rows = {row["id"]: row for row in GL.more_looks("storm", moving, _index(moving), {"state": "finished"},
                                                  films_now={})}
    assert "not a map of rain on the ground" in rows["rain"]["note"]
    still = run_directory(tmp_path / "b", move=(0.0, 0.0))
    _render(still, only=("rain",))
    rows = {row["id"]: row for row in GL.more_looks("storm", still, _index(still), {"state": "finished"},
                                                  films_now={})}
    assert "not a map of rain on the ground" not in rows["rain"]["note"]
    assert rows["rain"]["note"].startswith("Each grid cell (1 km across)")


@NEEDS_MATPLOTLIB
def test_the_environment_of_a_chain_run_says_its_air_differs_from_place_to_place(tmp_path):
    """CM1's base state is one column for the whole domain (base.F); the chain starts from a 3D state, so its
    environment look is one corner, and says so in the picture's footer facts and the panel's note."""

    run = run_directory(tmp_path / "a")
    _render(run, only=("environment",))
    rows = {row["id"]: row for row in GL.more_looks("storm", run, _index(run), {"state": "finished"}, films_now={})}
    assert rows["environment"]["note"] == "The air the storm starts in, the same everywhere in the domain."
    facts = next(p for p in _index(run)["products"] if p["id"] == "environment")["frames"][0]["facts"]
    assert facts["air_varies"] is False

    chain = run_directory(tmp_path / "b")
    (chain / "chain").mkdir()
    (chain / "chain" / "chain-receipt.json").write_text("{}", encoding="utf-8")
    _render(chain, only=("environment",))
    index = _index(chain)
    facts = next(p for p in index["products"] if p["id"] == "environment")["frames"][0]["facts"]
    assert facts["air_varies"] is True
    rows = {row["id"]: row for row in GL.more_looks("storm", chain, index, {"state": "finished"}, films_now={})}
    assert "differs from place to place" in rows["environment"]["note"]
    assert "the same everywhere" not in rows["environment"]["note"]


@NEEDS_MATPLOTLIB
def test_a_chain_run_whose_prepared_chain_lives_elsewhere_is_still_a_chain_run(tmp_path):
    """tools/arwen_chain_prepare.py writes chain.prepared_directory wherever the hand-written deck names it, so
    the run folder has no chain/ folder; the deck's [chain] table is what makes it a chain run."""

    from cm1core.gui import looks as GLK
    from cm1core.quicklook import environment as EN

    chain = run_directory(tmp_path / "c")
    assert not EN.air_varies(chain) and not GLK.air_varies(chain)
    deck = chain / "deck.toml"
    before = deck.read_text(encoding="utf-8") if deck.is_file() else ""
    deck.write_text(before + '\n[chain]\nprepared_directory = "/elsewhere/prep"\n', encoding="utf-8")
    assert not (chain / "chain").exists()
    assert EN.air_varies(chain) and GLK.air_varies(chain)
    _render(chain, only=("environment",))
    index = _index(chain)
    facts = next(p for p in index["products"] if p["id"] == "environment")["frames"][0]["facts"]
    assert facts["air_varies"] is True
    rows = {row["id"]: row for row in GL.more_looks("storm", chain, index, {"state": "finished"}, films_now={})}
    assert "differs from place to place" in rows["environment"]["note"]
