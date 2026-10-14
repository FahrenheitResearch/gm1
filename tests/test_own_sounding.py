"""Use your own sounding: the layouts GM1 reads, the conversion, each refusal, the numbers and the start.

- each layout (CM1's input_sounding, a University of Wyoming text listing, a SHARPpy file, a table with named
  columns) is told apart and gives the same air as the sounding it was written from;
- each refusal says what is wrong in plain words, with the line of the user's own file, and what to do;
- warnings (superadiabatic layers, supersaturation) do not refuse;
- a sounding that stops below the model top is offered made-up levels, and says where they begin;
- the start chosen from the sounding: the push for the shipped balloons, the bubble for the made environments;
- `gpuwm-cm1 sounding convert` and `gpuwm-cm1 preset fit|new --sounding FILE --start ...` (the frame and the
  start's place are the famous-day preset's own for the same balloon);
- the stored archive pages give the shipped balloon files byte for byte (development repository only).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from cm1core.cm1import import own_sounding as O
from cm1core.cm1import import sounding_formats as F
import own_sounding_support as S

ROOT = Path(__file__).resolve().parents[1]
SHIPPED = ROOT / "src" / "cm1core" / "data" / "soundings"
TOP = 19750.0


def _library():
    from cm1core.cartesian_door import resolve_initializer_library

    try:
        return resolve_initializer_library(None)
    except Exception:  # noqa: BLE001 - no wheel natives on this host
        return None


def _parser() -> bool:
    from cm1core.cm1import import namelist

    try:
        namelist.locate_tool()
    except Exception:  # noqa: BLE001
        return False
    return True


needs_library = pytest.mark.skipif(_library() is None, reason="the initializer library (getcape) is not installed")
needs_parser = pytest.mark.skipif(not _parser(), reason="the cm1-r21-namelist tool is not built")


def _close(reading: F.Reading, rows, *, theta=0.06, qv=0.06, wind=0.08) -> None:
    """The converted profile is the air the rows were written from, to the rounding of the listing."""

    profile = reading.profile
    assert profile.levels == len(rows) - 1
    assert profile.surface_hpa == pytest.approx(rows[0]["p"], abs=0.06)
    assert profile.surface_theta_k == pytest.approx(rows[0]["theta"], abs=theta)
    assert profile.surface_qv_gkg == pytest.approx(rows[0]["qv"], abs=qv)
    for k, row in enumerate(rows[1:]):
        assert profile.z_m[k] == pytest.approx(row["agl"], abs=0.51)
        assert profile.theta_k[k] == pytest.approx(row["theta"], abs=theta * (1.0 + row["agl"] / 4000.0))
        assert profile.qv_gkg[k] == pytest.approx(row["qv"], abs=qv)
        assert profile.u_ms[k] == pytest.approx(row["u"], abs=wind + 0.01 * row["spd"])
        assert profile.v_ms[k] == pytest.approx(row["v"], abs=wind + 0.01 * row["spd"])


# ------------------------------------------------------------------ the layouts


def test_cm1s_own_file_is_kept_byte_for_byte():
    text = S.made()
    reading = F.read(text)
    assert reading.format == "cm1" and not reading.converted and reading.text == text
    assert reading.profile.levels == 240 and reading.profile.top_m == 24000.0
    assert F.render(reading.profile) == text
    problems, warnings = F.check(reading.profile, top_needed_m=TOP)
    assert problems == [] and warnings == []


@pytest.mark.parametrize("knots", [False, True], ids=["m-s", "knots"])
@pytest.mark.parametrize("page", [False, True], ids=["text", "saved-page"])
def test_a_wyoming_listing_is_converted_with_its_own_speed_unit(knots, page):
    rows = S.measured()
    reading = F.read(S.wyoming(rows, knots=knots, page=page))
    assert reading.format == "wyoming" and reading.converted
    _close(reading, rows)
    said = " ".join(reading.conversions)
    assert ("knots" in said) is knots and "Potential temperature" in said and "mixing ratio" in said
    assert reading.station_m == S.STATION_M
    assert F.read(reading.text).format == "cm1"       # what it writes is what CM1 reads


def test_a_wyoming_level_with_a_blank_value_is_left_out_and_named():
    rows = S.measured()
    lines = S.wyoming(rows).splitlines()
    # blank the dewpoint of the fourth level (columns 22-28), as the archive does for a missing value
    lines[7] = lines[7][:21] + "       " + lines[7][28:]
    reading = F.read("\n".join(lines) + "\n")
    assert reading.profile.levels == len(rows) - 2
    assert reading.dropped == ({"line": 8, "why": "no dewpoint"},)


def test_a_sharppy_file_is_read_in_knots():
    rows = S.measured()
    reading = F.read(S.sharppy(rows))
    assert reading.format == "sharppy"
    _close(reading, rows)
    assert "knots" in " ".join(reading.conversions)
    missing = S.sharppy(rows).replace("%END%", " 5.00,  -9999.00,  -9999.00,  -9999.00,  -9999.00,  -9999.00\n%END%")
    assert F.read(missing).profile.levels == len(rows) - 1


def test_a_table_names_its_columns_and_units():
    rows = S.measured()
    _close(F.read(S.table(rows)), rows)
    metres = F.read(S.table(rows, header="P;HGHT;TMPC;DWPC;WDIR;WSPD (m/s)", speed=1.0, delimiter=";"))
    _close(metres, rows)
    kelvin = F.read(S.table(rows, header="pres height temp dewpoint wdir sknt", kelvin=True, pascal=True,
                            delimiter=" "))
    _close(kelvin, rows)
    said = " ".join(kelvin.conversions)
    assert "Pressure read as Pa" in said and "Temperature read as K" in said and "Dewpoint read as K" in said
    components = F.read(S.table_uv(rows))
    _close(components, rows, wind=0.002)
    assert "the file's own u" in " ".join(components.conversions)


def test_a_table_whose_wind_speed_has_no_unit_is_asked_about_not_guessed():
    rows = S.measured()
    text = S.table(rows, header="pressure,height,temperature,dewpoint,direction,speed")
    with pytest.raises(F.SoundingRefused) as refused:
        F.read(text)
    assert refused.value.code == "speed-unit" and refused.value.offer == {"speed_unit": ["kt", "ms"]}
    assert "knots or metres per second" in refused.value.text and refused.value.line == 1
    _close(F.read(text, speed_unit="kt"), rows)
    fast = F.read(text, speed_unit="ms")
    assert fast.profile.u_ms[-1] == pytest.approx(rows[-1]["u"] / S.KNOT, rel=0.01)


def test_a_file_listed_from_the_top_down_is_read_from_the_bottom_up():
    rows = S.measured()
    lines = S.table(rows).splitlines()
    reading = F.read("\n".join([lines[0], *reversed(lines[1:])]) + "\n")
    _close(reading, rows)
    assert any("top of the sounding first" in note for note in reading.notes)


# ------------------------------------------------------------------ refusals, in plain words with the line


def _refused(text: str, **kwargs) -> dict:
    record = O.examine(text, name="mine.txt", top_needed_m=TOP, **kwargs)
    assert record["ok"] is False and record["problems"], record
    return record["problems"][0]


def test_each_unreadable_file_says_why_and_what_to_do():
    lines = S.made().splitlines()
    assert _refused("")["code"] == "empty"
    assert _refused("abc\x00def")["text"] == "The file is not text."
    words = _refused("hello there\nthis is not a sounding\n")
    assert words["code"] == "columns" and "does not name pressure" in words["text"] and words["line"] == 1
    surface = _refused("\n".join(lines[1:]) + "\n")
    assert surface["code"] == "surface-line" and surface["line"] == 1
    assert "The surface line is missing" in surface["text"] and "three numbers" in surface["fix"]
    comment = _refused("# my sounding\n" + S.made())
    assert comment["line"] == 1 and "comment" in comment["text"]
    letters = _refused("\n".join([*lines[:9], "900.0 abc 12.0 1.0 2.0", *lines[9:]]) + "\n")
    assert letters["code"] == "format" and letters["line"] == 10 and "'abc'" in letters["text"]
    no_wind = _refused(lines[0] + "\n" + "\n".join(" ".join(line.split()[:3]) for line in lines[1:]) + "\n")
    assert no_wind["code"] == "no-wind" and no_wind["line"] == 2 and "u (toward the east)" in no_wind["fix"]
    short_line = _refused("\n".join([*lines[:5], "500.0 301.0 13.0 4.0", *lines[5:]]) + "\n")
    assert short_line["line"] == 6 and "4 number(s)" in short_line["text"]


def test_too_few_levels_and_heights_out_of_order():
    lines = S.made().splitlines()
    few = _refused("\n".join(lines[:2]) + "\n")
    assert few["code"] == "levels" and "1 level(s)" in few["text"]
    order = _refused("\n".join([*lines[:5], lines[3], *lines[5:]]) + "\n")
    assert order["code"] == "heights" and order["line"] == 6
    assert "(300 m) is not above the one before it (400 m)" in order["text"]
    below = _refused("\n".join([lines[0], lines[1].replace("100.0000", "  0.0000", 1), *lines[2:]]) + "\n")
    assert below["code"] == "heights" and below["line"] == 2 and "above 0" in below["fix"]


def test_units_that_look_wrong_are_named():
    lines = S.made().splitlines()
    celsius = _refused(lines[0].replace("300.000000", " 27.000000") + "\n" + "\n".join(lines[1:]) + "\n")
    assert celsius["code"] == "temperature-unit" and celsius["line"] == 1
    assert "looks like a temperature in C" in celsius["text"] and "(1000 / p) ^ 0.2854" in celsius["fix"]
    pascal = _refused("100000.0 300.0 14.0\n" + "\n".join(lines[1:]) + "\n")
    assert pascal["code"] == "pressure-unit" and "It looks like Pa" in pascal["text"]
    # plain temperature in K where potential temperature belongs: it falls with height
    profile = S.profile()
    temperatures = F.temperatures_k(profile)
    plain = [lines[0], *("%s %.3f %s" % (line.split()[0], temperatures[k + 1], " ".join(line.split()[2:]))
                         for k, line in enumerate(lines[1:]))]
    falls = _refused("\n".join(plain) + "\n")
    assert falls["code"] == "temperature-unit" and "looks like plain temperature in K" in falls["text"]
    humid = _refused("\n".join([*lines[:3], lines[3].replace(lines[3].split()[2], "85.000000", 1), *lines[4:]]) + "\n")
    assert humid["code"] == "moisture" and humid["line"] == 4 and "relative humidity" in humid["text"]
    gale = _refused("\n".join([*lines[:3], " ".join([*lines[3].split()[:3], "-9999.0", "-9999.0"]), *lines[4:]]) + "\n")
    assert gale["code"] == "wind" and gale["line"] == 4
    odd = _refused(S.table(S.measured(), header="pressure_psi,height,temperature,dewpoint,direction,speed_kt"))
    assert odd["code"] == "unit" and "psi" in odd["text"]


def test_a_dry_file_in_kg_per_kg_is_a_warning_not_a_refusal():
    lines = S.made().splitlines()
    rows = [" ".join([line.split()[0], line.split()[1], "%.6f" % (float(line.split()[2]) / 1000.0),
                      *line.split()[3:]]) for line in lines]
    record = O.examine("\n".join(rows) + "\n", name="dry.txt", top_needed_m=TOP)
    assert record["ok"] is True
    assert any(row["code"] == "moisture-unit" and "multiply the column by 1000" in row["text"]
               for row in record["warnings"])


def test_superadiabatic_layers_and_supersaturation_warn_and_do_not_refuse():
    lines = S.made().splitlines()
    fifth = lines[5].split()
    cooled = " ".join([fifth[0], "%.6f" % (float(fifth[1]) - 1.5), *fifth[2:]])
    soaked = " ".join([lines[30].split()[0], lines[30].split()[1], "30.000000", *lines[30].split()[3:]])
    record = O.examine("\n".join([*lines[:5], cooled, *lines[6:30], soaked, *lines[31:]]) + "\n", name="odd.txt",
                       top_needed_m=TOP)
    assert record["ok"] is True, record["problems"]
    by_code = {row["code"]: row for row in record["warnings"]}
    assert "first at line 6 (500 m)" in by_code["superadiabatic"]["text"] and "it runs" in by_code["superadiabatic"]["text"]
    assert "first at line 31" in by_code["supersaturated"]["text"] and "It runs." in by_code["supersaturated"]["text"]


def test_a_top_below_the_model_top_is_offered_made_up_levels():
    lines = S.made().splitlines()
    short = "\n".join(lines[:121]) + "\n"            # to 12 km
    record = O.examine(short, name="short.txt", top_needed_m=TOP)
    top = record["problems"][0]
    assert record["ok"] is False and top["code"] == "top" and top["line"] == 121
    assert "stops at 12.0 km and the model goes up to 19.75 km" in top["text"]
    assert top["offer"] == {"extend_to_m": 20500.0} and "19750 m" in top["fix"]
    extended = O.examine(short, name="short.txt", top_needed_m=TOP, extend=True)
    assert extended["ok"] is True and extended["top_m"] == 20500.0 and extended["levels"] == 120 + 17
    extension = extended["extension"]
    assert extension["from_m"] == 12000.0 and extension["levels_added"] == 17
    assert "made up, not measured" in extension["sentence"] and not extension["likely_below_tropopause"]
    # the added air has the top level's temperature at every height, and its wind and moisture
    profile = F.parse_cm1(extended["text"])
    temperatures = F.temperatures_k(profile)
    assert max(temperatures[121:]) - min(temperatures[120:]) < 0.05
    assert set(profile.u_ms[120:]) == {profile.u_ms[119]} and set(profile.qv_gkg[120:]) == {profile.qv_gkg[119]}
    # stopped in the troposphere: said, as a lid
    low = O.examine("\n".join(lines[:61]) + "\n", name="low.txt", top_needed_m=TOP, extend=True)
    assert low["ok"] is True and low["extension"]["likely_below_tropopause"]
    assert "acts as a lid at 6.0 km" in low["extension"]["sentence"]
    # too little measured to extend: the height needed is said
    stub = O.examine("\n".join(lines[:21]) + "\n", name="stub.txt", top_needed_m=TOP, extend=True)
    assert stub["ok"] is False and "offer" not in stub["problems"][0]
    assert "reaches 19750 m" in stub["problems"][0]["fix"]


# ------------------------------------------------------------------ names, words and the start


def test_a_run_name_stem_comes_from_the_file_name():
    assert O.label_from("My Sounding (OUN 2013).txt") == "My-Sounding-OUN-2013"
    assert O.label_from("D:\\soundings\\koun.input_sounding") == "koun"
    assert O.label_from("../../etc/passwd") == "passwd"
    assert O.label_from("***") == "sounding" and O.label_from(None) == "sounding"
    assert len(O.label_from("x" * 200 + ".csv")) == O.LABEL_CHARS


def test_the_sentence_follows_the_numbers_and_is_hedged():
    def said(cape, shear, cin=0.0, mu=None):
        return O.verdict({"cape_j_kg": cape, "shear_0_6km_m_s": shear, "cin_j_kg": cin,
                          "cape_most_unstable_j_kg": cape if mu is None else mu})

    assert said(20.0, 30.0)["sentence"] == "Stable air: a storm is unlikely."
    assert said(300.0, 30.0)["id"] == "weak"
    assert said(2500.0, 30.0, cin=150.0)["sentence"].startswith("Capped:")
    assert said(2500.0, 5.0)["id"] == "single" and said(2500.0, 15.0)["id"] == "multicell"
    rotating = said(2500.0, 30.0, cin=60.0)
    assert rotating["sentence"] == "Plenty of instability and shear for a rotating storm (a supercell)."
    assert "CIN 60 J/kg" in rotating["more"][0] and "not a forecast" in rotating["note"]
    assert "above the ground" in said(200.0, 30.0, mu=1800.0)["more"][0]
    assert O.verdict(None) is None


def test_the_start_is_the_push_for_every_shipped_balloon_and_the_bubble_for_made_air():
    ratios = []
    for path in sorted(SHIPPED.glob("*.input_sounding")):
        choice = O.start_choice(F.parse_cm1(path.read_text(encoding="ascii")))
        assert choice["recommended"] == "push", path.name
        ratios.append(choice["bubble_moisture"]["ratio"])
    assert len(ratios) == 9 and 0.42 < min(ratios) and max(ratios) < 0.78     # the 43 to 77 % the words state
    made = []
    for instability in (0.0, 0.25, 0.5, 0.75, 1.0):
        choice = O.start_choice(S.profile(instability))
        assert choice["recommended"] == "bubble", instability
        made.append(choice["bubble_moisture"]["ratio"])
    assert 0.80 < min(made) and max(made) <= 1.0                               # the 81 to 100 %
    choice = O.start_choice(S.profile(1.0))
    assert choice["close_call"] and "close call" in choice["why"]
    assert [option["id"] for option in choice["options"]] == list(O.STARTS)
    assert "often makes no storm" in choice["trap"] and "1.4 km" in choice["trap"]
    shallow = F.parse_cm1("\n".join(S.made().splitlines()[:11]) + "\n")
    assert O.start_choice(shallow)["recommended"] == "push"


@needs_library
def test_the_numbers_are_getcapes_on_the_storms_own_levels():
    record = O.examine(S.made(), name="made.input_sounding", top_needed_m=TOP, nz=40, dz_m=500.0)
    numbers = record["numbers"]
    assert record["ok"] and numbers["levels"] == {"nz": 40, "dz_m": 500.0}
    # the generator's own targets: 1650 J/kg and 21 m/s (tests/test_analytic_sounding.py holds these to 10 %)
    assert numbers["cape_j_kg"] == pytest.approx(1650.0, rel=0.1)
    assert numbers["shear_0_6km_m_s"] == pytest.approx(21.0, rel=0.1)
    for key in ("cin_j_kg", "lcl_m", "lfc_m", "el_m", "srh_0_1km_m2_s2", "srh_0_3km_m2_s2"):
        assert isinstance(numbers[key], (int, float)), key
    assert record["verdict"]["id"] == "supercell" and record["start"]["recommended"] == "bubble"


# ------------------------------------------------------------------ the commands


def _cli(*args: str, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "cm1core", *args], cwd=cwd, capture_output=True, text=True,
                          timeout=300, env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "src"),
                                            "CUDA_VISIBLE_DEVICES": ""})


def test_sounding_convert_writes_the_file_and_its_record(tmp_path):
    source = tmp_path / "My balloon.txt"
    source.write_text(S.wyoming(S.measured(), knots=True), encoding="ascii")
    out = tmp_path / "mine.input_sounding"
    done = _cli("sounding", "convert", str(source), str(out), "--top-m", "19750", "--json", cwd=tmp_path)
    assert done.returncode == 0, done.stderr
    record = json.loads(done.stdout)
    assert record["ok"] and record["format"] == "wyoming" and record["label"] == "My-balloon"
    assert record["written"] == str(out) and out.read_text(encoding="ascii") == record["text"]
    assert record["sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()
    kept = O.kept(out)
    assert kept is not None and kept["label"] == "My-balloon" and "text" not in kept
    assert O.source_words(kept) == "your sounding My balloon.txt, converted from University of Wyoming text listing"
    out.write_text(out.read_text() + "25000.0 700.0 0.0 0.0 0.0\n")
    assert O.kept(out) is None                      # the file was changed after its record was written
    plain = _cli("sounding", "convert", str(source), "--top-m", "19750", cwd=tmp_path)
    assert plain.returncode == 0 and "University of Wyoming text listing" in plain.stdout
    assert "How to start the storm: a warm bubble (--start bubble)" in plain.stdout


def test_sounding_convert_refuses_with_exit_2_and_writes_nothing(tmp_path):
    source = tmp_path / "short.csv"
    source.write_text(S.table(S.measured(top_m=12000.0)), encoding="ascii")
    out = tmp_path / "short.input_sounding"
    done = _cli("sounding", "convert", str(source), str(out), "--top-m", "19750", cwd=tmp_path)
    assert done.returncode == 2 and not out.exists()
    assert "Problem: The sounding stops at 12.0 km" in done.stdout and "Nothing was written." in done.stdout
    fixed = _cli("sounding", "convert", str(source), str(out), "--top-m", "19750", "--extend-top", cwd=tmp_path)
    assert fixed.returncode == 0 and out.is_file() and "That part is made up, not measured." in fixed.stdout
    needs_top = _cli("sounding", "convert", str(source), str(out), "--extend-top", cwd=tmp_path)
    assert needs_top.returncode == 2 and "--top-m" in needs_top.stderr
    unit = tmp_path / "unit.csv"
    unit.write_text(S.table(S.measured(), header="pressure,height,temperature,dewpoint,direction,speed"))
    asked = _cli("sounding", "convert", str(unit), "--json", cwd=tmp_path)
    assert asked.returncode == 2 and json.loads(asked.stdout)["problems"][0]["code"] == "speed-unit"
    assert _cli("sounding", "convert", str(unit), "--speed-unit", "kt", cwd=tmp_path).returncode == 0
    same = _cli("sounding", "convert", str(source), str(source), cwd=tmp_path)
    assert same.returncode == 2 and "OUT is the file being read" in same.stderr


@needs_parser
def test_a_preset_runs_on_a_listing_started_by_the_push_at_the_famous_days_own_place(tmp_path):
    """The Norman 2013 example listing, started with the push: the frame and the push's place are the
    famous-day preset's (which was made the same way from the same balloon)."""

    from cm1core.cm1run import own_start, presets as P

    listing = ROOT / "examples" / "soundings" / "norman-2013-05-20-wyoming.txt"
    preset = own_start.given(P.load("your-environment"), {}, {}, listing, "auto")
    famous = P.load("oun-2013-05-20")
    assert "iinit     =  12" in preset.namelist_text
    assert preset.record["gm1_settings"] == famous.record["gm1_settings"]
    for word in ("umove", "vmove"):
        line = next(line for line in famous.namelist_text.splitlines() if line.strip().startswith(word))
        assert line in preset.namelist_text
    given = preset.sounding_given
    assert given["label"] == "norman-2013-05-20-wyoming" and given["file"] == str(listing.resolve())
    assert given["own"]["format"] == "wyoming" and given["own"]["converted"] is True
    sentence = given["follow"]["sentence"]
    assert "push of rising air starts 4.8 km east and 6.0 km south of the centre" in sentence
    assert "15.4 m/s toward the east and 8.0 m/s toward the north" in sentence
    assert "worked out for a warm bubble" in sentence and "stays inside" not in sentence
    assert P.given_args(preset) == ["--sounding", str(listing.resolve()), "--start", "auto"]
    assert own_start.run_name(preset, "20260101-0000") == "norman-2013-05-20-wyoming-20260101-0000"
    assert P.examine(preset)["offered"] is True, P.examine(preset)["reasons"]
    # the made environment keeps the bubble; a stronger bubble is Customize's own 3 K
    made = tmp_path / "made-air.input_sounding"
    made.write_text(S.made(), encoding="ascii")
    bubble = own_start.given(P.load("your-environment"), {}, {}, made, "auto")
    assert "iinit     =  1," in bubble.namelist_text and not bubble.record.get("customized")
    strong = own_start.given(P.load("your-environment"), {}, {}, made, "strong-bubble")
    assert strong.record["gm1_settings"]["initialization.bptpert_k"] == 3.0
    # the push has its own strength and no bubble size
    pushed = own_start.given(P.load("your-environment"), {"push": 15.0}, {}, made, "push")
    assert pushed.record["gm1_settings"]["initialization.wmax_wnudge"] == 15.0
    with pytest.raises(P.PresetError, match="no bubble size"):
        own_start.given(P.load("your-environment"), {"bubble": 5.0}, {}, made, "push")
    # a famous day may try the warm bubble: the push's place becomes the bubble's
    back = own_start.given(famous, {}, {}, None, "bubble")
    assert back.record["gm1_settings"] == {"initialization.bubble_center_x_m": 4800.0,
                                           "initialization.bubble_center_y_m": -6000.0}
    with pytest.raises(P.PresetError, match="goes with --sounding"):
        own_start.given(P.load("your-environment"), {}, {}, None, "auto")


@needs_parser
def test_a_file_that_cannot_run_is_refused_by_the_preset_with_what_to_do(tmp_path):
    from cm1core.cm1run import own_start, presets as P

    short = tmp_path / "short.csv"
    short.write_text(S.table(S.measured(top_m=12000.0)), encoding="ascii")
    with pytest.raises(P.PresetError) as refused:
        own_start.given(P.load("your-environment"), {}, {}, short, None)
    assert "stops at 12.0 km" in str(refused.value) and "--extend-top --top-m 19750" in str(refused.value)
    unit = tmp_path / "unit.csv"
    unit.write_text(S.table(S.measured(), header="pressure,height,temperature,dewpoint,direction,speed"))
    with pytest.raises(P.PresetError, match="--speed-unit kt"):
        own_start.given(P.load("your-environment"), {}, {}, unit, None)


def test_the_two_example_files_are_what_the_tool_writes():
    import runpy

    # The tool stays in the development repository: where it is absent this test is skipped by the file it names.
    tool = runpy.run_path(str(ROOT / "tools" / "presets" / "make_sounding_examples.py"))
    assert tool["main"](["--check"]) == 0


def test_the_example_files_read_as_their_layouts():
    folder = ROOT / "examples" / "soundings"
    made = F.read((folder / "made-supercell.input_sounding").read_text(encoding="ascii"))
    assert made.format == "cm1" and made.text == S.made()
    listing = F.read((folder / "norman-2013-05-20-wyoming.txt").read_text(encoding="ascii"))
    shipped = F.parse_cm1((SHIPPED / "koun-2013-05-20-1800z.input_sounding").read_text(encoding="ascii"))
    assert listing.format == "wyoming" and listing.profile.levels == shipped.levels and not listing.dropped
    assert listing.profile.surface_theta_k == pytest.approx(shipped.surface_theta_k, abs=0.06)
    assert max(abs(a - b) for a, b in zip(listing.profile.qv_gkg, shipped.qv_gkg)) < 0.08


def test_the_stored_archive_pages_give_the_shipped_files_byte_for_byte():
    """The converter is the one the shipped balloon soundings were made with (development repository only: the
    pages as fetched are not distributed)."""

    index = json.loads((SHIPPED / "index.json").read_text(encoding="utf-8"))
    for row in index["soundings"]:
        page = (ROOT / row["response_stored"]).read_bytes().decode("utf-8", "surrogateescape")
        reading = F.read(page)
        assert reading.format == "wyoming"
        assert reading.text.encode("ascii") == (SHIPPED / row["filename"]).read_bytes(), row["filename"]


# ------------------------------------------------------------------ the guide


def test_the_soundings_guide_passes_the_copy_lint_and_is_linked():
    from cm1core.gui.copy_lint import lint_markdown

    guide = ROOT / "docs" / "SOUNDINGS.md"
    assert lint_markdown(guide) == []
    for other in ("README.md", "docs/QUICKSTART.md", "docs/TROUBLESHOOTING.md", "docs/RELEASE-NOTES.md"):
        assert "SOUNDINGS.md" in (ROOT / other).read_text(encoding="utf-8"), other
    trouble = (ROOT / "docs" / "TROUBLESHOOTING.md").read_text(encoding="utf-8")
    assert "**My sounding makes no storm.**" in trouble and "**My sounding file is refused.**" in trouble
    notes = (ROOT / "docs" / "RELEASE-NOTES.md").read_text(encoding="utf-8")
    release = notes.split("## 0.1.6\n")[1].split("\n## ")[0]
    assert "**A new card, Use my own sounding**" in release and "(SOUNDINGS.md)" in release
    assert "### Known limitations" in release and "has not been run" in release
    text = guide.read_text(encoding="utf-8")
    assert "43 to 77 %" in text and "81 to 100 %" in text and "under 80 %" in text


@needs_parser
def test_every_command_block_of_the_soundings_guide_runs(tmp_path):
    import runpy

    runner = runpy.run_path(str(ROOT / "tools" / "release" / "doc_commands.py"))
    record = runner["run_guide"](ROOT / "docs" / "SOUNDINGS.md", tmp_path / "guide", "fake", timeout_s=600)
    assert record["passed"], "\n".join(record["problems"])
    ran = [row for row in record["commands"] if "command" in row]
    assert len(ran) == 5
    run = json.loads((tmp_path / "guide" / "home" / "gm1-runs" / "my-sounding" / "run.json").read_text())
    assert run["case"] == "Your sounding: my-sounding"


# ------------------------------------------------------------------ what testing found (each fixed, each held here)


def _same_air(a: F.Profile, b: F.Profile) -> bool:
    """The same levels and values, whichever lines of the file they came from."""

    from dataclasses import replace

    return replace(a, lines=(), surface_line=0) == replace(b, lines=(), surface_line=0)


def test_a_cm1_file_the_engine_would_stop_on_is_written_again_with_the_same_numbers():
    """CM1's Fortran reads a file with a blank line, commas or extra words; the engine reads only the strict form
    (one surface line of three words, then lines of five, no blank line). Such a file was kept byte for byte and
    stopped at run start; now it is written again with the user's own numbers, and the note says what went."""

    text = S.made()
    lines = text.splitlines()
    variants = {
        "trailing blank line": text + "\n",
        "leading blank line": "\n" + text,
        "commas": "\n".join(",".join(line.split()) for line in lines) + "\n",
        "fourth number on the surface line": lines[0] + " 0.0\n" + "\n".join(lines[1:]) + "\n",
        "a note after a level": "\n".join([*lines[:5], lines[5] + "  measured", *lines[6:]]) + "\n",
        "a sixth number": "\n".join([*lines[:5], lines[5] + " 7.0", *lines[6:]]) + "\n",
        "windows line ends": text.replace("\n", "\r\n"),
    }
    assert F.strict_cm1(text)
    for why, variant in variants.items():
        assert not F.strict_cm1(variant), why
        record = O.examine(variant, name="mine.input_sounding", top_needed_m=TOP)
        assert record["ok"] is True, (why, record["problems"])
        assert F.strict_cm1(record["text"]), why
        assert F.parse_cm1(record["text"]) == F.parse_cm1(text), why          # the same numbers
        assert record["tidied"] is True and record["converted"] is False, why
    said = " ".join(O.examine(variants["trailing blank line"], name="x", top_needed_m=TOP)["notes"])
    assert "1 blank line(s)" in said and "same numbers" in said
    extra = " ".join(O.examine(variants["a note after a level"], name="x", top_needed_m=TOP)["notes"])
    assert "first on line 6" in extra
    assert O.examine(text, name="x", top_needed_m=TOP).get("tidied") is None   # a strict file is kept as it is


@needs_library
def test_a_tidied_cm1_file_gets_its_numbers():
    record = O.examine(S.made() + "\n", name="x", top_needed_m=TOP, nz=40, dz_m=500.0)
    assert record["numbers"] and "numbers_reason" not in record and record["verdict"]


def test_rows_of_numbers_with_no_header_say_so():
    rows = S.wyoming(S.measured()).splitlines()[4:]           # the number rows of a listing, its header left behind
    refused = _refused("\n".join(rows) + "\n")
    assert refused["code"] == "no-header" and refused["line"] == 1
    assert "11 numbers" in refused["text"] and "no header line" in refused["text"]
    assert "PRES HGHT TEMP DWPT" in refused["fix"] and "five numbers" not in refused["text"]
    table = S.table(S.measured()).splitlines()[1:]
    assert _refused("\n".join(table) + "\n")["code"] == "no-header"


def test_a_level_listed_out_of_place_is_put_in_order_by_pressure():
    rows = S.measured()
    lines = S.wyoming(rows).splitlines()
    moved = [*lines[:8], lines[11], *lines[8:11], *lines[12:]]           # one level three lines too early
    reading = F.read("\n".join(moved) + "\n")
    assert reading.dropped == () and reading.profile.z_m == F.read(S.wyoming(rows)).profile.z_m
    assert any("out of order (first at line" in note and "order of pressure" in note for note in reading.notes)
    # a mistyped pressure is not "put in order" (its height says where it belongs): the levels stay as listed
    typo = [*lines[:8], "%7.1f" % 1050.0 + lines[8][7:], *lines[9:]]
    kept = F.read("\n".join(typo) + "\n")
    assert kept.profile.levels == len(rows) - 1 and not any("order of pressure" in n for n in kept.notes)


def test_the_ground_is_the_lowest_level_with_a_temperature():
    rows = S.measured()
    lines = S.wyoming(rows).splitlines()
    whole = F.read(S.wyoming(rows)).profile
    # the archive's levels under the station: pressure and height, nothing else
    under = [*lines[:4], "%7.1f%7.0f" % (rows[0]["p"] + 20.0, S.STATION_M - 180.0), *lines[4:]]
    reading = F.read("\n".join(under) + "\n")
    assert reading.profile.z_m == whole.z_m and reading.profile.surface_hpa == whole.surface_hpa
    assert reading.dropped[0]["line"] == 5 and reading.dropped[0]["why"].startswith("below the ground (")
    # the ground's own wind missing: still the ground (CM1's layout holds no surface wind)
    no_wind = [*lines[:4], lines[4][:42] + " " * 14 + lines[4][56:], *lines[5:]]
    calm = F.read("\n".join(no_wind) + "\n")
    assert calm.profile == whole and calm.dropped == () and calm.warnings == ()
    # the ground's dewpoint missing: the next whole level is the ground, and a warning says how much higher
    no_dew = [*lines[:4], lines[4][:21] + " " * 7 + lines[4][28:], *lines[5:]]
    raised = F.read("\n".join(no_dew) + "\n")
    assert raised.profile.surface_hpa == pytest.approx(rows[1]["p"], abs=0.06)
    assert "Line 5, the lowest level with a temperature, has dewpoint missing" in raised.warnings[0]
    assert "took line 6" in raised.warnings[0] and "100 m higher" in raised.warnings[0]
    record = O.examine("\n".join(no_dew) + "\n", name="x", top_needed_m=TOP)
    assert record["warnings"][0]["code"] == "ground"


def test_air_colder_than_any_real_air_is_a_problem_not_a_traceback():
    lines = ["1000.0 300.0 14.0", *("%.1f 300.0 0.0 5.0 0.0" % (z * 500.0) for z in range(1, 59))]
    record = O.examine("\n".join(lines) + "\n", name="flat.txt", top_needed_m=TOP)
    problem = record["problems"][0]
    assert record["ok"] is False and problem["code"] == "temperature" and problem["line"] > 2
    assert "colder than any real air" in problem["text"] and "potential temperature" in problem["fix"]
    assert F.rslf(1.0e4, 20.0) == 0.0 and F.rslf(0.0, 300.0) == 0.0


def test_heights_in_km_are_named():
    lines = S.made().splitlines()
    km = [lines[0], *(" ".join(["%.3f" % (float(line.split()[0]) / 1000.0), *line.split()[1:]])
                      for line in lines[1:])]
    refused = _refused("\n".join(km) + "\n")
    assert refused["code"] == "height-unit" and "look like km" in refused["text"] and "by 1000" in refused["fix"]


def test_tables_with_marks_quotes_units_rows_and_mixed_wind_units():
    rows = S.measured()
    lines = S.table(rows).splitlines()
    # 9999 for a missing temperature: that level is left out, the column is still C, the line is named
    marked = [*lines[:5], ",".join([*lines[5].split(",")[:2], "9999", *lines[5].split(",")[3:]]), *lines[6:]]
    reading = F.read("\n".join(marked) + "\n")
    assert reading.dropped == ({"line": 6, "why": "no temperature"},) and reading.profile.levels == len(rows) - 2
    # quoted cells, as a spreadsheet writes them
    quoted = [lines[0], *(",".join('"' + cell + '"' for cell in line.split(",")) for line in lines[1:])]
    _close(F.read("\n".join(quoted) + "\n"), rows)
    # a units row under the header names the speed's unit, so it is not asked for
    plain = S.table(rows, header="pressure,height,temperature,dewpoint,direction,speed").splitlines()
    units = F.read("\n".join([plain[0], "hPa,m,C,C,deg,kt", *plain[1:]]) + "\n")
    _close(units, rows)
    assert any("Units read from line 2" in line for line in units.conversions)
    # m/sec is a speed unit
    _close(F.read(S.table(rows, header="pressure,height,temperature,dewpoint,direction,speed (m/sec)", speed=1.0)),
           rows)
    # u and v in different units are refused, not converted with one of them
    uv = S.table_uv(rows).replace("u [m/s]", "u [kt]")
    with pytest.raises(F.SoundingRefused) as refused:
        F.read(uv)
    assert refused.value.code == "unit" and "u and v" in refused.value.text


def test_a_blank_line_inside_a_listing_and_several_soundings_in_one_file():
    rows = S.measured()
    lines = S.wyoming(rows).splitlines()
    whole = F.read(S.wyoming(rows)).profile
    gap = F.read("\n".join([*lines[:10], "", *lines[10:]]) + "\n")
    assert _same_air(gap.profile, whole)
    two = ("99999 TST Test Observations at 00Z 01 Jan 2026\n" + S.wyoming(rows)
           + "\nStation information and sounding indices\n\n99999 TST Test Observations at 12Z 01 Jan 2026\n"
           + S.wyoming(S.measured(instability=0.2)))
    several = F.read(two)
    assert _same_air(several.profile, whole)
    assert any("holds 2 soundings; GM1 read the first one (99999 TST Test Observations at 00Z" in note
               for note in several.notes)


def test_the_warm_layer_near_the_ground_of_every_balloon_is_a_note_not_a_warning():
    for path in sorted(SHIPPED.glob("*.input_sounding")):
        record = O.examine(path.read_text(encoding="ascii"), name=path.name, top_needed_m=TOP)
        assert not [row for row in record["warnings"] if row["code"] == "superadiabatic"], path.name
    record = O.examine((SHIPPED / "kbmx-2011-04-27-1800z.input_sounding").read_text(encoding="ascii"), name="x",
                       top_needed_m=TOP)
    assert any("Nothing to fix" in note and "the lowest at 18 m" in note for note in record["notes"])


def test_small_mixing_ratios_are_printed_with_their_digits():
    lines = S.made().splitlines()
    rows = [" ".join([line.split()[0], line.split()[1], "%.6f" % (float(line.split()[2]) / 1000.0),
                      *line.split()[3:]]) for line in lines]
    why = O.start_choice(F.parse_cm1("\n".join(rows) + "\n"))["why"]
    assert "0.0 against 0.0" not in why and " g/kg" in why and "0.01" in why


def test_the_reason_numbers_are_missing_is_said_plainly():
    raw = ("the t=0 environment report could not be built from this run's own base state: built on this run's own "
           "2 mass levels; param.F needs at least four. A run that cannot state the CAPE ...")
    words = O.plain_reason(raw)
    assert words.startswith("too few levels") and "only 2" in words and "param.F" not in words
    assert "param.F" not in O.plain_reason("anything else at all")


@needs_library
def test_the_numbers_say_what_getcape_did_not_find():
    shipped = O.examine((SHIPPED / "koun-2011-05-24-1800z.input_sounding").read_text(encoding="ascii"), name="x",
                        top_needed_m=TOP, nz=40, dz_m=500.0)["numbers"]
    assert shipped["el_m"] is None and "first stops rising near 1.5 km, then rises again" in shipped["el_note"]
    stable = [line.split() for line in S.made(instability=0.0).splitlines()]
    dry = "\n".join(" ".join([*row[:2], "0.5" if len(row) == 3 else "0.1", *row[3:]]) for row in stable) + "\n"
    none = O.examine(dry, name="dry", top_needed_m=TOP, nz=40, dz_m=500.0)["numbers"]
    assert none["lfc_m"] is None and none["el_m"] is None and "el_note" not in none
    # a file that stops short: its own numbers up to its top, none that need air above it, and no sentence
    short = O.examine("\n".join(S.made().splitlines()[:41]) + "\n", name="short", top_needed_m=TOP)
    assert short["ok"] is False and short["verdict"] is None and short["start"]
    numbers = short["numbers"]
    assert numbers["partial_top_m"] == 4000.0 and numbers["shear_0_6km_m_s"] is None
    assert numbers["srh_0_3km_m2_s2"] is not None and numbers["storm_motion_m_s"] == [None, None]


@needs_parser
def test_a_famous_day_runs_on_another_balloon_with_its_own_push(tmp_path):
    """A preset that starts with the push, given another sounding: the frame and place are worked out for a bubble
    on that sounding, and it starts with its own push (it was refused: "the storm would drift too close")."""

    from cm1core.cm1run import own_start, presets as P

    other = tmp_path / "oun-1999.input_sounding"
    other.write_bytes((SHIPPED / "koun-1999-05-03-1800z.input_sounding").read_bytes())
    for start in (None, "push", "auto"):
        preset = own_start.given(P.load("oun-2013-05-20"), {}, {}, other, start)
        assert "iinit     =  12" in preset.namelist_text, start
        assert not [entry for entry in preset.record.get("customized") or [] if entry.get("knob") == "start"]
        assert "worked out for a warm bubble" in preset.sounding_given["follow"]["sentence"]
        assert P.examine(preset)["offered"] is True, (start, P.examine(preset)["reasons"])


@needs_parser
def test_a_cm1_file_with_a_blank_line_runs_from_its_tidied_copy(tmp_path):
    from cm1core.cm1run import own_start, presets as P

    mine = tmp_path / "mine.input_sounding"
    mine.write_text(S.made() + "\n", encoding="ascii")
    preset = own_start.given(P.load("your-environment"), {}, {}, mine, None)
    ran = Path(preset.sounding_path)
    assert ran != mine.resolve() and F.strict_cm1(ran.read_text(encoding="ascii"))
    own = preset.sounding_given["own"]
    assert own["tidied"] is True and own["original"] == str(mine.resolve())
    assert preset.sounding_given["label"] == "mine"
    plain = tmp_path / "input_sounding"
    plain.write_text(S.made() + "\n", encoding="ascii")
    unnamed = own_start.given(P.load("your-environment"), {}, {}, plain, None)
    assert unnamed.sounding_path != plain.resolve() and own_start.label_of(unnamed) is None
