# Observed soundings shipped with gpuwm-cm1

9 CM1 `isnd = 7` `input_sounding` files are shipped. Two (Norman 2013 and 2019) let the example decks
that name them run without a download: each deck names the file it was built for by SHA-256 in its header comment,
`gpuwm-cm1 examples NAME --copy DIR` copies the matching file beside the deck as `NAME.input_sounding`, and the run
receipt records `sounding_sha256`. All 9 are the soundings of the guided gallery's famous-environment
presets (`src/cm1core/presets/*/input_sounding`, byte for byte). `index.json` carries every number below.

| file | observation | SHA-256 | bytes | levels, top | CM1 hydrostatic check |
|---|---|---|---|---|---|
| `koun-2013-05-20-1800z.input_sounding` | Norman, Oklahoma (OUN, WMO 72357), 1800 UTC 20 May 2013 | `aa7d5afa3701ebf14f46221e532c144b01a94c4bc36baad0275db0c3da79e83d` | 7518 | 116 to 30.4 km | 0.58 hPa |
| `koun-2019-05-21-0000z.input_sounding` | Norman, Oklahoma (OUN, WMO 72357), 0000 UTC 21 May 2019 | `c172f77023e9c9bcfecfaf67d4f8e229bb3bb475fa10aa24f3d8b9389645b997` | 7128 | 110 to 32.9 km | 0.34 hPa |
| `koun-1999-05-03-1800z.input_sounding` | Norman, Oklahoma (OUN, WMO 72357), 1800 UTC 3 May 1999 | `2da08f23539b4b381e86113b647d54a89e41325a92634102faef32a0986c5517` | 4918 | 76 to 29.8 km | 1.29 hPa |
| `koun-2011-05-24-1800z.input_sounding` | Norman, Oklahoma (OUN, WMO 72357), 1800 UTC 24 May 2011 | `d72f8ee9189ca0eaefcb3e3cfc332a7ddfb844a346adf9027baf3b9e5c5acd56` | 7128 | 110 to 28.3 km | 0.52 hPa |
| `kbmx-2011-04-27-1800z.input_sounding` | Birmingham, Alabama (BMX, WMO 72230), 1800 UTC 27 April 2011 | `a7e8bd7b3efb89f184fae792b00e6de806bbcb7313b1f79ed7c4fa886c8250be` | 6348 | 98 to 27.9 km | 0.44 hPa |
| `kdnr-2015-06-02-0000z.input_sounding` | Denver, Colorado (DNR, WMO 72469), 0000 UTC 2 June 2015 | `9c557edb543f994e3e1d3f476b59b78afc801873353d610d1f082e174e5ac172` | 8883 | 137 to 32.2 km | 0.68 hPa |
| `kiln-2012-06-29-1800z.input_sounding` | Wilmington, Ohio (ILN, WMO 72426), 1800 UTC 29 June 2012 | `08b7bc5abd1d404707cd95a1ca14c8001f3b928b6a4d2b11e57c223775333dde` | 7583 | 117 to 30.8 km | 0.67 hPa |
| `ktlh-2004-09-16-0000z.input_sounding` | Tallahassee, Florida (TLH, WMO 72214), 0000 UTC 16 September 2004 | `394ba7900c473dca192abf851e058a3a47d290cff04c09ed8194b8617af72a6f` | 3488 | 54 to 20.1 km | 0.34 hPa |
| `kjan-2014-07-01-0000z.input_sounding` | Jackson, Mississippi (JAN, WMO 72235), 0000 UTC 1 July 2014 | `ececf9e35ea24b0b508d77da4b241e1d00576478037e35b9fa0a10ea21d43018` | 8948 | 138 to 32.2 km | 0.38 hPa |

## Source and rights

* **Observations:** NOAA National Weather Service operational balloon-borne rawinsonde launches. These are
  measurements made by a U.S. Government agency; works of the U.S. Government are not subject to copyright in the
  United States (17 U.S.C. 105). The sonde model is not recorded by the archive product used.
* **Retrieved from:** University of Wyoming, Department of Atmospheric Science, radiosonde archive,
  `type=TEXT:LIST` product (the page is kept byte for byte in the development repository, so each file is
  re-derived from it by `python tools/release/uwyo_to_input_sounding.py ship --check`; the pages are not
  distributed, since besides the data they carry the site's own page furniture and contact details):
  * `https://weather.uwyo.edu/wsgi/sounding?datetime=2013-05-20%2018:00:00&id=72357&src=UNKNOWN&type=TEXT:LIST`
    (retrieved 2026-09-14; response SHA-256 `94aae7f790e5092234e57fc6bced01c1162ccda280a6a82fc93fb7f597c22be2`, 16484 bytes; stored as `tests/data/uwyo/72357-2013-05-20T18.html`)
  * `https://weather.uwyo.edu/wsgi/sounding?datetime=2019-05-21%2000:00:00&id=72357&src=UNKNOWN&type=TEXT:LIST`
    (retrieved 2026-09-14; response SHA-256 `77cf0b8548ff71c82ec59840c1a837220b43551bdde89a7532980b96057c0365`, 16096 bytes; stored as `tests/data/uwyo/72357-2019-05-21T00.html`)
  * `https://weather.uwyo.edu/wsgi/sounding?datetime=1999-05-03%2018:00:00&id=72357&src=UNKNOWN&type=TEXT:LIST`
    (retrieved 2026-09-24T09:00:50Z; response SHA-256 `5a28566dfe578d03bca3a6a51efc61b80954104df9b6f6b796cbda0a754f9c75`, 13766 bytes; stored as `tests/data/uwyo/72357-1999-05-03T18.html`)
  * `https://weather.uwyo.edu/wsgi/sounding?datetime=2011-05-24%2018:00:00&id=72357&src=UNKNOWN&type=TEXT:LIST`
    (retrieved 2026-09-24T08:58:56Z; response SHA-256 `f279408e84036e1bb68e65d7876e7ad471595813d92e1b67c571e568630b841e`, 16422 bytes; stored as `tests/data/uwyo/72357-2011-05-24T18.html`)
  * `https://weather.uwyo.edu/wsgi/sounding?datetime=2011-04-27%2018:00:00&id=72230&src=UNKNOWN&type=TEXT:LIST`
    (retrieved 2026-09-24T08:59:02Z; response SHA-256 `38e85f65d825e84f2bdfa7e38ce4e102eac154f0abedde3effe62c19f6e22f30`, 15546 bytes; stored as `tests/data/uwyo/72230-2011-04-27T18.html`)
  * `https://weather.uwyo.edu/wsgi/sounding?datetime=2015-06-02%2000:00:00&id=72469&src=UNKNOWN&type=TEXT:LIST`
    (retrieved 2026-09-24T09:52:20Z; response SHA-256 `ae78e3d0e6ce49ac5e782f304c2ec5414470c9e8ff5a948a8b431669746a8f8c`, 17783 bytes; stored as `tests/data/uwyo/72469-2015-06-02T00.html`)
  * `https://weather.uwyo.edu/wsgi/sounding?datetime=2012-06-29%2018:00:00&id=72426&src=UNKNOWN&type=TEXT:LIST`
    (retrieved 2026-09-24T09:00:14Z; response SHA-256 `e59bc4cdd248058c15d8e6f9524bb9971a7314466850aa4a80de5693ddcd4cd2`, 16957 bytes; stored as `tests/data/uwyo/72426-2012-06-29T18.html`)
  * `https://weather.uwyo.edu/wsgi/sounding?datetime=2004-09-16%2000:00:00&id=72214&src=UNKNOWN&type=TEXT:LIST`
    (retrieved 2026-09-24T09:00:57Z; response SHA-256 `29e4fa30ed271adb64df67230521b178bea764af851a0b46bd565e2817893645`, 12045 bytes; stored as `tests/data/uwyo/72214-2004-09-16T00.html`)
  * `https://weather.uwyo.edu/wsgi/sounding?datetime=2014-07-01%2000:00:00&id=72235&src=UNKNOWN&type=TEXT:LIST`
    (retrieved 2026-09-24T09:55:05Z; response SHA-256 `aad70d4e096666890ae6546d0575c1acfea10020693e7e2382ce95043db3e448`, 18689 bytes; stored as `tests/data/uwyo/72235-2014-07-01T00.html`)
* The archive is credited as the retrieval source. No University of Wyoming value-added field (its MIXR, THTA,
  THTE, THTV, RELH columns or index block) is used: every value in the files is recomputed from the measured PRES,
  HGHT, TEMP, DWPT, DRCT and SPED columns.
* **Why each launch** (every page fetched while choosing was recorded with its numbers):
  * `koun-2013-05-20-1800z.input_sounding`: a well-known day's environment: the Norman midday balloon of 20 May 2013.
  * `koun-2019-05-21-0000z.input_sounding`: a well-known day's environment: the Norman evening balloon of 20 May 2019, 0000 UTC on the 21st.
  * `koun-1999-05-03-1800z.input_sounding`: the target was 0000 UTC 4 May; that balloon stopped at 9.7 km (30 levels), so the 1800 UTC special launch of 3 May is used (29.8 km).
  * `koun-2011-05-24-1800z.input_sounding`: the 1800 UTC special launch of 24 May 2011 is archived.
  * `kbmx-2011-04-27-1800z.input_sounding`: the 1800 UTC special launch of 27 April 2011 is archived.
  * `kdnr-2015-06-02-0000z.input_sounding`: the rule (surface T-Td >= 12 K, LCL >= 1.8 km, CAPE 800-2500 J/kg, 0-6 km shear >= 20 m/s, top >= 20 km, hydrostatic <= 1 hPa), first pass of a scan of 0000 UTC launches at DNR, LBF, AMA and DDC, May-June.
  * `kiln-2012-06-29-1800z.input_sounding`: of ILN, ILX and DVN at 1200 and 1800 UTC on 29 June 2012, the only launch with CAPE >= 3000 J/kg that reaches 20 km (ILX 1200 UTC stopped at 12 km).
  * `ktlh-2004-09-16-0000z.input_sounding`: a Gulf-coast launch in a landfalling tropical cyclone's rainbands with 0-1 km shear >= 10 m/s and a moist column that reaches 20 km (LIX 2005-08-29 0000 UTC has 0-1 km shear 16.8 m/s but CAPE 1125 J/kg; LCH 2005-09-24 stopped at 15.5 km).
  * `kjan-2014-07-01-0000z.input_sounding`: the rule (a July 0000 UTC launch from JAN, BMX, FFC or TLH with 0-6 km shear < 8 m/s and CAPE >= 2000 J/kg, top >= 20 km, hydrostatic <= 1 hPa), first pass of a scan of July 2012-2016; a 2019 pass was set aside because its archive page has 6467 levels.

## Conversion (the only processing)

`tools/release/uwyo_to_input_sounding.py`. Constants are CM1 r21.1 `src/constants.F` (g 9.81, rd 287.04,
cp 1005.7, rv 461.5, p00 1e5). For each measured level with complete data:

| quantity | formula |
|---|---|
| height | `z = HGHT - station elevation` (m above the station; CM1's `isnd = 7` axis is above a flat surface) |
| potential temperature | `theta = (TEMP + 273.15) * (p00 / (PRES*100))**(rd/cp)` |
| vapour mixing ratio | CM1's `rslf` evaluated at the dewpoint, written in g/kg |
| wind | `u = -SPED sin(DRCT)`, `v = -SPED cos(DRCT)` (archive speed unit is m/s) |

The first line is CM1's surface header (`p_sfc hPa, theta_sfc K, qv_sfc g/kg`); each following line is one measured
level above the surface. A level missing any of the six values, or not higher than the level below it, is dropped;
nothing was interpolated, smoothed, extended or filled. The dropped levels:
  * `koun-2013-05-20-1800z.input_sounding`: 10.2 hPa (31057.0 m MSL): missing DRCT, SPED
  * `koun-2019-05-21-0000z.input_sounding`: 7.2 hPa (33439.0 m MSL): missing DRCT, SPED; 7.1 hPa (33535.0 m MSL): missing DRCT, SPED
  * `koun-1999-05-03-1800z.input_sounding`: 10.6 hPa (30634.0 m MSL): missing DRCT, SPED
  * `koun-2011-05-24-1800z.input_sounding`: 13.8 hPa (28909.0 m MSL): missing DRCT, SPED
  * `kbmx-2011-04-27-1800z.input_sounding`: 44.2 hPa (21336.0 m MSL): missing nothing, not above the level below; 14.9 hPa (28313.0 m MSL): missing DRCT, SPED
  * `kdnr-2015-06-02-0000z.input_sounding`: 6.9 hPa (33892.0 m MSL): missing DRCT, SPED
  * `kiln-2012-06-29-1800z.input_sounding`: 10.3 hPa (31231.0 m MSL): missing DRCT, SPED
  * `ktlh-2004-09-16-0000z.input_sounding`: 52.6 hPa (20512.0 m MSL): missing DRCT, SPED
  * `kjan-2014-07-01-0000z.input_sounding`: 197.0 hPa (12497.0 m MSL): missing nothing, not above the level below; 8.5 hPa (32476.0 m MSL): missing DRCT, SPED

Every top exceeds any shipped deck's model top (20 km).

**Verification:** CM1's hydrostatic reconstruction of pressure from each file (`base.F`'s recursion from the
surface pressure) agrees with the measured pressures to within 1 hPa over the whole column, with one named
exception: `koun-1999-05-03-1800z.input_sounding`, 1.29 hPa. The excess is in the archive's own lowest layer: 964 hPa
at the station (345 m) and 925 hPa at 716 m are 13 m further apart than hydrostatic balance allows at that
temperature, so the reconstruction runs 1.0 to 1.3 hPa low from there up (0.2 hPa at 20 km), about the pressure
accuracy of a 1999 sonde; the 0000 UTC launch that evening stopped at 9.7 km, below any model top, so it cannot
stand in. The shipped Norman 2013 and 2019 files were also re-parsed by an independent implementation of CM1's
`base.F` `isnd = 7` reader and GM1's Rust parser (2026-09-14).

## Limits

Each file is a single-column observation used as a horizontally homogeneous base state, the standard
idealized-supercell method. A simulation started from one is not a forecast, hindcast or reconstruction of any
observed storm: each card says "This day's sounding in an idealized storm, not a reconstruction of that day's
storms."
