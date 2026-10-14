# Use your own sounding

A sounding is one column of air: how warm, how moist and how windy it is at each height. GM1 can grow an idealized
storm in the air of a sounding you bring: a weather balloon, a forecast sounding, or one you made yourself.

What you get is that air everywhere in the storm's area, over flat ground, with nothing changing in time except the
storm. It shows what kind of storm the air can make. It is not a forecast, and it is not a reconstruction of any
real day's storms.

## In the GUI

On **Pick a storm**, open the **Explore** group and press **Choose a sounding** on the card **Use my own sounding**.
The card opens in place, with five steps:

1. **Your sounding.** Press **Choose a file** (your browser reads the file, so it works for any file you can open,
   on Windows too), paste the sounding's text, pick one of the balloon soundings GM1 ships as a starting point, or
   type the path of a file on the computer running GM1.
2. **What GM1 found.** Which layout the file has, how it was converted, and anything wrong with it, in plain words
   with the line number and what to do. You can open and download the file CM1 reads.
3. **The air in your sounding.** A skew-T and hodograph, the numbers (below), and one sentence on the kind of storm
   such air usually makes.
4. **How the storm is started.** Three choices, one chosen for your sounding, with the reason (below).
5. **Next: how long can it run?** The usual screen: the storm is fitted to your graphics card, **Customize this
   storm** changes the grid, the area and the storm time, and **Start** runs it.

The run is named after your file (`my-balloon-20260101-1200`), the **Runs** list says "Your sounding: my-balloon",
and the run's folder keeps the file the model read as `input_sounding`. When your file was converted or written
again, your own file is kept beside it as `input_sounding.original`.

Back from **How long can it run?** opens the card again on the same sounding, with the start you picked.

Under WSL, a Windows file has another name inside Linux: `D:\soundings\balloon.txt` is
`/mnt/d/soundings/balloon.txt`. The card takes either one where it asks for a path. **Choose a file**
needs neither.

## From a terminal

```sh
gpuwm-cm1 preset environment --instability 0.5 --shear 0.5 --out my-sounding.txt
gpuwm-cm1 sounding convert my-sounding.txt
gpuwm-cm1 preset fit your-environment --budget 10m --sounding my-sounding.txt --start auto
gpuwm-cm1 preset new your-environment --budget 10m --root ~/gm1-runs --name my-sounding --sounding my-sounding.txt --start auto
gpuwm-cm1 run ~/gm1-runs/my-sounding
```

The first line only makes a sounding to try (use your own file instead). `sounding convert FILE` looks at a file and
prints what the card shows: its layout, anything wrong with it, the numbers, the sentence and the start. With a
second name it also writes the file CM1 reads:

```sh example
gpuwm-cm1 sounding convert balloon.txt balloon.input_sounding --top-m 19750
gpuwm-cm1 sounding convert table.csv table.input_sounding --speed-unit kt
gpuwm-cm1 sounding convert short.txt short.input_sounding --extend-top --top-m 19750
gpuwm-cm1 preset new your-environment --budget 1h --root ~/gm1-runs --sounding balloon.input_sounding --start push
```

`preset fit` and `preset new` take `--sounding FILE` in any of the layouts below, and `--start` says how the storm
is started: `push`, `bubble`, `strong-bubble`, or `auto` (chosen from the sounding, as the card does). Without
`--name`, the run is named after the file.

## The layouts GM1 reads

### CM1's own file

This is what CM1 calls `input_sounding`, and what GM1 runs. Text, numbers only, no comment lines:

```text
   966.0000     303.531982      17.939029           line 1, the ground: pressure (hPa), potential temperature (K), water vapour (g/kg)
    74.0000     303.038488      16.885028     0.5229     5.9772     then one line per level, from the ground up:
   265.0000     302.766910      15.499236     0.7147     8.1688     height above the ground (m), potential temperature (K),
   385.0000     302.609069      14.742552     0.7147     8.1688     water vapour mixing ratio (g/kg),
   489.0000     302.412879      14.267649     2.0471     8.8668     wind toward the east (m/s), wind toward the north (m/s)
```

(The words on the right are notes for this page; a real file has only the numbers. These are the first lines of the
Norman 20 May 2013 balloon GM1 ships: a wind from 185 degrees, a little west of south, blows toward the north.)

- Line 1 is the ground. It has three numbers and no wind: CM1 works out the wind at the ground from the first two
  levels above it.
- Heights are in metres above the ground, each higher than the one before.
- Temperature is **potential temperature** in K, not plain temperature. Near the ground it is about 300 K, and it
  rises with height.
- Water vapour is the mixing ratio in **g/kg** (about 5 to 20 near the ground in storm air), not kg/kg and not
  relative humidity.
- The file must reach the top of the model: 19750 m for the guided storms.
- CM1 itself also reads a file with blank lines, commas between the numbers, or words after them; GM1's engine
  reads only the plain form. GM1 writes such a file again with the same numbers, says what it took out, and keeps
  your own file beside the run's.

### A University of Wyoming text listing

The listing the University of Wyoming's upper-air archive shows for one balloon launch, saved as text or as the
whole page. It starts with a header and a line of units:

```text
   PRES   HGHT   TEMP   DWPT   DRCT   SPED
    hPa      m      C      C    deg    m/s
  966.0    345   27.4   22.4    185    5.1
  958.0    419   26.2   21.3    185    6.0
```

GM1 uses the pressure, height, temperature, dewpoint, wind direction and wind speed columns, and ignores the rest.
The speed column is `SKNT` (knots) in the older listings and `SPED` (m/s) in the newer ones; the units line says
which.

Copy the listing from its header line down. Rows of numbers alone are refused, since nothing says which column is
which. The lines a listing gives below the station's ground (a pressure and a height, nothing else) are left out,
and the card says so. When a file holds several soundings (a range of times), GM1 reads the first and says so.

### A SHARPpy file

The text file SHARPpy and many sounding sites write: a title, then the levels between `%RAW%` and `%END%`.

```text
%TITLE%
 OUN   130520/1800

   LEVEL       HGHT       TEMP       DWPT       WDIR       WSPD
-------------------------------------------------------------------
%RAW%
  966.00,    345.00,     27.40,     22.40,    185.00,     10.00
  958.00,    419.00,     26.20,     21.30,    185.00,     11.70
%END%
```

Each line is pressure (hPa), height above sea level (m), temperature (C), dewpoint (C), wind direction (degrees)
and wind speed in **knots**. `-9999` marks a missing value.

### A table with named columns

Any text table whose first line names its columns, separated by commas, semicolons, tabs or spaces:

```text
pressure_hPa,height_m,temperature_C,dewpoint_C,direction_deg,speed_kt
966.0,345,27.4,22.4,185,10.0
958.0,419,26.2,21.3,185,11.7
```

GM1 needs pressure, height, temperature, dewpoint and the wind, as direction and speed or as `u` and `v`. Common
names are understood (`p`, `pres`, `hght`, `z`, `temp`, `tmpc`, `dwpt`, `td`, `wdir`, `drct`, `wspd`, `sknt`). A
unit in a name is read: `speed_kt`, `wspd (m/s)`, `temp_K`, `pressure_Pa`, `height_ft`, or from a second line of
units under the names (`hPa,m,C,C,deg,kt`). Pressure in Pa and temperature in K are told from hPa and C by their
size. An empty cell, `-9999` or `9999` marks a missing value. Cells in quotes, as spreadsheets write them, are read.
`u` and `v` must be in the same unit.

The wind speed is the one thing GM1 does not guess. If its column does not say knots or m/s, the card asks, and the
command takes `--speed-unit kt` or `--speed-unit ms`.

## How a file is converted

For every layout but CM1's own, GM1 writes CM1's file with these rules, using CM1's own constants. They are the
rules the balloon soundings GM1 ships were made with.

| CM1 needs | From your file |
|---|---|
| Height above the ground (m) | Height minus the first level's height. The first level becomes the ground line. |
| Potential temperature (K) | T × (1000 hPa / p)^0.2854, with T in K |
| Water vapour mixing ratio (g/kg) | The saturation mixing ratio at the dewpoint (Bolton's formula, as CM1 computes it) |
| Wind toward the east and north (m/s) | u = −speed × sin(direction), v = −speed × cos(direction); the direction is where the wind blows from; 1 knot is 0.514444 metres per second |

A level missing any of the six values, or not higher than the level below it, is left out, and the card lists every
such line and why. Nothing is smoothed or filled in.

- Levels listed out of order are put in order of pressure, when their heights then rise too; the card says so.
- The ground is the lowest level with a temperature. Its wind may be missing, since CM1's file has no wind at the
  ground. If its dewpoint is missing, the next whole level becomes the ground, and the card warns how much higher
  that is.

## What GM1 checks

A file that cannot run is refused, with the line number and what to do:

| It says | What it means |
|---|---|
| The surface line is missing | Line 1 has five numbers, like a level. CM1 would read its first three as the ground's pressure, temperature and moisture. Add the three-number ground line. |
| Numbers with no header line | Rows of more than five numbers and no line naming the columns. Copy the listing from its header line down, or add a first line naming the columns. |
| Too few levels | CM1 needs the ground and at least two levels above it. |
| A height is not above the one before it | Put the levels in order from the ground up, each height once. |
| The heights look like km | CM1 reads metres: multiply the heights by 1000. |
| The air works out colder than any real air | The potential temperature does not rise with height far up. Check that column. |
| Temperature in C where potential temperature in K belongs | The second column must be potential temperature in K. Or give a table of pressure, height, temperature and dewpoint, which GM1 converts. |
| The surface pressure looks like Pa | CM1 reads hPa: divide by 100. |
| A mixing ratio looks like relative humidity | Give g/kg. |
| The sounding stops below the model top | See below. |

Some things only get a note or a warning, because they run. Air just above the ground that is warmer than the air
above it, as on a sunny afternoon, is in nearly every balloon sounding: a note, nothing to fix. A layer higher up
where potential temperature falls by more than 1 K is a warning (the model mixes it in its first minutes). So is a
level holding more water vapour than air can (the model turns the extra into cloud in its first step), and a file
whose mixing ratios are all below 0.1 (probably kg/kg: multiply by 1000).

### A sounding that stops too low

The guided storms' model goes up to 20 km (its highest level is at 19750 m), and CM1 stops on a sounding that ends
below the model top. If yours reaches at least 3 km, the card offers **Add levels up to the model top**
(`--extend-top --top-m 19750`): above your top level it adds air at that level's temperature, with its wind and
moisture, every 500 m. That is how the lower stratosphere roughly is, but it is made up, not measured, and the card
and the run's record say where it begins. If your sounding stops where it is still warmer than −40 °C, it probably
stopped below the tropopause; the made-up layer then acts as a lid at that height and storm tops come out lower than
in the real air. The card says so.

The guided screens do not lower the model top. In Advanced you can set the number of levels yourself.

## The numbers

The numbers come from the model's own code (CM1's parcel calculation, with GM1's wind numbers), on the levels the
storm will run on, for air lifted from the ground. Because the model's lowest level is half a level above the
ground, a calculation at full detail gives a slightly larger instability.

| On the card | Also called | What it tells you |
|---|---|---|
| Instability | CAPE, J/kg | The energy rising air can gain. Under about 500, storms are weak or absent; over about 2000, updrafts are strong. |
| Lid strength | CIN, J/kg | The energy needed to push surface air up to where it rises freely. Over about 100, the air is capped. |
| Cloud base | LCL | Where rising surface air first makes cloud. |
| Free rise from | LFC | Where rising air becomes warmer than its surroundings and keeps rising unaided. |
| Rising air stops near | EL | Roughly the storm's top. A dash with a sentence when it has none: rising air that is still rising where the file stops, or that stops low and rises again higher up. |
| Wind change, lowest 6 km | 0-6 km bulk shear | How much the wind changes between the ground and 6 km. Under about 10 m/s of shear, storms are short-lived; over about 20 m/s of shear, they can rotate. |
| Low-level turning | Storm-relative helicity, 0-1 and 0-3 km | How much the low-level wind turns relative to the moving storm. Larger values favour rotation near the ground. |

A dash means a number could not be found: no free rise for stable air, for example. For a file that stops short of
the model top, the card works the numbers out on the part it has, leaves out those that need air above it (the
shear to 6 km above a file that stops at 4 km) and gives no sentence until the file reaches the top.

The card works its numbers out on the guided storm's own levels (40 levels of 500 m). `sounding convert` does the
same with `--top-of` and the storm's namelist; with `--top-m` alone it uses 250 m levels up to that height, so its
numbers can differ a little from the card's.

The sentence under the picture ("Plenty of instability and shear for a rotating storm", "Stable air: a storm is
unlikely", "Capped: needs a strong push") is read from these numbers with the same thresholds as the **Make your
own environment** card. It is a rule of thumb from studies of idealized storms, not a forecast: the same air can
make another storm, or none.

## How the storm is started

Air that could make a storm still needs something to start it. GM1 offers the starts it runs the way CM1 does:

| Start | What it is |
|---|---|
| A push of rising air | Air is pushed gently upward in one spot for the first 20 minutes (CM1's updraft nudging, `iinit = 12`). It lifts the air from the ground up. The famous-day storms start this way. |
| A warm bubble | A bubble of air 1 K warmer than its surroundings, 10 km in radius, centred 1.4 km above the ground (CM1's classic start, `iinit = 1`). |
| A stronger warm bubble | The same bubble at 3 K. |

**The known trap: on a real sounding, a warm bubble often makes no storm.** CM1's bubble warms nothing at the ground
and most at 1.4 km up. In real air the moisture usually falls off quickly above the ground, so the bubble lifts
drier air than the surface air the instability number is counted from. None of the nine balloon soundings GM1 ships
made a storm from the warm bubble on a grid a graphics card fits (CM1 itself made none from it on the one balloon
tried), and all nine made one from the push.

GM1 picks the default from that measured difference. It compares the water vapour at 1.4 km with the water vapour
at the ground:

- under 80 %: the push of rising air. The nine shipped balloons are at 43 to 77 %.
- 80 % or more: the warm bubble. The made environments, whose storms start from a bubble, are at 81 to 100 %.

This is a line between two groups of runs, not a law. Near 80 % the card says it is a close call. A stronger bubble
lifts harder but still starts with the air at 1.4 km. If no storm forms, try the other start; Results says so when
a warm bubble made no storm on air the card said could make one.

The rule was then checked on the card's own storm (1 km grid, 2 hours) on an RTX 4090. The Birmingham 27 April
2011 balloon (77 %, a close call) made a supercell for the whole 2 hours from the push, and nothing from the warm
bubble. The Norman 20 May 2013 listing in `examples/soundings/` (70 %) made a strong storm from the push that had
rained itself out after about an hour, and nothing from the bubble. A made moist sounding (93 %) made a splitting
pair of supercells from the bubble, and stable air made nothing, as the card said. Each 2-hour run took about a
minute.

With either start, the view follows the storm. GM1 works out from your sounding's winds how a storm started by a
warm bubble will move and where it should start so that it stays inside the area (the storm-track check), and the
fit screen states both, for example:
"The view moves with the storm at 15.4 m/s toward the east and 8.0 m/s toward the north (ground-relative), and
the push of rising air starts 4.8 km east and 6.0 km south of the centre". The check
follows a bubble only, so for the push the fit screen says the frame and place were worked out for a bubble: a
storm from the push can move another way. On the Birmingham balloon it drifted south and was 12 km from the edge
after 2 hours, so on a longer run watch the edges.

## Where to get soundings

- **Weather balloons.** The University of Wyoming's upper-air archive lists every launch as text
  (`weather.uwyo.edu`, the "Text: List" output). Save the listing or the page and give it to GM1 as it is.
  Balloons launched by the U.S. National Weather Service are U.S. Government observations.
- **Sounding tools.** SHARPpy and SounderPy write files GM1 reads: SHARPpy's own text file, and SounderPy's CM1
  output.
- **Your own WRF or ArWen run.** The card **Use a sounding from my WRF or ArWen run** takes a column straight from
  a history file ([`WRF-DOWNSCALING.md`](WRF-DOWNSCALING.md) is the other route, which uses the whole run).
- **Made air.** **Make your own environment** and `gpuwm-cm1 preset environment` write a sounding with a chosen
  instability and wind shear.
- **The ones GM1 ships.** Nine balloon soundings of well-known days are in the card's list. Download one from the
  card, change it in a text editor, and bring it back.

Two example files are in the repository's `examples/soundings/` folder: `made-supercell.input_sounding` (CM1's own
layout, made by GM1) and `norman-2013-05-20-wyoming.txt` (a text listing worked back from the shipped Norman
balloon of 20 May 2013, a U.S. Government observation). Both are free to use. The Norman listing's storm on the
1 km grid rains itself out after about an hour; for one that lasts 2 hours, pick the Birmingham 27 April 2011
balloon in the card's list.

## If it does not work

[`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) has "My sounding makes no storm" and "My sounding file is refused".
