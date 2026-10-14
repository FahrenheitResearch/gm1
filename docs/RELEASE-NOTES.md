# gpuwm-cm1 release notes

## 0.1.6

Your own sounding. A new card grows a storm in the air of a sounding file you bring: a weather balloon, a forecast
sounding or one you made, in CM1's own layout, as a University of Wyoming listing, a SHARPpy file or a table. The
engine is 0.1.5's (ArWen 2.8.9), and every storm that 0.1.5 runs, 0.1.6 runs the same way; the hurricanes, any grid
spacing on the thunderstorm presets, the 100 m supercell and ground friction in Customize stay as 0.1.5 has them.

### What is new

- **A new card, Use my own sounding**, in the gallery's Explore group. Choose a sounding file (your browser reads
  it, so any file you can open works, on Windows too), paste its text, type a path, or start from one of the nine
  balloon soundings GM1 ships. The card says what it found, shows the air as a skew-T and hodograph with the model's
  own numbers (CAPE, CIN, LCL, LFC, EL, 0-6 km shear, 0-1 and 0-3 km storm-relative helicity) and one hedged
  sentence on the storm such air usually makes, picks how to start the storm, and goes on to **How long can it
  run?**, where the grid, the area and the storm time are fitted and can be changed as for any storm. Back from that
  screen opens the card again on the same sounding.
- **Four file layouts.** CM1's own `input_sounding`; a University of Wyoming text listing (`PRES HGHT TEMP DWPT ...`,
  as text or as the saved page, wind in knots or m/s as its units line says); a SHARPpy text file (`%RAW%` to
  `%END%`); and a table whose first line names its columns (pressure, height, temperature, dewpoint, and the wind as
  direction and speed or as u and v). The converted file is shown and can be downloaded. The conversion is the one
  the shipped balloon soundings were made with, and gives those nine files byte for byte from their archive pages.
  A wind speed whose column does not say knots or m/s is asked about, not guessed. Levels out of order are put
  in order of pressure, the lines a listing gives below the station's ground are left out and named, units on a
  second header line are read, and a CM1 file with blank lines, commas or words after its numbers is written again
  with the same numbers, since GM1's engine reads only the plain form.
- **What cannot run is refused in plain words, with the line number**: a missing surface line, rows of numbers with
  no header line, too few levels, heights out of order or in km, temperature in C where potential temperature in K
  belongs, a surface pressure in Pa, a column that is not what its place says. The warm air just above the ground
  that afternoon balloons have is a note; a large fall of potential temperature higher up and air holding more
  vapour than it can are warnings, since they run. A sounding that stops below the model top (19750 m for the
  guided storms) can be given made-up levels above its top, at the top level's temperature; the card and the run's
  record say where they begin, and say when the sounding probably stopped below the tropopause.
- **How the storm is started is a choice, with a default from the sounding.** A push of rising air (CM1's updraft
  nudging, as the famous-day cards use), CM1's warm bubble, or the bubble at 3 K. On a real sounding the warm bubble
  often makes no storm: it is centred 1.4 km above the ground, where real air is usually much drier than at the
  ground. The default follows what GM1 has measured: where the water vapour at 1.4 km is under 80 % of the
  surface's, as in all nine shipped balloons (43 to 77 %), the push; otherwise, as in the made environments (81 to
  100 %), the bubble. It is a line between two groups of runs, and the card says when a sounding is close to it.
  When a warm bubble makes no storm on air the card said could make one, Results says to try the push.
- **The view follows the storm your sounding makes.** The storm-track check runs on your sounding, and the fit
  screen states the motion it chose and where the storm starts. For the Norman 20 May 2013 balloon, the card's run
  has the same motion and starting place as the famous-day card for that balloon. The check follows a warm bubble
  only, so for the push the screen says the frame was worked out for a bubble; on a long run, watch the edges.
- **A famous-day card runs on another sounding.** `--sounding FILE` on a card that starts with the push (a famous
  day) was refused as drifting out of the area; its frame and place are now worked out on the new sounding, as for
  the card above, and it keeps its own push.
- **The run is yours.** It is named after your file, the Runs list says "Your sounding: NAME", the run folder keeps
  the file the model read (`input_sounding`) and your own file when it was converted (`input_sounding.original`),
  and Results says whose air it is.
- **From a terminal.** `gpuwm-cm1 sounding convert IN [OUT]` checks a file and writes CM1's `input_sounding`
  (`--speed-unit kt|ms`, `--extend-top --top-m M`). `gpuwm-cm1 preset fit|new ID --sounding FILE` takes the same four
  layouts, and `--start push|bubble|strong-bubble|auto` starts the storm the chosen way on any card that starts from
  a bubble or a push (a hurricane card refuses it). `gpuwm-cm1 preset options ID --start push` lists the settings
  as they are with that start.
- **A new guide**, [`docs/SOUNDINGS.md`](SOUNDINGS.md): the layouts with examples, the conversions, where to get
  soundings, the starts and the trap, and what the numbers mean. Two example files are in `examples/soundings/`.
- **Smaller fixes.** The fit's first line, Advanced mode's summary and Results' "This run" line say "An 80 × 80 km
  area" for an area said with a vowel (the gust front's own, for example), as the grid spacing already did ("An
  82.9 m grid"). On a storm that runs on a sounding file (this card's, or the column the WRF or ArWen card keeps),
  Customize says the air is the file's in place of the two sliders of Make your own environment, which did not
  change that air.

### What was measured

- **Six storms on an RTX 4090** with ArWen 2.8.9 (gpuwm 2.8.9), each the card's own storm (the Make your own
  environment preset: a 1 km grid, 120 × 120 km, 40 levels of 500 m, 2 hours), started from the GUI as the card
  starts it:
  - The Birmingham 27 April 2011 balloon as a University of Wyoming listing in knots, given made-up levels above
    18.9 km where it stopped (77 %, a close call). From the push, a supercell for the whole 2 hours: updrafts up to
    46 m/s, 62 dBZ at 1 km, updraft helicity up to 2055 m²/s². From the warm bubble, no storm (no updraft over
    0.9 m/s).
  - The Norman 20 May 2013 listing in `examples/soundings/` (70 %). From the push, a storm by 10 minutes (updrafts
    up to 67 m/s, 59 dBZ) that had rained itself out by about 60 minutes (8 dBZ left at 2 hours). From the warm
    bubble, no storm (1.9 m/s).
  - A made moist sounding in CM1's layout (93 %). From the warm bubble, a splitting pair of supercells from
    20 minutes: updrafts up to 62 m/s, 64 dBZ at 1 km.
  - Stable air with an inversion near the ground, as a table with named columns in knots (CAPE 0). From the warm
    bubble, nothing (0.6 m/s), as the card said.
  - What the card said beforehand, the kind of storm and the start, held in all six (measured).
- **Staying in the area.** The main updrafts of the storms from the made sounding and from the Norman listing kept at
  least 42 km from every edge. The storm from the push on the Birmingham balloon drifted south against the moving view by
  about 6 m/s and was 12.5 km from the south edge at 2 hours, with its rain cut at the east edge from about 1 hour;
  at that drift it would leave the area at about 2 h 35 min (worked out from the drift, not run).
- **Time, memory and disk** of the card's storm on that card: 45 to 47 s of model time and 53 to 57 s from Start to
  Finished, plus about 30 s for the pictures (the fit said about 70 s on a new install and 57 s once a run had been
  timed); 1.22 GB of GPU memory at the peak, where the fit said 1.10 GB, and 4.15 to 4.21 GB of disk, where it said
  3.9 GB (measured).
- **On a CPU, with this version.** The conversion gives the nine shipped balloon files byte for byte from their
  stored archive pages. On the card's levels, the card's CAPE is within 3 % of each balloon's record for seven of
  the nine; the Denver balloon's is 10 % above its record and the Tallahassee balloon's 7 % below. All nine get the
  push as their default start.

### How to update

Run the install line again: on Windows the PowerShell line, on Linux the `curl` line (`docs/INSTALL.md`, "Update
and uninstall"); with pip, `pip install --upgrade gpuwm-cm1`. The engine stays at ArWen 2.8.9 (gpuwm 2.8.9 and
gpuwm-data 2.8.9). Close GM1 and let a running storm finish first; the installer stops with a message while one
runs. Your storms stay where they are. Finish a storm with the GM1 version that started it. The new card needs
nothing else. The two example files are in the repository and the source package (`examples/soundings/`), not in
the installed package; the card's list of shipped balloons and `gpuwm-cm1 preset environment --out FILE` give files
to try.

### Known limitations

- **One run of each, on one card, on the card's 1 km storm.** Every GPU number above comes from one 2-hour run of
  each case on one RTX 4090, at 1 km with 500 m levels. A sounding of your own on a finer grid, over a longer time
  or on another card has not been run. The runs used a development build of this work, made before its last fixes
  (the messages, the reading of untidy files and the wording of the push's frame); the 0.1.6 package itself has
  not run a storm on a sounding of the user's own.
- **The view follows a bubble's track.** GM1's storm-track check follows a warm bubble only. For the push, the
  frame and the starting place are worked out for a bubble at that spot, and the fit screen says so: a storm from
  the push can move another way, as on the Birmingham balloon above. Results offers "Run it 1 hour longer" without
  saying how near the storm is to an edge.
- **The default start is a line between two groups of runs.** The shipped balloons measure 43 to 77 % and the made
  environments 81 to 100 %, so the gap is narrow; near 80 % the card says it is a close call. That none of the nine
  balloons made a storm from the bubble and all nine did from the push was measured on the famous-day cards' grids
  (250 m, or 500 m when fitted); on the card's 1 km storm the push has been run on two balloons.
- **Not run on a GPU:** the 3 K bubble, a capped sounding with real instability under its lid (so the sentence
  "Capped: needs a strong push" is untested), a SHARPpy file, a famous-day card on another balloon (now offered,
  checked on a CPU only), and a CM1 file GM1 wrote again (blank lines, commas or extra words; checked on a CPU only).
- **The card's storm is a 1 km quick look at every time.** About 1 hour and Overnight fit the same 1 km, 2-hour
  storm as About 10 minutes, as for the Make your own environment card since 0.1.2; a finer grid, a larger area or
  a longer storm time is a Customize choice.
- **The numbers depend on the levels.** The card works them out on the storm's own 40 levels of 500 m;
  `sounding convert --top-m` uses 250 m levels and the balloon records a 1 hPa grid, so the shear and helicity
  differ a little among the three (Norman 20 May 2013: 0-6 km shear 28.1 m/s on the card, 26.5 m/s in the record).
  The card's CAPE is for air lifted from the ground; the environment picture in Results gives the mixed-layer CAPE,
  so the two can differ.
- **The estimates run a little under** for the card's storm: memory by about 11 % and disk by about 7 % (above).
- **What the card does not do:** lower the model top for a short sounding (it adds made-up levels; Advanced sets
  the levels); read rows of numbers with no header line (refused, since the speed unit would be a guess); let you
  pick one of several soundings in one file (it reads the first and says so); turn kg/kg into g/kg for you (it
  warns); notice a mistyped pressure aloft in a listing; or take a CM1 file with its heights out of order (refused).
  A warm bubble that keeps its relative humidity (`maintain_rh`) or sits lower is not offered. The card's picture
  of your sounding carries GM1's "simulated" mark, though its footer says it is a sounding file, not a GM1 run.
- The known limitations of 0.1.5, 0.1.4, 0.1.3, 0.1.2, 0.1.1 and 0.1.0 still apply ("Known limitations" under
  each, and "Not in v0.1" and "Further known limits" under "0.1.0", below), except that the 100 m supercell is no
  longer greyed (since 0.1.4) and ground friction is no longer greyed in Customize (since 0.1.5).

## 0.1.5

Hurricanes. Two hurricane presets join the gallery, made from CM1 r21.1's own hurricane case, and ground friction
becomes a choice in Customize. The engine is 0.1.4's (ArWen 2.8.9), and every storm that 0.1.4 runs, 0.1.5 runs the
same way; any grid spacing on the thunderstorm presets and the 100 m supercell stay as 0.1.4 has them.

### What is new

- **Hurricane, quick look, 8 km**, a new card under Other storm types: CM1 r21.1's `hurricane_3d_cpm` case with 23
  settings changed to what GM1 has (an even 8 km grid of 128 x 128 x 59 cells over 1024 km with repeating sides,
  4 days, a fixed 7.5 s time step, GM1's turbulence scheme in place of the boundary layer scheme, a constant ocean
  drag of 0.0024 with a heat and moisture exchange of 0.0012, the lowest three levels equally spaced, 250 cloud
  droplets per cubic centimetre, no restart files along the way), on CM1's own moist tropical sounding. It is
  CM1's hurricane case with GM1's physics choices, not CM1's stock hurricane, and the card says so. It was run
  for its whole 4 days from the guided GUI on an RTX 4090 and compared with CM1 r21.1 run with the same changed
  settings but its own adaptive time step; the numbers are under "What was measured" below. The card's picture
  is that run's wind at the lowest level at hour 90. The two hurricane cards lead Other storm types, so both
  show before "Show all".
- **Hurricane, 4 km**, a second card: the same case on a 4 km grid of 256 x 256 x 59 cells, an overnight budget.
  CM1 has not been run with these settings at 4 km, so the card, its note and its pictures say it is not
  compared. It was run for its whole 4 days from the guided GUI on the same card and engine as the 8 km one; its
  card gives the numbers, and its picture is that run's wind at the lowest level at hour 90.
- **A hurricane takes a fixed time step.** CM1's hurricane file lets the time step grow with the storm
  (`adapt_dt = 1`, up to twice `dtl`). With it GM1's hurricane stops early or makes a much weaker storm (a known
  issue, below); with a fixed step the same storm stays close to CM1's until the last day. The hurricane cards
  take `adapt_dt = 0` with `dtl = 7.5`, `--set adapt_dt=1` on a hurricane card is refused with the reason, and the
  import report says "Runs differently" for any namelist with the adaptive step on stretched levels, with that
  fix. A run on the adaptive step cannot be continued, and Results says why instead of offering it.
- **A hurricane keeps its own grid.** Any grid spacing (0.1.4) is for the thunderstorm presets. A hurricane card
  offers its own grid only: Customize shows it as a choice, with no slider and no box to type in, and
  `--custom grid=` with another spacing is refused in plain words.
- **Settings not yet compared with CM1 no longer hold a storm back.** Guided mode used to offer only storms with
  nothing under "Not yet checked against CM1". Such a storm now runs, and its card and the budget screen say
  which settings are not yet compared: "Runs, not yet compared with CM1: ...". A refusal still blocks, and so does
  a setting that asks for an acknowledgment (Thompson). The hurricanes have been run and are offered this way; the
  two squall-line presets would be too, and each stays greyed with "not run yet" until it has been run on a GPU as
  it is set up. The rule also changes one thing on every other storm: see "Also in this release" below.
- **A hurricane run has its own pictures and words.** The wind (0 to 80 m/s ground-relative) and the pressure at
  the lowest model level (as hPa below the pressure at the area's edge, 0 to 70; the lowest level, not sea level,
  and the picture says so) in place of the looks made for one thunderstorm, which the More looks row leaves out;
  GM1's runs stay inside both ranges. Rain from 1 mm to 400 cm, which holds the most rain the runs made in one
  cell. Each of these and the reflectivity shows the whole area and, beside it, the inner 500 km around the
  middle of the area, marked as a square on the whole-area panel. The wind colours start at 10 m/s
  ground-relative and a white line marks hurricane force (33 m/s ground-relative); the picture places the
  fastest wind from the pressure centre ("12 km east of the pressure centre", for example). The pressure has a
  line every 4 hPa and colours placed by the square root of the drop, so the wide shallow low shows as well as
  the centre;
  the rain colours step tenfold (a logarithmic scale). Every range is fixed and printed on the picture, and the
  numbers beside it are the run's own. Also: a time series that leads with the lowest pressure and the fastest
  wind, in hours; storm time in hours and days on Watching, Results and the pictures themselves, header and
  footer ("T+81 h (3 d 9 h)"); and, in place of "No storm yet", what the swirl is doing and when the eye is
  usually clear. Each picture's footer says whether CM1 has been run with that run's settings: "compared with
  CM1 on this 8 km case (measured)" for the 8 km card's settings (grid, levels, time step, drag, sea and cloud
  droplets), "not compared with CM1 at these settings" for any other.
- **The fit keeps a hurricane whole.** Its area is never made smaller than 1000 km (the starting swirl is 500 km
  in radius) and its storm time never shorter than 3 days (the eye forms on the second day); when it does not fit
  a time, the reason says so. Customize offers 3 to 6 days of storm time (a time longer than GM1 has run says
  "not run yet"), a picture every 1 to 12 hours and the sea surface temperature (26 to 31 °C; only 28 °C has
  been run). The budget screen says a hurricane's stretched levels as they are ("59 levels, 50 m at the bottom
  to 500 m aloft"). On the command line `preset options hurricane-8km` lists the times in days and hours with
  the minutes `--custom` takes; `--set tsk0` (the sea knob owns it), `--set` of a levels word (`stretch_z`,
  `ztop`, `dz_bot`, `dz_top`, `str_bot`, `str_top`), `--sounding`, `--instability` and `--shear` are refused on a
  hurricane card in plain words, each naming the knob or the Advanced route. Results offers "Run it 1 day
  longer" for a hurricane, and says when that day is past what GM1 has run.
- **The 8 km card on a small card.** Where the storm does not fit about 1 hour, the card says so and opens on
  Overnight: an 8 GB card that also drives a desktop is estimated at about 65 minutes even at the shortest
  3 days, and at about 87 minutes for the card's 4 days.
- **CM1's 3-D hurricane namelist imports** once ten of its settings are replaced with what GM1 has (a uniform grid,
  GM1's turbulence scheme, a constant ocean drag, equally spaced lowest levels and 250 cloud droplets per cubic
  centimetre); as CM1 ships it, it is still refused, and the axisymmetric one cannot run. The 0.1.0 notes'
  "21 of CM1's 23 shipped namelists are refused" holds for those namelists as CM1 ships them.
- **The hurricane grids are timed on an RTX 4090 out of the box.** The record of measured speeds now carries
  both hurricane grids on an RTX 4090, from the parts of the runs when the card was their own: 0.0198 s a step
  at 8 km and 0.0642 s a step at 4 km (measured). A fresh install on that card says about 15 minutes for the 8 km
  card and about 50 minutes for the 4 km one; before, it priced the 8 km grid at the reference computer's
  slower floor, about 37 minutes. Other cards are priced as before, and a computer's own run of a grid replaces
  the record's number. A grid within 0.5 % of the same number of cells on an RTX 4090 with the same rain scheme
  is priced at the same point.
- **The rate rule.** The fastest of a grid's recent runs prices it (since 0.1.4), not the median of its runs:
  runs made while other work shared the card had made the next estimate too long, and the fit had shortened a
  storm that fits (the 8 km hurricane's 4 days, priced at 58 minutes, were cut to 3 days 9 h for a 1 hour
  budget). With the fastest run the same fit prices the 4 days at about 15 minutes. This changes the time
  estimate of every storm that has been run more than once on a card, not only a hurricane's.
- **Smaller fixes.** The fit's first line gives the cells for every spacing of your own that is not a round
  number, as it already did when the area came out at a fraction of a km ("A 30.0 × 30.0 km area (448 × 448
  cells) at 67 m"), and says "An 82.9 m grid" and "An 8 km grid". `docs/INSTALL.md` gives the disk the largest
  storms take: the 100 m supercell and the hurricanes.

### Also in this release: ground friction in Customize

Not hurricane work, but it comes from the same rule. Ground friction (`bbc = 3`, `sfcmodel = 1`, `cecd = 1`) is
now a choice in Customize this storm on every storm that has the panel; it was greyed. What has been measured
is what the 0.1.2 notes give: CM1's supercell with ground friction against CM1 r21.1 on an RTX 4090 only, where
nine statistics are 0.986 to 1.018 of CM1's and max vertical vorticity is 1.072. It has not been run on an RTX
3080 and not compared in any other storm. The note under the switch says this, the budget screen lists the
storm under "Runs, not yet compared with CM1", and the import report's row for these settings now gives the
same numbers in place of "has not been compared with CM1 in a storm". WSM6 and Thompson stay greyed in
Customize, as before.

### What was measured

Each hurricane preset was run once, for its whole 4 days, from the guided GUI on an RTX 4090 on ArWen 2.8.9, with
a development build of the hurricane work and the same namelists this version ships, byte for byte. CM1 r21.1 ran
the 8 km case on a CPU.

- **8 km, against CM1.** GM1's run of this preset on an RTX 4090 with ArWen 2.8.9 (gpuwm 2.8.9), on a fixed 7.5 s
  time step, is compared with a CM1 r21.1 run with the same changed settings but its own adaptive time step (from
  10 s, as CM1's file has it). Through hour 54 the two stay within about 6 hPa and 7 m/s ground-relative at the
  lowest level. GM1 reaches 951.2 hPa at hour 90, with its fastest wind, 62.9 m/s ground-relative, at hour 93; CM1
  reaches 954.9 hPa and 62.0 m/s ground-relative at hour 81. GM1's storm holds near its peak after it: 954.0 hPa
  and 58.8 m/s ground-relative at hour 96; CM1's storm weakens after it: 971.7 hPa and 50.7 m/s ground-relative at
  hour 96 (all measured).
- **The last day.** From hour 81 to hour 96 GM1's storm goes from 954.6 hPa and 61.4 m/s ground-relative to
  954.0 hPa and 58.8 m/s ground-relative (in those hours its lowest pressure is 951.2 hPa, its fastest wind
  62.9 m/s ground-relative, its radius of maximum wind 20 km throughout); CM1's goes from 954.9 hPa and 62.0 m/s
  ground-relative to 971.7 hPa and 50.7 m/s ground-relative (radius of maximum wind 20 to 28 km). So GM1's storm
  holds its strength through the last 15 hours while CM1's weakens (measured).
- **The eye.** Both runs have a clear eye and eyewall by hour 48. In both runs the radius of maximum wind is
  20 km or less from hour 21: in GM1's run to the end (hour 96), in CM1's to hour 90, then 28 km (both measured,
  from the 3-hourly frames).
- **4 km.** GM1's run of this preset on an RTX 4090 with ArWen 2.8.9 (gpuwm 2.8.9) ran its whole 4 days:
  989.6 hPa and 37.8 m/s ground-relative at the lowest level at hour 48, 965.6 hPa and 58.5 m/s ground-relative
  at hour 72, and 965.5 hPa and 52.4 m/s ground-relative at hour 96; it was strongest at hour 90, with 952.3 hPa
  and 69.9 m/s ground-relative (measured). The run with ArWen 2.8.0 (gpuwm 2.8.0) ended at 958.7 hPa and
  61.8 m/s ground-relative. CM1 has not been run with these settings at 4 km, so this storm has nothing to be
  compared with.
- **Time.** On an RTX 4090 the 8 km card's 4 days took 24 minutes (46080 steps; other work shared the card for
  the first 12 minutes), and the 4 km card's took 49 minutes (46080 steps, 0.0642 s a step).
- The hour-by-hour table of both 8 km runs (pressure, wind, radius of maximum wind and warm core) is in
  `docs/TROUBLESHOOTING.md`, under "How close is GM1's hurricane to CM1's?".

### How to update

Run the install line again: on Windows the PowerShell line, on Linux the `curl` line (`docs/INSTALL.md`, "Update
and uninstall"); with pip, `pip install --upgrade gpuwm-cm1`. The engine stays at ArWen 2.8.9 (gpuwm 2.8.9 and
gpuwm-data 2.8.9). Close GM1 and let a running storm finish first; the installer stops with a message while one
runs. Your storms stay where they are. Finish a storm with the GM1 version that started it. A hurricane run needs
about 7.5 GB of disk at 8 km and about 30 GB at 4 km (4 days, a picture every 3 hours).

### Known limitations

- **One run each.** Every hurricane number above comes from one run of each card on one RTX 4090, and so do the
  two measured speeds; the 8 km speed comes from the part of that run after other work left the card. No other
  card has run a hurricane.
- **The 8 km storm differs from CM1's in the last day.** Through hour 54 the two stay close; from hour 81 GM1's
  storm holds near its peak while CM1's weakens, and at hour 96 they are 954.0 hPa against 971.7 hPa. The CM1 run
  used its own adaptive time step, GM1 a fixed 7.5 s one.
- **The adaptive time step.** With CM1's own adaptive setting GM1's hurricane stops early or makes a much weaker
  storm, a known issue: 8 km runs on it stopped at hour 20.8 and at hour 29.1, and the one that ran to the end
  reached 995 hPa and 35 m/s ground-relative where CM1 reaches 955 hPa and 62 m/s ground-relative (measured with
  ArWen 2.8.0 (gpuwm 2.8.0)). It has not been measured again on ArWen 2.8.9; the hurricane cards refuse it.
- **The 4 km storm is not compared with CM1.** CM1 has not been run with these settings at 4 km.
- **Times past 4 days and other sea temperatures** are offered but have not been run: Customize says "not run
  yet" beside 5 and 6 days, and only 28 °C has been run.
- **The released package itself has not run a hurricane on a GPU.** The runs above used a development build of
  the hurricane work with the same namelists; the card pictures were drawn from those runs, and their records name
  that build's version number (0.1.2).
- **Not in this version:** CM1's stock hurricane as written (it needs horizontal grid stretching, a boundary layer
  scheme and separate horizontal and vertical mixing lengths), the axisymmetric hurricane, a 2 km hurricane
  preset, and CM1's wind-dependent ocean drag (`cecd = 3`), which the import still refuses. The diagnostic that
  stopped such a run before its first step now takes the run's own roughness length; lifting the refusal waits
  for a GPU run of CM1's supercell with that drag, compared with CM1's own.
- The known limitations of 0.1.4, 0.1.3, 0.1.2, 0.1.1 and 0.1.0 still apply ("Known limitations" under each, and
  "Not in v0.1" and "Further known limits" under "0.1.0", below), except that the 100 m supercell is no longer
  greyed (since 0.1.4) and ground friction is no longer greyed in Customize.

## 0.1.4

Any grid spacing in guided mode. The model and the engine (ArWen 2.8.9) are 0.1.3's, and every storm that 0.1.3
runs, 0.1.4 runs the same way. What changed is how guided mode and `gpuwm-cm1 preset` choose the grid, the time
step and the area, and what they say about them.

### What is new

- **Any grid spacing.** Guided mode's **Customize this storm** takes any grid spacing (quick picks at 100 m, 250 m,
  500 m, 1 km and 2 km, a slider, or a number you type: 67 m) and any height for the levels, and
  `gpuwm-cm1 preset fit|new ID --custom grid=67 --custom levels=125` does the same from a terminal. The time step
  follows the spacing (CM1's 6 s per km across and per 500 m up) and is shortened where needed so every saved picture
  and the end of the run land on a whole step. Where the domain moves with the storm (`imove = 1`, as in CM1's
  supercell) the step is also a small odd number over a power of two, so the running sum of the domain's position
  never rounds: 0.402 s at 67 m becomes 0.390625 s (25/64), a few percent shorter as a rule and a fifth at most.
  Where it does not move, 0.402 s becomes 0.4 s. The area keeps its size, rounded up to whole cells; the levels keep
  the model top to the nearest whole level. The fit then sizes the area to the card and says the result exactly
  ("67 m grid, 74.5 × 74.5 km, 1112 × 1112 cells").
- **A typed decimal spacing runs.** CM1 holds dx, dy and dz in a 4-byte REAL, and a run starts only with a length
  that word holds exactly. A typed 82.9 m, or levels every 130.3 m, is given to the run as the nearest such length
  (82.9000015258789 m in the namelist, which is what CM1 itself makes of 82.9) and is still said as 82.9 m.
- **Unusual is not refused.** With a choice of your own, a run that fits only in an area smaller than the storm was
  set up for is offered, with a note that the storm may leave an area this small and an offer to move the view
  with the storm. A domain larger than the largest GM1 has run (39 million cells) is offered with a note instead of
  being turned down. What is still not offered is what cannot run, with the reason and what would fit: more GPU
  memory, RAM or disk than is free even on the smallest area a storm can start in (16 km, or its warm bubble's
  width); more than 256 levels or fewer than 8; fewer than 8 cells across; more cells than one restart file holds
  (16 GiB, about 60 million cells); more time than the time picked; and, on a storm that runs a sounding file, a
  storm that leaves its area (the run's own start refuses that one). Where the reason names a value that would be
  taken, the screen has a button that sets it, and a finest grid it names has been priced first, so it does fit.
- **What the fit offers, Start takes.** The famous days start from a push of rising air; the fit now follows that
  storm's track as the run's own start does, so it no longer offers a small area that Start then refuses. When the
  largest area that fits is one the storm leaves, the reason says that, with the area the storm needs and a coarser
  grid that covers it.
- **A run is named for the spacing it has.** A 150 m run started from the 1 km preset is
  `supercell-150m-custom-...` and is listed as "Supercell, 150 m grid (customized from Supercell, quick look,
  1 km)".
- **An area of your own follows the storm at every storm time.** With 30, 45 or 90 minutes of storm time the view
  stayed on the preset's own motion, and the storm drifted to the edge of a small area. The view now moves as the
  storm-track check predicts, and the screen says that and no more: the prediction is not exact.
- **A small area says what is known about its edges.** Whenever the area is smaller than the one the storm was set
  up for (64 km for the supercell), or than the room the storm-track check gives the storm, the fit shows one note,
  whether or not the view follows the storm: "the storm may leave an area this small; pictures near the edges then
  show edge effects", with the check's own timing as a prediction. Nothing on the screen says the storm stays
  inside. On an area you chose, the note has a one-tap "Use N km" for the smallest larger area without the note,
  when one fits the card.
- **Watching says the grid.** The header line of a run gives its spacing and area beside its name
  ("67 m grid, 22 × 22 km").
- **A grid's own measured speed is its fastest recent run.** GM1 prices a storm it has run before on this card at
  that run's step time. Other work on the card during a run only slows it, and the median of several runs made
  later estimates too long; the fastest of the last five is used.
- **As long as it takes.** A fourth time beside About 10 minutes, About 1 hour and Overnight (up to 10 hours):
  no limit (`--budget any`).
- **The 100 m supercell is offered again**, with a 0.5 s time step and a picture every 5 minutes. Up to 0.1.3 it was
  greyed: it stopped after six steps: with its 0.6 s step, CM1's running sum of the moving domain's position
  (`domainlocy`) rounds in the last place, 10.8 m against 10.799999999999999 m, and GM1's run checks that sum
  exactly. The check is unchanged (it belongs to the sources the CM1 comparison was measured on); the step is now
  one it holds exactly, and every step guided mode picks for a moving domain is such a step.
- **`gpuwm-cm1 estimate` and the Estimate screen name a time step that would stop a moving domain**, with the step
  at which the run would stop and a time step that works.

### What was measured

- On an RTX 4090 on ArWen 2.8.9, as this version ships: a 67 m storm from Customize's defaults at About 1 hour
  (a 49.3 km area, 736 × 736 cells, 6,912 steps of 0.390625 s) ran its 45 minutes in 46 minutes against 60
  estimated, at 0.39 s a step, with 19.0 GB of GPU memory against 20.9 GB estimated; the 100 m supercell (44 km,
  440 × 440 cells with 80 levels, 0.5 s steps) ran its first 20 minutes, 2,400 steps, at 0.32 s a step against
  0.28 estimated, with 14.4 GB against 15.0 GB; an 82.9 m grid ran 10 minutes (1,280 steps of 0.46875 s at
  0.054 s a step, 2.7 GB against 2.9 GB); and a 150 m grid with 153 levels every 130.3 m ran 5 minutes at
  0.067 s a step against 0.060 estimated, with 3.8 GB against 3.2 GB estimated. None stopped. The storm grew the
  same at each spacing (strongest updraft 1.3 m/s at 10 minutes and 15 m/s at 20 minutes; 66 m/s at 45 minutes at
  67 m). Another run was on the card for the last three.
- Earlier, during development, on the same card on ArWen 2.8.0: spacings of 33 m (the first 10 minutes), 67 m
  (45 minutes, twice), 100 m (30 minutes), 150 m with levels every 100 m (43 minutes), 333 m (2 hours, and
  30 minutes on an area of 361 cells across), 500 m (2 hours) and 700 m (the cloud from one bubble, 20 minutes):
  none stopped, and at 333 m the storm's timing, updraft and radar echo fall beside the 250 m, 500 m and 1 km
  runs'. Memory came within 10 % of the estimate. Time was within about 10 % at 67 m, 333 m and 500 m; a mature
  storm ran up to a quarter slower than estimated at 100 m, the 33 m run about a third slower, and the run with
  200 levels took about twice its estimate. Another run shared the card during these.
- The edges of a small area, 67 m, 45 minutes of storm time: in a 34 km area with the view moving as predicted the
  storm moved east faster than predicted and its rain was cut off by the east edge (with the preset's own motion it
  reached the west and north edges); in the 49 km area the fit gives that card from Customize's defaults, with the
  preset's own motion, the storm's core stayed inside and the rain ahead of it reached the east edge at 45 minutes.
- No run finer than 100 m has been compared with CM1. CM1 agreement itself was measured at 1 km.

### How to update

Run the install line again: on Windows the PowerShell line, on Linux the `curl` line (`docs/INSTALL.md`, "Update
and uninstall"); with pip, `pip install --upgrade gpuwm-cm1`. The engine stays at ArWen 2.8.9 (gpuwm 2.8.9 and
gpuwm-data 2.8.9). Close GM1 and let a running storm finish first; the installer stops with a message while one
runs. Your storms stay where they are. Finish a storm with the GM1 version that started it.

### Known limitations

- **Time estimates for tall columns.** The time estimate has no term for the number of levels, only a note on a fit
  with more than 80 levels. The two measured points disagree: 200 levels took about twice the estimate on 2.8.0,
  153 levels took 11 % longer on 2.8.9, both with another run on the card. In the 153-level run the GPU memory
  also came out a fifth over its estimate (3.8 GB against 3.2 GB).
- **Small areas.** The note on a small area is a warning, not a forecast: the storm-track prediction does not match
  the real motion of the 67 m storm that was measured, and with the view following the storm a 34 km area still lost the storm's
  rain at the east edge. Even the 49 km area the fit gives a 24 GB card at 67 m let the rain reach the east edge by
  45 minutes. On a 16 GB card a 67 m area of your own gets no "Use N km" button, because no area without the note
  fits that card.
- **A suggestion can lead to a second refusal.** A one-tap suggestion is not always the last step: in a sweep of
  1820 choices, 11 of 56 suggestions led to another refusal. A grid that fits the memory can then be refused for
  time on the 1 hour budget; on the 100 m supercell the grid named after a refusal for the restart file's size
  can then be refused for memory on a 24 GB card; with decimal levels a suggested grid can be a few metres too fine for the storm's room, and
  the next reply names the one that fits.
- **`gpuwm-cm1 import` and the Check screen** do not yet name a time step that would stop a moving domain: they
  still report such a namelist as runnable. `gpuwm-cm1 estimate` says it.
- The known limitations of 0.1.3, 0.1.2, 0.1.1 and 0.1.0 still apply ("Known limitations" under each, and "Not in
  v0.1" and "Further known limits" under "0.1.0", below), except that the 100 m supercell is no longer greyed.

## 0.1.3

A documentation release: the pages that lead beyond the GUI, each saying exactly how far it has been checked.
Nothing in the model changed. The package's code is 0.1.2's, on the same engine (ArWen 2.8.9), so every storm runs
as in 0.1.2 and every number under "0.1.2" below stands.

### What is new

- **The README** now says what GM1 has done, each line with the file in this repository that backs it, and has a
  section "Beyond the GUI" that lists the ways past the GUI: decks, nesting, `gpuwm-cm1 chain`,
  `gpuwm-cm1 reproduce` and the public datasets of the 4 m tornado run.
- **`docs/GM1-DECKS.md`**: GM1's own TOML decks, start to finish (`gpuwm-cm1 examples --copy`, `check`,
  `cartesian --plan`, `cartesian --integrate`, then `quicklook` and `cm1out` on the frames); what a deck run does
  not get (GUI progress, `status`, `stop`) and how to stop one; a table of all 40 example decks with the grid, the
  estimated GPU memory, the smallest card and what `check` and `cartesian --plan` say on a CPU (the 0.1.0 and 0.1.3
  wheels say the same); and recipes for a 100 m supercell, 20 m layers near the ground, the near-ground physics of
  the tornado runs, a squall line with repeating sides and the moving domain. Every command on the page runs in the
  test suite, or, where it needs a GPU, is checked option by option against the program.
- **`docs/NESTING.md` and `examples/tornado-4m/`**: what a nest is, a three-stage recipe (parent, continuation,
  nest), every `[nest]` key, memory, disk and run times, and the decks of the 4 m run, edited to end before its time
  step failed. Two variants need no file from outside: `norman/`, the same grids on the shipped 24 May 2011 Norman
  sounding, and `small/`, a 20 m nest for a 16 GB card. The start of `small/` was run on GM1 0.1.0 on an RTX 4090
  (installed by pip): the 448 x 448 x 80 parent at 100 m from 0 to 1440 s (0.40 s a step, 8.3 GiB on the card
  against 5.78 GiB in the plan), then a 500 x 500 x 80 child at 20 m inside it from 1260 to 1500 s (2.0 s a step,
  16.4 GiB against 12.97 GiB). Every stage ended with status 0 and the child stayed finite. What that run did not
  show is under "Known limitations".
- **The tornado verdict and its scorer**: `docs/GM1-TORNADO-CRITERION-V2.md` (the criterion as it was fixed before
  scoring), `docs/GM1-TORNADO-VERDICTS-V2.md` (the verdict table, and what a reader can check) and
  `tools/criterion_v2/` (the scorer, the recorded per-sample tables of every scored run, and `check_recorded.py`,
  which decides all 17 recorded verdicts again in a few seconds). The 4 m run rates TORNADO before 3312 s: one
  vortex tracked for 455 s from T+47:36.9, peak 10 m ground-relative wind about 128 m/s (129.0 m/s in the last
  record before the time step limit was passed), median radius of maximum wind 128 m. It is one run, made on a
  development build that predates 0.1.0 and scored after the fact; CM1 has no nesting, so there is no CM1 run to
  compare it with. The live criterion in the wheel (`[nest.tornado_criterion]`) is the older rule, and it did not
  call this run a tornado.
- **Eight small source tools** under `tools/` that the nesting pages use: place a nest (`nest_place.py`), read a
  plan (`plan_summary.py`), preview a window, backward trajectories through the window or through the child's
  frames, check the nest's path, replay the live check and report a sounding. They are unsupported, are not in the
  pip package, and run from a clone with the Python GM1 is installed in.
- **In the package** the code is 0.1.2's. Only the version and text changed: the README and the documents the
  sdist carries (the decks page is new among them), and the comments of one example deck
  (`cartesian-nested-tornado-window`, which now names `tools/nest_window_trajectories.py`).

### Corrections to the earlier notes

The sections below are left as they were published; this list corrects them.

- **`cecd = 2` and `3`** were described as not implemented. The drag itself is there: a GM1 deck may declare them
  and the plan admits it (`examples/cartesian-surface-enthalpy-flux-over-water.toml` uses `cecd = 3`), though no run
  with them has been made on a GPU. The import refuses them when the semi-slip ground is on (`bbc = 3`,
  `sfcmodel = 1`), because every imported run also writes the 10 m wind swath, which is written for constant drag,
  and such a run would stop before its first step. The import's message still says "only cecd = 1 (constant drag
  coefficient) is implemented".
- **The moving domain's time step.** A moving domain with a time step such as 0.6 s stops after a few steps with a
  domain-position message whose remedy (reduce the grid or the device budget) is wrong: the cause is the time step.
  `docs/GM1-DECKS.md`, "The moving domain", has the rule.
- **Initialization selectors**: 55 of 57 selector values are checked against CM1 at time zero
  (`gpuwm-cm1 doctor --science`), not 30 of 32.
- **Thompson (`ptype = 3`)** needs `nt_c = 100.0` as well as `--acknowledge-thompson`: GM1's Thompson always runs
  100 droplets per cubic centimetre, and CM1's own file has 250, which is refused with that fix line.
- **Nests, the two-part turbulence models and the Lagrangian extras** were listed as "not part of the GUI release".
  They are in the wheel and run from decks; they are not in the GUI.
- **Which turbulence model the tornado runs used.** `docs/GM1-CARTESIAN-RUNTIME.md` recommends the two-part model
  `sgs_model = 4` for tornado work on a semi-slip ground. The project's own tornado runs, the 4 m run included, used
  CM1's TKE scheme (`sgs_model = 1`); the two-part models have no storm comparison with CM1.
- **A Morrison run inside a live nest.** `docs/GM1-CARTESIAN-RUNTIME.md` said none had been made. The 4 m run was
  one, on a development build; on a release build the only one is the short RTX 4090 run above (240 s of nested
  time, finite, no vortex). No comparison is possible, since CM1 has no nesting. `gpuwm-cm1 doctor --science` and
  every Morrison run's receipt still print the old sentence ("any Morrison run inside a live nest on a GPU" under
  "not checked"): that text is part of a sealed record of every Morrison run and changes only together with it.
- **`nz > 80` with WSM6** needs a capacity overlay, and the tool that builds one is a development tool, not in the
  public repository: on an installed GM1 keep `nz` at 80 or below with WSM6, or use Morrison, which runs up to 256
  levels without an overlay.
- **Tools named in the documents.** Some documents told readers to run development tools that are not in the public
  repository. Those references are reworded: each now says the tool is a development tool, or gives the
  `gpuwm-cm1` command that does the same job.

### How to update

Nothing in the model changed, so 0.1.2 runs every storm as 0.1.3 does. To update anyway, run the install line
again: on Windows the PowerShell line, on Linux the `curl` line (`docs/INSTALL.md`, "Update and uninstall"); with
pip, `pip install --upgrade gpuwm-cm1`. Close GM1 and let a running storm finish first; the installer stops with a
message while one runs. Your storms stay where they are. Finish a storm with the GM1 version that started it.

The nesting examples, the small tools and the scorer are not in the package. Clone the repository beside your
install (`git clone https://github.com/FahrenheitResearch/gm1`) and run them with the Python GM1 is installed in;
`docs/NESTING.md` says which Python that is for the installer and for pip.

### Known limitations

- **The small example's nest did not stay on the storm** in its one test (run on GM1 0.1.0 on an RTX 4090).
  Twenty-five minutes of storm time was too early for a vortex: the child was started by hand on rotation of 0.003
  per second, where `tools/nest_place.py` proposed no start, followed a weak eddy on the outflow and moved 2.3 km in
  four minutes, away from the storm, and the live check said NO_TORNADO. `small/` has not been run past 24 minutes
  of storm time, so whether it ever makes tight rotation with the Norman sounding is not known; its 300 x 300 child
  and `norman/` have not been run at all. The guide says not to start a nest before the placement tool proposes a
  start.
- **A nest following a real vortex has not been shown on a released build.** The 4 m run did it on a development
  build that predates 0.1.0, and no 0.1.x build has repeated it. After 3312 s that run's time step was too long for
  its winds, and nothing after 3312 s is cited. No nested run on a card smaller than 32 GB has made a tornado.
- **Outsiders cannot yet re-derive the verdict from the public datasets.** The headline cannot yet be re-derived
  from them: its genesis time, its 455 s and its 129.0 m/s record rest on window records that are not public, and
  three of the four datasets cover the last minutes, after the limit was passed. What can be re-derived is
  narrower: the 162 public one-a-second records from before 3312 s, converted and scored, give TORNADO for the last
  161 s before the limit (peak 126.7 m/s). The verdicts page has the commands.
- **The decks and recipes are checked on a CPU only.** None of the 40 example decks as written, and none of the
  recipes, has been run on a GPU for these pages; the decks that have run are the first stages of `small/`, above.
- **Unchanged since 0.1.0:** `gpuwm-cm1 quicklook FRAMES` and `gpuwm-cm1 cm1out FRAMES` write to `quicklook/` and
  `cm1out/` beside the frames folder, not inside it, although their help says "default: TARGET/quicklook" (the
  guides give every run a folder of its own; `--out DIR` chooses the place). A refused `gpuwm-cm1 edit` exits with
  status 2 but leaves the refused value in the run folder's namelist, so later edits of that folder fail until the
  key is set back.
- The known limitations of 0.1.2, 0.1.1 and 0.1.0 still apply ("Known limitations" under each, and "Not in v0.1"
  and "Further known limits" under "0.1.0", below).

## 0.1.2

An engine update. GM1 now runs on ArWen 2.8.9 (gpuwm 2.8.9 and gpuwm-data 2.8.9) instead of 2.8.0, for
correctness. Everything 0.1.1 fixed on Windows is kept, and the GUI, the presets and the commands work as in 0.1.1.
WSM6 storms give 0.1.0's output word for word; Morrison storms now match CM1 r21.1 at 0.985 to 1.050 of CM1's (0.1.0:
0.986 to 1.046); Thompson's storm changed with an engine fix and has not yet been compared. Checked on an RTX 4090
and an RTX 3080, which give the same output as each other.

### Engine update: ArWen (gpuwm) 2.8.9

GM1 now requires ArWen 2.8.9: gpuwm 2.8.9 and gpuwm-data 2.8.9 (`>=2.8.9,<2.8.10`). 0.1.0 and 0.1.1 required
2.8.0. The move is for correctness: on RTX 50-series cards ArWen 2.8.9 makes the compiler divide where it had been
multiplying by a reciprocal, and Thompson's rain collecting graupel now reads the table WRF builds instead of
reading past its end. Before a run, GM1 checks that the installed ArWen files for the selected scheme match the
SHA-256 digests recorded for gpuwm 2.8.9, and refuses the scheme if they differ.

Measured on an RTX 4090 and an RTX 3080, each run made once with this release and once with GM1 0.1.0 on gpuwm
2.8.0 on the same card. The two cards give the same output as each other, word for word, with both engines, in
every run made on both (the 1 km supercell preset, CM1's supercell namelist with Morrison and with WSM6, and a
stopped and resumed storm), so the WSM6 and Morrison results below hold for both cards:

- **WSM6 (`ptype = 7`)**: CM1's supercell namelist gives GM1 0.1.0's output word for word at every output of its
  2 hours, on both cards, so WSM6's comparison with CM1 is unchanged from 0.1.0.
- **Morrison (`ptype = 5`)**: no longer word for word. ArWen rebuilt its Morrison kernel for speed. In the 250 m
  supercell on the RTX 4090, one number in 74 million first differs, by one unit in the last place, after 25 minutes
  of storm time, and the storm then grows that difference as it grows any other; both cards part from 0.1.0 at the
  same output. The storm's mature-phase means stay within 1.4 % of 0.1.0's. Against CM1 r21.1 on CM1's supercell namelist
  the ten mature-phase statistics are 0.985 to 1.050 of CM1's on an RTX 4090 and an RTX 3080 (re-measured on ArWen
  2.8.9; 0.1.0 published 0.986 to 1.046). Every statistic stays inside its release band (10 % for the updraft statistics, 15 %
  for the rest), so the coverage report still says "Same as CM1"; max vertical vorticity, at 1.0503, is just past
  GM1's 5 % target, and the report's Morrison row now names it. Max w at 900 s and 1800 s (4.100 and 29.273 m/s) and
  the strongest updraft's column at 5 km (within one cell of CM1's at every 15-minute output) are as in 0.1.0. With
  ground friction (`cecd = 1`, RTX 4090 only), nine statistics are 0.986 to 1.018 of CM1's and max vertical
  vorticity is 1.072 (0.1.0: 0.987 to 1.013, and 1.091).
- **Thompson (`ptype = 3`)**: its storm differs from 0.1.0's, because ArWen corrected a table read (rain collecting
  graupel). On CM1's supercell namelist on the RTX 4090, over the mature phase, max rain is 8 % higher and the cold
  pool 4 % smaller; the timing is unchanged. It has not yet been compared with CM1 or WRF on this engine. It still
  runs only after an acknowledgment, and its output is still unverified.
- **Speed and memory**: a step takes 1 to 4 % less time on the RTX 4090 and about 3 to 6 % less on the RTX 3080
  (single runs; the RTX 3080 also drove a desktop, and its two GM1 0.1.0 runs of the same preset differ by 7 %).
  Peak card memory is unchanged for Morrison and WSM6 on both cards, and about 340 MiB lower for Thompson on the
  RTX 4090. The first run compiles its kernels about 1 s faster on the RTX 4090.
- **Restarts**: a storm stopped at 30 minutes and resumed to 2 hours gives the uninterrupted storm's frames word
  for word after the resume, as in 0.1.0, on both cards. Finish a storm with the GM1 version that started it.

| Scheme | Pinned to gpuwm 2.8.9 bytes | GPU revalidation on 2.8.9 (RTX 4090 and RTX 3080) |
|---|---|---|
| Morrison (`ptype = 5`) | yes | re-measured: 0.985 to 1.050 of CM1 r21.1 on its supercell namelist, the same on both cards. Output differs from 0.1.0's in the last place from about 25 minutes of storm time. |
| WSM6 (`ptype = 7`) | yes, including the 256-level capacity tier | passed on both cards: CM1's supercell namelist gives 0.1.0's output word for word for 2 hours. The 256-level capacity tier was not run on 2.8.9. |
| Thompson (`ptype = 3`) | yes | RTX 4090 only: runs and stays finite for 2 hours; its storm differs from 0.1.0's (the corrected table read). Not compared with CM1 or WRF on this engine. |
| WDM6, aerosol-aware Thompson | yes | none planned; not offered. |
| YSU, MYNN, RRTMG, New Tiedtke | yes | none planned; no GM1 run couples them. |

The tables under "0.1.0" below are 0.1.0's own measurements, on gpuwm 2.8.0 or earlier. For this release the
Morrison numbers above replace 0.1.0's, WSM6's stand, and the speed tables were measured on an earlier ArWen.

**Not run on ArWen 2.8.9.** These stand as measured on 2.8.0 or earlier, or not at all:

- **RTX 50-series cards.** An RTX 4090 and an RTX 3080 have run on 2.8.9; no RTX 50-series card has. On RTX
  50-series cards ArWen's notes expect WSM6 and Morrison output to differ in the last place from the other two
  cards' (the division fix above), so GM1's comparison with CM1 on those cards is unmeasured.
- **On the RTX 3080:** Thompson, ground friction and the 250 m supercell preset (it needs about 16 GB; the card
  has 10 GB). These were run on the RTX 4090 only.
- **The 256-level capacity profile.** A 237-level column with the profile was admitted on 2.8.0; the admission run
  has not been repeated on 2.8.9, and the profile's memory and register measurements were taken on the 2.8.0
  kernel.
- **WSM6's column check against WRF v4.6.1.** It graded the kernel text of gpuwm 2.8.0 and earlier. The 2.8.9
  text differs only in how seven divisions by a constant are written; the check runs on a GPU and has not been
  repeated.
- **Thompson's 120-s column against WRF v4.6.1**, and any Thompson storm against CM1.
- **The GUI** with a stop and a continue; **the nine dynamics cases** held against CM1; **every preset to its first
  picture**; **the famous-storm and `iinit = 12` runs**; **the low-level path's refusal of a card another
  program holds**; **WSM6 on a tiled grid** and **Morrison at 512 x 512 x 75** for speed.
- **The `iinit` comparisons** in `docs/CM1-USERS.md` and **the speed tables** were measured before the move to
  2.8.9.

### How to update

Run the install line again: on Windows the PowerShell line, on Linux the `curl` line (`docs/INSTALL.md`, "Update
and uninstall"). It moves gpuwm and gpuwm-data to 2.8.9 along with GM1. With pip: `pip install --upgrade
gpuwm-cm1`, which does the same. Close GM1 and let a running storm finish first; the installer stops with a message
while one runs. Your storms stay where they are. A storm started with an earlier GM1 cannot be continued by 0.1.2.

### Known limitations

- `gpuwm-cm1 doctor --science` still quotes the 0.1.0 measurements; the release notes give the 2.8.9 ones.
- No RTX 50-series card has run GM1 on ArWen 2.8.9, and Thompson's storm has not been compared with CM1 or WRF on
  this engine ("Not run on ArWen 2.8.9", above).
- The known limitations of 0.1.1 and 0.1.0 still apply ("Known limitations" under "0.1.1", and "Known
  limitations", "Not in v0.1" and "Further known limits" under "0.1.0", below).

## 0.1.1

A release of fixes for Windows. The model, the engine pin (gpuwm 2.8.0 and gpuwm-data 2.8.0) and every number in
the tables are those of 0.1.0, and everything under "0.1.0" below describes 0.1.1 too. Verified on Windows 11 with an
RTX 3080.

### Fixed on Windows

- **GM1 opens your browser.** 0.1.0 opened a File Explorer window (showing Documents) instead of the browser, and
  the link had to be copied from the console. 0.1.1 opens the link in the Windows default browser. The console says
  "Opened" only when that worked, and otherwise prints the link for you to open.
- **Opening GM1 again goes to the running storm.** A page that was just opened goes straight to the storm that is
  running, and the first screen and Runs say "A storm is running" with a **Watch it** button. Starting GM1 while it
  is already open brings its page up in the browser again.
- **The link survives a restart.** A tab or a bookmark from an earlier start keeps working: GM1 keeps the link's key
  with your runs folder, readable by you only, so a link stays good until you ask for a new one with `gpuwm-cm1 gui
  --new-token`. A page opened with an old key, or with none, says "This link is out of date" and how to get the
  right one.
- **Folders are shown as Windows paths**, the way File Explorer names them
  (`\\wsl.localhost\Ubuntu-24.04\home\YOUR-NAME\gm1-runs`), in the console, the doctor, Runs and Results, with an
  **Open folder** button. The installer ends by naming the folder your storms will be in.
- **Storm sizes, Delete and Keep pictures only.** Runs shows the size of each storm and the total. **Delete**
  removes a storm. **Keep pictures only** removes its large model files and keeps the pictures; such a storm can no
  longer be continued. Both ask first, naming the storm and its size, and both are refused while the storm is
  running or while GM1 is still drawing its pictures.
- **The doctor's disk line names the Windows drive.** Under WSL2 it gives the free space of the Windows drive that
  holds Ubuntu's disk as well as the Linux figure, and judges by the smaller of the two.
- **Smaller wording fixes.** The "Starting" sentence goes once the storm is running, and "Stopping at the next step"
  goes once it has stopped. The per cent done agrees with the storm time shown beside it. The first card no longer
  says "about 2 minutes on most GPUs" above your own GPU's time. The installer says "Updating" when it updates.

The running-storm page, the kept link, the storm sizes, Delete and Keep pictures only work the same way on Linux.

### How to update

Run the install line again: on Windows the PowerShell line, on Linux the `curl` line (`docs/INSTALL.md`, "Update
and uninstall"). With pip: `pip install --upgrade gpuwm-cm1`. Close GM1 and let a running storm finish first; the
installer stops with a message while one runs. Your storms stay where they are. If a File Explorer window that 0.1.0
opened is still on the screen, close it.

### Known limitations

The known limitations of 0.1.0 all apply to 0.1.1 unchanged; this release fixes none of them. They are listed under
"0.1.0" below ("Known limitations", "Not in v0.1" and "Further known limits"). In short:

- Morrison (`ptype = 5`) and WSM6 (`ptype = 7`) run. Thompson (`ptype = 3`) runs only after an acknowledgment and its
  output is unverified on this release's engine. NSSL and every other scheme are refused by name.
- 21 of CM1's 23 shipped namelists are refused; the supercell and the squall line run.
- Some guided storms are greyed, each with its reason: the squall-line presets, the 100 m supercell, and any storm
  that needs more memory than the card has free.
- GM1 does not check whether another program is already using the GPU.
- Surface drag runs with `cecd = 1` only.
- Restarts do not cross GM1 versions: finish a storm with the version that started it.
- On Windows, closing the GM1 window closes the GUI, never a running storm. Shutting down, restarting or sleeping the
  computer, or `wsl --shutdown`, stops a storm.
- Reaching the GUI from Windows is measured in WSL2's default network mode only.

## 0.1.0

GM1 (the `gpuwm-cm1` package) runs CM1 r21.1's idealized storm cases on one NVIDIA GPU (Linux or WSL2), on uniform
grids from 100 m to 2 km, with CM1-format output and a local web GUI. Bring a CM1 `namelist.input`: GM1 says, setting
by setting, what it runs as CM1 does, runs it, and writes CM1's own output files. The column physics run through
ArWen, the Apache-2.0 GPU weather engine published as the `gpuwm` package. `Development Status :: 3 - Alpha`.

0.1.0 is the first release. The fidelity and speed tables below were measured on its code with gpuwm 2.7.4, before
the engine pin moved to 2.8.0; with gpuwm 2.8.0 the fidelity runs give the same words (see "Engine pin").

gpuwm-cm1 is an independent port. It is not affiliated with or endorsed by the University Corporation for
Atmospheric Research (UCAR) or the NSF National Center for Atmospheric Research (NCAR).

Start with `docs/QUICKSTART.md` (a first storm in the GUI) or `docs/CM1-USERS.md` (your own namelist). Install:
`docs/INSTALL.md`: one line in PowerShell on Windows (it sets up WSL2 and Ubuntu, and a GM1 shortcut), one line in
a Linux terminal, or `pip install gpuwm-cm1`, which installs CuPy for CUDA 13 and NVRTC with it. When something does
not work: `docs/TROUBLESHOOTING.md`, `gpuwm-cm1 doctor --plain` (one fix line per problem) and `gpuwm-cm1 doctor
--report` (one block to paste into a bug report).

### Supported microphysics

- **Morrison (`ptype = 5`)**: runs; on CM1's own supercell namelist it matches CM1 r21.1 within a few per cent
  ("How close to CM1", below).
- **WSM6 (`ptype = 7`)**: runs, through ArWen's WSM6. The coverage report says "Runs differently": CM1 forces
  `eqtset = 1` for WSM6, and GM1 runs `eqtset = 2` with WSM6's heating converted to constant volume.
- **Thompson (`ptype = 3`)**: runs only after an acknowledgment (`--acknowledge-thompson`). Its one check, a 120-s
  column against WRF v4.6.1's own Fortran, was made on an earlier ArWen release; gpuwm 2.8.0 changes Thompson's
  numerics, so that check has not yet been repeated on the engine this release is pinned to. Treat Thompson output
  as unverified.
- **WDM6 and aerosol-aware Thompson**: not offered. CM1 r21.1 has no `ptype` for either.

**Engine pin.** 0.1.0 requires gpuwm 2.8.0 and gpuwm-data 2.8.0 (`>=2.8.0,<2.8.1`). Before a run, GM1 checks that the
installed ArWen files for the selected scheme match the SHA-256 digests recorded for gpuwm 2.8.0, and refuses the
scheme if they differ. Status of each scheme on that pin:

| Scheme | Pinned to gpuwm 2.8.0 bytes | GPU revalidation on 2.8.0 |
|---|---|---|
| Morrison (`ptype = 5`) | yes | passed on an RTX 4090: CM1's supercell namelist (from the command line, and in the GUI with a stop and a continue), an `iinit = 12` namelist and a famous preset give the same words as with gpuwm 2.7.4 on the same card, and the ten statistics against CM1 r21.1 are unchanged (0.986-1.046). Not yet repeated on an RTX 5090. |
| WSM6 (`ptype = 7`) | yes, including the 256-level capacity tier | passed on an RTX 4090: CM1's supercell namelist with `ptype = 7` gives the same words as with gpuwm 2.7.4. A 237-level column with the 256-level capacity profile is admitted and completes a step; its numerical and resource measurements are still to be made. |
| Thompson (`ptype = 3`) | yes | needs a new measurement (2.8.0 changes its numerics). |
| WDM6, aerosol-aware Thompson | yes | none planned; not offered. |
| YSU, MYNN, RRTMG, New Tiedtke | yes | none planned; no GM1 run couples them. |

- **Every other scheme** (NSSL, P3, Kessler, Goddard, GSR, RE87, ISHMAEL) is refused by name.

### Known limitations

- **Some guided presets are greyed** ("not offered") with the measured reason: the squall-line presets until the SGS
  closure's repeating-side families match CM1, the 100 m supercell (in 0.1.0 it stops early on a domain-position
  check; being fixed), and any preset that needs more memory than the card has free.
- **`gpuwm-cm1 run` and the GUI do not check whether another program is already using the GPU.** Run on a card with
  enough free memory. (The low-level `gpuwm-cm1 cartesian --gpu-uuid` path does check.)
- **The first run on a machine compiles GPU kernels** (about 10 s) without printing a notice.
- **Surface drag** (`bbc = 3`, `sfcmodel = 1`) runs with `cecd = 1` (constant drag coefficient) only. The import
  refuses `cecd = 2` and `3`, which 0.1.0 does not implement; set `cecd = 1`.
- **NSSL microphysics is refused.** It is the first item for v0.2 (it blocks 17 of 21 real namelists collected for
  this release).
- **21 of CM1's 23 shipped namelists are refused** (LES, hurricane, 2-D, single-column, DNS and terrain cases); the
  supercell and squall line run.
- **Slurm and PBS job scripts** are written but have not yet run on a real cluster.
- **6 CPU tests fail on hosts without AVX-512.** In the source tree,
  `tests/test_cartesian_restart.py::test_published_frames_and_restarts_are_the_pre_profiling_bytes` (its six profiler
  and publication-mode variants) compares a CPU run's published files with digests recorded where numpy ran its
  AVX-512 kernels. numpy's AVX2 and SSE kernels round some float32 results differently, so on an AVX2-only CPU the
  second frame's words differ (and with SSE only, the first frame's too). The GPU path is unaffected.
- **The GPU path has been tested on RTX-class cards only** (RTX 3080, 4090, 5070 Ti and 5090); data-centre GPUs
  have not been tested.
- Further limits are listed under "Further known limits" below.

## Starting a storm, the famous presets and real-weather downscaling

- **CM1's other ways to start a storm.** Every CM1 r21.1 `iinit` arm was audited against GM1, and the 3D arms GM1
  lacked or ran differently were added, word for word with CM1's source:
  - `iinit = 12`, **updraft nudging** (Naylor and Gilmore 2012; `init3d.F:1370-1396`, `misclibs.F:539-585`,
    `sound.F:564-590`), which about half of the real CM1 namelists collected for this release use;
  - `iinit = 9`, forced convergence (Loftus et al. 2008), and `iinit = 10`, momentum forcing (Morrison et al. 2015);
  - `iinit = 7`, CM1's tropical-cyclone vortex, word for word at time zero;
  - `iinit = 8` and `irandp = 1` now use CM1's own random numbers (the gfortran stream `init3d.F:427-440` seeds), so
    they match CM1 at time zero instead of "running differently"; `iinit = 3`, 5 and 11 put their perturbations
    where CM1 does when a namelist does not say otherwise.
  The words CM1 keeps inside `init3d.F` (the nudging centre, strength, size and time window; the convergence and
  momentum-forcing words) are settings under CM1's own names with CM1's values as defaults, in the Settings
  screen's "Updraft nudging", "Forced convergence" and "Momentum forcing" groups. Against CM1 r21.1 on the stock
  1 km deck with `iinit = 12`, max w agrees to 0.002 m/s through 7 minutes. `iinit = 4` (the moist benchmark) and
  CM1's other analytic soundings (`isnd` 1-4, 8, 10-14, 18-23) are built as CM1 builds them but shown only
  (`iconly = 1`): GM1's run from `isnd` 4, 10 and 11 and from `iinit = 4` does not match CM1's yet, and its run from
  the others has not been compared with CM1's yet. A run starts from `isnd` 5, 7, 17 or 9 (whose first 15 minutes
  matched CM1's).
- **The nine famous-environment presets are offered.** None makes a storm from CM1's warm bubble on a grid a GPU
  fits, so they start with CM1's updraft nudging, with CM1's own words (a 10 m/s target in an ellipsoid 20 km across, fading
  from minute 15, gone at minute 20), centred where the storm-track check recommends. All nine made a storm on the
  RTX 3080 at 1 km (80-158 s each). Two were run in CM1 r21.1 as well (Norman, 20 May 2013 and 3 May 1999): CM1
  makes the same storms, max w within 1 %, in the same column, and just as short-lived. Each card
  says the storm is started by a gentle push of rising air, what its storm did in the run behind its picture, and
  that it is the day's sounding in an idealized storm, not a reconstruction of that day's storms.
- **Downscale (3D) on real weather.** The chain takes the history's cell length on the ground (its map scale over
  the box) instead of requiring a map scale within 1 % of 1, and can drive the box with water vapour
  only (WRF's `have_bcs_moist = .false.`) under GM1's own microphysics when GM1 does not run the history's scheme.
  With the same limits otherwise, a box over the ocean off California in the ArWen control run of the 25 December
  2025 atmospheric river passes every check, and ran for 2 h on the RTX 3080 (`docs/WRF-DOWNSCALING.md`, "A worked
  example"). Its outermost row equals WRF's target each step; the relaxation zone does not pull pressure toward
  the driving run as well as it does wind, temperature and moisture (recorded, not hidden). Boxes over the Sierra
  and the Central Valley are refused, each with its reason.
- **The live nest holds its outermost row** to the parent each step, as WRF's `spec_bdy_final` does.
- **The looks, independently checked.** 1,501 numbers from the looks were recomputed from the netCDF output alone
  and all agree. Clouds from above uses CM1's dry-air density; Morrison runs keep CM1's rain and hail swaths
  (`output_srs`, `output_sgs`), as CM1 does; the environment look of a downscaled run does not say the air is the
  same everywhere; the looks' "about N s" estimates come from measured costs.

## The GUI's looks, starter storms and WRF or ArWen input

- **The GUI's design.** Near-black glass panels, the ink and hairline scales, a cyan accent and focus ring, violet
  for "new". Every text on every screen measured at WCAG AA against what is actually painted behind it, at 1440 and
  400 px.
- **More looks** on Results, from any run, with no extra setup, each with its own "What am I looking at" and a line on
  what the grid spacing can show: cold air near the ground and gust fronts, rain so far, clouds seen from above (a
  simplified visible-light picture from the model's cloud water and ice, lit by a low sun; not satellite data), a
  slice through the strongest updraft, hail near the ground so far, the strongest near-ground spin so far, the
  storm's environment (skew-T, hodograph, CAPE, CIN, shear, helicity, storm motion), air rising into the storm (runs
  that follow air parcels) and the swirl near the ground (a film). Guided Results opens on the four favourites;
  the rest sit in one "More looks" row. Drawn on the CPU: all ten per-frame looks take 3.8 s per frame at 1 km on one
  core.
- **Save as film** for any look that changes with time: an animated PNG always, an MP4 with the optional extra
  (`pip install "gpuwm-cm1[film]"`, imageio-ffmpeg, whose separately licensed ffmpeg program is run, never bundled),
  else a GIF. The 1 km swirl film (217 pictures) takes about a minute.
- **More starter storms**, in four groups: Start here (the quick 1 km supercell stays first), Explore (make your own
  environment with two sliders, storms that split, multicells, a storm in almost calm air), Other storm types (a
  line of storms, a gust front, a warm bubble growing into a cloud at 250 m, the squall lines) and Famous
  environments (nine public-domain NWS balloon soundings of well-known days, each "this day's sounding in an
  idealized storm, not a reconstruction of that day's storms"). Every offered card ran on an RTX 3080 for its picture
  and time. The famous cards start their storms with CM1's updraft nudging (above): CM1's warm bubble makes none on a
  3080-sized grid, and CM1 r21.1 itself makes none from the bubble on the 20 May 2013 deck either (the same updraft to
  0.01 m/s).
- **An environment from a WRF or ArWen run** (the column route): open a `wrfout` or ArWen history, pick a time and a
  point on a map of 0-6 km shear or CAPE, and GM1 writes CM1's `input_sounding` from that one column (destaggered,
  winds turned to earth directions, theta from T or THM, heights from PH+PHB), previews its skew-T and hodograph, and
  runs it as an idealized storm. `gpuwm-cm1 sounding times|from-wrf` and `gpuwm-cm1 import NAMELIST --wrf FILE`; in
  guided mode the card "Use a sounding from my WRF or ArWen run". One column, the same everywhere: no terrain, no
  boundary forcing.
- **Downscale a WRF or ArWen run (3D)**, in advanced mode only: the ArWen chain (WRF's ndown in GM1's terms)
  reads WRF histories as well as ArWen's; `gpuwm-cm1 chain inspect|check|new`; a screen to choose the box on a map,
  the ratio and the time span, with every admission check in plain words. Its outermost row is held to the driving
  run each step as WRF's `spec_bdy_final` does. **Verification status:** host-tested, and four device runs on the
  RTX 3080 (a steady hold, a box in an ArWen idealized supercell history, a box in a GM1 1 km run, and the
  real-weather box above). `docs/WRF-DOWNSCALING.md` has both routes, the limits and the roadmap for terrain and
  mapped coordinates.

## Running CM1 namelists

- **CM1's own storm, within a few per cent.** Each scheme's constant-pressure latent heating reaches CM1's
  constant-volume equations as CM1's does, and CM1's supercell namelist gives the ten mature-phase statistics at
  0.986-1.046 of CM1 r21.1's, where CM1 against a copy of itself nudged by 0.0001 K gives 0.984-1.003 (measured).
- **CM1's SGS start-up**: `ramp_sgs`/`ramp_time` are honoured, and step 1's TKE starts as CM1's does.
- **Periodic sides**: the momentum and acoustic faces of repeating north-south or east-west sides match CM1's.
- **`gpuwm-cm1 import`**: any CM1 r21.1 namelist, with no CM1 build. Older CM1 keys are read under their r21.1
  names; r22.0 keys are judged by value. The six-category coverage report lists every refusal in one pass. `ptype`
  5 is Morrison, 7 is WSM6, 3 is Thompson after an acknowledgment; any other scheme is refused by name.
- **`gpuwm-cm1 edit`**: change namelist words and the settings CM1 keeps in Fortran (the bubble, cold pool, line
  thermal, parcel seed box), then check again; the difference from your file is kept.
- **Run control**: `run`, `stop`, `pause`, `resume`, `status`, a per-GPU queue, crash detection that knows which host
  a run is on, and `jobscript` for Slurm and PBS with the output writer and pictures as sidecars on the GPU node's
  CPUs. The wind swaths and ground canvases are carried in restarts, so a paused and resumed run is bit-identical to
  one that never stopped (measured on the RTX 3080 in all 37 carriers).
- **`gpuwm-cm1 cm1out`**: CM1's own output files from a run's frames. netCDF laid out as `writeout_nc.F` writes it
  (CM1's dimensions in km, names, units and 71 global attributes) or GrADS with CM1's descriptors, following
  `output_format` and `output_filetype`; `cm1out_stats`; `cm1out_pdata` for parcels; and for Morrison, `dbz` and
  `cref` from CM1's `calc_refl10cm` on the CPU. A t = 0 frame is written. A CM1 user's xarray script ran unchanged
  on real CM1 output and on GM1's.
- **`gpuwm-cm1 estimate`**: GPU memory, locked host memory, disk and wall time against what is free now (a desktop
  display can hold several GB of the card), a storm-track check, and "Measure speed".
- **`gpuwm-cm1 gui`**: a local web app on 127.0.0.1 with a token, in two modes over one engine.
  - *Guided* (the default for a first run): the starter presets (a 1 km supercell and the supercell at 250 m; the
    supercell at 100 m and the squall lines at 1 km and 250 m are greyed in 0.1.0), each fitted to your GPU's free
    memory and a time you pick (about 10 minutes,
    about 1 hour, overnight); one Start; a live picture; results on four looks (reflectivity, spin near the ground,
    updraft at 5 km, the wind swath), each with a "What am I looking at" panel. A preset that cannot run here is
    greyed with the reason.
  - *Advanced*: the namelist editor, the coverage report, every setting, the estimator's numbers, run control with
    Continue or extend, job scripts, output choices, slices and a column skew-T, the reproduce bundle, and "Show
    command" beside every action. The mode is remembered per user and never changes a run folder.
- **`gpuwm-cm1 quicklook`**: the viz 2.0 pictures from each frame, on the CPU.
- **`gpuwm-cm1 doctor --plain`**: is this computer ready, in plain words with one fix line per problem.
- **`gpuwm-cm1 examples --cm1`**: CM1's supercell and squall-line namelists, shipped unchanged under UCAR's notice.
- **Packaging**: a `manylinux_2_28` wheel for Python 3.12-3.14, built in a container; no CUDA toolkit or
  `CUDA_PATH` needed.

## How close to CM1

| Case | What was compared | Result |
|---|---|---|
| CM1's supercell namelist, Morrison (`ptype = 5`), 1 km, 2 h | ten mature-phase statistics (3600-7200 s) against CM1 r21.1 `ptype = 5` | 0.986-1.046; CM1's twin 0.984-1.003. Max w 4.100 / 29.273 m/s at 900 / 1800 s against 4.102 / 29.348. The strongest updraft's column at 5 km is within one cell of CM1's at every 15-min output. |
| The same, pointwise | w after the first rain | departs from CM1 beyond the twin (nrmsd 0.029 at 1800 s against 0.0005): ArWen's WRF-lineage Morrison is not CM1's `morrison.F`. |
| The same namelist with WSM6 (`ptype = 7`) | against CM1 `ptype = 7` | 9-21% stronger. CM1 forces `eqtset = 1` for WSM6; GM1 runs `eqtset = 2` with WSM6's heating converted to constant volume. The report says "Runs differently". |
| Thompson (`ptype = 3`) | one 120-s column | Runs only after an acknowledgment. |
| CM1's supercell namelist with surface drag (`bbc = 3`, `sfcmodel = 1`, `cecd = 1`) | against CM1 r21.1 with the same namelist | Drag is claimed only with `cecd = 1`, CM1's constant drag coefficient: nine of the ten mature-phase statistics within 1.3% of CM1's (0.987-1.013), max vertical vorticity 1.091; the strongest updraft's column at 5 km within one cell of CM1's at every 15-min output; the drag slows the lowest-level mean wind as CM1's does, to 0.0001 m/s for the first 30 minutes. Drag moves CM1's own ten statistics by 2% at most on this case. The report still says "Not yet checked against CM1". `cecd = 2` and `3` are refused by the import (see "Not in v0.1"). |
| Stretched levels with cloud, `lspgrad = 1`, rigid walls, `irbc = 1`, repeating sides | | "Not yet checked against CM1" in the report. |

## Speed on the same case (measured)

| Case | GM1 | CM1 r21.1 | Ratio |
|---|---|---|---|
| CM1's supercell, 1 km, 2 h, 0.58 M cells | RTX 3080: 95.7 s | 8 OpenMP threads: 259.6 s | 2.7× |
| The same physics at 250 m, 256 × 256 × 80, 5.24 M cells | RTX 3080: 0.2283 s per step | 8 threads: 1.807 s per step | 7.9× |
| 100 m case, 128 × 128 × 80 stretched, 900 s | RTX 3080: 138.2 s | 12 threads: 1,558.8 s | 11.3× |

CM1: a Core Ultra 7 270K Plus, gfortran -O2, OpenMP. GM1:
before the latent-heating fix, which added about 7% per step. After the fix, the imported stock namelist runs its 2 h
in 70-72 s on the RTX 3080 from launch to exit. NCAR's OpenACC work on CM1 covers the dry solver; official CM1 with
moist physics is still CPU-only, and GM1 makes no speed comparison with the GPU work.

## What fits (estimated from measured runs; Morrison; 80 levels)

| Grid spacing | Largest uniform area | Card |
|---|---|---|
| 1 km, 500 m | every shipped CM1 convection namelist (0.8 GiB measured) | 8 GB and up |
| 250 m | 64 km (3.5 GB measured); 100 km on a headless 8 GB card; 120 km on 12 GB | 8-12 GB |
| 100 m | 60 km on 16 GB; 78 km on 24 GB | 16 GB and up |

`gpuwm-cm1 estimate` uses the memory actually free on your card.

## Not in v0.1

| | Why |
|---|---|
| `stretch_x`, `stretch_y` | Refused: every horizontal operator assumes uniform spacing. Planned for v0.2. |
| NSSL, P3, Kessler, Goddard, GSR, RE87, ISHMAEL microphysics | Refused by name. NSSL is the first v0.2 item (it blocks 17 of 21 real namelists collected for this release). |
| `iinit` 4 (the moist benchmark), `isnd` 4, 10, 11 | Shown only (`iconly = 1`): the starting state is CM1's, but GM1's run from it does not match CM1's yet. `iinit` 7, 9, 10 and 12 (updraft nudging) run. |
| `isnd` 1, 8, 12-14, 18-23 (CM1's other analytic soundings) | Shown only (`iconly = 1`): the starting state is CM1's, but GM1's run from it has not been compared with CM1's yet. |
| Radiation, PBL, cumulus schemes, `sfcmodel` 2-7 | Refused or not coupled. |
| `cecd` 2 and 3 with `sfcmodel = 1` (wind-dependent drag) | Refused by the import: only `cecd = 1` (constant drag coefficient) is implemented in 0.1.0; set `cecd = 1`. |
| `eqtset = 1` dynamics | GM1 runs `eqtset = 2` only. |
| Reading or writing CM1 restart files | The formats differ. GM1's own restarts continue GM1 runs. |
| Fortran edits (`init3d.F`, `base.F`), heat-flux initiation | The bubble, cold pool, line thermal and parcel seed box are settings instead. |
| LES, hurricane, 2-D, single-column, DNS and terrain namelists | 21 of CM1's 23 shipped namelists are refused (the supercell and squall line run). |
| Native Windows, macOS, more than one GPU, MPI | Linux and WSL2 on one NVIDIA GPU. |
| GPUs older than compute capability 7.5 (V100), drivers older than R580 | CUDA 13 does not support them. |
| Submitting batch jobs from the GUI | The GUI writes the job script for you to submit. |
| 3-D views, WebGL | v0.2. Films of the picture looks are available now (Save as film). |
| `thompson-aerosol-mp28` | Not offered: it reaches routines adapted from Numerical Recipes, whose licence does not permit redistribution, pending a legal review; no CM1 `ptype` selects it. |

## Further known limits

- **The squall-line presets are greyed** ("not offered") until the SGS closure's repeating-side families
  match CM1; CM1's squall-line namelist imports in advanced mode with those rows under "Not yet checked".
- **Statistics come from the frames.** `cm1out_stats` has CM1's names at each output time, not at `statfrq`.
- **Not written yet:** `kmh`, `kmv`, `khh`, `khv` and `rain2`; the report lists them.
- **Parcels restart with each segment**: they are not carried in restarts, the parcel `dbz`, `qsl` and `qsi` are not
  filled, and there is no t = 0 parcel frame.
- **Restarts do not cross GM1 versions.** A restart written by an earlier GM1 is refused by Continue, so finish a
  run with the version that started it.
- **The job script has not run on a real Slurm or PBS cluster yet.**
- **Estimates:** wall times on cards GM1 has not run on are scaled by memory bandwidth ("rough"). The storm-track
  check places CM1's supercell about 10 km west of where GM1 runs it, so its recommended frame and bubble are
  optional. Host memory for the pictures is not priced (about 3.3 GB per 39 M-cell frame for Morrison reflectivity).
- **Gallery pictures** come from 10-minute fitted runs, and the 100 m card shows the 250 m picture.
- **WSL2**: reaching the GUI from Windows is measured in WSL2's default network mode only.
- **Presets:** the famous presets' cards were made from 1 km runs (their 250 m bases need about 9.7 GB of GPU
  memory). The fitted time estimates of two starter presets were off by 25-53 % and the gust front's 30 % high;
  the famous presets' wall-time estimate is 51-57 % low for Wilmington and Jackson (the other seven within 10 %). The
  Wilmington card starts one storm, not a line. CM1's calm-air option (`iwnd = 0`) is refused by the run-start check,
  so the calm presets use 2 m/s of shear.
- **Drawing the looks:** all ten looks cost about 8 s per frame at 5 M cells on one core, and up to 15 s for a mature
  storm's frame (reflectivity's melting-layer step, CM1's `calc_refl10cm`), against a 15 s aim; no 100 m frame was
  timed.
- **3D downscaling** has run on one real-weather box, offshore and flat; its relaxation zone leaves pressure about as
  far from the driving run as the interior.
- **Old glibc:** a CM1 built with glibc 2.40 or older rounds `coshf` differently, so its `iinit = 10` forcing differs
  from GM1's (and from a newer CM1 build) in the last bit of some words.
- **`nz > 80` with WSM6** needs a capacity overlay (`docs/ARWEN-WSM6-CAPACITY-256-AUTHORITY.md`).

## Capabilities of the `cartesian` door (for GM1 decks)

`gpuwm-cm1 cartesian DECK` runs GM1's own TOML decks, which reach capabilities the CM1 importer does not offer.
`gpuwm-cm1 doctor --science` prints the per-key version of this table.

| Capability | Status |
|---|---|
| Uniform-Cartesian dycore: RK3, `psolver = 3` acoustics, 5th-order advection, SGS/TKE, dry-mass correction, upper damping | Validated against CM1 at operator level, and on CM1's supercell namelist as a whole storm |
| ArWen WSM6 column physics | Validated at the column seam against WRF |
| Morrison | Matches CM1 `ptype = 5` on CM1's supercell namelist (above); other namelists not checked |
| Initialization selectors | 30 of 32 selector values checked against CM1 at time zero |
| Lateral boundaries: open-radiative, periodic, `irbc = 2`, `roflux`, `nudgeobc`, `irdamp = 2`, `hrdamp` | Validated against CM1; rigid walls and `irbc = 1` not checked |
| Coriolis, beta plane, adaptive time step | Validated against CM1 |
| Parcels (`parcel.F`) | Validated against CM1 |
| Budgets, pressure decomposition | Device-tested; not compared with a CM1 run end to end |
| Restarts | Bit-identical pause and resume on the GPU |
| Live nests, two-part SGS on stretched decks, Lagrangian extras | GM1-only; not part of the GUI release |
| Simulated reflectivity | WSM6: on the GPU, ArWen's 10 cm operator. Morrison: CM1's `calc_refl10cm` on the CPU in `cm1out` |

## Licensing notes

- **gpuwm-cm1** is Apache-2.0 (`LICENSE`).
- **CM1 r21.1** is the mathematical authority, used under UCAR's release grant, reproduced in `NOTICE`, which must be
  kept. CM1's supercell and squall-line namelists ship unchanged as examples under that grant. No other CM1 source,
  table, executable or runtime asset is bundled.
- **ArWen** (the `gpuwm` package, an installed dependency, not bundled) is Apache-2.0.
- **Shipped soundings** are NOAA/NWS rawinsonde observations (U.S. Government works), as recorded in
  `cm1core/data/soundings/PROVENANCE.md`.
- **NVIDIA CUDA**: the wheel bundles no NVIDIA component; installing it installs NVIDIA's own wheels (CuPy's CUDA 13
  toolkit and NVRTC) as dependencies, under the NVIDIA CUDA Toolkit EULA.
- **DejaVu fonts** for the pictures ship with their licence (`cm1core/quicklook/fonts/LICENSE-DejaVu.txt`).

## Authority and attribution

The mathematical authority is the official CM1 r21.1 release tarball, SHA-256
`dc49fe84531056d1ae6249b37a5e3ee453fd96861c3b6bafd63828d92e64edf7`. CM1 is the work of George H. Bryan and NCAR. The
port does not mix scientific source from another CM1 tree. ArWen supplies column physics through a measured,
versioned seam.

## Verification

The GPU checks below were made on this release's code with gpuwm 2.7.4, before the engine pin moved to 2.8.0. With
gpuwm 2.8.0 on an RTX 4090, these runs give the same words as with gpuwm 2.7.4 on that card: CM1's stock supercell
(also in the GUI with a stop and a continue), the famous preset, the `iinit = 12` namelist, WSM6, and the seven
fixed-step dynamics cases (74 of 74 frames); the two adaptive-step dynamics cases stay within their recorded bounds
of CM1 (see "Engine pin").

- **Dynamics:** the nine dynamics cases held against CM1 give the same words on the RTX 3080 every time they are run
  (102 of 102 frames).
- **CM1's stock supercell:** a fresh Python 3.12 venv gives the max w series 4.10 / 29.27 m/s at 900 / 1800 s, and so
  does the GUI with a stop and a continue.
- **A famous preset:** Norman, 20 May 2013 with updraft nudging gives the series its card was made from.
- **An `iinit = 12` namelist in the GUI:** CM1's supercell namelist with `iinit = 12`, run through the GUI's advanced
  mode, gives max w within 0.997-1.034 of CM1 r21.1's own run at every saved time.
- **The real-weather chain box:** the offshore box of the atmospheric-river run gives the same bytes as its first run,
  in every field both write.
- **CPU suite:** on a Linux computer without AVX-512, the only failures are the AVX-512 pins under "Known limitations".
- **Walkthrough:** 64 screens at 1440 and 400 px: 0 errors, 0 overflow.
- **Build.** `tools/release/build_manylinux.sh` in `quay.io/pypa/manylinux_2_28_x86_64` (AlmaLinux 8.10, glibc 2.28,
  Rust 1.95.0) on a Linux build host: the sdist, the wheel built from its extraction,
  `auditwheel repair` (the Rust tools and `libcm1_state.so` need nothing newer than glibc 2.28), and
  `tools/inspect_artifacts.py --require-manylinux`: 20 of 20 checks pass. The build is reproducible: two builds
  of one commit give byte-identical wheels and sdists.
- **Install, a CM1 user's first hour** (`tools/release/smoke_cm1_user.sh`, fresh environments, `CUDA_PATH` and
  `CUDA_HOME` unset): on an RTX 3080 under WSL2 with Python 3.12 (venv and pip) and 3.14 (uv): doctor ready, CM1's
  supercell namelist imported, run for its 2 h (about 70 s), converted by `cm1out`, and read by a CM1 user's own
  xarray script; on a Linux computer with no GPU, with Python 3.14 (venv and pip) and 3.12 (uv): doctor names the
  missing driver, import, estimate, the presets and the GUI server work (the GUI process maps no CUDA library), and the 3080 run's
  folder converts and draws there. Every one prints the same max w series: 0, 4.10, 29.27, 49.79, 56.05, 60.34,
  59.85, 58.37, 55.09 m/s at 0-7200 s.
- **Both guides, command by command, for real** on the RTX 3080 (`tools/release/doc_commands.py --mode real`):
  QUICKSTART's 7 commands and CM1-USERS' 22 commands and its xarray block, every one exit 0. The run stopped at 1 h
  and resumed, and the run paused at its 30-min restart and resumed, give the uninterrupted run's max w series. The
  CPU suite runs both guides against the fake runner (`tests/test_docs_guides.py`), with `preset fit` for every preset
  and budget on three GPU fixtures.
- **The GUI, both modes, headless** (Chromium driving the GUI of the installed wheel on the RTX 3080 computer, its
  first launch):
  - beginner: the guided gallery (five cards; the squall lines greyed until the SGS closure is matched, the 100 m card greyed with the free
    memory it needs), Supercell quick look, About 10 minutes, Start: the first picture 18 s after Start, finished in
    73 s, Results on the four looks with their panels;
  - advanced: CM1's supercell namelist imported (exit 0), the coverage report, the namelist and all 583 settings,
    Sounding, Domain and time, Settings (restarts every 30 min, netCDF beside GrADS), Estimate (the recommended GPU
    budget), Start, Stop after the 30-min restart, Continue from it to 2 h, Results (a slice and a column skew-T),
    Commands; the CM1 user script on its `cm1out.nc` prints the uninterrupted run's numbers;
  - 0 console or page errors, 0 failed or outside requests, 0 px of horizontal overflow, no action without its Show
    command, at 1440 and 400 px; 27 screens, 54 screenshots and a contact sheet.
- **CPU suite in the wheel environment** (the wheel in a fresh Python 3.12 venv, a source checkout, no GPU): on a
  Windows 11 computer under WSL2 whose CPU has AVX-512, every test passed but one, a file-time race in the GUI's "is
  this estimate stale" check, since fixed (it could make the Domain screen ask for Estimate again on WSL2). Skips are
  tests that need a GPU, a built CM1 or local evidence that is not distributed.
