# Your first storm with GM1

GM1 runs the storm model CM1 on your NVIDIA graphics card. This page takes you from nothing to a finished storm in
the GUI, in about 15 minutes, most of it the install. You do not need to know CM1: the GUI picks the settings for
you and says what each picture shows. If you already use CM1 and have your own namelist, read
[`CM1-USERS.md`](CM1-USERS.md) instead.

## 1. What you need

- A computer with an NVIDIA graphics card (RTX 20 series or newer) with 8 GB of memory or more, running Windows 10
  or 11, or Linux.
- The NVIDIA driver, version 580 or newer. On Windows, the normal Windows driver is all GM1 needs.
- About 7 GB of free disk: 2.5 GB for GM1 and about 4 GB for the first storm (on Windows add about 2 GB for Ubuntu,
  all on drive C:).

### Common questions

- **Mac?** No. Apple computers have no NVIDIA graphics card.
- **AMD or Intel graphics?** No. GM1 needs an NVIDIA card: RTX 20 series or newer, with 8 GB of memory or more.
- **A laptop?** Yes, if it has such an NVIDIA card.
- **Is anything sent over the internet?** No. GM1 runs entirely on your computer, has no account and sends nothing
  anywhere; the internet is used only to install and update it. The GUI page is reachable only from your own
  computer, with the key in its link.
- **Can I leave it running overnight?** Yes, if the computer does not sleep or restart meanwhile (see section 6).

## 2. Install

**Windows**: open PowerShell (Start menu, type PowerShell; not Command Prompt) and paste this line:

```powershell install
irm https://raw.githubusercontent.com/FahrenheitResearch/gm1/main/install.ps1 | iex
```

If Windows does not have WSL2 (Ubuntu in Windows) yet, the installer sets it up: click Yes when Windows asks,
restart the computer, choose a user name and password when Ubuntu asks, and paste the same line again. If WSL is
there but Ubuntu is not, Ubuntu asks for its user name and password in the same window and the install goes on. When
it is done, the Start menu and the desktop have a **GM1** shortcut, and the Start menu has **GM1 check**.

**Linux**: open a terminal and paste this line:

```sh install
curl -LsSf https://raw.githubusercontent.com/FahrenheitResearch/gm1/main/install.sh | sh
```

Both download about 2.5 GB, mostly NVIDIA's own libraries, and need no Python of your own. If you already use
Python, pip works too, on Linux, in a Python 3.12 to 3.14 environment (see [`INSTALL.md`](INSTALL.md)).

## 3. Check your computer

The installer already ran this check. To run it again on Windows, open **GM1 check** from the Start menu. On Linux,
type:

```sh
gpuwm-cm1 doctor --plain
```

It checks the driver, the graphics card and its free memory, the disk and the memory, and which storms fit your
computer, and prints `ok` or `FIX` for each, with one line saying how to fix anything that is missing. When everything
is `ok`, you are ready.

## 4. Open the GUI

On Windows, open **GM1** from the Start menu or the desktop. On Linux, open a new terminal (so it finds GM1) and
type:

```sh
gpuwm-cm1 gui
```

The GUI opens in your browser, and the GM1 window says so: "Opened the link in your browser". If it could not, the
window says "Open this link in your browser:" with the link, like `http://127.0.0.1:8765/?token=...`: copy it into
your browser. The link stays the same each time you start GM1, so a tab or a bookmark from an earlier start still
works.

Closing the browser is always fine, and so is closing the GM1 window (on Linux, the terminal): that closes the GUI,
never a running storm, which keeps going until it finishes. Open GM1 again to see it: when one storm is running, the
page opens straight on **Watching**, and otherwise the first screen leads with "A storm is running", its name, how far
it is and a **Watch it** button. Opening GM1 while it is open brings the page that is already open up in your browser
again. On Windows, shutting down or restarting the computer, or `wsl --shutdown`, stops a running storm (Run it longer
or Resume continues it later from its last saved point).

The GUI starts in **Guided** mode. The switch at the top right changes to **Advanced**, which shows every setting;
you can switch back at any time, and nothing about your storms changes when you do.

## 5. Pick a storm

The first screen, **Pick a storm**, shows the storms GM1 can run in four groups. Only **Start here** is open on a
first launch; each other group shows one row of cards and a **Show all** button that opens it in place.

| Group | Cards |
|---|---|
| Start here | Supercell, quick look, 1 km (the quickest to try; each card says about how long it takes on your GPU); Supercell, 250 m; Supercell, 100 m, overnight (greyed in this version) |
| Explore | Make your own environment; storms that split in two; clusters of short-lived storms; a storm in almost calm air |
| Other storm types | A line of storms; cold air spreading out as a gust front; a warm bubble growing into a cloud; the two squall lines |
| Famous environments | The air measured by weather balloons on well-known days (Norman, Oklahoma; Birmingham, Alabama; Denver, Colorado; Wilmington, Ohio; Tallahassee, Florida; Jackson, Mississippi), each named by its station and date |

Each card has a picture from a real run and says about how long it takes on your card. A card that cannot run on
your computer is greyed out with one line saying why, for example that it needs more free graphics memory than your
card has. Only storms GM1 runs the way CM1 does are offered.

A famous-environment card uses that day's balloon sounding in an idealized storm: the same air everywhere, flat
ground, and a gentle push of rising air for the first 20 minutes to start the storm, as CM1's updraft nudging does (from
a warm bubble these balloons' air made no storm on the grids a GPU fits; measured on an RTX 3080). It is not a reconstruction of that day's storms, and every such card says so, along with what its storm did in
the run behind its picture: some last the whole two hours, others rain themselves out within the hour. CM1 r21.1
makes the same storm, with the same life, from two of these balloons (measured). The
balloons ship with GM1, so Advanced can import a card's namelist and sounding (in the installed package's
`cm1core/presets/ID/` folder) and try a finer grid or another start.

**Make your own environment** has two sliders: how unstable the air is, and how much the wind changes with height.
Under them, one sentence says which kind of storm such air usually makes (from Weisman and Klemp's studies), with the
model's own numbers for the air you made. The picture is the nearest setting that has been run, and its caption says
which. The same air from a terminal: `gpuwm-cm1 preset environment --instability 0.5 --shear 0.5`.

Choose **Supercell, quick look, 1 km**.

## 6. Pick how long it may run, and press Start

The next screen asks **How long can it run?** with three buttons: **About 10 minutes**, **About 1 hour** and
**Overnight (up to 10 hours)**. Choose **About 10 minutes**.

GM1 fits the storm to your card and that time. Under **What you get** it says, in one sentence, the area, the grid
spacing, the storm time, about how long it will take and how much memory and disk it needs. Under **What was changed
to fit** it lists anything it made smaller so the storm fits; for the quick look, nothing. Every time and size is an
estimate from speeds GM1 has measured on real runs, and says so.

For **About 1 hour** or **Overnight**, set the computer not to sleep (Windows: Settings, System, Power, Sleep: Never
while plugged in). A restart or sleep stops the storm; **Run it longer** or Resume continues it from its last saved
point.

To change the storm before it starts, press **Customize this storm** (or **Customize** on its card in the gallery).
You can pick the grid spacing, the height levels, the area, the storm time and how often a picture is saved; the air
(the same two sliders as **Make your own environment**, on the storms that use made air); how the storm is started;
random bumps at the start; Earth's rotation; and following air parcels. Each row says its CM1 name in small grey
letters, and the line at the top of the panel says again how long it will take, how much GPU memory and how much disk,
as you change things. What you set stays as you set it: GM1 fits only the rest to your card and time, and lists your
changes under **What you changed**. If your choices do not fit the time you picked, the screen says which time does
(**Use About 1 hour**) or offers to let GM1 choose a setting again; when the storm would drift out of the area it
names the area that keeps it inside, and which one of your changes moves it out. A choice guided mode cannot offer yet (WSM6 or
Thompson rain and hail, ground friction) is greyed with the reason and a button that makes the run folder with it
set and opens it in Advanced; **All settings (Advanced)** makes the run folder
with your choices and opens every CM1 setting, and nothing runs until you press Start there. **Reset to the preset**
puts everything back.

Press **Start**.

## 7. Watch it

The **Watching** screen shows the storm time (up to 2 hours of storm), the time left, and one large picture that
updates each time the model saves a moment of the storm (every 5 minutes of storm time). The first 10 to 60 seconds
are spent preparing your card; the first picture follows a few seconds later. On an RTX 3080 the whole quick look
takes about 1 to 2 minutes.

**Stop** stops the storm at the next step. The pictures so far are kept. A storm stopped before its first saved
point cannot be continued later, so GM1 asks first.

The pictures are drawn on the computer's processor after the GPU saves each moment. On a fast card the GPU can finish
before the pictures do: Watching then says how many are still being drawn, and they appear by themselves.

## 8. Read the results

When the storm finishes, **Open the results** shows four pictures, each with a time slider and **Play**:

- **Reflectivity**: what a weather radar would see 1 km above the ground: rain and hail, heaviest in red and
  purple. A hook-shaped tail at the back of the storm marks where its rotation wraps rain around the updraft.
- **Spin near the ground**: how fast the air spins about a vertical axis at the lowest model level, red one way and
  blue the other, with the wind drawn as streaks.
- **Updraft at 5 km**: rising air 5 km up. A supercell keeps one long-lived updraft.
- **Wind swath**: the strongest near-ground wind each place has had since the start, so it grows as the storm moves.

Beside each picture, **What am I looking at** explains it with your storm's own numbers. Every picture is labelled
SIMULATED: it is a model, not an observation.

Under the four tabs, **More looks** has the rest, drawn from the same run with no extra setup: cold air near the
ground and gust fronts, rain so far, clouds seen from above, a slice through the strongest updraft, hail near the
ground so far, the strongest near-ground spin so far, the storm's environment (its sounding's skew-T and hodograph,
with CAPE and shear), air rising into the storm (when the run followed air parcels) and the swirl near the ground.
Each has its own **What am I looking at**, with a line on what the grid spacing can and cannot show. **Save as
film** under a picture makes a film of that look on the computer's processor: an animated picture always, and an
MP4 when the film add-on is installed. The installers add it; with pip it is `pip install "gpuwm-cm1[film]"`.

## 9. Find the files

**Open the files** says where the storm's folder is, with an **Open folder** button that shows it in your file
manager. On Windows the folder is inside Ubuntu, so it is named as File Explorer names it
(`\\wsl.localhost\Ubuntu-24.04\home\YOUR-NAME\gm1-runs\STORM`), with the Linux path under it. The output is a
folder of netCDF files in CM1's own format, which any CM1 tool reads, with lines to copy for `ncview` and for Python.

Storms take disk space: a quick look takes a few GB, a fine one tens of GB. **Runs** on the left lists every storm
with its size, and the total at the top. **Delete** removes a storm's whole folder. **Keep pictures only** deletes
its big files and keeps its pictures and its results page; such a storm can no longer be run longer, and no new
pictures can be drawn from it. Both ask first, and neither is offered while the storm is running.

**Run it 1 hour longer** continues the same storm from where it ended. **Show everything** switches to Advanced on
the same storm: its settings, the check against CM1, the estimate and the log.

Next, try **Supercell, 250 m** with **About 1 hour**: the same storm with much finer detail. It needs a lot of disk
(up to about 120 GB); **Ready to start** shows how much, and makes the storm smaller when your disk has less free.

## 10. The same from a terminal

Everything the GUI does is a `gpuwm-cm1` command, and **Show command** under **Start** shows it. The guided Start
is the last two of these lines:

```sh
gpuwm-cm1 preset fit supercell-1km --budget 10m
gpuwm-cm1 preset new supercell-1km --budget 10m --root ~/gm1-runs --name my-first-storm
gpuwm-cm1 run ~/gm1-runs/my-first-storm
```

`preset fit` only says what Start would make on your card; `preset new` writes the storm's folder, and `run` runs
it. A customized storm adds one `--custom KNOB=VALUE` per change, for example
`gpuwm-cm1 preset fit supercell-1km --budget 10m --custom grid=500 --custom parcels=on`, and
`gpuwm-cm1 preset options supercell-1km` lists what can be changed and the choices. `gpuwm-cm1 preset list` shows every storm the gallery offers, and whether it is offered here:

```sh
gpuwm-cm1 preset list
gpuwm-cm1 status ~/gm1-runs/my-first-storm
```

The GUI keeps its storms in `~/gm1-runs` unless you start it with `gpuwm-cm1 gui --root FOLDER`. Every command the
GUI can run is listed in [`COMMANDS.md`](COMMANDS.md).

## If something goes wrong

- **A card says it does not fit.** Close other programs using the graphics card, or pick a shorter time; the fit
  makes the storm smaller for a shorter time. On Windows the desktop itself keeps 4-5 GB of the card.
- **Waiting for the GPU.** Another storm is running on the card. This one starts when it ends.
- **My storm stopped overnight.** The computer slept or restarted for updates, or (Windows) the GM1 window was
  closed. Open GM1, open the storm and continue it with **Run it longer**.
- **The pictures are blank at first.** A storm takes 20 to 30 minutes of storm time to grow from its first bubble of
  rising air; until then there is little to see. Keep watching.
- **The GUI does not open.** Check that the GM1 window (or the terminal running `gpuwm-cm1 gui`) is still open, and
  open the exact link it printed, with its token.
- **"This link is out of date."** The page was opened without its key, or with an old one. Start GM1 again (Start
  menu: GM1) and use the page it opens.
- **Anything else.** Run `gpuwm-cm1 doctor --plain` (on Windows: **GM1 check** in the Start menu) and follow its fix
  lines. In the GUI, **System** on the left shows the same check. [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) has the
  common problems with their fixes, and how to ask for help.
