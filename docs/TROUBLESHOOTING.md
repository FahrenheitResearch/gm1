# When GM1 does not work

Start here: run the check. On Windows, open **GM1 check** from the Start menu: it runs the check in a window and
saves `gm1-report.txt` on your desktop for a help request. On Linux, open a terminal and type:

```sh
gpuwm-cm1 doctor --plain
```

(On Windows you can also type `wsl` in PowerShell to reach Ubuntu, then type the same line. The Ubuntu app's name in
the Start menu varies, for example "Ubuntu 24.04 LTS".)

It checks the driver, the graphics card, GM1's own files, Python, the disk and the memory, and which storms fit your
computer. Everything marked `FIX` has one line under it saying exactly what to do. Most problems are below, each with
its fix.

## The NVIDIA driver, and WSL2 on Windows

**"No NVIDIA driver found", "No NVIDIA GPU is visible inside WSL2", or the driver is older than R580.** GM1 needs the
NVIDIA driver 580 or newer (CUDA 13).

- On Windows: install the newest driver for your card on **Windows** from
  <https://www.nvidia.com/Download/index.aspx> (or the NVIDIA App), restart the computer, then in PowerShell run
  `wsl --update` and then `wsl --shutdown`, and open Ubuntu again. WSL2 uses the Windows driver: never install an
  NVIDIA driver inside Ubuntu. If you did, remove it in Ubuntu with `sudo apt purge 'nvidia-driver-*'`, then run
  `wsl --shutdown` in PowerShell.
- On Linux: install driver 580 or newer from <https://www.nvidia.com/Download/index.aspx> (on Ubuntu:
  `sudo ubuntu-drivers install nvidia:580`), then restart the computer and check that `nvidia-smi` says 580 or newer.

**"The NVIDIA driver answers, but its CUDA library (libcuda) cannot be loaded".** The driver is half installed or
was partly removed. Reinstall it as above (on Windows, the Windows driver, then `wsl --shutdown`), and restart.

**A storm fails with "No NVIDIA graphics card can be used".** The NVIDIA driver is missing or older than R580, as
above. GM1 also says so when it opens. Run `gpuwm-cm1 doctor --plain` for the exact fix.

**"Older than the RTX 20 series".** CUDA 13 runs on cards of compute capability 7.5 and newer: RTX 20 series and
later, GTX 16 series, and data-center cards from the T4 on. A GTX 10 series card or older cannot run GM1.

**The Windows installer stops at WSL with "virtualization" or error 0x80370102.** Virtualization is turned off in
the computer's firmware. Check: Task Manager, Performance, CPU, "Virtualization". If it says Disabled, restart into
the BIOS or UEFI settings, turn on Intel VT-x (or "Intel Virtualization Technology") or AMD SVM, save, start Windows
and paste the installer line again.

**Ubuntu asks for a user name and password.** That is Ubuntu's first start: choose any name and password (the
password is not shown while you type it). When the installer asked for it in its own window, type `exit` at the
prompt that follows and the install goes on; otherwise paste the installer line again in PowerShell.

**Doctor says RAM is low (Windows).** Ubuntu gets only part of the computer's memory. In PowerShell paste this line,
then `wsl --shutdown`, and open GM1 again:

```powershell install
Add-Content "$env:USERPROFILE\.wslconfig" "`n[wsl2]`nmemory=16GB"
```

If that file already has a `[wsl2]` line, do not add a second one: open it with
`notepad "$env:USERPROFILE\.wslconfig"` and put `memory=16GB` on the line under `[wsl2]`.

## Installing

**The installer stopped with an error.** Paste the same line again: a dropped download is the most common cause, and
the installer picks up where it stopped. If it stops again at the same place, ask for help (below) with its last
lines.

**"Only N GB is free" or "No space left on device".** The install needs about 2.5 GB and the first storm about 4 GB
more (on Windows add about 2 GB for Ubuntu, all on drive C:). Free some space and paste the line again.

**Drive C: is still full after I deleted storms (Windows).** Ubuntu's virtual disk does not shrink by itself. In
PowerShell: `wsl --shutdown`, then `wsl --manage Ubuntu --set-sparse true` (with your distribution's name from
`wsl --list` if it is not Ubuntu). Windows then gives the space back over time.

**`gpuwm-cm1: command not found`.** Open a new terminal (the installer added GM1's folder for new terminals), or type
`export PATH="$HOME/.local/bin:$PATH"`.

**pip says "No matching distribution found for gpuwm-cm1".** GM1 runs on 64-bit Intel or AMD Linux with Python 3.12
to 3.14. On Windows, pip in PowerShell cannot install GM1: use the Windows installer line. On Linux, use the
installer line instead: it brings its own Python.

**pip says "externally-managed-environment", or venv says "ensurepip is not available".** Ubuntu and Debian keep
their own Python for themselves. Use the installer line instead: it needs no Python of yours. To stay with pip, type
`sudo apt install python3-venv` and make a virtual environment first, as `INSTALL.md` shows (Ubuntu 22.04 and Debian
12 have a Python older than 3.12: use the installer line there).

**"A storm (or the GM1 window) is running" when updating.** The installer does not replace GM1 under a running
storm. Close the GM1 window and wait for the storm to finish (or stop it in GM1), then paste the line again.

**Extra `cm1-...` and `gm1-...` commands.** The installer also puts GM1's own helper programs on `PATH`
(`cm1-state`, `cm1-render`, `cm1-r21-namelist` and a few more). GM1 uses them itself; you only need `gpuwm-cm1`.

**Repair an install.** Run the installer line with `GM1_REINSTALL=1`, which reinstalls every package:

```sh install
curl -LsSf https://raw.githubusercontent.com/FahrenheitResearch/gm1/main/install.sh | GM1_REINSTALL=1 sh
```

## Starting the GUI

**The browser did not open.** The GM1 window says which happened. "Opened the link in your browser" means Windows
(or your Linux desktop) took the link; look for a new tab in your default browser, which may be behind other windows.
"The browser was not opened" is followed by "Open this link in your browser:" and the link, like
`http://127.0.0.1:8765/?token=...`: copy it into your browser exactly as printed (in Windows Terminal, hold Ctrl and
click it). The part after `token=` is the key to the page. Over ssh GM1 never opens a browser; see `INSTALL.md` for
the ssh line that brings the page to your own computer.

On Windows GM1 asks Windows to open the link in your default browser (`rundll32.exe url.dll,FileProtocolHandler`,
then `cmd.exe /c start`, then PowerShell's `Start-Process`). If no browser comes up at all, set a default browser in
Windows Settings, Apps, Default apps. GM1 0.1.0 used `explorer.exe`, which on some computers opened a File Explorer
window showing Documents instead of the browser: update GM1 (paste the installer line again) and close that window.

**"This link is out of date."** The page was opened without its key or with an old one: a link typed without the
`?token=` part, a new browser profile, or a link from before `gpuwm-cm1 gui --new-token`. Start GM1 again (Start
menu: GM1) and use the page it opens. The link is kept with your runs folder, so it stays the same from one start to
the next; `gpuwm-cm1 gui --new-token` makes a new one, after which every earlier link and tab stops working.

**I opened GM1 again and my storm is not there.** It is: with one storm running the page opens on Watching, and
otherwise the first screen leads with "A storm is running" and a **Watch it** button. **Runs** on the left lists
every storm, running or finished.

**"Port 8765 is in use".** Another program uses that port. Opening GM1 again while it is already open shows the
page that is open (with its link), and GM1 tries the next nine ports by itself when another program has 8765. When
all ten are taken, it says so: pick another port, `gpuwm-cm1 gui --port 8800`.

**The GM1 window says "GM1 stopped" or closes at once.** Read the lines above "GM1 stopped". In Ubuntu, run
`gpuwm-cm1 doctor --plain` and follow its fix lines.

**Closing things.** Closing the browser is always fine, and so is closing the GM1 window (on Linux, the terminal):
that closes the GUI, never a running storm, which keeps going until it finishes. Open GM1 again to see it. Shutting
down, restarting or sleeping the computer stops a running storm, and on Windows so does `wsl --shutdown` (continue it
later with Run it longer or Resume). For long storms, set the computer not to sleep.

## Running storms

**A storm card is greyed.** The card says why in one line. Most often the card needs more free graphics memory than
yours has right now: close other programs that use the graphics card (games, video, another storm), or pick a shorter
time, which makes the storm smaller. On Windows the desktop itself keeps 4-5 GB of the card. `gpuwm-cm1 doctor
--plain` lists, under Presets, every storm that does not fit this computer and why.

**"Supercell, 100 m" is greyed.** It needs about 9.4 GB of free graphics memory at its smallest size (a 46 km
area), so on a smaller card, or one a desktop is using, the card says how much it needs and how much is free. Earlier versions
greyed it on every computer: with its 0.6 s time step the run stopped after six steps, at its check of where the
storm's area is on the ground. Its time step is now 0.5 s, which that check holds exactly, and it is offered
wherever it fits. On a smaller card, "Supercell, 250 m" shows the same storm in fine detail, or use **Customize
this storm** to pick a grid your card has room for.

**Estimate says the time step stops the run with a moving domain.** In Advanced, a namelist whose domain moves
(`imove = 1` with `umove` or `vmove` not zero) needs a time step that is a small odd number over a power of two:
6, 3, 1.5, 0.75, 0.5, 0.375 s. CM1 keeps the place of a moving domain as a running sum, one step at a time, and GM1
checks that sum exactly; with a step such as 0.6 s or 0.4 s the sum rounds in its last digit after a few steps and
the run stops there. The Estimate screen (`gpuwm-cm1 estimate`) names the step at which it would stop and a time
step that works; set `dtl` to it. Guided mode picks such a step by itself. A domain that does not move takes any
time step that divides its output times.

**I typed a grid spacing and it says "Too big for this GPU".** The line under it says how much free graphics memory
that spacing needs on the smallest area a storm can start in and the finest spacing that does fit your card, which
GM1 has priced before naming it. Press the **Use ...** button beside it, or close other programs that use the
graphics card. Thinner height levels need memory too: with levels every 100 m the same card holds an area less than
half as wide as with levels every 500 m.

**It says the storm leaves the largest area that fits.** A storm that runs on a sounding file (a famous day, the
cloud from one bubble) is not started in an area it would leave before the end. When the largest area your card or
your time allows at that spacing is too small for the storm, the line says so, says the area the storm needs, and
names a coarser spacing that covers it, with a button to use it. A shorter storm time also helps.

**Under "Good to know before you start" it says the storm may leave an area this small.** The run fits your card
only in an area smaller than the one the storm was set up for, or than the room GM1's storm-track check gives it.
It still runs, but once the storm reaches an edge, what happens near the sides is the sides' doing more than the
storm's. The note stays even when the view moves with the storm, because where the storm goes is a prediction: on
an RTX 4090 a 67 m storm left a 34 km area within 45 minutes with the view moving as GM1 predicted.
**Keep this area and follow the storm** moves the view as GM1 predicts the storm moves (the change is listed under
**What you changed**). When the area is your own choice, the note names the smallest larger area without it, with
a button to use it, when one fits your card. Or pick a coarser grid or a longer time so a larger area fits, or go
ahead and read the pictures near the edges with that in mind.

**Levels thinner than about 78 m are refused.** The model is built for at most 256 levels, and the storms' 20 km
top divided by 256 is 78 m. The message names the thinnest levels that fit. Levels thicker than about 2.7 km are
refused too: the model needs at least 8 of them.

**The squall lines are greyed.** They are not offered yet on any computer: nobody has run them on a GPU as they
are set up here. Each card also says which of its settings have not yet been compared with CM1 (the sides of the
area that repeat, for one); those settings run, and no longer hold a storm back by themselves. The hurricanes say
the same and are offered: they have been run.

**My hurricane looks ragged for the first day.** That is what this storm does. It starts from a weak, wide swirl
with no rain. During the first day scattered rain cells form around it and the swirl weakens and loses its shape;
it then tightens, and an eye usually appears on the second day: GM1's run of the 8 km card and CM1 r21.1's run of
the same changed settings (CM1 on its own adaptive time step) both have a clear eye and eyewall by hour 48
(measured). Watching and Results say so beside each early picture. If there is still no eye after three days, look
at the wind and the pressure at the lowest level: a storm that is slowly deepening is only late. At 8 km the
eyewall is a few grid cells wide, so it is drawn wider and weaker than a finer grid would show.

<!-- hurricane-runs:tightening -->
In both runs the radius of maximum wind is 20 km or less from hour 21: in GM1's run to the end (hour 96), in CM1's
to hour 90, then 28 km (both measured, from the 3-hourly frames).
<!-- /hurricane-runs -->

**How close is GM1's hurricane to CM1's?** What was compared, and what the two runs did:

<!-- hurricane-runs:card_8km_measured -->
GM1's run of this preset on an RTX 4090 with ArWen 2.8.9 (gpuwm 2.8.9), on a fixed 7.5 s time step, is compared with
a CM1 r21.1 run with the same changed settings but its own adaptive time step (from 10 s, as CM1's file has it).
Through hour 54 the two stay within about 6 hPa and 7 m/s ground-relative at the lowest level. GM1 reaches 951.2 hPa
at hour 90, with its fastest wind, 62.9 m/s ground-relative, at hour 93; CM1 reaches 954.9 hPa and
62.0 m/s ground-relative at hour 81. GM1's storm holds near its peak after it: 954.0 hPa and
58.8 m/s ground-relative at hour 96; CM1's storm weakens after it: 971.7 hPa and 50.7 m/s ground-relative at hour 96
(all measured).
<!-- /hurricane-runs -->

<!-- hurricane-runs:hours_81_to_96 -->
From hour 81 to hour 96 GM1's storm goes from 954.6 hPa and 61.4 m/s ground-relative to 954.0 hPa and
58.8 m/s ground-relative (in those hours its lowest pressure is 951.2 hPa, its fastest wind
62.9 m/s ground-relative, its radius of maximum wind 20 km throughout); CM1's goes from 954.9 hPa and
62.0 m/s ground-relative to 971.7 hPa and 50.7 m/s ground-relative (radius of maximum wind 20 to 28 km). So GM1's
storm holds its strength through the last 15 hours while CM1's weakens (measured).
<!-- /hurricane-runs -->

<!-- hurricane-runs:guide_8km -->
| Hour | GM1, ArWen 2.8.9 (gpuwm 2.8.9), fixed 7.5 s step: hPa, m/s ground-relative, radius of maximum wind km, warm core K at its height km | CM1 r21.1, adaptive step from 10 s: hPa, m/s ground-relative, radius of maximum wind km, warm core K at its height km |
|---|---|---|
| 0 | 1006.0, 14.9, 84, 1.7 at 0.0 | 1006.0, 14.9, 84, 1.7 at 0.0 |
| 12 | 1006.5, 13.2, 92, 5.3 at 15.3 | 1007.0, 14.6, 132, 5.6 at 16.2 |
| 24 | 1005.6, 15.8, 12, 4.0 at 10.3 | 1004.8, 17.7, 12, 4.0 at 10.3 |
| 36 | 996.1, 29.0, 20, 4.6 at 8.2 | 999.1, 24.3, 20, 4.3 at 14.8 |
| 48 | 979.0, 47.0, 20, 7.8 at 8.8 | 983.9, 41.1, 20, 6.0 at 9.2 |
| 60 | 975.0, 51.0, 20, 9.5 at 7.8 | 970.1, 51.8, 20, 9.3 at 8.8 |
| 69 | 961.9, 59.6, 20, 10.0 at 7.8 | 965.4, 56.3, 20, 9.6 at 8.2 |
| 72 | 962.7, 55.8, 20, 10.0 at 7.8 | 963.4, 55.5, 20, 9.8 at 8.2 |
| 81 | 954.6, 61.4, 20, 10.7 at 15.3 | 954.9, 62.0, 20, 10.8 at 13.3 |
| 84 | 952.3, 62.6, 20, 11.0 at 13.3 | 956.5, 60.8, 20, 12.3 at 15.3 |
| 87 | 951.6, 61.8, 20, 13.9 at 15.3 | 958.2, 59.6, 20, 11.6 at 15.3 |
| 90 | 951.2, 61.8, 20, 13.6 at 15.3 | 961.0, 57.5, 20, 12.3 at 15.3 |
| 93 | 952.0, 62.9, 20, 15.1 at 15.3 | 968.6, 49.4, 28, 10.3 at 15.8 |
| 96 | 954.0, 58.8, 20, 16.5 at 15.3 | 971.7, 50.7, 28, 8.6 at 13.8 |
<!-- /hurricane-runs -->

<!-- hurricane-runs:warm_core -->
Warm core: how much warmer the centre is than its surroundings, as potential temperature: the mean within 40 km of
the centre (the mean of the averages around each ring one grid cell wide) minus the average of the 300 to 450 km
ring, at the height where that difference is largest, and that height. The height is not the same all through a run:
from hour 39 in GM1's run it is 7.8 to 8.8 km up through hour 78 and 13.3 to 15.3 km up from hour 81, and in CM1's
run it is 8.2 to 9.2 km up through hour 72 and 11.8 to 15.8 km up from hour 75. A number from the middle of the
storm and one from near its top are two different layers, so these numbers are not one curve: read a rise or a fall
only between hours at about the same height.
<!-- /hurricane-runs -->

**My own hurricane namelist stopped early, or its import report lists the time step.** CM1's hurricane file
lets the time step change with the storm (`adapt_dt = 1`). The import report says so under "Runs differently";
the hurricane cards use a fixed 7.5 s step (`adapt_dt = 0`, `dtl = 7.5`), and so can you. In guided mode
`--set adapt_dt=1` on a hurricane card is refused with the reason, and Advanced can still run it.

<!-- hurricane-runs:known_issue_adaptive_step -->
With CM1's own adaptive setting GM1's hurricane stops early or makes a much weaker storm, a known issue: 8 km runs
on it stopped at hour 20.8 and at hour 29.1, and the one that ran to the end reached 995 hPa and
35 m/s ground-relative where CM1 reaches 955 hPa and 62 m/s ground-relative (measured with ArWen 2.8.0 (gpuwm
2.8.0)).
<!-- /hurricane-runs -->

**The pictures are blank at first.** A storm takes 20 to 30 minutes of storm time to grow from its first bubble of
rising air, and the first 10 to 60 seconds of a run are spent preparing the graphics card. Until then there is
little to see: keep watching.

**A storm failed with an out-of-memory message.** GM1 fits a storm to the graphics memory free when you press Start,
and does not check whether another program takes some of it later. Close the other programs that use the card and
start the storm again.

**"Waiting for the GPU".** Another storm is running on the card. This one starts when it ends.

**My storm stopped overnight.** The computer slept or restarted for updates, or (Windows) `wsl --shutdown` was run.
Open GM1, open the storm and continue it with Run it longer. For long storms, set the computer not to sleep
(Windows: Settings, System, Power, Sleep: Never while plugged in).

**The pictures stop at T+0:35:00 although the storm ran 2 hours.** On a fast graphics card the model finishes before
the computer's processor has drawn every picture. Watching and Results say how many are still being drawn; they
appear by themselves within a minute or two.

**"An MP4 needs the optional film add-on".** The installers add it; paste the installer line again to put it back.
With pip: `pip install "gpuwm-cm1[film]"`. The animated picture works without it.

## Where things are

**Your storms** are in the folder `gm1-runs` in your home folder (`~/gm1-runs`), unless you started the GUI with
`gpuwm-cm1 gui --root FOLDER`. **Runs** in GM1 says where that is and has an **Open folder** button; a storm's
Results page has one for that storm. On Windows they are inside Ubuntu, and GM1 names the folder as File Explorer
does, `\\wsl.localhost\Ubuntu-24.04\home\YOUR-NAME\gm1-runs`: paste that into File Explorer's address bar, or
under **Linux** in its left column open Ubuntu, then `home`, your user name and `gm1-runs`. The GM1 window prints the
same name when it starts, and so does `gpuwm-cm1 doctor --plain`.

**The disk is filling up.** **Runs** lists each storm's size and the total. **Delete** removes a storm; **Keep
pictures only** removes its big files and keeps the pictures. On Windows the storms are on Ubuntu's own disk, a file
on a Windows drive: the Disk line of `gpuwm-cm1 doctor --plain` (and **System** in GM1) gives the free space of both,
and count on the smaller. Windows does not hand space back to that drive by itself when storms are
deleted; the space is free for new storms at once.

**GM1 itself** is in `~/.local/share/uv/tools/gpuwm-cm1` (installer) or in the environment you made (pip). Its
downloaded packages are in `~/.cache/uv` (installer) or `~/.cache/pip` (pip).

## Update and uninstall

**Update**: paste the installer line again, on Windows in PowerShell, on Linux in a terminal. With pip:
`pip install --upgrade gpuwm-cm1`.

**Uninstall** on Windows, in PowerShell (WSL and Ubuntu stay installed; it also removes the GM1 shortcuts):

```powershell install
& ([scriptblock]::Create((irm https://raw.githubusercontent.com/FahrenheitResearch/gm1/main/install.ps1))) -Uninstall
```

On Linux:

```sh install
curl -LsSf https://raw.githubusercontent.com/FahrenheitResearch/gm1/main/install.sh | sh -s -- --uninstall
```

Uninstalling also removes the 2.5 GB of downloaded packages. Your storms in `~/gm1-runs` are kept; delete the folder
when you no longer want them. On Windows, to remove Ubuntu too, and everything in it including your storms, type in
PowerShell: `wsl --unregister Ubuntu`. With pip: `pip uninstall gpuwm-cm1`, then `pip cache purge` frees pip's 1.4 GB
download cache.

## Asking for help

If none of the above fixes it, open an issue at <https://github.com/FahrenheitResearch/gm1/issues/new/choose> with
the doctor report:

- **On Windows**, open **GM1 check** from the Start menu. It saves `gm1-report.txt` on your desktop and copies it:
  paste it into the issue, or drag the file into it.
- **On Linux**, paste what this prints:

```sh
gpuwm-cm1 doctor --report
```

It is one block with GM1's and its parts' versions, your system, the graphics card, every check and the newest
storm's state and error. It names no user and no computer: your home folder reads `~`. Say what you did and what
happened; a screenshot helps. If `gpuwm-cm1` itself is not found, paste the installer's last lines instead.
