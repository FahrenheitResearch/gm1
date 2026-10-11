# GM1-ARWEN-CHAIN: a GM1 Cartesian domain driven by an ArWen history

**Status: implemented, host-tested, and run on one GPU (an RTX 3080, sections 8
and 9): a steady hold, a 200 m box driven by an idealized ArWen (gpuwm) supercell
history, a 200 m box driven by GM1's own 1 km run, and a 1 km box driven
by a real-weather ArWen history: the Sierra atmospheric-river run of
2025-12-25, offshore, 01-03 UTC.** It reads WRF's and ArWen's own histories
(`wrf_history`: classic and netCDF-4, one or many records per file) and is
packaged as `gpuwm-cm1 chain` and the GUI's "Downscale a WRF or ArWen run (3D)"
screen. The real-weather box is admitted by the same limits as the others through two
lanes: GM1's spacing is the history's true spacing over the box (the
map scale), and a history whose microphysics GM1 does not run gives its water
vapour alone (WRF's `have_bcs_moist = .false.`). Users read
`docs/WRF-DOWNSCALING.md`.

The chain is WRF's `ndown` expressed in GM1 terms. A standalone, finer GM1
Cartesian domain takes its initial state from a saved ArWen (GPU WRF) history
frame. Its lateral boundaries are specified and relaxed toward a
**time-varying** state read from the following frames, interpolated linearly
in time. It is the live nest (`CM1-NEST-SEAM.md` §5) with an offline parent.

## 1. Authorities, each for what it decides

| question | authority | where |
|---|---|---|
| GM1 dynamics | CM1 r21.1, unchanged | every GM1 operator; base state from `src/base.F:495-607` (isnd=7) |
| where forcing enters the step | WRF `solve_em.F` placement, as the live nest has it | `CM1-NEST-SEAM.md` §5.2, `nest_live.WRF_PLACEMENT` |
| boundary value at an interval start | WRF specified lane: the **driving** state (not the domain's own) | `main/ndown_em.F:846-848` `stuff_bdy` |
| boundary tendency | `(next - this)/interval_seconds` | `main/ndown_em.F:321,969-972`; `share/module_bc.F:2938` |
| relaxation target | `bdy + dtbc*bdy_tend` | `share/module_bc.F:1300-1320` |
| clock | `dtbc = 0` when the boundary record is read; `dtbc += dt` before each step | `share/input_wrf.F:1833-1835`; `dyn_em/solve_em.F:377` |
| Davies weights | `lbc_fcx_gcx` specified branch with `spec_exp = 0`, which gives the nested branch's numbers | `dyn_em/module_bc_em.F:1297-1341` |
| vertical transfer | CM1 `interp_on_restart`, per column, on each ArWen column's own heights above ground | `src/restart_read.F:564-609, 3302-3470`; ghosts `src/param.F:7952-7953, 8165-8173` |
| order of transfer | vertical on a parent-resolution window, then horizontal | `external/RSL_LITE/force_domain_em_part2.F:126,141-147` |
| horizontal transfer | the live nest's limited, block-mean-preserving capture on each C-grid stagger | `nest_downscale.downscale_level_staggered` (`CM1-NEST-SEAM.md` §5.1) |
| heights | `(PH+PHB)/g - HGT` on w levels; a mass level is the mean of its two faces | WRF `g = 9.81` (`share/module_model_constants.F:17`) |
| Coriolis | CM1 applies `fcor` to the full wind (`src/solve2.F:633-650`); `f = 2·EOMEG·sin(lat)` | `module_model_constants.F:83` |
| ArWen's `T` | `total_theta - 300` | `gpuwm/io/wrfout.py:822`, `gpuwm/core/state.py:958-961` |
| end-of-step reset of the specified rows | `spec_bdy_final`, specified lane: u, v, t, ph, mu, and moist only for `P_QV` or with `have_bcs_moist` (not w) | WRF v4.6.1 `dyn_em/solve_em.F:4531-4716`, `share/module_bc.F:2066-2216` |
| a history's hydrometeors when GM1 does not run its scheme | WRF's default real-data moist lane, `have_bcs_moist = .false.`: qv relaxed, specified and reset; every other moist species `flow_dep_bdy` after each RK stage (zero inflow, zero-gradient outflow), no relaxation | WRF v4.6.1 `Registry/Registry.EM_COMMON:2945`; `dyn_em/solve_em.F:2256-2258, 2337-2349, 4622-4623`; `share/module_bc.F:2335-2456` |
| length of a map step | a `DX` step is `DX/m` on the earth (`m` = map distance / earth distance) | WRF's metric terms, e.g. the `msf*` factors `spec_bdy_final` itself takes (`share/module_bc.F:2066`) |

CM1 r21.1 has no lateral boundary forcing (`nest_boundary` module docstring), so
the chain is a declared extension whose boundary authority is WRF.

## 2. Decisions that are GM1 choices (recorded in every receipt, `arwen_chain.GM1_CHOICES`)

* **Coordinates and scope.** GM1 is flat and has no map-factor terms, so the
  chain admits a window only where that description is honest:
  * terrain relief under `chain.terrain_relief_limit_m`;
  * GM1's spacing the history's true spacing over the window,
    `DX/(ratio*m)` with `m` the window-mean `MAPFAC_M` (`m = 1`, and so
    `DX/ratio` exactly, for `MAP_PROJ = 0`), and the map factor's variation
    over the window, `max |MAPFAC_M/m - 1|`, under
    `chain.map_factor_departure_limit`. The limit is unchanged in value and in
    meaning, GM1's horizontal length error: with a spacing of `DX/ratio` the
    error would be `|MAPFAC_M - 1|`; with the window's own scale the part of the
    error that is the same over the window is removed and the check measures
    what remains;
  * `MAP_PROJ` 1, 2 or 3 with `DX = DY`, or `MAP_PROJ` 0: an idealized
    history (WRF's `ideal.exe` cases, gpuwm's own idealized runs, GM1 frames
    from `chain frames-from-gm1`) is flat, unmapped and doubly Cartesian
    already, so it is GM1's own geometry. Its Coriolis
    parameter is its `F` field, else `2·EOMEG·sin(XLAT)`, else none;
  * an odd ratio of 3 or more; GM1's `dx` the binary32 word of `DX/(ratio*m)`
    (exactly `DX/ratio`, and exact in binary32, when `m = 1`);
  * the window at least 3 ArWen cells inside ArWen's domain.

  GM1's axes are ArWen's grid axes, and U/V are ArWen's grid-relative components,
  copied without rotation. The projection, true latitudes, relief and map factor
  are recorded.
* **Vertical.** Heights are measured above ground. Below ArWen's first scalar
  level CM1 extends the column by a constant, which is CM1's own rule; the count
  of GM1 levels affected is recorded. GM1's highest scalar level must be at or
  under the history's last mass level, and GM1's top face under the history's
  top face, in every column the transfer reads (the window and one history cell
  around it); `chain new` lowers GM1's top to fit. Above the last level CM1's
  constant extension would hold the Exner function constant with height, out
  of hydrostatic balance by `g dz / (cp θ)` per metre: the first device run that
  let GM1's top sit 1 km above the history's last level gave a pressure burst
  15 times the driving run's (section 8.2).
* **Thermodynamics.** The chain remaps full θ = T+300 and full Exner
  π = ((P+PB)/p00)^(rd/cp), using CM1's constants. GM1's own `th0`/`pi0` are
  subtracted afterwards on the device's words. No hydrostatic rebalance is
  applied, as CM1's `interp_on_restart` applies none; the domain-mean π′ is
  measured and recorded.
* **Base state.** CM1's isnd=7 sounding of the t0 domain-mean column, on GM1's
  scalar levels. The surface record is the mean PSFC, TH2 (or T2 with PSFC) and
  Q2.
* **Moisture** (`chain.moisture`). Two lanes, both WRF's:
  * `all-species` (the default, absent from the document): species are matched
    by name to GM1's scheme (`qv→QVAPOR … nci→QNICE …`). The history's
    `MP_PHYSICS` must be one GM1 runs the way CM1 does: 6 (WSM6), 8 (Thompson)
    or 10 (Morrison). Any other is refused with the scheme named (`WSM3 (3)`,
    `P3 (50)`, `NSSL 2-moment (17)` …), because its species would be poured
    into another scheme's categories; the refusal names the vapour-only lane.
    This is WRF with `have_bcs_moist = .true.` (or a nest).
  * `vapour-only`: WRF's own default real-data lane, `have_bcs_moist =
    .false.`. Water vapour is the only moist field taken: its initial state,
    its relaxation and specified rows, and its end-of-step reset. Every other
    species of GM1's scheme starts at zero (as `real.exe` from an analysis
    without hydrometeors does) and, after each RK scalar stage, takes WRF's
    `flow_dep_bdy` on the specified rows: zero where the normal velocity at the
    row's outer face flows in, the first interior row's value where it flows
    out (GM1 reads the sign of the stage's `rru`/`rrv`, CM1's time-averaged
    mass flux; WRF reads `ru_m`/`rv_m`). No relaxation. Any history scheme is
    admitted; its hydrometeors are measured at t0 (column paths over the
    window) and recorded as left out. This is the honest path for P3, which
    GM1 does not run (`column_component_blockers.py` `p3-mp50`): P3's one free
    ice category (mass, number, rime mass and rime volume) has no counterpart
    among Morrison's, WSM6's or Thompson's cloud ice, snow and graupel, so
    mapping it would be a choice with no authority. What it costs is spin-up:
    the history's clouds and rain are not in the box at t0 and do not enter
    through its sides, and GM1 makes its own (section 9).
  `USE_THETA_M = 1` requires `THM` beside `T`, checked consistent with a dry
  `T`.
* **Surface record.** `PSFC`, `TH2` (or `T2` with `PSFC`) and `Q2` when the
  history has them; otherwise WRF's own `phy_prep` extrapolation of `P+PB` to
  the ground and the lowest level's θ and qv (CM1's constant extension). The
  source of each is recorded.
* **Ground between saved times.** `HGT` may differ between saved times by float
  rounding only (1 cm, `arwen_chain.HGT_TOLERANCE_M`). Stock WRF 3.7 rewrote
  `HGT` with 2.5 mm differences between records of one file; its first record's
  ground differed by up to 5.2 m in 25 cells, which is refused (start at the
  next saved time).
* **TKE** is not transferred. CM1's restart-grid path does not read `tkea`
  (`restart_read.F:1207`), and WRF's TKE has no boundary call
  (`solve_em.F:2378-2416`).
* **Time.** Frames must be at one cadence, and that cadence must divide the GM1
  step and be no longer than the GM1 domain's crossing time at the fastest
  ArWen wind in the window. There is no extrapolation past the last frame.
* **Frame.** A translating or adaptive GM1 frame is refused.
* **Mass.** `apmasscon` does not run: the domain takes its mass through the
  specified zone, as a nest child does.
* **End of step.** After microphysics and before `solve3` copies the new state
  into the time-level-n arrays, the specified rows are set to
  `value + dtbc*tend`: WRF's `spec_bdy_final`, which WRF runs at the end of
  `solve_em` (WRF v4.6.1 `dyn_em/solve_em.F:4531-4716`; ArWen
  `ingest/lateral_bc.py` `apply_state_boundary_values`;
  `arwen_chain_runtime.SPEC_BDY_FINAL`). The chain is WRF's specified lane:
  u, v, π′ (for WRF's t, ph and mu), θ and each transferred species; not w,
  which WRF resets only in a nest (`:4566`). The reset is the live
  nest's own `NestLateralForcing.finalize_specified` (one kernel,
  `gm1_nest_spec_apply` modes 2 and 3), which the live nest now runs too, in
  WRF's nested form (every field, `CM1-NEST-SEAM.md` §5.2 and §5.12); an earlier
  version reset π′, θ and the species only, and u and v stayed on their rows
  to float accumulation (8e-4 m/s RMS, 8.6). A species target is held at zero
  or above (GM1's choice: the reset follows `pdefq`, and the product rounds a
  few 1e-12 below zero where both saved times hold none). Without the reset,
  microphysics acted on the specified rows every step and nothing returned
  them: 8.2's box on the newer toolchain measured the outer row 6.4 K (θ) and
  14 g/kg (rain) from the driving run after 30 min (8.6).
* **Upper damping.** CM1's Rayleigh damping, if selected, relaxes toward the t0
  base state for the whole run. This is a named limit, not a transcription.

## 3. What was built

| piece | file |
|---|---|
| classic netCDF (CDF-1/CDF-2, ArWen's default writer) record decoder → LE f32 + `frame.json`; refuses netCDF-4/HDF5 (remedy `nccopy -k nc6`) and CDF-5 | `rust/cm1-state/src/arwen_history.rs`, bin `gm1-arwen-history-extract` |
| frame reader, admission (every refusal below), per-column CM1 vertical rule, horizontal capture, sounding, prepared products, WRF table pair | `src/cm1core/arwen_chain.py` |
| device: `ChainLateralForcing` (the live nest's forcing with WRF offline tables), `ChainDriver` (initial state, tables at boundary times, `dtbc`), runtime class skipping `apmasscon` | `src/cm1core/arwen_chain_runtime.py` |
| `[chain]` table (whole or absent; absent is not projected, so all 62 example hashes are unchanged) | `src/cm1core/config.py` |
| door: host refusals, prepared-chain + sounding binding before the device opens, runtime class, attach after the initializer, `arwen_chain` receipt | `src/cm1core/cartesian_door.py` |
| prepare tool | `tools/arwen_chain_prepare.py` |
| one reader of WRF and ArWen histories: CDF-1/-2/-5 and netCDF-4, many records per file, ArWen told from WRF, frames with `ArwenFrame`'s interface, the Rust extractor's frame format word for word | `src/cm1core/wrf_history.py` |
| admission as named checks (`admission_report` runs every one it can and says which it could not reach; `admit_frames` raises the first as before) | `src/cm1core/arwen_chain.py` |
| the packaged command: `gpuwm-cm1 chain inspect \| check \| new \| frames-from-gm1`; `chain new` writes an ordinary run folder (`gm1.cm1-run/v1`, the namelist imported as any namelist is, `[chain]` in the deck) that `gpuwm-cm1 run` runs | `src/cm1core/chain_door.py` |
| a chained run writes no restart and keeps `[chain]` through a re-import; the storm-track admission does not apply to it | `src/cm1core/cm1run/launch.py`, `src/cm1core/cartesian_door.py` |
| GUI, advanced mode: "Downscale a WRF or ArWen run (3D)" (`POST /api/chain/{open,check,new}`) | `src/cm1core/gui/chain.py`, `gui/static/js/chain/`, `gui/copy/chain.json` |
| the map-scale spacing (`window_map_scale`, `chain_spacing_m`, `box_mass_slice`; the spacing and map-factor checks) | `src/cm1core/arwen_chain.py`, `src/cm1core/chain_door.py` |
| `chain.moisture` (`all-species` / `vapour-only`), the left-out condensate record, `flow_dep_bdy` for the species the history does not drive; `--moisture` on `chain check/new`; Moisture and GM1 microphysics on the screen | `src/cm1core/config.py`, `arwen_chain.py`, `arwen_chain_runtime.py`, `chain_door.py`, `gui/chain.py`, `gui/static/js/chain/screen.js` |
| `spec_bdy_final` as one kernel for the live nest and the chain (`gm1_nest_spec_apply` modes 2 and 3; `NestLateralForcing.finalize_specified`, called by the child runtime before `solve3`) | `src/cm1core/nest_live_kernels.py`, `src/cm1core/nest_live.py` |

### Refusals (each names its wrong result in its message)

| group | codes |
|---|---|
| frames | `GM1-CHAIN-FRAME`, `-FRAME-COUNT`, `-FRAME-GRID`, `-CADENCE`, `-WINDOW` |
| time | `-INTERVAL-STEP`, `-DURATION`, `-CROSSING` |
| geometry | `-PROJECTION`, `-SPACING`, `-PLACEMENT`, `-TERRAIN`, `-MAP-FACTOR`, `-CORIOLIS` |
| physics and variables | `-MICROPHYSICS`, `-SPECIES`, `-VARIABLE`, `-THETA-M`, `-COLUMN`, `-MODEL-TOP` |
| run binding | `-PREPARED`, `-RESTART`, `-INITIAL-CONDITION`, `-SOUNDING` |
| document | `CM1-CFG-CHAIN-TIME`, `-RATIO`, `-ZONES`, `-NEST`, `-GRID`, `-LATERAL`, `-MOVING-FRAME`, `-ADAPTIVE-STEP`, `-CORIOLIS`, `-INITIALIZATION`, `-DURATION`, `-MOISTURE` |

## 4. Host tests

**`tests/test_arwen_chain.py` (28 tests)**
* The per-column CM1 map and interpolation equal `nest_vertical.build_cm1_vertical_map` and `cm1_vertical_interpolate` raw-word, on random columns.
* A steady uniform state is reproduced to within one float32 ulp. The ulp is the two-plane `do_interp` sum; the horizontal capture of a flat level is exact.
* A steady state's tables have exactly zero tendency, and the Davies reference then adds exactly nothing.
* The boundary target is linear in time to 2e-6 relative. Its value is the driving state, and interval n+1 starts on frame n+1 word for word.
* Frame strips equal the initial state's boundary rows word for word.
* A field linear in height crosses column-varying, stretched ArWen levels to float32 rounding.
* The T, P+PB, species-name and perturbation mappings round-trip.
* The GM1 boundary u equals the ArWen face value, and horizontally linear u is reproduced.
* Every refusal fires.
* A Python-written CDF-2 file goes through the built Rust extractor into the Python reader, raw-word.

**`tests/test_arwen_chain_document.py` (16 tests)**
* `[chain]` parsing, projection and whole-table rules, and each document refusal.
* The door binding with its own sounding, another sounding, and another clock.
* Restart and iconly refusals.
* `ChainDriver` on a NumPy stand-in for CuPy: the t0 state lands in both time levels as perturbations, tables load at steps 0 and 60 equal to `interval_tables`, `dtbc` restarts at each boundary time and counts each step's end, and stepping past the last frame is refused.

**`tests/test_arwen_chain_wrf.py`**: WRF-shaped netCDF-4 histories
(three records, `USE_THETA_M = 1` with `THM`, `d02`) admitted and prepared for
WSM6, Thompson and Morrison; every classic container; `MAP_PROJ = 0` as GM1's
geometry, with `F` or without; WSM3, P3 and NSSL refused by name; WRF and ISO
time spellings; frames selected by time across records and files, and two
domains' files refused; the surface fallback; the report going past a refusal;
GM1's top under the history's last level; a record read from the file preparing
what its extracted frame prepares, word for word.

**`tests/test_chain_door.py`**: `chain inspect`, `check` (exit 0 / 3,
every check in plain words), `new` (an ordinary run folder, the typed line
echoed, a second `new` refused) and `frames-from-gm1` (a history the chain
admits and prepares), on the command line.

**`tests/test_gui_chain.py`**: the screen's three POSTs walk the API
table (dry run = same line, no file changed; `chain new` replays from a shell
to the same folder), a refused box in plain words, a made run started and its
Results served, the verification label pinned.

**`tests/test_wrf_history.py`**: the reader; the Rust extractor's frame
format equal word for word on synthetic files and, with `GM1_WRF_REAL_FILES`,
on the real Sierra ArWen, Katrina WRF 3.7 and WK82 ArWen files.

**Device record:** the RTX 3080 runs of section 8 are held to their criteria,
and on a GPU the steady hold is run again (kept with the measurement records,
not shipped).

**Map scale, moisture and the end-of-step reset** (`tests/test_arwen_chain.py`, `tests/test_arwen_chain_document.py`,
`tests/test_nest_spec_bdy_final.py`): a map factor the same over the window is
admitted with GM1's spacing `DX/(ratio*m)` and nothing left to measure, and
refused at the spacing check with `DX/ratio`; one that varies over the window
by more than the limit is refused; a P3 history is refused in the all-species
lane with a remedy naming vapour-only, and admitted in the vapour-only lane
with its left-out hydrometeor paths in the ratio the frames hold; the prepared
chain carries qv alone; the driver's tables are u, v, w, π′, θ and qv, the
hydrometeors start at zero, and `flow_dep_bdy` gives zero on inflow and the
first interior row on outflow on all four sides (a level with each flow
reversed); `chain.moisture` is optional, not projected when absent, and
refused by value; the specified-lane reset covers u, v, π′, θ and the species
and not w; the live nest's reset covers every carrier (w included), holds a
species at zero, touches nothing inside the rows or in the time-level-n
arrays, and runs once per step just before `solve3`; the kernel's modes 2 and
3 equal their CPU reference word for word (g++, every lattice, side, spec
zone 1 and 2, linear and shuffled thread order) and, with
`GM1_NEST_DEVICE_TESTS=1`, on a device.

**Rust:** `arwen_history` unit tests (CDF-1 and CDF-2 record/static decode; HDF5, CDF-5, missing record and existing output refused) and the binary's usage test.

## 5. First real case: worked plan

**Before the map-scale and vapour-only lanes, no real-weather history at hand passed admission**
(read-only, a box over the highest ground and a box over water in each):

| history | what refused it then |
|---|---|
| Sierra ArWen, 2025-12-25 00-03Z (3 km Lambert, 300 × 270 × 49, CDF-2, `MP_PHYSICS` 50) | P3 is not a GM1 scheme; the map scale is 1.28-1.52 % from 1 everywhere in the domain (true latitudes 29.1 and 49.1, the domain 35-43 N), over the 1 % limit; saved hourly, so a small box's crossing time (15-23 min) is shorter than the saving interval; over the Sierra, 3623 m of relief. Over water the terrain check passes. |
| stock WRF 3.7, 2005-08-28 00-09Z (30 km Mercator, 90 × 73 × 29, CDF-1, `MP_PHYSICS` 3) | WSM3; Mercator with true latitude 0 puts the map scale 5.8-24 % from 1; saved every 3 h against a 2 h crossing; its first record's ground differs by 5.2 m (from the second saved time on, that check passes). Over the Gulf the terrain check passes. |

**With those lanes a Sierra box is admitted, with every limit unchanged.** Two of the
Sierra refusals were not limits on the air but on what GM1 could express, and
the lanes give GM1 the expression instead of loosening the check:

* **Map scale.** The 1 % limit bounds GM1's horizontal length error. With a
  spacing of `DX/ratio` GM1's error is `|MAPFAC_M - 1|`, and that is what
  the check would measure. The Sierra domain lies wholly between its true latitudes,
  so `MAPFAC_M` is 0.9848-0.9872 everywhere (35.3-42.8 N): of 271,210,275 boxes (every size
  from 12 to 200 cells, every position), **none** is within 1 % of 1
  (`unit_map = 0`; smallest departure 1.33 %). No box anywhere could have passed; moving the box nearer a
  true latitude is impossible inside this domain. GM1 takes the history's true
  spacing over the box, `DX/(ratio*m)` with `m` the box-mean map factor (the
  length a `DX` step has on the earth, WRF's metric), so the part of the error
  that is the same over the box is gone, and the check measures what remains,
  `max |MAPFAC_M/m - 1|`, against the same 1 %. For the admitted box it is
  0.07 % (GM1 cells 1014.79 m, map scale 0.98542).
* **P3.** GM1 does not run P3 (`column_component_blockers.py` `p3-mp50`), and
  the all-species lane still refuses it by name. GM1 adds WRF's own real-data
  moist lane, `have_bcs_moist = .false.` (§2, Moisture): water vapour alone is
  taken and driven; clouds and rain start at zero and form in GM1. Nothing of
  P3 is mapped onto another scheme.

The crossing, terrain, placement, model-top, Coriolis and every other check are
unchanged. The search (a window survey and a land survey,
ratio 3, the hourly saves, the fastest `|U|`, `|V|`
of any level at 00-03Z; the arithmetic of the checks on the box's own cells,
where the checks also read the ring of cells around the box, so the survey only
finds candidates and `chain check` decides):

| region | boxes | what the unchanged checks say |
|---|---|---|
| everywhere | 271,210,275 | terrain ≤ 50 m: 6,201,215; crossing ≥ 60 min: 103,215,080; both, and the map scale's variation ≤ 1 %: **847,050**, all over the ocean |
| the Central Valley (land-majority boxes, sizes 12-200 step 4) | 62,351,579 | 2,540 are flat enough (the largest 16 × 28 cells, near Sacramento, 48 m of relief), but the fastest wind crosses any of them in at most 24.4 min against the hourly saves; every land box large enough to be crossed in 60 min has at least 705 m of relief (the foothills). **None passes**: the Valley is too narrow for an hourly history. `chain check` on that largest flat box (`--window 105,104,16,28`) refuses it on the crossing (25.6 min) and on the ground (96 m, counting the ring of cells around the box that the transfer reads) |
| offshore Pacific | smallest 22 × 52 cells | passes; the run below took a larger box, 36 × 66 cells, cells 6-41 east and 40-105 north (36.4-38.2 N, 125.3-124.0 W, `LANDMASK` 0 everywhere), whose fastest wind crosses it in 68 min |

`gpuwm-cm1 chain check --history .../wrfout --window 6,40,36,66 --ratio 3
--start 2025-12-25_01:00:00 --scheme morrison-mp10 --moisture vapour-only`
admits it on every check (without `--moisture vapour-only` it is refused
on the microphysics alone);
the run is §9.2. A storm-scale real case (5 min histories, a scheme GM1 runs)
still needs a history made for it, as below.

### Step A: flat-terrain smoke

* Case: WRF 4.6.1 inputs from GFS, 2026-07-20 00Z. d02 is 3 km, 240×160×49, `MP_PHYSICS` 6, and `HGT = 0`
  everywhere, so it is flat by construction. `wrfinput_d01/d02`, `wrfbdy_d01`
  and `namelist.input` are present.
* ArWen run: d02 with `history_interval` 300 s for 2 h, a history selection that
  carries P/PB, PH/PHB, U/V/W, T, every WSM6 species, MAPFAC_M, XLAT, PSFC, TH2
  and Q2 (ArWen's `severe` preset plus the `minimal` metadata), and the Rust
  classic writer.
* GM1 window: ratio 5 → 600 m. `parent_i0 = 100`, `parent_j0 = 60`, span 30×30,
  so the GM1 grid is 150×150 cells (90 km). `dt_s = 3.0` divides 300 s. Set
  `terrain_relief_limit_m = 0.0` and `fcor_s_inv` to the value the prepare tool
  prints if it refuses. The test checks mechanism, not tornado scale.

### Step B: first storm-scale real case (Mayfield/Cayce, 2021-12-11)

* Case: an ArWen configuration of the 2021-12-10/11 Mayfield tornado
  (`mp_physics = 6`, which matches GM1 `wsm6-mp6`). The domains are d01
  3 km, d02 1 km (450×450), d03 500 m (300×300) and d04 100 m.
* ArWen: set d03 `history_interval_s = 300` over 04–06Z (the corridor
  crossing), with the history selection above.
* GM1 window: from d03 (500 m), ratio 5 → **100 m**. Take 80×80 d03 cells
  (40 km) centred on the d04 box, so the GM1 grid is 400×400 cells with the
  stretched deck. `dt_s` must divide 300 s. `spec_zone = 1`,
  `relax_zone = 5`.
* Relief and map factor: run the prepare step first. It measures and refuses;
  declare the limits only after reading the measured values (western Kentucky
  and the Mississippi valley are expected to be tens of metres, but that is not
  yet measured).
* Crossing time: 40 km at ~40 m/s is 1000 s, which exceeds 300 s, so the window
  is admitted. At 900 s cadence it would be refused.

## 6. Commands

```bash
# host: what the history is, every admission check in plain words, then the run folder
gpuwm-cm1 chain inspect HISTORY_FILE_OR_FOLDER --map-png map.png
gpuwm-cm1 chain check --history HISTORY --window I0,J0,NI,NJ --ratio 5 [--start T] [--end T] [--every S]
gpuwm-cm1 chain new   --history HISTORY --window I0,J0,NI,NJ --ratio 5 ... --out RUNDIR
# device: an ordinary run
gpuwm-cm1 run RUNDIR
# GM1 driving GM1 (the nest-equivalence and spin-up references): a GM1 run's frames as a history
gpuwm-cm1 chain frames-from-gm1 GM1_RUNDIR --out FRAMES_DIR [--every S]

# the lower-level route of the first version still works: a Cartesian document with [chain]
python3 tools/arwen_chain_prepare.py chain.toml
python -m cm1core cartesian chain.toml --output R/receipt.json --history-dir R/history \
    --integrate --sounding <prepared_directory>/input_sounding

# the Rust extractor stays the oracle of the Python reader (tests/test_wrf_history.py)
cargo build --release --manifest-path rust/cm1-state/Cargo.toml --bin gm1-arwen-history-extract
GM1_WRF_REAL_FILES=a:b:c python3 -m pytest tests/test_wrf_history.py -q
```

The device runs of section 8 were made by scripts kept with the measurement
records (not shipped), each run with a memory budget taken from the free memory
measured just before.

Device verification that should come first, before the real case:

1. **Steady-state hold.** Prepare the synthetic steady frames (`tests/test_arwen_chain.py::make_frame`, two frames 300 s apart), run 60 steps, and require the boundary rows to stay at their t0 words and the interior drift to be only GM1's own adjustment of the base-state/π′ residual (`initialization.domain_mean_pi_perturbation_max_abs` in the receipt).
2. **Chain vs live nest.** Publish a GM1 parent every step as ArWen-format frames (converter still to write, §7), chain a child from them with `interval = parent dt`, and compare it with the live nest driven by the same parent. They differ by construction in exactly one place: WRF's specified lane takes the driving value at the interval start, where the nest takes the child's own. The comparison measures that difference.
3. **Receipt checks.** Tables are loaded at every boundary step (`arwen_chain.intervals_loaded`), and `lateral_forcing.acoustic_small_steps_specified` = steps × nsound.

## 7. What remains for a production chain

1. **Device execution on a real-weather history.** Done on one GPU with
   idealized and GM1-driven histories (§8) and a real-weather ArWen
   history (the Sierra atmospheric river, offshore, vapour only, §9.2). A
   storm-scale real case, with 5 min histories from a scheme GM1 runs, still
   needs a history made for it (§5, Steps A and B).
2. **Rust remap.** The vertical and horizontal remap runs on the host in NumPy, as `tools/nest_downscale_init.py` does. GM1 keeps regridding in Rust, so the production remap should move to `cm1-state` with this NumPy code as its raw-word oracle.
3. **Prepare cost.** Each boundary frame is captured on the full GM1 lattice and then stripped. That is memory- and time-heavy for 400×400×nz×37 frames; capture only the strips' donor windows.
4. **Terrain and map factors.** Chain into GM1's terrain lane with a real terrain mapping. Map factors: GM1 takes the box-mean map scale (the uniform part of the error) and refuses only when the scale varies by more than the limit across the box; map-factor terms inside GM1 (a scale that varies across a large box) are not done.
5. **Translating and adaptive frames.** Sample the window along the frame's ground track, and give winds frame-relative boundary values.
6. **Restart and resume.** Persist the interval position and the prepared identity in the restart, and skip the initial-state write on resume.
7. **Surface and land.** TSK, soil, land use and roughness from ArWen for GM1's surface layer. The document's own surface deck is used today.
8. **Hydrostatic consistency.** Offer a measured rebalance of π′ to CM1's discrete hydrostatics (WRF's `rebalance_driver`, `ndown_em.F:710`). Today the residual is only recorded; §8.3 measures its cost, a first-minute pressure adjustment 2.9 times the settled rate at 6 s sampling, gone within about 70 s.
9. **Upper damping.** A WRF-like w-only damping or a time-varying reference instead of CM1 Rayleigh toward t0.
10. **Container coverage.** Done in Python (`wrf_history`, netCDF4): netCDF-4, CDF-5 and multi-record files. The Rust extractor still reads classic files only and stays the oracle.
11. **TKE and scheme moments.** TKE transfer from an LES parent (`km_opt` 2/3) and number-concentration unit audit per scheme.
12. **A GM1-history → frame converter.** Done: `gpuwm-cm1 chain frames-from-gm1` (§8.3 uses it). The nest-equivalence comparison with the live nest remains.
13. **Terrain and mapped coordinates for users** (the order `docs/WRF-DOWNSCALING.md` gives): (a) a map-factor lane, so boxes on a Lambert or Mercator grid away from the true latitude are admitted with their lengths corrected rather than refused (done for the box-mean scale; a scale varying by more than 1 % across the box is still refused); (b) chaining into GM1's terrain lane, so a box over hills keeps them; (c) the land surface from the history (item 7); (d) a translating box that follows a storm (item 5).

## 8. Device runs on an RTX 3080: a steady hold and two boxes

One GPU: an RTX 3080 (10 GB), each run under 10 minutes with a
memory budget taken from the free memory measured just before it. The scripts,
run records and analyses are kept with the measurement records, not shipped.
The analyses compare child and driver on the driver's cells: the
child block-averaged, the driver taken onto GM1's levels by the chain's own
transfer. "Forced zone" is the outer driver cell of the box (spec_zone 1 +
relax_zone 4 = 5 child cells at ratio 5); "interior" is the rest.

### 8.1 (a) Steady hold

Two synthetic idealized frames 300 s apart holding one state (3 km, WSM6), a
15 × 15-cell box at ratio 3: 45 × 45 × 20 cells of 1 km, 60 steps of 5 s (one
interval), 11.7 s of wall time.

* The specified rows kept their t0 words: 0 words changed in u, v, w, θ and p.
* Interior drift: 5.2e-4 m/s in u, 2.6e-4 in v, 1.4e-4 in w, 0 in θ, 0.078 Pa
  in p; π′ moved by 2.5e-7 against a t0 domain-mean residual of 3.0e-7, so the
  drift is GM1's own adjustment of that residual.
* Tables were loaded at step 0, and 480 acoustic small steps were specified.

### 8.2 (b) A box in an ArWen supercell

gpuwm 2.7.4 ran its own Weisman-Klemp (1982) quarter-circle supercell case on
the 3080, used as it ships except for WSM6 in place of its Kessler scheme:
96 × 96 × 40 at 1 km, dt 6 s, 1 h in 10.5 s of wall time, a wrfout history
every 300 s through gpuwm's own writer (CDF-2, 13 records, `MAP_PROJ = 0`;
max w 47 m/s at 30 min). `gpuwm-cm1 chain inspect | check | new` read it
through `wrf_history` and put a 24 × 24-cell box on the storm at ratio 5:
120 × 120 × 38 cells of 200 m, dt 1.2 s, 00:30 to 01:00. `gpuwm-cm1 run` ran it
in 75.7 s.

Relaxation, RMS of child minus ArWen on ArWen's cells (forced zone / interior):

| t (s) | u (m/s) | θ (K) | w (m/s) | qv (g/kg) |
|---|---|---|---|---|
| 300 | 0.73 / 2.36 | 0.56 / 1.78 | 0.90 / 2.84 | 0.045 / 0.154 |
| 600 | 1.25 / 3.51 | 1.02 / 2.82 | 1.50 / 4.36 | 0.071 / 0.364 |
| 900 | 1.52 / 4.33 | 1.18 / 2.87 | 1.81 / 4.80 | 0.078 / 0.602 |
| 1200 | 1.80 / 5.44 | 1.31 / 2.80 | 2.23 / 5.90 | 0.120 / 0.581 |
| 1500 | 1.96 / 4.87 | 1.58 / 3.10 | 2.52 / 5.06 | 0.179 / 0.596 |
| 1800 | 1.70 / 4.40 | 1.35 / 3.08 | 2.10 / 4.86 | 0.158 / 0.602 |

The forced zone stays 2-5 times closer to the driving run than the interior,
where the 200 m storm goes its own way (max w 50-66 m/s against ArWen's 47-49).

Conservation over 30 min (trapezoid over the 60 s frames): inside the forced
zone, the dry-air mass budget closes to 3.3e-5 of the mass (1 % of the gross
lateral exchange) and the water budget, with 6.8e9 kg of surface
precipitation, to 0.32 % of the water (1.4 % of the precipitation). Over the
whole box the budget does not close, by 0.28 % of the mass and 23 % of the
water: the specified and relaxation rows move the edge toward the driving
state, which puts mass and water in without a flux through the outer face
(WRF `module_bc` does the same).

Spin-up: the pre-registered measure is max |dπ′/dt| on the driver's cells over the
first interval, child against driver, at most 2. It is **3.2: not met as
written.** At 60 s sampling the child's first interval's rate equals its
later ones (1.15e-5 against 1.13e-5 s⁻¹) and its first minute is calmer than
minutes 2-5 (1.8e-5 against 4.8e-5 s⁻¹). At 6 s sampling (the probe of
8.6) the first minute does carry an adjustment: 4.7e-4 s⁻¹ against 1.5e-4 s⁻¹
over 1-5 min on GM1's cells (3.2 times; 2.3 times on the history's cells),
gone within about 70 s. It is the same acoustic adjustment of an interpolated
state that is not rebalanced as in (c) (7.8), and it is shorter than a 60 s
frame, so the 60 s frames do not see it. Its rate is
GM1's own level (GM1's 1 km run in 8.3: 2.2e-5 s⁻¹), while ArWen's 1 km
pressure field changes several times more slowly (3.6e-6 s⁻¹ in the first
interval, 6.8e-6 later). The max w ratio is 1.4.

**A defect this run found and fixed.** An earlier run of this box let GM1's top (20 km)
sit above the history's last mass level in the box (18.9 km). CM1's constant
extension then held the Exner function constant over the top kilometre, and
the child's first-interval pressure rate was **15 times** ArWen's (5.4e-5 s⁻¹;
2.8e-4 s⁻¹ in its first minute). The model-top check now requires GM1's highest
scalar level under the history's last mass level and its top face under the
history's top face, and `chain new` lowers GM1's top to fit (19 km here); the
ratio fell to 3.2 and the first-minute rate to 1.8e-5 s⁻¹.

### 8.3 (c) A box in GM1's own 1 km run

GM1's stock 1 km supercell (Morrison, fixed frame, frames every 60 s) ran
90 min in 55 s; `chain frames-from-gm1` turned its frames into an idealized
history. A 36 × 36-cell box at ratio 5, 180 × 180 × 40 cells of 200 m, dt 1.2 s,
was chained on the storm from 01:05 to 01:30 with saved times every 300 s (93 s
of wall time); a second child ran the first 300 s with frames every 6 s (31 s).
Because the driver's frames exist every 60 s, the child can be compared with the
driver between saved times as well as at them.

Relaxation at the boundary times, RMS on the driver's cells: forced zone
against the driver / interior against the driver (zone against the relaxation
target equals zone against the driver at these times):

| t (s) | u (m/s) | θ (K) |
|---|---|---|
| 300 | 0.77 / 2.02 | 0.51 / 1.39 |
| 600 | 1.06 / 3.23 | 0.64 / 2.24 |
| 900 | 1.31 / 3.73 | 0.93 / 2.51 |
| 1200 | 1.36 / 3.82 | 1.07 / 2.52 |
| 1500 | 1.27 / 3.74 | 1.00 / 2.28 |

Conservation over 25 min inside the forced zone: dry-air mass to 2.8e-4 of the
mass, water to 0.14 % of the water (with 5.2e9 kg of surface precipitation);
over the first 300 s at 6 s sampling, 2.1e-5 and 0.048 %.

Spin-up: max |dπ′/dt| on the driver's cells in the first 5 min at 60 s
sampling, child 3.4e-5 against driver 2.2e-5 s⁻¹: **1.6, met**; max w 63.5
against 57.6 m/s (1.1). At 6 s sampling the child's first minute reaches
1.9e-4 s⁻¹, 2.9 times its own rate at 4-5 min (6.6e-5 s⁻¹), and is gone
within about 70 s: the acoustic adjustment of an interpolated state that is not
rebalanced hydrostatically (7.8). The storm: the child's max w 55-69 m/s
against the driver's 56-60 in the box; its 2-5 km updraft helicity peaks at
3300 m² s⁻² at 8 min against the driver's 700-850, then settles at 900-1700.

### 8.4 Verdicts against the pre-registered criteria

| criterion | verdict |
|---|---|
| (a) steady hold: specified rows at their t0 words, interior drift only GM1's adjustment | met |
| boundary relaxation: forced zone closer to the driver than the interior at every saved time, u, θ, w, qv, in (b) and (c) | met |
| conservation inside the forced zone: mass to 1e-3 of the mass, water to 1 % of the water, in (b) and (c) | met |
| spin-up, child ≤ 2 × driver | (c) met (1.6); (b) **not met (3.2)** on the pre-registered measure (the driver on GM1's heights); 1.7 on ArWen's own moving levels. Both carry a first-minute acoustic adjustment at 6 s sampling, about 3 times their later rate, gone within about 70 s (8.2, 8.3, 8.6) |
| specified rows at WRF's target at the end of every step (`spec_bdy_final`) | met after the fix of 8.6: θ to 5e-5 K, qv to 1e-8; before it, 6.4 K and 2 g/kg off after 30 min |
| `chain new` → `gpuwm-cm1 run` → Results | met: both boxes ran as ordinary runs; the (b) box's cm1out and quick-looks were drawn and shown on the Results screen |

### 8.5 What these runs do not show

* A real-weather history: none at hand passed admission then (§5); §9.2 runs one.
* The live-nest equivalence of §6's first version (chain against live nest
  from the same parent): not run.
* Terrain, map factors, a translating frame, a restart: all refused by design.
* Only WSM6 and Morrison ran on the device; Thompson is admitted on the host
  only.

### 8.6 The same runs on a newer CUDA toolchain, with an analysis written apart from the chain

(a), (b) and (c) were also run on the same RTX 3080 with a newer CUDA toolchain
(CUDA runtime 13.2, NVRTC 13.3, against 12.9 and 12.8 for 8.1-8.3). The
prepared products of (b) are byte-identical across the two toolchains; the
integrated fields differ by about 1e-2 after a minute (another compiler), and two
runs on one toolchain are bit-identical. The analysis was written separately
from the chain's code; it and its numbers are kept with the measurement records,
not shipped.

* **WRF `module_bc` on the device.** The chain's own boundary arena gives WRF's
  target `bdy + dtbc*bdy_tend` at every child frame. Before the fix, row 0
  (the specified row) held u and v to 8e-4 and 2e-4 m/s RMS (float
  accumulation), but θ was 0.28 K RMS and 6.4 K at most from the target, qv
  2 g/kg, rain 14 g/kg, π 1e-3 after 30 min, growing with time: microphysics
  acted on the specified rows and nothing restored them (section 2, "End of
  step"). After the fix, row 0 is at the target to float rounding (θ 7e-6 K
  RMS, 5e-5 K at most; qv 1e-8), and rows 1-3 depart from the target
  monotonically inward in every frame, as WRF's weights 1, 2/3, 1/3 make them
  (u: 0.63, 1.04, 1.58 m/s; θ: 0.46, 0.79, 1.30 K, RMS over frames).
* **(b) after the fix.** Inside the forced rows the dry-air budget closes to
  1.2e-5 of the mass (0.3 % of the gross exchange) and water to 0.39 % of the
  water (1.7 % of the 6.8e9 kg of rain); 66 s of wall time. Spin-up
  unchanged: the 6 s probe's first minute is 3.2 times minutes 1-5 (8.2).
* **(c) after the fix** (8.3's GM1 parent, the child and the 6 s probe
  run again): forced zone against interior at the saved times u 0.75/2.01 ...
  1.27/3.57 m/s and θ 0.50/1.38 ... 1.05/2.38 K; inside the forced rows mass
  2.2e-4 of the mass and water 0.09 %; spin-up 1.58, max w 1.11; 99 s.
* **(a) after the fix:** pass, 0 specified words changed in u, v, w, θ, p.


## 9. Device runs on an RTX 3080: the live nest's reset and a real-weather box

The same RTX 3080, each run under 10 minutes with a memory budget of 90 % of
the free memory measured just before it. Scripts, run records and numbers are
kept with the measurement records, not shipped.

### 9.1 The live nest's end-of-step reset (`spec_bdy_final`)

CM1's Weisman-Klemp sounding (isnd 5, iwnd 2) and
warm bubble, a 48 × 48 × 40 parent at 600 m (dt 3 s) and a ratio-3 child
(16 × 16 parent cells, 200 m, dt 1 s, spec_zone 1, relax_zone 4, WSM6) whose
west side runs through the updraft, so cloud and rain form on its specified
rows (child max qc 1.4 g/kg, max w 4.3 m/s at 15 min). 600 parent steps
(30 min); every 5th parent step the child's specified row on each side
(`spec_bdyupdate`'s ownership) is compared with the parent captured onto the
child's lattice by the nest's own operator, which at the end of a parent step
is WRF's target `bdy + dtbc*bdy_tend` (`share/interp_fcn.F:2583-2584`). Two legs
in one process: the code as it ships, and the same code with `finalize_specified` made a
no-op (the live nest without the reset). Max |specified row − captured parent| over
the 30 min:

| carrier | with the reset | without it | parent on those rows, max |
|---|---|---|---|
| u, v, w (m/s) | 1.9e-9, 1.5e-8, 6.0e-8 | 1.7e-5, 8.6e-6, 4.3e-6 | 31.8, 15.0, 7.6 |
| θ (K) | 7.5e-9 | **0.66** | 3.0 (perturbation) |
| π′ | 7.3e-12 | 6.3e-4 | 5.1e-4 |
| qv, qc, qr (g/kg) | 0, 3.6e-9, 8.9e-13 | **0.24, 0.24, 0.020** | 14.0, 1.8, 0.50 |

With the reset every carrier's specified row ends each parent step at the
interpolated parent to float rounding; without it, microphysics moved θ by up to
0.66 K and the water species by up to 0.24 g/kg there (the same defect §8.6
measured in the chain). 1800 child-step resets ran
(`spec_bdy_final.steps`); 89 s of wall time per leg, 2.0 GB of device memory.
The kernel's modes 2 and 3 also passed their device word-identity tests
(`GM1_NEST_DEVICE_TESTS=1`, `tests/test_nest_spec_bdy_final.py`,
`tests/test_nest_live_operators.py`: 2 passed). The chain's steady hold (§8.1)
passes again with the reset (0 specified words
changed in u, v, w, θ, p over 60 steps; interior drift 5.1e-4 m/s in u, 0 in θ).

### 9.2 A 1 km box in the Sierra atmospheric river (real weather, offshore)

History: the ArWen control member of the 2025-12-25 Sierra atmospheric-river
showcase (3 km Lambert conformal, true latitudes 29.1 and 49.1 N, 300 × 270 ×
49, P3, saved hourly).
Box: history cells 6-41 east and 40-105 north (36 × 66 cells; 36.4-38.2 N,
125.3-124.0 W; all ocean), ratio 3: **108 × 198 × 40 GM1 cells of 1014.79 m on
the ground** (map scale 0.98542, varying 0.07 % across the box), dz 500 m,
dt 6 s, Morrison, **water vapour only**, 01-03 UTC (three saved times), CM1's
1 km supercell namelist for everything the chain does not set. `chain check`
admitted it on every check (§5); `chain new` then `gpuwm-cm1 run`: 1200 steps
in **87 s** of wall time (0.061-0.065 s/step; peak device memory 1.95 GiB against the estimate's 1.36 GB), 25 frames every 300 s. Two probes
from the same prepared chain: the first 20 min with frames every 60 s (27 s)
and the first 5 min every 6 s (20 s). The analysis takes the
driver onto GM1's lattice by the chain's own transfer and compares cell for
cell.

What the vapour-only lane left out at 01 UTC (the box's mean column paths):
0.43 kg m⁻² of hydrometeors (cloud 0.15, rain 0.07, P3 ice 0.21) against
21.6 kg m⁻² of vapour, 2.0 % of the water. GM1's first frame equals the
transfer (θ and qv RMS 0, u and v 1e-6 m/s, p 1.5e-3 Pa; w 2.4e-3 m/s).

**Boundary relaxation.** RMS of GM1 − ArWen at the saved times, by distance
from the side in GM1 cells: row 0 (specified) / rows 1-4 (relaxed) / rows 5-14 /
rows 15 and in:

| t | u (m/s) | v (m/s) | w (m/s) | θ (K) | qv (g/kg) | p (hPa) |
|---|---|---|---|---|---|---|
| 02 UTC | 0.09 / 0.60 / 1.14 / 1.53 | 0.10 / 0.64 / 1.14 / 1.47 | 0.00 / 0.28 / 0.46 / 0.63 | 0.00 / 0.40 / 0.70 / 0.76 | 0.00 / 0.069 / 0.142 / 0.166 | 0.015 / 0.151 / 0.161 / 0.163 |
| 03 UTC | 0.09 / 0.80 / 1.32 / 1.53 | 0.08 / 0.75 / 1.23 / 1.41 | 0.00 / 0.39 / 0.43 / 0.48 | 0.00 / 0.47 / 0.72 / 0.75 | 0.00 / 0.089 / 0.153 / 0.185 | 0.027 / 0.169 / 0.151 / 0.132 |

Row 0 is at ArWen to the last bit for θ and qv (the end-of-step reset); its u
and v are not zero because the mass-point average takes the first relaxed face
too. For u, v, w, θ and qv the error grows inward at both saved times. Against
the relaxation target itself (the saved times interpolated linearly), in all
24 frames every 300 s, the relaxed rows are closer than both interior bands for
u, v, θ and qv, and for w in 23 of 24 (at 02:45 UTC rows 5-14 are 0.36 m/s
against 0.38); θ's error peaks at rows 5-14 rather than deeper in during the
first 20 min. Pressure does not follow: GM1's pressure differs from ArWen's by
about 0.15 hPa everywhere in the box, and from 02:05 UTC on the relaxed rows are
no closer than the interior (0.169 against 0.132 hPa at 03 UTC). The pre-registered
criterion names u, θ, w and qv; pressure is reported, not passed.

**Conservation.** Inside the forced zone (5 GM1 cells in, where only GM1's
numerics act), trapezoid over the frames:

| span, sampling | dry-air residual / mass | / gross lateral exchange | water residual / water | / surface precipitation |
|---|---|---|---|---|
| 2 h, 300 s | −1.2e-5 | −0.77 % | 1.5e-4 | 0.13 % of 4.7e10 kg |
| 20 min, 60 s | −4.8e-6 | −0.69 % | −3.7e-5 | (1.5e8 kg) |
| 5 min, 6 s | 1.8e-6 | 0.44 % | 8.4e-6 | (no rain yet) |

Over the whole box the budget does not close, by 2.1e-3 of the mass and 0.22 %
of the water, as in §8.2: the specified and relaxed rows move the edge toward
ArWen, which puts mass in without a flux through the outer face (WRF
`module_bc` does the same).

**Spin-up.**

* The pre-registered measure of §8, max |dπ/dt| on the history's cells over the
  first interval at the history's sampling: GM1 2.54e-7 s⁻¹ against ArWen's
  2.87e-7: **0.89, met** (≤ 2); 0.63 in the second hour.
* The box-mean |dp_sfc/dt| inside the forced zone (lowest level): hourly, GM1
  0.75 and 0.57 hPa/h against ArWen's 0.95 and 0.52. Sampled more finely, GM1's
  pressure carries the oscillation of a state moved onto a new grid and not
  rebalanced (§7 item 8): at 300 s, 2.1 hPa/h over the first 30 min and 1.0 over
  the last hour; at 60 s, 7.6 over the first 5 min and 3.0 at 15-20 min; at 6 s,
  93 over the first minute and 35 at 4-5 min. ArWen's history is hourly, so it
  has no finer counterpart; the adjustment halves within minutes and the
  300 s-sampled rate settles within about an hour.
* Clouds and rain, which the vapour-only lane starts at zero: GM1's box-mean
  hydrometeor path reaches half of ArWen's mean (0.81 kg m⁻²) at 15 min and
  all of it at 30 min; at 02 and 03 UTC it is 1.26 and 0.93 kg m⁻² against
  ArWen's 1.17 and 0.82. The first rain reaches the sea after about 16 min; the
  box-mean precipitation is 0.81 mm in the first hour against ArWen's 1.31
  (62 %) and 1.41 mm in the second against 1.83 (77 %).
* Max w: 8.3 m/s at t0 (ArWen's, transferred), 14.3 m/s at 5 min (1.7 times
  ArWen's 8.4) and 20.8 m/s at 20 min; 14-16 m/s at 02-03 UTC against ArWen's
  7.3-8.8. GM1 at 1 km resolves this air's convection (most-unstable CAPE
  578 J/kg in the box-mean sounding) more strongly than ArWen at 3 km with P3;
  that is a difference between two models at two spacings, not a shock.

### 9.3 Verdicts, and what these runs do not show

| criterion | verdict |
|---|---|
| live nest: specified rows at WRF's target at the end of every step | met: every carrier to float rounding (1e-8), against 0.66 K and 0.24 g/kg without the reset |
| a real-weather window admitted by the unchanged limits | met for the Sierra history offshore (847,050 boxes over the ocean; none in the Central Valley, by the crossing check); the map-scale check measures GM1's remaining length error once GM1 takes the box's scale (§5), and P3 enters only as water vapour |
| boundary relaxation: the forced zone closer to the driver than the interior | met for u, v, w, θ, qv at both saved times; against the target every 300 s, u, v, θ, qv in 24 of 24 frames and w in 23; not for pressure from 02:05 UTC (GM1's pressure differs from ArWen's by about 0.15 hPa RMS everywhere in the box, relaxed rows included; the box-mean difference is about 0.07 hPa) |
| conservation inside the forced zone: mass to 1e-3 of the mass, water to 1 % of the water | met: 1.2e-5 and 1.5e-4 over 2 h |
| spin-up, child ≤ 2 × driver on the pre-registered measure | met (0.89); max w 1.7 times ArWen's in the first 5 min, 2.4 in the first hour (GM1's own convection) |

Not shown: agreement with observations (none was compared); the history's
clouds and rain at t0 and through the sides (left out by the lane: 2 % of the
water, and GM1 took 30 min to make its own); terrain or land (the box is ocean);
a history saved more often than hourly (air that crosses the box in 68 min sees
the sides move linearly between saves an hour apart); the box's west side is
6 cells from ArWen's western edge and the transfer reads from cell 5, just
inside the 5 rows WRF relaxes by default (the history does not record ArWen's
`spec_bdy_width`); the check asks for 3 cells, so it is admitted, but the
box's western air is close to ArWen's own boundary-driven air.

### 9.4 Checks made separately from the chain's code (RTX 3080 and host)

The scripts and their outputs are kept with the measurement records, not shipped; the WRF source read is
`wrf-model/WRF` at tags `v4.6.1` and `v4.7.1`, fetched read-only.

* **Repeatability.** The Sierra box (6,40,36,66, ratio 3, Morrison, vapour only, 01-03 UTC) was prepared and run
  a second time from the same sources (budget 2222 MiB, 90 % of the measured free
  memory): 2 h in 95 s of steps (101 s with start-up), sampled device peak 1.95 GiB. The prepared initial
  state, the boundary arena and every history frame compared are byte-identical between the two runs.
* **Admission, recomputed from the wrfout with netCDF4 alone** (no GM1 code): relief 0 m (the box and
  its ring are all ocean, `HGT` 0); `MAPFAC_M` 0.984948-0.986074 over the ring, mean 0.985423, variation
  6.60e-4 (0.066 %), distance from 1 0.01505; GM1 spacing `float32(3000/(3*0.985423))` = 1014.792236 m;
  `MAPFAC_M` equals the Lambert-conformal map factor computed from `XLAT`, `TRUELAT1/2` alone to 3.3e-8, so
  the projection is what the attributes say; fastest |U| 23.45 and |V| 49.24 m/s over 01-03 UTC, crossing
  4080 s (68.0 min) against 3600 s; f at the box's mean latitude 37.32 N within the declared tolerance.
  Every number equals the check's. No admission limit changed value (`DEFAULT_TERRAIN_LIMIT_M` 50,
  `DEFAULT_MAP_FACTOR_LIMIT` 0.01, the crossing rule unchanged). The map-factor measure follows the
  spacing: with `DX/ratio` GM1's length error would be `|m - 1|`; GM1 takes
  `DX/(ratio*m_window)` and its length error is `|m/m_window - 1|`. Both bound the same quantity (GM1's
  cell length against the history's cell length on the ground) by the same 1 %, so this is not a relaxed
  limit.
* **`spec_bdy_final` and `flow_dep_bdy` against a hand-built case.** Both WRF routines
  (`share/module_bc.F:2066-2216` and `:2335-2456`, identical in v4.6.1 and v4.7.1) were transcribed loop
  for loop in 1-based WRF indices and run on random fields beside `gm1_nest_spec_apply` modes 2 and 3 on
  the device (u, v, w and mass staggers, spec_zone 1 and 2, width 5, the field inside a padded owner) and
  beside `ChainLateralForcing.after_scalar_stage` (random `rru`/`rrv` with zeros in them): 16 of 16 and 6 of
  6 cases word-identical (4,312 and 992 points set), the owner's padding untouched.
* **The real run's specified rows.** At every 300 s frame the outermost row of u, v, θ, π (from `pp3d`)
  and qv equals `value + dtbc*(end - start)/3600` computed here from the prepared strips, to one float
  rounding (θ 3.1e-5 K, π 3e-8), against 1.2-4.5 m/s (u, v), 0.54-1.6 K and 2.7e-4-9.1e-4 kg/kg on the first
  relaxed row. Hydrometeors on outer inflow faces are zero except where microphysics formed them after the
  last stage's `flow_dep_bdy` (0-4 % of inflow points, at most 0.08 g/kg; none equals the interior row, so
  none is a wrongly signed outflow copy), which is WRF's order too.
* **The saved `prs` and `rho` on the outermost row are from before the reset.** GM1 computes them after
  microphysics (CM1 `mp_driver.F`'s equation of state) and the reset follows, so on that row the history's
  `prs` differs from `p00*(pi0+pp3d)^(cp/Rd)` by up to 9.5e-4 in π (about 0.3 %); on every other row they
  agree. WRF has the same order (`calc_p_rho_phi` after microphysics, v4.6.1 `solve_em.F:4186`, before
  `spec_bdy_final`), so no word changes; read pressure on that row from `pp3d`.
* **Relaxation, by an operator independent of the chain's transfer** (each GM1 cell takes its history
  column, linear in height between the history's mass-level heights, pressure linear in ln p): at 02 UTC,
  rows 1-4 / rows 15 and in, u 0.60/1.51 m/s, v 0.64/1.45 m/s, θ 0.40/0.76 K, qv 0.071/0.166 g/kg, p
  0.141/0.156 hPa; at 03 UTC p 0.169/0.148 hPa (the relaxed rows further from ArWen than the inside). The
  figures of §9.2 are reproduced; the pressure verdict stands as reported.
* **CUDA graphs.** A chain run refuses `--cuda-graphs` (`CM1-CARTESIAN-CUDA-GRAPHS-UNPROVEN`), so the
  chain's reset and `flow_dep_bdy` never run captured. The live nest does: a 5 min WK82 nest (600 m parent,
  ratio-3 child) with `--cuda-graphs off`, `on` (291 child replays) and `verify` (294 child captures
  compared, 0 mismatches) wrote byte-identical parent and child history.
* **Line citations.** WRF line numbers in this record are v4.6.1's (the version the rest of GM1's records
  cite); v4.7.1 moves `solve_em.F` by 5-9 lines. `flow_dep_bdy` ends at `module_bc.F:2456`, and the end-of-step
  block is `solve_em.F:4531-4716`.
