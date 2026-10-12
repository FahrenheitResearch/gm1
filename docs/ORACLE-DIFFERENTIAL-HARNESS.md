# The CM1/GM1 differential harness

The real CM1 r21.1 binary is the oracle. This harness configures CM1 and
gpuwm-cm1 from **one** physical description, runs both, and reports where and
when they part.

**Where the harness is.** `tools/oracle/` holds transcriptions of CM1 source and
needs a built CM1 r21.1, so it stays in the development repository: it is not in
the public repository or the wheel, and the commands on this page cannot be run
from either. This page is published so that a reader can see exactly how the
comparisons quoted in the release notes were made. The case descriptions the
harness reads are public, under `examples/oracle/`.

In the development repository, everything below runs from a source checkout
with `PYTHONPATH=src`.

```
PYTHONPATH=src python3 tools/oracle/differential.py <subcommand>
```

---

## 1. Pick the horizon before you pick the duration

```
PYTHONPATH=src python3 tools/oracle/differential.py horizon \
    --term acoustic --nx 32 --ny 32 --nz 40 --dx 1000 --dy 1000 --dz 500 --dt 1.0
```

```
acoustic
  timescale   2.77778 s
  horizon     278 s  =  278 large steps  (1668 acoustic substeps)
  formula     tau = min(dx,dy)/c, c = 360 m/s (misclibs.F:883-885); horizon = 100*tau
  because     A sound wave crosses one cell in dx/c. ...
```

Terms: `acoustic`/`psolver`, `advection`/`metric`/`diffusion`, `boundary`,
`gravity-wave`/`damping`, `surface-drag`, `microphysics`, `coriolis`,
`radiation`. Omit `--term` to print them all; add `--json` for a receipt.

Two terms answer **`differential_unsuitable: true`** and say what to use
instead:

* **coriolis**: 1/f is hours; a storm run cannot see it. Use the analytic
  inertial oscillation (uniform f, no other forcing, initial `(u,v)=(U,0)`,
  compare against `u = U cos ft`, `v = -U sin ft`).
* **radiation**: diurnal. Compare offline against `run/RRTMG_LW_DATA` and
  `run/RRTMG_SW_DATA` for a fixed profile.

`microphysics` deliberately returns `20*dt`, with the rationale that you must
**initialise a saturated profile** rather than wait for a bubble to build a
cloud.

## 2. Write the case once

A case is `[case]` + `[cm1]` + `[gm1]`. `[cm1]` speaks CM1 r21.1's own
namelist words, because CM1 is the authority; `[gm1]` holds only the keys CM1
has no word for (the lane, the CUDA tile shape, the named ArWen component).
See `examples/oracle/acoustic-warm-bubble-32.differential.toml`.

Both refuse by key: `[cm1]` refuses anything not in CM1's
physical-configuration inventory, `[gm1]` refuses anything CM1 *does* have a
word for.

```
PYTHONPATH=src python3 tools/oracle/differential.py translate CASE.toml --out DIR
```

writes `DIR/namelist.input`, `DIR/<case>.toml` and `DIR/translation.json`.
The translation receipt names every CM1 selector this port cannot carry; if
it is non-empty the two runs are **not the same physical problem** and the
emitted TOML carries an `UNVALIDATED` comment saying so.

## 3. Run each side

```
PYTHONPATH=src python3 tools/oracle/differential.py cm1 CASE.toml OUTDIR
PYTHONPATH=src python3 tools/oracle/differential.py gm1 CASE.toml OUTDIR [--remote HOST]
PYTHONPATH=src python3 tools/oracle/differential.py run CASE.toml OUTDIR [--remote HOST]
```

`cm1` copies the authority `run/` directory (executable plus the lookup tables
the schemes open by name, symlinked) into `OUTDIR`, writes the namelist there
and runs it. **Nothing is ever written under the authority tree.**

The GPU lives on another machine, so `gm1 --remote HOST` scps the config out,
runs `gpuwm-cm1 cartesian --integrate --history-dir`, and brings the frames
back.

### Running a checkout's gpuwm-cm1 on a host with no cargo

A CUDA host that has no Rust toolchain (or an installed wheel that predates
the current tree) can run **this checkout's** `cm1core` from source against
the node's own cupy, with the two native artifacts built here. Name the remote
layout first; there are no built-in defaults:

| Variable | Meaning |
|---|---|
| `GM1_REMOTE_ROOT` | directory on the host that receives `cm1core/` and `native/` (or pass `deploy --root`) |
| `GM1_REMOTE_PYTHON` | interpreter on the host with cupy installed |
| `GM1_REMOTE_ARWEN` | the admitted ArWen tree on the host, put on `PYTHONPATH` |

Build once, deploy once:

```
export GM1_REMOTE_ROOT=<remote-dir> GM1_REMOTE_PYTHON=<remote-venv>/bin/python \
       GM1_REMOTE_ARWEN=<remote-arwen-tree>
cd rust/cm1-state && cargo build --release --bin cm1_forecast_publish --lib
cd ../..                     # cargo names the binary after its source file
PYTHONPATH=src python3 tools/oracle/differential.py deploy <user>@<cuda-host>
```

Then `gm1 --remote <user>@<cuda-host>` and `run --remote <user>@<cuda-host>`
work. `deploy` rsyncs `src/cm1core` and scps
`cm1_forecast_publish` + `libcm1_state.so` into `$GM1_REMOTE_ROOT/native/`,
and `run_case` passes them as `--library` / `--history-publisher`, which is
what `cartesian_door.py` documents for a source checkout. It also puts
`$GM1_REMOTE_ARWEN` on `PYTHONPATH`; without the admitted ArWen tree the engine
pin refuses with `CM1-ARWEN-ESTATE-DRIFT`. A missing variable refuses by name
before anything is copied.

The CM1 side looks for the built oracle at `work/native-reference/cm1r21.1`
in this checkout, then in a sibling `mirror` checkout; set
`CM1_AUTHORITY_ROOT` (or pass `--authority`) to point anywhere else.

Frames come back with `rsync --exclude history.cm1fhs`: the sealed envelope
duplicates the raw arena the comparator reads, so copying it doubles the
transfer for nothing.

**Share the card.** Check `nvidia-smi` first, keep runs short, and never kill
a process you did not start.

## 4. Compare

```
PYTHONPATH=src python3 tools/oracle/differential.py compare CM1DIR GM1DIR/frames \
    [--interior N] [--tolerance FIELD=NRMSE] [--json OUT.json]
```

Per field, per history time: `max|d|` and its `(k,j,i)`, `rmse`, normalised
`rmse`, `|d|/peak`, and the first time any field crosses tolerance.
`--interior N` trims N cells from each lateral edge, which separates an
interior disagreement from a lateral-boundary one without another run.

**Read `|d|/peak`, not `nrmse`, for fields that are zero over most of the
domain** (a warm bubble's `thpert`, a quiescent `w`): the RMS denominator is
dominated by background there and turns a small error into a scary number.

### What the two sides look like on the wire

* **CM1** writes `output_format=1`: raw little-endian IEEE **binary32**, no
  container, described by the `.ctl` files beside it. The harness forces the
  output selectors so CM1 writes exactly the fields GM1 publishes.
* **CM1's precision**: the shipped `src/Makefile` selects gfortran with **no
  `-DDP`**, so CM1's prognostic arrays are binary32, the same width as GM1's.
  Both sides therefore cross this harness as f32 words and no precision
  conversion is introduced. `_difference` widens to f64 only to compute the
  statistics.
* **GM1** frames carry `history-carriers.f32` (a raw concatenated f32 arena),
  `history-field-table.tsv` and `history-publication.json`. The harness reads
  the raw arena, checking every declared offset against the file's own size.

**Bit identity is not the expectation.** Different codes, different order of
operations, different hardware. What must hold is that the disagreement starts
at rounding and **does not grow**.

### The field pairing, and the one trap in it

`compare.FIELD_MAP` / `compare.CM1_NAME`:

| GM1 carrier | CM1 variable |
| --- | --- |
| `th3d` | **`thpert`**, not `th` |
| `prs` | `prs` |
| `rho` | `rho` |
| `u3d`, `v3d`, `w3d` | `u`, `v`, `w` at native stagger, no interpolation |
| `closure.tkea` | `tke` |
| `q3d[0..5]` | `qv qc qr qi qs qg` |

`th3d`'s field-table meaning string says "current potential temperature" and
its values are the **perturbation**: on the isnd=5 base state it runs
-1.2e-3 .. 0.96 K where CM1's `th` runs 300 .. 489 K. Pairing it with `th`
reports `nrmse = 1.0` at every frame, which reads as a catastrophic dycore
failure and is only a name.

## 5. The key map, and the gap register derived from it

```
PYTHONPATH=src python3 tools/oracle/differential.py keymap          # the map
PYTHONPATH=src python3 tools/oracle/differential.py keymap --gaps   # the gap register
PYTHONPATH=src python3 tools/oracle/differential.py keymap --key icor
```

The map is the record `src/cm1core/records/CM1-GM1-KEYMAP.json`, authored and
**checked** by `tools/build_cm1_gm1_keymap.py` against the packaged authority
inventory three ways: every mapped CM1 key exists; every CM1
physical-configuration key is accounted for; every gpuwm-cm1 key in
`config.py`'s `_SCHEMA` is either mapped or listed as having no CM1 word.

583 CM1 namelist fields; 272 of them configure physics; **175 of those map
to a gpuwm-cm1 TOML key and 97 do not** (18 more map from the output and
parcel groups; the record's `counts` hold these numbers). That 97 is a
*derived* gap register (the authority inventory minus what this port can
express), not an assertion about coverage. Regenerate it after changing
`config.py`:

```
python3 tools/build_cm1_gm1_keymap.py
```

### One translator

The translation lives in `src/cm1core/cm1import/translate.py`, which the CM1
namelist importer uses too; `spec.DifferentialCase.translate` hands it the
base deck plus the case's `[cm1]` overrides and `HARNESS_OUTPUT` as the
&param9 selection. It needs no cm1.exe: an unmapped key is judged against
what GM1 integrates (the key map's `gm1_behaviour`, below), r21.1's code
defaults come from `records/CM1-R21-NAMELIST-DEFAULTS.json`, and
`cm1_run.base_deck` reads the shipped decks from `upstream/cm1-21.1/pristine`
when no build exists. Only running CM1 needs the build.

### Transforms that are not the identity

* **`zd` -> `closures.upper_damping_depth_m` is not a copy.** CM1's `zd`
  (`param.F:8655`) is the **height of the base** of the Rayleigh layer; GM1's
  `upper_damping_depth_m` (`rust/cm1-state/src/rayleigh_profiles.rs:186-188`)
  is its **depth below the model top**. `depth = ztop - zd`. CM1's shipped
  supercell deck `zd = 15000` on a 20 km top is GM1's `5000`. An identity map
  puts the damper in the wrong place, and the run still finishes.
* `rstfrq` -> `run.restart_interval_s` takes the absolute value: CM1's
  negative `rstfrq` is a disable sentinel, GM1 carries a positive cadence.
* `imoist`/`ptype` -> `physics.component`: GM1 names the scheme behind its
  ArWen seam; the translator derives CM1's integers back from the name.
* On `stretch_z = 0` the translator emits **none** of the param6 vertical-grid
  words: CM1 overwrites `ztop` on that branch and `config.py` documents them
  as optional with exactly those values, so an unstretched grid states them by
  omission rather than by declaring a coordinate the run does not have.
* On `stretch_z = 1` the `zd` transform needs the **generated** model top, so
  `cm1import.translate.model_top_m` builds the column through `cm1_vertical_grid` -- one
  generator, the same one the run uses -- and takes `param.F:8193`'s
  `sigmaf(nk+1)` from it rather than assuming the declared `ztop`.
  `stretch_z = 3` and `4` read their levels from a file
  (`param.F:8019-8026,8085-8089`), so the namelist does not determine the top
  and the translation refuses rather than place the damper by guess.
* A stretched CM1 history writes `zdef N levels` with the heights on the
  following lines, not `zdef N linear`. `grads.read_ctl` reads both and
  carries the explicit heights on `GradsDataset.levels`; `z0`/`dz` are then
  only the first height and the first spacing and describe nothing else.

---

## The stretched-coordinate case

`examples/oracle/stretched-column-32.differential.toml`, term `metric`, 100
large steps at dt = 1 s on 32x32x40 at 1 km with `stretch_z = 1`: 100 m layers
at the ground thickening to 300 m aloft under an 8 km top, `dz = 200` as the
`param.F:8170` metric reference and not any layer's thickness. Curves in
`evidence/oracle/stretched-column-32/`, summary in its `receipt.json`.

CM1 writes its own scalar and w levels into the GrADS descriptors, so the
coordinate is checked before any dynamics is:
`tests/test_cm1_vertical_grid.py` holds this port's generator to those levels
to the descriptor's printed precision, and that test needs no GPU.

`max|d|/peak`, t = 20 s -> t = 100 s: `prs` 1.2e-4 -> 8.6e-5, `rho` 1.1e-4 ->
6.1e-5, `u3d` 5.5e-4 -> 2.4e-4, `v3d` 2.3e-3 -> 1.1e-3, `q3d:qv` 1.2e-4 ->
1.3e-4, `th3d` 1.5e-3 -> 1.9e-3, `w3d` (near zero) 8.7e-1 -> 5.6e-2. `tke` and
all five hydrometeors identically zero in both. These are the magnitudes the
uniform worked example already reaches, which is the claim: stretching the
column costs nothing in agreement.

`negative-control-uniform-base-state.txt` beside them is the same GM1 run with
the declared coordinate not reaching the Rust initializer, so the base state
stood on the uniform column while the operators ran on the stretched one:
`prs` 1.3e-1 and `rho` 1.1e-1, **constant in time**. A wrong vertical
coordinate does not diverge, it offsets, and that is what to look for.

CM1's other two stretched branches read their levels from a file rather than
generating them, so the level list is declared once, on the GM1 side, as
`[gm1] "grid.levels_m"` -- the key map carries it as GM1-only -- and `cm1_run`
writes CM1's `input_grid_z` from it. Two more cases use that:

* `explicit-w-levels-32` (`stretch_z = 4`, `param.F:8072-8132`) lists exactly
  the faces the analytic case generates, so the two describe one column
  reached two ways. Every field matches the analytic case digit for digit at
  every frame.
* `explicit-scalar-levels-32` (`stretch_z = 3`, `param.F:8005-8064`) is a
  different column -- 128 m to 512 m under a 13184 m top -- and reaches the
  same agreement: `prs` 1.1e-4 -> 5.7e-5, `rho` 1.0e-4 -> 4.1e-5.

**`stretch_z = 3` needs a column whose recursion is exact.** `param.F:8025`
builds faces by `sigmaf(k+1) = 2*sigma(k) - sigmaf(k)`, a first-order
difference equation in binary32: each face carries the representation error of
every level below it and the errors alternate rather than cancel. Feed it a
list that is merely close to a nice column and the end-segment layer depths
come out differing in the sixth digit, and `CM1-CFG-VERTICAL-FORNBERG` refuses
it -- correctly: one such column's measured upper weights are
`(1.8749927, -1.2499900, 0.3749973)` against the `(1.875, -1.25, 0.375)` the
compiled kernels carry inline, some sixty f32 ulp out. Choosing every face a
multiple of 64 m makes the recursion exact and the refusal goes away.

---

## What the worked example measured

`examples/oracle/acoustic-warm-bubble-32.differential.toml`, 280 large steps
at dt = 1 s on 32x32x40 at 1 km / 500 m, CM1 r21.1 against gpuwm-cm1 on an
RTX 5070 Ti. Full curves in
`evidence/oracle/acoustic-warm-bubble-32/`.

| field | t = 20 s | t = 280 s | shape of the curve |
| --- | --- | --- | --- |
| `prs` | 1.2e-4 | 7.4e-5 | flat |
| `rho` | 9.8e-5 | 5.3e-5 | flat |
| `u3d` | 5.4e-4 | 3.8e-4 | flat |
| `v3d` | 2.3e-3 | 1.1e-3 | flat |
| `w3d` | 9.1e-1 | 2.3e-2 | **decaying** |
| `th3d` | 1.5e-3 | 3.1e-2 | **growing, 21x** |
| `q3d:qv` | 1.9e-4 | 2.0e-3 | **growing, 10x** |
| `closure.tkea`, `qc qr qi qs qg` | 0 | 0 | identically zero in both |

(numbers are `max|d| / peak|CM1|`)

Pressure, density and the horizontal momentum agree at rounding level and stay
there through 1680 acoustic substeps. No condensation occurs in either code,
so nothing here is microphysics.

**The scalars do not.** `thpert` and `qv` diverge monotonically, and from
t = 120 s the largest difference sits at `(k,j,i) = (6..7, 17, 29)` every
frame -- two cells inside the east lateral boundary, in the lowest few
kilometres -- while the interior (`--interior 8`) shows no monotonic growth at
all. The growth begins after the bubble's signal reaches the lateral boundary.
This is a **scalar open-lateral-boundary difference**, not an interior dycore
difference.

**Cause, found and fixed.**  The scalar advection
tendency kernel (`cm1_adv5_scalar_tendency_v1`) and the final-stage moisture
pdef total (`cm1_rk_pdef_total_v1`) wrote only `3 <= i < ni-3`,
`3 <= j < nj-3`, into a buffer zeroed first, so the outer three cells of every
side got no scalar advection at all -- theta (advected as `(th0-300)+th'`,
so also no vertical advection of the base state), qv and every hydrometeor
row.  CM1's `advs` advects every cell, with an open side's inflow face-flux
copy and divx-piece subtraction (`adv.F:214-340`), and `pdefx1/x2`,
`pdefy1/y2` run over every cell with `bcs(dum)`'s halo (`pdef.F:49-166`).  A
whole-domain launch now does the same; a tile-local launch (`obc=0`) keeps the
interior schedule.  `tests/test_cm1_flat_stretched_native_words.py` holds the
edge cells to CM1's own compiled `advs` word for word, and
`evidence/oracle/open-boundary-ring` is the full-model check
(`examples/oracle/edge-coldpool-100.differential.toml`): the edge-band
difference from CM1 fell from 10-45 times the interior level to the interior
level.  The acoustic-warm-bubble evidence above predates the fix.

The decisive control -- the same case with periodic laterals -- **cannot be
run**: `config.py` seals the Cartesian lane to the tuple `(2,2,2,2,1,4)` and
refuses `wbc=ebc=sbc=nbc=1`, although `cuda_sound3_periodic.py`,
`cuda_psolver3_periodic.py` and `cuda_anelp_periodic.py` are all in the tree.

Two other things the worked example ran into:

* `initialization` `(isnd=5, iwnd=0)` is admitted by `config.py:816-826` and
  **refused by the runtime**: `gm1_idealized_residency.py` says "the analytic
  Weisman-Klemp sounding lane admits iwnd=2 only". The door and the
  initializer disagree, so the pair is configurable and unrunnable. The case
  carries `iwnd=2` because of it.
* The Cartesian lane forces `physics.component = "wsm6-mp6"` and
  `equation_set = 2` (`config.py:1666-1679`), so **there is no dry
  configurable lane**. The cheapest decisive differential this port can run is
  a moist one kept short enough that the dycore is what the curve measures.

## Initialization at model time zero

`tools/oracle/initialization_differential.py` is the same harness run at the
cheapest decisive horizon there is: zero.

Initialization is the one part of the model whose correctness does not need a
trajectory.  The state at t = 0 either is the state CM1 builds or it is not,
and a difference is visible before a single step.  So both sides stop before
stepping: CM1 runs one large step and writes its initial output time, and
`gpuwm-cm1` runs `initialization.iconly = 1`, which builds the same state on
the host and writes it in the runtime's own frame layout with no CUDA context
at all.  The whole comparison is therefore GPU-free and runs in a couple of
minutes.

```
PYTHONPATH=src python3 tools/oracle/initialization_differential.py \
    --out evidence/oracle/initialization \
    --library rust/cm1-state/target/release/libcm1_state.so
```

Twenty-eight cases: every `iwnd` branch, every implemented `iinit` arm, both
`ibalance` branches, `iorigin`, `maintain_rh`, `imoist`, and the `isnd=7` and
`isnd=17` external-sounding lanes.  Twenty-seven are bit-identical or inside
the stated tolerance; the twenty-eighth is `irandp=1`, which cannot be
compared word for word because CM1 draws from gfortran's generator.  The
per-selector verdict lands in `src/cm1core/records/CM1-INITIALIZATION-VALIDATION.json`,
and every run receipt and `gpuwm-cm1 doctor` report the unvalidated ones by
name.

---

## Things the harness got wrong, and now does not

**"Unrepresentable" meant "differs from CM1's distributed deck".** An unmapped
CM1 key was reported only when its value differed from `run/namelist.input`,
so a key every shipped deck sets -- `ramp_sgs = 1` -- was dropped in silence
although GM1 did not run it, and a key outside the r21.1 inventory (a newer
CM1's) was skipped outright. The key map now records, for every unmapped
state-affecting key, what GM1 integrates (`gm1_behaviour`: the one value it
runs, or the selector values under which CM1 reads the key at all), and the
translator names a key whenever the two differ, names any key the inventory
does not hold, and names `eqtset` for a scheme CM1 runs only at eqtset=1
(src/param.F:936-948). `ramp_sgs`/`ramp_time` now cross as
`closures.ramp_sgs`/`closures.ramp_time`
(evidence/oracle/supercell-fidelity).

**`nsound` is derived, never declared.** `misclibs.F:818-915` with
`README.namelist:453-455`: since cm1r19 CM1 has never read the acoustic
small-step count from a namelist. It derives it from `dtl` and the grid before
the first step (`cm1.F:3054`) and again on every change of `dt`. gpuwm-cm1
declares it, so a case that names one was free to hand the two codes a
different number of acoustic small steps per large step, and the
`acoustic-warm-bubble-32` case did exactly that, 6 against CM1's own 4. A
divergence curve produced that way is measuring the schedule as well as the
term. `spec.translate` now derives the count with `cm1_dtsmall` and refuses a
`[gm1] runtime.acoustic_substeps` that disagrees with it.

**CM1's metadata record is not four words wide.** `writeout.F:9536-9700`
writes one record per variable per output, and *which* variables depends on the
run: `adt` under `adapt_dt=1`, `umove`/`vmove`/`domainlocx`/`domainlocy` under
`imove>=1`, up to four more under radiation. The reader assumed four, so every
translating-frame run had `umove` read as a model time and the frame matcher
paired the wrong times, silently, because 12.5 is a plausible number of
seconds. `writeout.F:9721-9739` writes the list that says what the stride is;
`grads.open_history` now reads it.

---

## Worked dynamics cases

`examples/oracle/` ships four cases for CM1's `param2`/`param3` dynamical
words, with their measured divergence curves in
`evidence/oracle/dynamics-cm1-words/` and a receipt in
`evidence/validation/cm1-dynamics-cm1-words-v1/`.

| case | term | how the horizon was chosen |
| --- | --- | --- |
| `coriolis-fplane-32` | `icor=1` | not `1/f`. A differential is decisive once the term's contribution, `f*V*t`, exceeds the disagreement the two codes already have. 280 s gives 2e-2 of the wind against a 5.2e-5 floor. |
| `coriolis-fplane-moving-32` | `icor=1` with `imove=1` | as above; the point is `solve2.F:645,647`'s frame add-back, not a longer run. |
| `beta-wide-512` | `betaplane=1` | the domain, not the duration. Over 32 km `beta0` moves `f` by 0.3x the floor; over 512 km by 12% of `fcor`, which separates cleanly. |
| `adaptive-timestep-32` | `adapt_dt=1` | long enough for the controller to leave the 10% growth cap, reach the `2*dtl` ceiling and ramp down onto 14 output times. |

`coriolis-adaptive-32` runs the last two together.

Four controls ship beside them, and both codes run each one:

| control | what it is the other half of |
| --- | --- |
| `coriolis-fplane-32-icor0` | the `icor=0` floor and no-op score for `coriolis-fplane-32`; the fixed-step member of `adaptive-timestep-32`; the `alph=0.60` member of the alph pair |
| `coriolis-fplane-moving-32-icor0` | the same for `coriolis-fplane-moving-32` |
| `beta-wide-512-beta0` | the f-plane member of `beta-wide-512` |
| `coriolis-fplane-32-icor0-alph050` | the `alph=0.5` member; `switch_response.py` against `coriolis-fplane-32-icor0` |

The receipt was captured again on an RTX 4090 (sm_89) after its source pins
went stale.  Each table in `evidence/oracle/dynamics-cm1-words/` is built from
nine CM1 runs and nine GM1 runs, one of each per case above, compared in every
CM1 x GM1 pairing a table names.  `capture-rtx4090.json` records, for each
GM1 run, the device, the digest of every Python source the run executed, and
the digests of the emitted configuration.
The development repository holds the runbook (`GM1-GPU-VERIFY-DYNAMICS-RECEIPT.md`,
not in the public tree): the nine runs, the
commands, the cost, and what the sealed receipt must say afterwards.

`beta-wide-512` needs `--base-deck examples/oracle/supercell-with-betaplane.namelist.input`:
CM1's supercell deck does not declare `betaplane`, and CM1 reads only the keys
its deck declares.

---

## 6. When the trajectory comparison cannot settle an option

A divergence curve cannot tell a correct implementation of a CM1 option from
an implementation that does nothing, if the option moves the solution by less
than the two codes already disagree by. Measure the **response** instead:

```
PYTHONPATH=src python3 tools/oracle/switch_response.py \
    --control DIR_OF_THE_OFF_RUN --variant DIR_OF_THE_ON_RUN
```

Each directory is one `differential run` output (`cm1/` beside `gm1/frames/`),
and the two must differ in exactly one namelist word. It prints, per field and
per shared history time,

```
dCM1 = CM1(on) - CM1(off)        dGM1 = GM1(on) - GM1(off)
```

`|dCM1|` says how much the option is worth on this case; `|dGM1|` says how
much gpuwm-cm1 moves; `|dGM1 - dCM1| / |dCM1|` is the fraction of CM1's answer
this port does not reproduce. A port that ignores the option scores 1.0 and is
flagged `GM1 DID NOTHING`, no matter how close its trajectory looked.

`CM1History.read` takes CM1's one-based `cm1out_NNNNNN` number, and CM1's
frame 1 is t = 0.  Until the dynamics re-capture, the tool passed the
zero-based position instead, so every `dCM1` it printed was CM1's response one
output interval before its label.  Responses measured before that fix
(`tests/test_switch_response.py`) should be measured again before they are
relied on.

Read the last column only while `|dCM1|` is larger than the two codes'
baseline disagreement on the same case. Below that it is measuring trajectory
separation, not the option -- which is what
`evidence/oracle/boundary-families/README.md` says about the open-boundary
companions, and why those operators are settled a second way, raw-word against
their own CPU transcription.

---

## 9. CM1 `&param9` is now on both sides

`HARNESS_OUTPUT` (`src/cm1core/oracle/spec.py`) selects the CM1 output fields
gpuwm-cm1 publishes, and the emitted gpuwm-cm1 document now carries the
matching `[output]` table.  Both models therefore write the same list of
fields and the comparator has both members of every pair.

`compare.FIELD_MAP` gained the `&param9` products, whose names are CM1's own
`writeout.F` `name_output` strings on both sides:

```
uh  xvort  yvort  zvort  pv
sws svs sps srs sgs sus shs
rain prate  cape cin lcl lfc  th  pwat lwp
```

Two are deliberately absent from `FIELD_MAP` even though both models write
them: `dbz` and `cref`.  CM1 forms reflectivity with its own
`src/mp_wsm6.F`/`src/mp_radar.F` operator and gpuwm-cm1 with ArWen's audited
WSM6 10 cm operator, so pairing them would report the difference between two
transcriptions as a model difference.  `cm1_output_selection` carries an
`UNVALIDATED` marker that says exactly that, and so does every run receipt and
`gpuwm-cm1 doctor`.

A `[cm1]` table may not set an `output_*` word: the harness owns the selection
on both sides, and `load_case` refuses one by name.

**Reading a diagnostic's curve.**  A `&param9` product is a pure function of
the completed state, so its difference is the state's difference put through
the operator.  Where a case has a real signal in the field, the nrmse is the
statistic to read.  Where it does not -- `uh`, `svs`, `zvort` and `pv` in a
warm-bubble case with no organised rotation -- the nrmse is the ratio of two
very small numbers, and what matters is the absolute `max|d|` against
`cm1_peak` and whether the ratio grows or decays.  A wrong operator grows.
