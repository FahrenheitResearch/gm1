# Configurable Cartesian GM1 runtime

`tools/run_cartesian_gm1.py` initializes a fresh sounding and warm-bubble state
through the Rust CM1 initializer and runs the CUDA operators on an explicit
Cartesian grid. Horizontal dimensions and all three spacings are arguments.
The large step is an argument too. `dt=0.5 s`, three RK stages and six
acoustic substeps are the defaults; every stage, the final-RK eqtset2
held-rate removal included, takes the step as a kernel argument, exactly as
CM1 does (`solve2.F:145` declares `dt` a default-REAL variable, not a
parameter). Two bounds remain, and both are CM1's own:

- the acoustic small step `dt/nsound` may not exceed `0.60*min(dx,dy)/360 m/s`
  (`misclibs.F:867`, the psolver=3 branch of `dtsmall`). A declared
  (`run.dt_s`, `runtime.acoustic_substeps`) pair that violates it is refused at
  the configuration door, naming the substep count CM1's own `dtsmall` would
  have derived;
- the advective Courant number of the live state may not reach 1.50. The norm
  is `calccflquick`/`calccfl` (`misclibs.F:2726-2814`, `misclibs.F:2584-2718`)
  and the stop is `misclibs.F:2714`, which a serial `adapt_dt=0` CM1 reaches
  through `statpack.F:469` at statistics cadence whenever `stat_cfl=1`,
  CM1's own shipped default. (`cm1.F:3020-3023` is *not* this authority:
  it is `adapt_dt=1`-only, `#ifdef MPI`-only, and its `stopit` is cleared at
  `cm1.F:3035` before any reader.) GM1 measures it once before the first large
  step of an integration and again at every history frame: earlier and no less
  often than CM1, which is GM1's own cadence choice and not a transcription.
  When the stop is reached mid-run GM1 does what CM1 does: `cm1.F:3517-3520`
  forces `dostat`/`dowriteout` so the tripping state's output is written, and
  only then does `cm1.F:4385-4392` stop the model. The frame that reached the
  stop is therefore published, the integration receipt is carried on the
  `CartesianCourantStop` that ends the run, and
  the `gpuwm-cm1 cartesian ... --integrate` door writes that receipt with the status
  `STOPPED_AT_CM1_ADVECTIVE_COURANT_STOP` before refusing.

`run.dt_s` must still divide `run.duration_s` and `run.history_interval_s`
exactly: a fixed-timestep clock cannot place a frame between two steps.

This is an execution and memory baseline under numerical qualification. It
does not yet establish a native-CM1 trajectory, forecast skill, viewer
integration, or a released product.

Restart identity is no longer on that list. The `cartesian` door writes one
complete `CM1FCP1` restart per `run.restart_interval_s` into `--restart-dir`
and continues from one with `run.irst`/`run.rstnum` and `--restart-from`;
`run.iconly = 1` builds the initial condition, writes it and stops
(README.namelist:227-235, cm1.F:2660,4341-4350). What a restart carries, and
why a stopped-and-resumed run lands on the same words as one that was never
interrupted, is in `src/cm1core/cartesian_restart.py` and proved in
`tests/test_cartesian_restart.py`; the device-side form of the same proof is
`tests/test_cuda_cartesian_restart.py` and is GPU-gated.

## Requested configuration

- 400 cells east-west, 400 north-south, 237 vertically: 37,920,000 mass cells.
- Uniform 125 m spacing: 50 km by 50 km horizontal extent, 29.625 km model top.
- Source sounding: pristine CM1 r21.1
  `run/config_files/cpm_RadConvEquil/input_sounding`, which reaches 40 km.
  The earlier 20 km sounding cannot initialize this model top; the source
  profile is not extrapolated.
- Full-column WSM6 physics uses the explicit
  `gm1.arwen-wsm6-column-capacity-256/v1` dependency profile. See
  [the capacity authority record](ARWEN-WSM6-CAPACITY-256-AUTHORITY.md).

## Column component and dense ice

`physics.component` selects the column scheme. Four are bound, each with its
own CM1 `q3d` scalar axis, persistence inventory and compiled column depth:

| `physics.component` | CM1 `ptype` | ArWen `mp_physics` | `numq` | levels | measured? |
|---|---:|---:|---:|---|---|
| `wsm6-mp6` (default) | 7 | 6 | 6 | 2..80, or 2..256 with the capacity overlay | yes |
| `morrison-mp10` | 5 | 10 | 10 | 2..256, no overlay (and the overlay is refused) | **no** (GPU-run, not vs CM1) |
| `thompson-mp8` | 3 | 8 | 8 | 2..256, no overlay | **no** |
| `thompson-aerosol-mp28` | none | 28 | 11 | 2..256, no overlay | **no, and not against CM1** |

Every bound component now executes on this lane, not only WSM6: startup warms
the selected scheme, the eqtset=2 stage terms and the post-physics EOS read
CM1's six mass rows, and all `numq` rows are advected, published to `qa`,
restarted and injected into a nest by name. Number concentrations never enter
the EOS, buoyancy loading, pdefq or the dry-mass ledger (the CM1 citations are
in [the seam record](../src/cm1core/records/CM1-ARWEN-SEAM.md)). This is proven
on CPU host doubles. Morrison has also run on GPUs, single-domain only:

* **RTX 5090, 2026-09-16, stability only.** A 512x512x75 stretched
  (`stretch_z = 1`, 20-320 m) 100 m bubble storm with `sgs_model = 4`,
  `bbc = 3` and `ihail = 1` integrated 5400 s (10800 steps at 0.5 s, 12772 s
  wall) with every carried field finite; max w 50 m/s at 5400 s
  (the run's `receipt.json` and `metrics.log`). That tree
  predates the stretched-grid number-advection fix merged in `fbd129e`, so its
  four number concentrations did not advect: the run says the scheme stays
  stable in a storm, not that its trajectory is right.
* **RTX 4090, 2026-09-17, after the fix.** The stretched-profile device test
  carries all ten Morrison rows the way the uniform family does, and a
  256x256x75 360 s stretched storm (`quick-256-morrison.toml`) is
  word-identical to the pre-fix tree for every frame before the first nonzero
  number concentration, differs from that frame on, and is finite throughout
  (8 PASS, 0 FAIL, 1 SKIP; the verification bundle's
  `morrison/morrison.txt`).

No Morrison run inside a live nest has been made on any GPU, and there is no
differential against native CM1 `ptype=5`.

An unmeasured component is selectable, not hidden: its run receipt and
`gpuwm-cm1 doctor` both carry the token `GM1-UNVALIDATED-CAPABILITY` and the
list of what was not checked (`cm1core.component_validation`). All four have been dispatched once on a seeded column
(`evidence/micro/wsm6-dense-ice-ihail/column-component-smoke.json`); that is
what a binding runs, not what it computes correctly. Five further ArWen
schemes -- Kessler, Milbrandt-Yau, NSSL two-moment, WDM6 and P3 -- are compiled
inside the admitted estate and refused here by naming the single thing that
holds each (`cm1core.column_component_blockers`).

`physics.ihail` is CM1's own `&param2` selector and reaches whichever scheme
is bound: `0` gives a graupel-like dense ice category, `1` a hail-like one.
It is optional, and **an undeclared value now binds CM1's own shipped `1`**
for `wsm6-mp6` and `morrison-mp10` (`run/namelist.input:56`). This lane
executed `0` until that move, so a WSM6 run's words -- and its 10 cm
reflectivity -- changed with it; declare `physics.ihail = 0` to get the
previously recorded branch. Neither Thompson has a
dense-ice branch at all, and both refuse a declared value by name. What each setting
replaces, and the measured fall-speed consequence, is in
[the seam record](../src/cm1core/records/CM1-ARWEN-SEAM.md) and in
`evidence/micro/wsm6-dense-ice-ihail/ihail-differential.json`.
It is optional, and an undeclared value binds the branch recorded for the
selected component, because CM1 supplies no default of its own
(`src/param.F:1567-1580` stops the model outside `0..1`): `wsm6-mp6` binds `0`,
which is what this lane has always executed, and `morrison-mp10` binds `1`,
which is what CM1 ships `ptype=5` with in
`run/config_files/supercell/namelist.input:54-55`. A component with no
recorded default is refused by value. What each setting replaces, the
fall-speed consequence, and the GPU-free proof that the selected constants are
the ones the compiled kernel evaluates, are in
[the seam record](../src/cm1core/records/CM1-ARWEN-SEAM.md).
`physics.ndcnst` is CM1's `&param3` constant droplet concentration and is
consumed by `morrison-mp10` only.

Shipped documents:

* `examples/cartesian-supercell-morrison-hail.toml` -- CM1's own supercell
  deck physics: `ptype=5`, `ihail=1`, `ndcnst=250`, in the translating frame.
* The tornado- and nest-oriented decks now select `morrison-mp10`:
  `examples/cartesian-nested-storm-following-mesh.toml`,
  `examples/cartesian-tornado-lagrangian-analysis.toml` and the
  100 m parent decks with 11 m and 20 m (5 m lowest layer) children. A single-moment
  scheme fixes the rain intercept, so downdraft evaporation over-cools the cold
  pool, and cold-pool strength controls tornadogenesis; double-moment is the
  tornado-research standard. WSM6 validation and evidence decks are unchanged,
  and so are the decks that declare `output_dbz` (Morrison has no admitted
  reflectivity formula here yet). **GPU run of these decks: pending** --
  Morrison has run single-domain on GPUs (above), never inside a nest.
* `examples/cartesian-supercell-thompson.toml` -- the same storm on CM1's
  `ptype=3`.
* `examples/cartesian-supercell-thompson-aerosol.toml` -- the same storm on a
  scheme CM1 does not have, on ArWen's own fallback aerosol initial state.

## Execution

Use the versioned capacity overlay on `PYTHONPATH` and the built Rust library
with the grid and Rayleigh-profile entry points. The selected GPU is checked
under both its UUID lock and the existing GM1 directory mutex.

```sh
CUDA_VISIBLE_DEVICES="$GPU_UUID" \
PYTHONPATH="$CAPACITY_OVERLAY:$GM1_ROOT/src" \
python tools/run_cartesian_gm1.py \
  --nx 400 --ny 400 --nz 237 --dx 125 --dy 125 --dz 125 \
  --dt 0.5 --nsound 6 --tile-size 64 \
  --warmup-steps 1 --steps 3 \
  --library "$CM1_LIBRARY" --sounding "$SOUNDING_40KM" \
  --gpu-uuid "$GPU_UUID" --lock "$GM1_DIRECTORY_LOCK" \
  --capacity-profile gm1.arwen-wsm6-column-capacity-256/v1 \
  --output "$NEW_RECEIPT"
```

The output must be a new path. `--plan` prints the CPU-side allocation plan.
Initialization and the component's real driver backing precede the measured
model budget. The budget also reserves tile-dependent WSM6 scratch and a
128 MiB margin. Physical wind views share their owned padded arrays;
immutable geometric metrics share storage only after raw-word source proofs.
Temporary acoustic and scalar workspaces are released between phases.

The timing covers synchronized complete steps. Initialization, compilation
warmup, final field scans, rendering, and output persistence are excluded.
VRAM is sampled at phase boundaries, so the reported peak is a sampled peak.
Each successful receipt identifies the input sounding, Rust binary, loaded
Python source hashes, kernel/operator receipts, step times, and field checks.

## Rotation of the base state: `dynamics.lspgrad`

With `dynamics.coriolis = 1` alone, CM1's f-plane term
(`src/solve2.F:639-653`, `fcor x (V + Vmove)`) acts on the whole wind, and the
base-state hodograph turns inertially because nothing balances it. CM1 r21.1
measured that in a no-storm 8x8x115 periodic column (semi-slip drag,
f = 1e-4 1/s, frame (8.5, 0.5) m/s): 0-1 km storm-relative helicity went from
232.4 to 171.9 m2/s2 in 5400 s. `dynamics.lspgrad = 1` is CM1's
`src/solve1.F:590-612`: right after the Rayleigh assignment it subtracts
`fcor x (V0 + Vmove)` from the held u/v tendency, so rotation acts on `V - V0`
only; the same column then kept 224.5 m2/s2, and the remaining drift is the
surface drag.

- Absent is CM1's own `lspgrad=0` and is not projected, so no existing
  document's `sha256` moves. `0` and `1` are admitted; `2`, `3` and `4` are
  refused by name (`CM1-CFG-LSPGRAD-UNPORTED`), because each takes its forcing
  from ug/vg, gradient-wind or ulspg/vlspg profiles this lane's closure does
  not carry.
- `1` needs `coriolis = 1` with `beta_plane = 0`
  (`CM1-CFG-LSPGRAD-WITHOUT-FPLANE-CORIOLIS`) and is refused beside
  `closures.do_adapt_move` (`CM1-CFG-LSPGRAD-ADAPTIVE-MOVE`), which rebases
  `u0`/`v0` under a forcing formed once.
- The closure forms two per-level words from its own compact, frame-relative
  `u0`/`v0` once, and refuses them unless they are raw-word identical to the
  gfortran-proven CPU operator. One kernel then applies them on every tile, so
  the held tendency is `((Rayleigh - ul) + stress)`, CM1's accumulation order.
  A nest child inherits the selection and forms its words on its own levels.
- A momentum budget publishes the term as `ub_lspg`/`vb_lspg`, a GM1
  extension: CM1 r21.1's 3-D budget has no slot for it, so its own u/v budget
  does not sum (its domain-average `ub_lspg`, domaindiag.F:7466-7480, is the
  `ulspg` profile, which lspgrad=1 leaves at zero).
- Measured against CM1 on an RTX 5090 (2026-09-22),
  with `examples/oracle/balance-column-lspgrad1-cd005.differential.toml`
  (an archived column with a drag coefficient `cnstcd = 0.005` and `ramp_sgs = 0`)
  and its free-slip twin. **What passes:** the lspgrad switch response. At
  600 s, `V(lspgrad=1) - V(lspgrad=0)` of the domain-mean wind agrees with
  CM1's to 0.0057 m/s between 3 and 12 km and 0.028 m/s between 1 and 3 km,
  against a CM1 response of 2.28 and 0.98 m/s there (0.25 % and 2.8 %), and
  the kernel is word-for-word the CPU operator on the device
  (`tests/test_cuda_gm1_solve1_lspgrad1_closure_gpu.py`). **What fails, and
  not because of lspgrad:** the 0-1 km SRH tolerance. In this laminar,
  horizontally uniform periodic column CM1 keeps every column bit-identical
  and its subgrid TKE identically zero aloft (free-slip: TKE 0 and SRH 232.5
  m2/s2 at every output to 5400 s). GM1's columns drift apart from roundoff
  (horizontal standard deviation of u about 5e-6 m/s at 20 s, growing about
  e-fold per minute), it spins subgrid TKE up from zero (0.5 m2/s2 by 600 s
  with a free-slip bottom) and the resolved flow then overturns: the
  free-slip column stops at about 1500 s on an EOS refusal, and the drag
  column's 0-1 km SRH is 4.6 m2/s2 below CM1's at 600 s and 421 m2/s2 below
  at 5400 s. The same happens with `lspgrad = 0` and without a moving frame.
  **Where the asymmetry comes from** (phase-by-phase probe of the 8x8 and
  16x16 free-slip column on the 5090): every physical rectangle is uniform
  until step 1, RK stage 1, scalar stage, where `th3d` first differs, on
  exactly the outer three-cell ring. The scalar advection tendency kernels
  (`cm1_adv5_scalar_tendency_v1` in the uniform, terrain-stretched and flat
  stretched images, and `cm1_rk_pdef_total_v1` on the final stage) write only
  `3 <= i < ni-3, 3 <= j < nj-3`, so theta and every moisture row get no
  advection at all in the ring, on every lateral family (measured on open
  laterals too: zero nonzero ring words, interior active). CM1's `advs`
  advects every cell `i = 1..ni` (adv.F), with its open-side inflow edits.
  Next, at RK stage 2, the held u/v tendencies differ on the boundary faces:
  `cm1_mom5_advu_v1`/`advv_v1` advanced only faces `2..ni` and applied the
  open-side inflow corrections whatever the lateral family, where CM1 under
  `wbc = 1` (`ibw = 0`, param.F:6678) advects faces 1 and `ni+1` and applies
  none (adv.F `advu` `i1`/`i2`, `doitw`). Both were lateral-boundary defects
  of the advection, not of the closure and not of lspgrad. The ring was
  fixed first and the faces after it: with both, the free-slip
  column's 64 columns stay bit-identical through 1800 s on the RTX 3080
  (horizontal standard deviation exactly 0 at every level and 60 s frame, no
  EOS refusal), where with the ring fix alone it grows from 4e-5 m/s at 60 s
  to 101 m/s at 1440 s and stops. The uniform (`stretch_z = 0`) lane also
  needed its acoustic step to advance the periodic boundary faces before the
  column pass forms the divergence (sound.F:386-406 before :446-530); a
  12x12x115 uniform-lane column at 100 m stopped before 19 s without that and
  stays bit-identical to 1800 s with it. A uniform column cannot see how the
  boundary face itself is advanced: the flat (stretched) lane's acoustic step
  gave a periodic face the open side's radiation update instead of the
  pressure gradient until it took the family (sound.F:274-302,
  param.F:9182-9219), which a bubble beside a periodic corner shows. The
  SGS closure's lateral edges are still the open ones on every family
  (evidence/oracle/periodic-momentum-faces). The column still drifts from CM1's: 0-1 km
  SRH 229.1 -> 226.7 m2/s2 over 600-1800 s against CM1's 232.5, with a
  horizontally uniform TKE (0.88 m2/s2 at 60 s, decaying) where CM1's stays
  zero. That TKE matches the fresh-run step-1 turbsmag start
  (`cuda_cartesian_closure.py:1034-1049`); this run does not
  separate it from other causes. CM1 alone measures the drift at that drag: 0-1 km
  SRH 232.5 -> 215.9 m2/s2 over 5400 s at `cnstcd = 0.005` (7.1 %), and
  232.5 -> 217.0 with `ramp_sgs = 0`.
- Also on the 5090 (in review of the lspgrad = 1 port): in a nested run (120x120x40
  900 m parent, 9:1 child, `coriolis = 1`, `lspgrad = 1`, frame (12.5, 3)
  m/s) the parent's and the child's u and v at 10 km stay at the base state
  word for word through 10 child steps, while with `lspgrad = 0` the Coriolis
  term turns them; the child forms the same per-level words as the parent. On
  the column with `output_ubudget`/`output_vbudget`, `ub_lspg`/`vb_lspg` are
  horizontally uniform and equal `-f(V0+Vmove)`/`+f(U0+Umove)` of the
  sounding to 1.2e-8 m/s2. A 20 s run stopped at 10 s and resumed writes a
  bit-identical restart envelope and bit-identical prognostic and budget
  carriers (`ub_lspg` included); only the surface-swath extrema (`sps`,
  `sus`, `sws2`/`sps2`/`sus2`) differ after the resume, which CM1 carries in
  its restart file (restart_write.F:696-750) -- a separate, pre-existing gap,
  closed on 2026-09-23: the parent restart now carries the
  swaths (section "Checkpoint and restart", "The swaths").
- At the time of that comparison GM1 ran no SGS startup ramp, so the CM1
  reference was run with `ramp_sgs = 0`. `closures.ramp_sgs`/`closures.ramp_time`
  now select CM1's ramp (src/turb.F:240-262, `cuda_cartesian_sgs_ramp`); an
  undeclared pair is CM1's compiled `ramp_sgs = 0` (src/cm1.F:355-356).

## Latent heating at eqtset=2

CM1's eqtset=2 microphysics finish rebuilds pressure and theta at fixed density
(src/mp_driver.F:603-617), which is right only for constant-volume heating --
what CM1's own Morrison and Thompson produce under eqtset=2. ArWen's schemes
heat at constant pressure, so `cuda_cartesian_moist_heating` converts every
component's step to CM1's constant-volume form (and runs CM1's `satadj2` after
Morrison) before the rebuild. Before 2026-09-23 the lane kept about 0.78 of
every phase change's latent heat and CM1's own supercell deck grew a storm
about 40 % weaker than CM1's (evidence/oracle/supercell-fidelity). A fresh
start also no longer seeds its first-step TKE with turbsmag, which CM1 does
only for `interp_on_restart` (src/init3d.F:2688-2760).

## The lower boundary

This lane ran with CM1's free-slip rigid bottom, `bbc=1`, which is what CM1's
own supercell deck uses (`run/namelist.input:70`). `sfcstress`
(`src/turb.F:4536-4545`) copies `t13`/`t23` from `k=2` down to `k=1` there, so
the lowest row of the implicit vertical momentum solve
(`src/turbtend.F:1454`, `src/turbtend.F:2224`) sees no surface stress at all
and the ground generates no horizontal vorticity. That is the right deck for a
mid-level mesocyclone and the wrong one for a vortex that has to intensify at
the ground.

Five keys in `[closures]`, under CM1's own namelist names, select the other
two lower boundaries: `bbc`, `sfcmodel`, `isfcflx`, `cecd` and `cnstcd`. They
are optional as a group and required as a group -- declare all five or none.
A document that declares none of them is the free-slip deck and is
byte-identical to what it was before the keys existed; a document that
declares some of them is refused by the first missing key
(`CM1-CFG-LOWER-BOUNDARY-PARTIAL`), because a half-declared surface would take
the rest from this port's default and integrate a boundary the document never
named.

- `bbc` -- 1 free slip, 2 no slip, 3 semi slip (`README.namelist:574-580`).
  `boundaries.bottom` is the same CM1 selector under this port's earlier
  spelling and must agree with it.
- `sfcmodel` -- 0 or 1. CM1 stops when `bbc=3` has no surface model
  (`src/param.F:2073-2088`) because the semi-slip stress is `ust**2` times the
  surface wind direction and only a surface layer computes `ust`.
- `isfcflx` -- 0 or 1. With 1 the surface exchanges heat and moisture with the
  air above it; see "The surface layer" below. It requires `sfcmodel=1`,
  because `src/turb.F:645-649` calls `sfcflux` only inside that block and the
  coefficients it multiplies are written there.
- `cecd` and `cnstcd` -- 1 is the constant drag coefficient branch of
  `src/sfcphys.F:120-136`, which is what idealized surface-drag studies use;
  2 and 3 make the drag a function of the 10 m wind over water.
  `cnstcd` fixes the roughness length `znt = 10/(exp(karman/sqrt(Cd)) - 1)`,
  and `ust` follows the lowest model level wind through the log layer. CM1
  imposes no upper bound on `cnstcd` and neither does this port: a `Cd` above
  one simply puts the roughness length above the 10 m reference height, and
  the log-layer wind ratio stays strictly positive because both logarithms
  still take arguments above one. What is refused is the range where the
  roughness length stops existing in binary32 -- below `2.0325873e-05`, where
  `exp(karman/sqrt(Cd))` overflows and `znt` is exactly zero, and at or above
  `4.5036004e+13`, where `karman/sqrt(Cd)` drops below `2**-24`, `exp` of it
  rounds to exactly `1.0` and `znt` is `10/0`.

Every wind that reaches the surface is ground-relative. `src/turb.F:296-307`
forms `ua+umove`, `va+vmove` before any surface quantity is computed. In the
translating frame this lane runs in, applying drag to the carried
frame-relative wind instead would drag the wrong wind, so the frame velocity
is part of the surface operator's selection and is refused rather than
defaulted.

With `bbc=3` the lowest model level also stops taking its dissipative heating
from the interior `dissten` average: `src/solve1.F:932-935` moves the interior
loop's first index to `k=2` and `src/solve1.F:950-988` heats `k=1` with the
surface-layer rate `ust**3*log((dz1+znt)/znt)/(karman*dz1)` instead, so the
kinetic energy the surface stress removes comes back as heat at the level that
lost it.

`examples/cartesian-supercell-surface-drag.toml` is the translating-frame
supercell deck with the semi-slip boundary selected.

### Near-wall subgrid models for tornado LES

`closures.sgs_model` selects CM1's LES closure (`README.namelist:334-352`).
This lane integrates three of r21.1's seven values: 1, the Deardorff TKE
closure, and CM1's two two-part near-wall variants, 3 and 4. Smagorinsky (2)
and the NBA models (5, 6) are refused by name. The two-part models keep the TKE
closure unchanged and add a near-wall piece that is confined to
`gamwall = 1-0.5*(1+tanh((z-150 m)/25 m))` (`src/param.F:8869-8888`, CM1's
hard-coded constants), i.e. roughly the lowest 250 m. CM1 ramps them in
linearly between model times 1800 s and 3600 s (`src/turb.F:264-270`).

- `sgs_model = 4`, Bryan (2020), is the one to use for tornado LES on a
  semi-slip bottom. `t2pcode` (`src/turb.F:6078-6267`) sets a horizontally
  uniform mean-shear eddy viscosity. At each wall-layer level, that
  viscosity's flux divergence balances the domain-mean tendency of all other
  terms, divided by the log-law shear of the domain-mean `ust` and `znt`.
  `turbuz`/`turbvz` add that flux to the vertical stress tendency. With the
  Deardorff closure alone, the lowest few levels of a 20 m or 5 m stretched
  grid carry too little subgrid stress, so the resolved near-ground wind comes
  out over-sheared (the LES log-layer mismatch). That profile sets the
  near-ground inflow a vortex tilts and stretches. Model 4 requires `bbc = 3`
  and `sfcmodel = 1`, and `sfcmodel=1` imposes a neutral surface layer
  (`src/sfcphys.F:529-535`).
- `sgs_model = 3`, Sullivan et al. (1994), in r21.1 only scales `kmh`/`kmv`
  by the isotropy factor `gamk` (`src/turb.F:1612-1628`). Its mean-shear
  viscosity is computed but never copied into the array `turbuz` reads. Its
  restart writing is refused, because this lane's restart carries no `gamk`
  profile.
- `closures.t2p_avg` is optional and may only be declared with 3 or 4. Only
  CM1's domain average (1) is ported. `t2p_avg = 2` would need CM1's running
  time averages and is refused.

The domain means are CM1's serial binary64 sums, one device thread per level,
so they match the Fortran order word for word. In a heterogeneous storm domain
they are means over the whole domain, environment included. A live nest child
inherits the selection and averages over its own window only. CM1 has no
nests, so that window is this port's choice; nothing here measures how the
window changes the profile.
`examples/cartesian-stretched-tornado-two-part-sgs.toml` is the stretched
tornado-grade grid with `sgs_model = 4`.

#### When model 4's `kmw` is zero

After the ramp has started, `kmw` can be exactly zero at one wall-layer face or
at all of them. That is a legitimate CM1 r21.1 outcome: `t2pcode` clips the
profile itself, and for identical inputs GM1 computes CM1's words, zeros
included (see *What is shown* below). It does not by itself explain why a
particular run's profile is zero at a particular time.

`t2pcode` sets each wall-layer face `k+1` from level `k`
(`src/turb.F:6238-6245`) from three terms:

- `d2pm1`: the dot product of the level's mean wind `(uavg, vavg)` with the
  mean stress terms `(t13avg, t23avg)`, over the mean speed `savg`;
- `d2pm3`: the dot product of the mean wind with the two-part flux
  `(ufw, vfw)` at level `k`, times `rfavg*rdz*mh(k)`, over `savg`;
- `tem`: `t2pfac*gamwall(k+1)` over the product of `rdz*mh(k)`, the log-law
  shear `dsdz` (floored at `1e-12`) and `rfavg`.

`kmw(k+1)` is then `tem` times `(0 - d2pm1) + d2pm3`, floored at zero.

- `d2pm1` is the component of the domain-mean tendency at level `k` along the
  domain-mean ground-relative wind `<U>(k)`. It counts every term except the
  two-part flux, over the step just finished: `src/solve3.F:342-352`
  subtracts that step's own two-part tendency from `(u3d-ua)/dt`. It is
  negative where the other terms slow the mean wind along its own direction
  (surface drag does) and non-negative where they do not.
- `-d2pm3` is `t2pcode`'s estimate of the along-wind tendency the two-part
  flux through the face below gives level `k`. The dot2p block adds
  `rdz*mh*(dum2(k+1)-dum2(k))` with `dum2(k) = rf*ufw(k)`
  (`src/turbtend.F:1771-1802`), so the lower face contributes
  `-rf*ufw(k)*rdz*mh/rho`; `d2pm3` takes the density from the upper face,
  `rfavg = <rf(k+1)*rr(k)>`. `d2pm3` is zero at `k = 1`, where `ufw(1) = 0`,
  and above any zero face, because a zero `kmw(k)` makes `ufw(k)` and `vfw(k)`
  zero. Above a live face its sign is, up to rounding, the sign of
  `<U>(k).(<U>(k)-<U>(k-1))`, the along-wind component of the mean shear.
  Positive, the face below carries momentum down out of level `k`; negative,
  it brings momentum up into it. That component can be negative only where
  the speed drops with height, but a speed drop does not make it negative:
  `<U>(k-1) = (6,0)` under `<U>(k) = (0,5)` gives `+25`.
- `tem` is positive at every wall-layer level once `t2pfac >= 0.001`. The ramp,
  the `gamwall` taper, the log-law shear `dsdz` and the density scale `kmw`.
  None of them can change its sign.

So `kmw(k+1)` sizes the flux through face `k+1`, against the log-law shear, to
make up `t2pfac*gamwall(k+1)` of what the other terms and the face below take
from level `k` along the mean wind, `d2pm3 - d2pm1`. At full weight that would
hold the level's mean wind steady. B20's viscosity is never negative: where the
other terms and the face below take nothing net, making it up would need a flux
against the shear, and CM1 sets zero instead.

**The rule.** Because `tem > 0`, `kmw(k+1) = 0` exactly when
`(0.0-d2pm1)+d2pm3 <= 0`, that is, exactly when `d2pm3 <= d2pm1`. Every
pattern of zeros is this one rule, applied face by face from the ground up:

- **Every face is zero** exactly when `d2pm1 >= 0` at every wall-layer level
  `k = 1..ntwk-1`. At `k = 1` no flux arrives from below, so `kmw(2) = 0`
  exactly when `d2pm1(1) >= 0`. A zero face passes no flux, so each face above
  it meets the same test. `ufw`, `vfw` and the two-part tendency are then
  zero as well: CM1 has switched the near-wall model off for that step.
- **A zero face above a live face** (`kmw(k) > 0`, `kmw(k+1) = 0`) has one of
  two causes:
  - `d2pm1(k) >= 0` and `d2pm3(k) <= d2pm1(k)`: the other terms do not slow
    level `k`, and they add at least what the face below carries down out of
    it. Under zero or negative along-wind shear (`d2pm3 <= 0`) this holds
    whenever `d2pm1 >= 0`; under positive shear, where the mean wind speeds up
    with height, it holds once `d2pm1 >= d2pm3 > 0`;
  - `d2pm1(k) < 0` and `d2pm3(k) <= d2pm1(k)`: the other terms slow level `k`,
    and the face below brings at least that much momentum up into it. That
    needs `d2pm3 < 0`, a negative along-wind shear component, as above a
    shallow outflow jet.
- **A zero face under live faces** (`kmw(k) = 0`, `kmw(k+1) > 0`): no flux
  arrives through the zero face, so `d2pm1(k) < 0` alone makes the face above
  it live again. A surge that speeds up only the lowest level zeroes `kmw(2)`
  and leaves the faces above it live.

In a storm domain `d2pm1` is everything else that changes the domain-mean wind
at that level: the resolved vertical momentum flux (downdrafts carrying faster
air down, for example), the subgrid stress and surface drag, and, on open
lateral boundaries, the momentum carried across them and the pressure force on
them.

For `t2p_avg = 1` nothing else masks the profile. The low-wind-speed zeroing at
`src/turb.F:6414-6419` belongs to `t2pcodetavg` (`t2p_avg = 2`, refused here).
The LES-subdomain mask at `:1522-1535` needs `cm1setup = 4`. Before 1801.8 s
(`t2pfac < 0.001`) there is no profile. The step a run resumes on integrates
no two-part flux either (the restart skip).

**Time.** `t2pfac` scales only `tem`, so the ramp can neither create nor clear
a zero. `d2pm1` is recomputed every step from the step before, with that
step's own two-part tendency removed, so a live profile does not keep itself
alive. A run's zeros can come and go between output times only because its
`d2pm1` (and, above a live face, `d2pm3`) changes.

**Diagnostics.** CM1 publishes `t2pm1 = -d2pm1*tem` and `t2pm3 = d2pm3*tem`
(`src/turb.F:6250-6251`, `src/domaindiag.F:4493-4531`). GM1 publishes neither
and keeps no `d2pm1`, `d2pm3` or `tem`; a B20 restart carries no two-part word,
and `output_km` is refused. Two CPU tools, both outside the runtime package
(so neither enters the runtime source identity a restart is checked against),
fill that gap:

- `tools/capture_two_part_reductions.py --at T ... --out DIR -- cartesian
  DECK.toml ...` runs the door in-process and, on the first step at or after
  each time `T`, saves the words `cm1_t2p_b20_profile_v1` read and wrote;
- `python3 tools/oracle/two_part_kmw_terms.py DIR/*.npz` replays them,
  prints `d2pm1`, `d2pm3`, `tem` and the branch of the rule above that decided
  each face, and says whether the replay reproduced the run's words. `d2pm1`
  does not depend on `logf`. A live face's value does, through `tem`, so a
  device run's live words can differ from a host replay by CUDA's `logf`
  alone.

**What is shown, and what is not.** `tests/test_two_part_kmw_clip.py` compiles
CM1's own `getgamk`/`t2pcode`, cut whole from the pristine `turb.F`, with the
Makefile's options at `-O2` and `-O0`. When `CM1_R21_1_NATIVE_BUILD` names a
native CM1 build, it also links that build's object code. It runs four
synthetic storm-like domains over the 20 m stretched column, each built to
exercise one branch of the rule:

- drag spin-down: every face is live;
- a cold pool with downdrafts mixing faster air down: every face is zero.
  These fields do not depend on time. They are zero at every ramp value
  tested, from 1900 s to 5400 s, which shows only that the ramp cannot clear
  a zero;
- a shallow outflow jet: a face above a live face is zero under negative
  along-wind shear;
- gust surges at two levels: the lowest face is zero under live faces, and a
  face above a live face is zero under positive shear.

On all four, GM1's CPU reference and the packaged CUDA text give CM1's words,
zeros included, and CM1's own `t2pm1`/`t2pm3` are the oracle's words. A
two-step chain shows that a live profile does not carry into a step whose
other terms clip it. Its two-part tendency and `solve3` subtraction are CM1's
statement text, not the whole compiled `turbuz`/`solve3`, and the rest of that
step is the test's own update, not GM1's runtime path.
`tests/test_two_part_kmw_capture.py` runs the real `CartesianClosure` and
`CartesianTwoPart` through the host stand-in, tiled and untiled, with live
faces and two kinds of zero, and holds the capture and replay to the runtime's
words. `tests/test_two_part_kmw_clip_gpu.py` is the opt-in device
check of the four domains. It uses no tolerance: device `logf` must lie within
3 units in the last place of the host's, and the live words must be the host
reference's words for the device's own `logf`.
`tests/test_two_part_kmw_clip_device_host.py` proves that check on the host
with a shifted `logf`.

None of this replays a GM1 run. By the rule, an all-zero profile after the
ramp, on a step that is not a resume step, means `d2pm1 >= 0` at every
wall-layer level, provided the device computed that step as the kernel text
does. For a particular run that is an inference until its reductions are
captured and replayed as above; how large its `d2pm1` was, at which levels,
and what in the storm set it are not known without that capture.

## Simulated radar reflectivity

`[physics] output_dbz` is CM1's own namelist word (`README.namelist:1366-1369`,
"reflectivity (dBZ)?"). It is off by default and optional per run. With it on,
every published history frame carries three more carriers under CM1's own
output names:

| carrier | shape | what it is | CM1 |
| --- | --- | --- | --- |
| `dbz` | `[nz, ny, nx]` | simulated 10 cm reflectivity | `writeout.F:4089-4107` |
| `cref` | `[ny, nx]` | composite (column-maximum) reflectivity | `writeout.F:1081-1091` from `misclibs.F:2259-2292` |
| `dbz_level` | `[ny, nx]` | one chosen plan-view level | no CM1 word; the level is named in metres and resolved by `misclibs.F:2454-2466` |

`dbz_level` appears only when `[physics] dbz_level_m` is declared; declaring
it with `output_dbz` off is refused, because the level would be named in the
document and never computed.

**What it costs.** Without it a frame carries twelve carriers and about
sixteen float32 words per gridpoint. `dbz` is exactly one more word per
gridpoint; `cref` and `dbz_level` are one word per *column* each, which is
`1/nz` of a word per gridpoint apiece. On the shipped 60x60x40 deck that is
1.05 words out of 16.16, or 6.5% of a frame. A run that does not declare
`output_dbz` publishes exactly the frame it always did, byte for byte, and
prices exactly the device memory it always did.

**Where it is computed.** On the device, inside the microphysics call, by the
scheme that owns the size distributions the formula depends on: WRF's
`refl10cm_wsm6` (`mp_wsm6.F90:2275-2444`) through the declared ArWen seam.
CM1 refuses `output_dbz` per `ptype` for exactly that reason
(`param.F:1583-1603`) and so does this port: a component with no admitted
formula is refused by value, naming the dBZ field the simulated hydrometeors
would never have scattered. The composite fold and the level slice are a
separate CM1-transcribed kernel (`cuda_composite_reflectivity.py`), verified
without a GPU by compiling its own packaged source text with `g++` and
comparing raw words against the host transcription
(`tools/oracle/composite_reflectivity_kernel_oracle.py`).

**When it is computed.** Only on the steps whose completed state is about to
be written. CM1 raises `getdbz` at the output cadence (`cm1.F:3214-3237`) and
leaves it down otherwise; nothing in the forecast reads reflectivity. A frame
whose reflectivity belongs to an earlier model time is refused rather than
published beside this step's winds.

**Reading it.** `storm_diagnostics` adds the column-maximum and chosen-level
maxima and the ground area above 20/40/50 dBZ to every diagnostic row, and
`tools/analyze_storm_run.py` reports peak composite reflectivity, when the
echo first reached the reporting threshold, and the largest convective-core
area. `cm1-render --product reflectivity` draws the composite and
`--product reflectivity-level --level K` one level, both on the standard NWS
dBZ scale; `gm1-view --science-layer reflectivity_dbz` uses the same scale.

`examples/cartesian-supercell-reflectivity.toml` is the translating-frame
supercell deck with reflectivity turned on.

## How the pre-WSM6 moisture repair is spread over the device

`runtime.pdefq_launch` selects how CM1's positive-definite repair
(`src/misclibs.F:1037-1117`, the hard-wired `pdscheme=1` branch) is laid out
across thread blocks. It is optional, and it selects an execution layout, not a
CM1 branch: both admitted values run the same arithmetic in the same order and
produce raw-word identical `q3d` and `asq`.

That is worth stating precisely, because it is what makes the key safe to leave
undeclared. `pdefq` orders exactly three things:

- `misclibs.F:1065-1081`, the four accumulations over `k` (`t1`, `a1`, `t2`,
  `a2`), which `misclibs.F:1065-1068` resets at every `i`, so they belong to one
  `(i,j)` column alone;
- `misclibs.F:1082`, which adds `a2` and then subtracts `a1` from the running
  `budj`. Fortran evaluates that left to right, `(budj+a2)-a1`, so it is a
  recurrence over `i`, not a sum, and cannot be regrouped;
- `misclibs.F:1086-1088`, which accumulates `budj` times `tem` into `asq`,
  south-to-north.

Nothing else is ordered. The `do i` loop carries only `budj`; a column reads
read-only carriers plus its own `q3d` column and writes only its own `q3d`
column. So the columns may be evaluated in any order, and both layouts keep the
two carries above serial.

- `row-serial` is the original transcription: one thread per physical row `j`,
  which walks `i` and `k` itself. Its parallelism is `nj` alone, so its cost
  scales as `ni*nk` and is independent of `nj`: a 448-row domain resolves the
  whole repair to four thread blocks whatever `ni` and `nk` are.
- `column-parallel` (the default) launches one thread per `(i,j)` column
  (`ni*nj` of them, indexed so a warp reads consecutive `i`) and then runs
  `misclibs.F:1082`'s carry as a separate ordered west-to-east scan over the
  `(a1,a2)` pairs the column kernel wrote. The same 448-row domain launches
  1,792 blocks.

The layout is recorded in every runtime receipt as `pdefq_launch`, with the
exact grids under `pdefq_launch_plan`, so a run says which one it used.
`tools/oracle/wsm6_pdefq_layout_oracle.py` compiles the packaged CUDA kernel
text with a host C++ compiler and IEEE-identical intrinsic substitutions and
checks both layouts against the source-order CPU oracle raw word for raw word,
in forward and reverse thread order; the mutexed device test in
`tests/test_cartesian_pdefq_launch_layout.py` does the same on real hardware.

## Closure tiles

The closure (`CartesianClosure.held_tendencies` and `tke_rk`) is the only
horizontally decomposed dynamics phase; every other dynamics phase already
runs on the whole domain. `runtime.tile_nx`/`tile_ny` choose its tile, and they
are **optional as a pair**:

- both declared (16, 32, 64 or 128): the tiled closure every document ran
  before the keys became optional. A document that declares them keeps its
  exact canonical bytes and configuration hash.
- both left out: the closure runs as **one whole-domain tile**, and the WSM6
  block follows it unless `physics_block_nx`/`_ny` are declared. The door
  prices the device peak (dynamics phase peak + closure profiles + surface
  state + reflectivity residents + the largest of the held-tendency tile, the
  TKE stage and one physics block, `cartesian_door.closure_decomposition_peak_bytes`)
  and admits the untiled closure only when it fits `runtime.memory_budget_mib`.
  Otherwise it refuses with `CM1-CARTESIAN-UNTILED-BUDGET`, naming the budget
  that would fit and the largest explicit tile that fits the declared one. It
  never falls back to a tile silently. The runtime record carries
  `tile_selection` for an undeclared tile and nothing new for a declared one.
- one side only, `0`, `"auto"` or any other extent: refused by name
  (`CM1-CFG-CARTESIAN-TILE-PAIR`, `CM1-CFG-UNSUPPORTED-VALUE`).

Why it matters: the host launches the closure's operations once per tile. On
the host stand-in (`tools/oracle/host_cupy.py`), one held-tendency tile costs
about 54 copies, 53 raw-kernel launches, 22 array operations and one stream
wait, and each of the three TKE stages costs 12 copies and one launch per tile,
so a large step pays roughly 170 CuPy launches per closure tile, plus one WSM6
call (component construction, staging copies, post-physics EOS and a registry
drain) per physics block. A 512 x 512 grid at 64-cell tiles runs 64 of each;
untiled runs one. The price is device memory: the untiled held-tendency
workspace is a whole-domain volume set (8.06 GiB at 512 x 512 x 75, against
155 MiB for a 64-cell tile), for a priced peak of 14.2 GiB instead of 6.3 GiB.

**Words.** The closure operators read at most three cells either side of the
cell they write; each tile gathers that halo from the resident state and
scatters only the cells it owns, so the decomposition is invisible to every
published word. `tools/oracle/cartesian_tiling_identity.py` proves it without a
device: it drives the real `CartesianClosure` (its own compile calls, tile
loop, gather/scatter and launchers) through a NumPy stand-in for CuPy whose
kernels are the packaged CUDA text compiled by g++, twice -- tiled and untiled
-- and compares every `held` and `tke` carrier after every phase over two large
steps. `tests/test_cartesian_untiled_closure.py` runs it for the default
supercell selection and for bbc=2, bbc=3 with sfcmodel/isfcflx, set_znt,
irdamp=2, hrdamp=1, rterm=1, lspgrad=1, sgsmodel=3/4 and Morrison: raw-word identical.
The WSM6 column kernel and the post-physics EOS kernel index one column only,
so the physics block is invisible to their words as well.

**The one exception: cecd=2/3.** CM1's wind-dependent water roughness
(`src/sfcphys.F:138-194`) iterates `znt` from the previous step's value. The
tiled closure writes each tile's advanced `znt` back into the resident carrier
before the next tile gathers its halo, so a halo column at a seam starts its
iteration from a roughness another tile already advanced, and the surface
stress on the seam face -- which averages that column's `ust` -- differs in the
lowest level at tile seams (pinned by
`test_the_water_roughness_iteration_is_the_one_seam_dependent_path`). The
untiled closure starts every column from the previous step, as CM1's own
column loop does. Declared-tile documents keep computing exactly what they
computed; a cecd=2/3 document that drops its tile keys computes the CM1-ordered
words instead.

Benchmarking on a device: `tools/benchmark_cartesian_untiled.py DOCUMENT
--workdir DIR --steps N` runs the document tiled and untiled through the timing
harness, one after the other, reports seconds per step and compares every
carrier of the published history arena word for word.

## How wide one WSM6 call is

`runtime.physics_block_nx` and `runtime.physics_block_ny` size the rectangle
one WSM6 call covers. They are optional; absent, the WSM6 block follows
`runtime.tile_nx`/`tile_ny`, which is the decomposition every document ran
before the keys existed.

Tying the two together was an accident of history, not a requirement. The
closure tile is sized by a stencil: it reads three cells either side and pays
for a halo. WSM6 is column physics: `src/mp_driver.F:1469-1479` is called once
per physical column inside CM1's own `i`/`j` sweep, and no column reads
another, so the block has no stencil and no halo. What its size decides is how
many times the per-call staging, launch and drain are paid, and how much
component scratch one call needs: a pure per-call-overhead against per-call
footprint trade, with nothing scientific in it.

At 448x448 with a 64x64 closure tile the step makes 49 WSM6 calls; at 224x224
it makes four. The per-call cost is independent of `nz`, so on a deep grid the
overhead is paid 49 times over regardless of how much work each call does.
Against that, one call's own scratch grows as the block area:
`cartesian_wsm6_scratch_reserve` prices it, the door refuses a block past
4096 cells on a side, and the runtime receipt records
`physics_block_shape_yx`, `physics_block_count` and
`physics_block_follows_closure_tile` so a run says which decomposition it used.

## The surface layer

`bbc=3` asks a surface layer for `cd`, `znt` and `ust`. CM1's
`getcecd` (`src/sfcphys.F:81-195`) has four branches and they are not
interchangeable; five more `[closures]` keys, optional and required as their
own group, select between them:

- `set_znt = 1` with `cnst_znt` -- impose the roughness length
  (`src/sfcphys.F:83-99`). `src/init_surface.F:246-254` writes it into every
  column after the land-use table, and `Cd = (karman/ln(10/z0))**2` follows.
  This is the branch a study specified by land cover wants: roughness length
  is what a surface has, and a drag coefficient is what a reference height
  implies. Note the expression takes `ln(10/z0)` and `ln(za/z0)` with no `z0`
  added to either height, which is *not* the `cecd=1` form.
- `set_ust = 1` with `cnst_ust` -- impose the friction velocity
  (`src/sfcphys.F:102-118`) and invert the roughness length out of the log law
  that would have produced it. `ust` is then not recomputed.
- `cecd = 1` with `cnstcd` -- the constant drag coefficient, as before.
- `cecd = 2` or `3` -- the drag depends on the 10 m wind over water and is
  iterated against the wind it predicts (`src/sfcphys.F:138-194`): 2 is
  Deacon's line, 3 is Fairall (2003) at low wind and Donelan (2004) at high
  wind, saturating at `2.4e-3` above 25 m/s. Over land both read the roughness
  length the land-use table left in `znt`. `src/sfcphys.F:189-191` stops CM1
  when the iteration has not converged in twenty passes; a kernel cannot stop
  a model, so it raises a flag the host reads at the same point in the step.

`src/sfcphys.F:81` tests `set_znt` first, then `set_ust`, then `cecd`. CM1
leaves its own check for both switches at once commented out
(`src/param.F:1311-1327`); this port refuses that case, because the declared
friction velocity would never reach the surface while the receipt named it.

With `isfcflx = 1` the same call also writes the enthalpy exchange
coefficients (`src/sfcphys.F:197-236`) -- `cnstce`, Deacon's line or Drennan's
constant over water, and `Ce = Cd` over land -- and `sfcflux`
(`src/sfcphys.F:341-361`) turns them into

    thflux = Ch * s10 * ( tsk/pi_sfc - th0 - tha )
    qvflux = Cq * s10 * ( rslf(psfc,tsk) - qv ) * mavail

where `psfc` is the three lowest mass pressures extrapolated to the ground
through the `src/param.F:9038-9040` Fornberg weights (`src/turb.F:292`).
`src/turbtend.F:588-596` puts them in as the lower boundary of the implicit
vertical scalar diffusion, for potential temperature and water vapour only:
`src/solve1.F:1208,1217` pass `iflux=1` for those two and
`src/solve1.F:1222,1233` pass `0` for every other scalar. With `isfcflx=0`
that boundary evaluates the same `M(temc,0.0f)` it always did, so a free-slip
or drag-only run is unchanged word for word.

## The initial surface

A `[surface]` table carries CM1's `param12` initial surface conditions under
CM1's own names: `initsfc`, `tsk0_k`, `tmn0_k`, `xland0`, `lu0`, `season` and
`oceanmodel`. Like the lower boundary it is optional as a table and required
as a whole; an absent table is CM1's own shipped deck
(`run/namelist.input:130-136`) and is not projected into the canonical
document, so every identity sealed before it existed is unchanged.

`src/init_surface.F` runs five parts in an order that matters:

1. the `initsfc` branches (`:67-145`) -- 1 constant, 2 the WRF sea breeze with
   land between 3/8 and 5/8 of the domain, 3 rough west and smooth east, 4 a
   coastline at the centre;
2. the soil slab and the horizontal surface length scale (`:152-203`);
3. `dosfcflx` (`:207-227`), which is what actually switches the scalar lower
   flux on;
4. `getlanduse` (`:239-243`, `:441-466`), which **overwrites** `znt`,
   `mavail`, `emiss`, `thc`, `albedo` and `xland` from `lu_index`;
5. the `set_znt` override (`:246-254`) and the reality checks (`:270-322`).

Step 4 is why `initsfc=3` does not do what its name promises. The land-use
table runs after the `initsfc` branches and assigns the roughness length of
`lu0` to every column, so the rough-west/smooth-east contrast never reaches a
CM1 run either. This port reproduces that rather than correcting it, and says
so per run: the history receipt carries
`initsfc3_roughness_overwritten_by_landuse`. `set_znt=1` is re-imposed after
the table, so that one does survive.

`LANDUSE.TBL` itself is packaged as `records/CM1-LANDUSE-TBL.json`, with the
bytes of CM1 r21.1's own `run/LANDUSE.TBL` pinned in the record, because a GPU
port has no working directory beside an executable to read it from.

`oceanmodel` admits only 1, the fixed sea-surface temperature `sfcmodel=1`
requires (`README.namelist:946`). `oceanmodel=2` is the Pollard mixed layer,
which makes the sea surface prognostic so it cools under the storm; no GM1
carrier evolves it, so it is refused by name rather than silently held fixed.

`examples/cartesian-surface-prescribed-roughness.toml` imposes a 0.05 m
roughness over grassland with no enthalpy flux;
`examples/cartesian-surface-enthalpy-flux-over-water.toml` runs a 29 C sea
surface with Fairall/Donelan drag and the enthalpy flux on.

## What the surface was measured against, and what it was not

Every surface term above is raw-word identical to CM1 r21.1's own statements
compiled by gfortran at `-O0` and `-O2` with `-ffp-contract=off`, on the
Fortran's own input words, for the host transcription *and* for the packaged
CUDA kernel text compiled for the host with IEEE-identical `_rn`
substitutions: all four `getcecd` branches, the enthalpy exchange
coefficients over land and water, `turb.F:292` `psfc`, `rslf`, the `sfcflux`
computed branch, and the `turbtend.F` implicit column with the surface flux
both on and off. `tools/oracle/surface_layer_harness.f90` and
`tools/oracle/turbsz_lower_flux_harness.f90` are those statements;
`tools/oracle/surface_layer_kernel_oracle.py` and
`tools/oracle/turbsz_lower_flux_kernel_oracle.py` run the packaged kernel
text against them.

The one kernel this work changed that was already in production -- the
implicit scalar diffusion -- was checked the other way round as well. Both
images, the one committed before this work and the one in the tree, were
compiled on an RTX 5070 Ti and run on the same inputs with the surface flux
switched off and its array deliberately filled with values the kernel must
never read: 1728 words, raw-word identical
(`evidence/validation/gm1-scalar-turbulence-free-slip-identity-v1/receipt.json`).
The free-slip lane's run-to-run bit reproducibility is unchanged.

What that does *not* establish, and what `gpuwm-cm1 doctor` and every run
receipt say by name:

- the device comparison covers one card and one grid size. On an RTX 5070 Ti
  (sm_120, `evidence/validation/gm1-surface-layer-device-v1/receipt.json`) the
  `cecd=1` constant-drag branch, the `set_znt=1` prescribed-roughness branch
  and the `isfcflx=1` enthalpy flux over them are **bit-identical** to the
  host words -- so on that card they are raw-word identical to CM1's own
  statements -- while `set_ust=1` and the wind-dependent water drag differ by
  at most **3 units in the last place** through CUDA's `expf` and `logf`,
  `cecd=3` being the largest because its convergence loop carries the
  difference round again. No other architecture, driver or CUDA runtime has
  been compared;
- no multi-step `cm1.exe`-versus-GM1 differential was run with a surface layer
  on, so the composition of drag, surface fluxes and a storm over many steps
  is unmeasured against the reference model;
- `src/sfcphys.F:439-447` accumulates `qvflux` into `qbudget(8)`; this port
  carries no such accumulator, so a moisture budget closed over an
  `isfcflx=1` run will not balance by the surface term;
- `src/turb.F:651-656` `sfcdiags` and the `hfx`/`qfx`/`t2`/`q2`/`br`/`zol`
  diagnostics are not ported and no output carries them.

## Grid refinement: the optional `[nest]` table

CM1 r21.1 has no grid nesting: no namelist word, no README section, no
`CHANGES` entry and no parent/child code in `src`; its only moving-mesh
capability is the whole-domain adaptive move of `src/azimavg.F:4376-4528`.
GM1's nest is a **declared extension**, recorded in full in
`src/cm1core/records/CM1-NEST-SEAM.md`.

A document declares one refined child grid by adding one table. Absent, the
run is exactly the single-domain run it was: the table is not projected into
the canonical document, so every sealed configuration identity is unchanged.

```toml
[nest]
refinement_ratio      = 9      # odd, >= 3; child dx = grid.dx_m / this
time_refinement_ratio = 9      # child dt = run.dt_s / this; that many child
                               # steps per parent step
parent_i0             = 45     # 0-based parent cell of the child's SW corner
parent_j0             = 45
parent_span_x         = 30     # whole parent cells the child covers
parent_span_y         = 30
spec_zone             = 1      # rows whose tendency the parent sets outright
relax_zone            = 4      # Davies ramp width, measured from the edge
feedback              = 1      # 0 one-way, 1 two-way
acoustic_substeps     = 6      # the child's own nsound, checked against CM1's
                               # dtsmall on the CHILD's spacing and step
follow_storm          = true   # re-centre on CM1's own pressure centre
follow_interval_s     = 270.0  # CM1's adapt_move_frq cadence
follow_max_speed_m_s  = 13.5   # fastest the nest may follow; whole parent
                               # cells per re-centring derived from dx
# follow_tracker      = "cm1-pressure-held-offset"  (default; see below)
# follow_on_edge      = "hold"   # wrf-surface-circulation only; absent is "stop"
# follow_hold_limit   = 4        # required with "hold": held samples before relocating
# follow_relocation_radius_m = 36800.0 # required with "hold"; at least
                               # (hold_limit + 1) * speed * interval + one child
                               # cell (18325 m here); twice that is recommended
```

With `follow_tracker = "wrf-surface-circulation"`, `follow_on_edge = "hold"`
keeps a nested job running through tracker samples out of reach -- no
cyclonic circulation in the window, the maximum on the search-window edge, or
a centroid beyond the search reach -- by keeping the previous centre, so the
nest holds its position, and through a move into the corral (cancelled per
axis, as WRF cancels it). The search reach is `follow_max_speed_m_s *
follow_interval_s` times the follow intervals since the last accepted centre,
and every `hold` window is one child cell wider than its reach (WRF truncates
its window edges to whole cells, which held a vortex 5 per cent under the
declared speed on a quarter of its samples; `stop` keeps WRF's edges). What
that guarantees, exactly: a vortex whose displacement from the last
*accepted* centre is within `held + 1` reaches is found again. It says
nothing about a vortex at or over the declared speed -- the widened window
grows no faster than the vortex -- so expect alternating held samples near
that speed and, over it, a relocation every `follow_hold_limit + 1`
intervals. After `follow_hold_limit` consecutive held samples the next one
that would be held relocates the centre to the strongest circulation within
`follow_relocation_radius_m` of it. That radius must reach one child cell
past the relocating sample's `(follow_hold_limit + 1)`-interval reach and
should be at least twice that: the smallest admitted radius has no margin (a
vortex 2.5 per cent over the declared speed stops the run at its first
relocation), and raising `follow_hold_limit` instead never catches a faster
vortex, it only enlarges the radius required. The nest moves to the
relocated centre one parent cell per parent step. The run stops by name only
when the child's lowest-level winds are non-finite across the window or
within one smoothing radius of its maximum (`CM1-CARTESIAN-NEST-FOLLOW-LOST`),
when a relocation finds no circulation inside its radius
(`CM1-CARTESIAN-NEST-FOLLOW-RELOCATION`; its remedy names the radius first),
or after
`follow_hold_limit` follow intervals of consecutive corral-cancelled parent
steps (`CM1-CARTESIAN-NEST-FOLLOW-CORRAL`). Every held sample and relocation
is listed with its parent step, window and reason in the receipt's
`storm_following` block (`src/cm1core/records/CM1-NEST-SEAM.md` 5.8). The
held-sample and corral counts cross a restart with the tracker centre
(`nest.follow.hold.*` in the child's restart, section 5.10), so a resumed nest
searches the reach the uninterrupted run would and relocates when it would.

`follow_smoothing_radius_m` has no default because no value derived from the
search reach works for every storm (measured in the same record): a box
narrower than the orbit of a sub-vortex lets the sub-vortex capture the
tracker (200 m against a sub-vortex 350 m out stopped a 20 m nest at 75 s),
and a box many times wider than a compact vortex's core cannot centre it (600
m on a 60 m core stopped at the first follow sample). To follow a tornado's
parent circulation through multiple-vortex phases, declare a radius of the
order of that circulation's core. `hold` is not a substitute: on the same
sub-vortex case with a 200 m box it keeps the run alive for ten minutes, but
the tracked centre stays on the sub-vortex, 219-272 m from the parent
circulation on every sample, where a 600 m box follows it within 51 m. Use
the radius to follow the storm and `hold` to survive the samples that still
fall out of reach.

`examples/cartesian-nested-storm-following-mesh.toml` is a complete worked
profile: a 100 m storm-following nest inside a 900 m supercell parent.

`gpuwm-cm1 cartesian <config> --output <receipt> --plan` prices the nest
without opening a device: the plan carries the child's geometry, the clock
CM1's own `dtsmall` derives for it, the child's full `RuntimeConfig`, the
Davies weights, the coupling schedule and a separate `nest_allocation` block.

Four things decide whether a nest is admitted, and all four are CM1's own
numbers:

- **`src/misclibs.F:857-868`**: the psolver=3 acoustic small step is
  `0.60*min(dx,dy)/360 s`, so refining the horizontal by `r` divides it by `r`.
- **`src/misclibs.F:883-885`**: refining time by the same `r` leaves
  `dbldt/dtsm` unchanged, so the child needs the **same** `nsound` the parent
  does. Refining space alone multiplies both the required `nsound` and the
  advective Courant number of `src/misclibs.F:2714` by `r`; the plan reports
  that multiplier by value.
- **`src/misclibs.F:835`** with `:887-911`: `max_nsound = 192` and the stop
  above it.
- **`src/cm1.F:744-777`**: the C-grid staggering that makes placing the child
  on whole parent cells a conservation requirement rather than a convenience.

`--integrate` runs a fixed nest concurrently with its parent on one device (validation and the still-refused cases: `src/cm1core/records/CM1-NEST-SEAM.md` 5.4-5.5):
after every parent step the child's boundary tables are rebuilt from the
parent's new state, the child takes `time_refinement_ratio` steps with the
Davies relaxation added once per step to its held tendencies and the specified
zone re-imposed inside every acoustic small step (WRF `solve_em.F`'s
placement), and `feedback = 1` writes the child's interior back. Child frames
are published under `<history-dir>/nest-child/` at the parent's history
cadence; the receipt's `nest_execution` block records the placement, the time
interpolation, the feedback and the WRF citations. A nested run writes a
restart for both domains at every restart time (`restart-NNNNNN/nest-child/`
beside the parent's) and `run.irst = 1` resumes both, the tracker's state,
its `hold` state and the tornado criterion's ledger included
(`src/cm1core/records/CM1-NEST-SEAM.md` section 5.10). What still refuses, by
name: `iconly`, an adaptive step, an adaptive frame and the beta plane
(section 5.4). A nest may be added to a resume of a single-domain restart: the
child is built from the restored parent.

**The child's own surface swaths.** When `[diagnostics]` selects any of
`output_sws`..`output_shs`, the child keeps the same swaths on its own grid
(`nest_live.child_swath_selection`; `output_uh/vort/pv` and `tornado_report`
stay the parent's), folded at the end of every child step
(`solve3.F:855-1136`) and published in each child frame as `sws`, `svs`, `sps`
... . They are frame-relative only: no ground-fixed `sws2` twin, because the
child's lattice moves inside the parent as well as with the frame; each child
frame's placement document (`nest-child/placements/`) puts the plane on the
ground. A nest move shifts the swath with the child, so the words over ground
the child still covers are unchanged, and the exposed strip takes the parent's
swath of the same name through the same capture as every other exposed row
(`nest_move`, THE SHIFT AND THE FILL). The child restart carries the planes
(`nest.swath.*`, `restart_write.F:692-757`) and a resume continues them word for
word. Since 2026-09-23 the parent's own swaths are in the parent restart
too (the `swath` group, section "Checkpoint and restart"), so after a resume
the parent's swath -- and any strip a later move exposes in the child, which
takes the parent's words -- continues from the start of the run as well.

### Inspecting a nest window before spending the GPU on it

A `[nest]` table costs GPU hours to get wrong and nothing to check. The window
can sit off the storm, the Davies relaxation zone can cut through the updraft
and the ratio can price a child that does not fit on the card, and all three
are visible from a parent frame that has already been written.

```
python3 tools/nest_window_preview.py CONFIG.toml FRAME_DIR OUT_DIR \
    [--carrier w3d,th3d] [--level-index K]
```

reads the same document a run would read, captures one published parent frame
onto the child mesh its `[nest]` table declares, and writes both the parent's
own view of the covered region and the child's, as raw `f32` planes with a
manifest. The operator is the nest's own
`nest_boundary.interpolate_parent_to_child`, so what the preview shows is what
the child's boundary rows would actually be handed.

What it is **not** is a forecast. Interpolation adds no information: a child
plane carries the parent's structure on a finer mesh and nothing finer, and the
manifest says so under `claim_limit` so a plane cannot be lifted out of context
and read as a refined forecast. What refinement buys is what the child's own
dynamics produce from that state, and that needs the child to run.

`cm1core.nest_downscale` is the operator behind it. The nest's reference
implementation is a pure-Python nest of lists at roughly a microsecond per
point, which is hours per restart on a real grid; `downscale_level` is the same
arithmetic over NumPy, and because every step of the reference is elementwise
(no reduction whose summation order could change), the rewrite is bit-preserving
rather than merely close. `assert_matches_reference` compares `uint64` views
rather than values within a tolerance, and the tests hold it at ratios 3, 5, 7
and 9, on a flat field where the reference never forms its quotient at all, and
at a step function where the extremum limiter is the only thing doing any work.
`require_identical_vertical_decks` is what keeps the vertical a copy rather than
an operator: the nest refines the horizontal only, so a child of a `stretch_z=1`
parent carries a vertical deck whose `sha256` equals the parent's, and unequal
decks refuse by name instead of being silently interpolated onto.

## Checkpoint and restart

CM1 saves "all information that is carried from one timestep to the next in
the cm1rst* files" (`README.restart:12-15`). The Cartesian lane does the same,
into one `CM1FCP1` envelope per restart number:

```
gpuwm-cm1 cartesian <config> --output <receipt> --integrate \
    --history-dir <frames> --restart-dir <restarts>
```

`run.restart_interval_s` is CM1's `rstfrq` (`run/namelist.input:30`).
`src/cm1.F:3186` decides `dorestart` from the model time the step about to run
will reach, `src/cm1.F:4246` writes the file only after that step completed,
`src/cm1.F:4247-4252` advances `rsttim` by `rstfrq` past the new model time,
and `src/cm1.F:301`/`:4310` number the files from one upward. All four are
here, in that order, and a frame that falls on the same step is published
first, exactly as `cm1.F` writes output before `restart_write`.

Files are named `cm1rst_NNNNNN.cm1fcp` after `README.restart:33-35`, each with
a `cm1rst_NNNNNN.json` sidecar carrying the contract, artifact and inventory
hashes a resume is admitted against. Both are written into a dot-prefixed
staging directory and renamed into place only once both are on disk and
synchronized.

To resume, add CM1's own selectors:

```toml
[restart]
irst                 = 1      # README.namelist:227; param.F:718-719 clamps {0,1}
rstnum               = 3      # README.namelist:229-232; the file set to resume
restart_reset_frqtim = true   # README.namelist:1488-1510; cm1.F:2711-2714
```

**What is carried.** Not a maintained list: every mutable allocation of the
runtime's device-buffer registry, enumerated at the boundary. On the 32x32x40
example that is 65 carriers and 10,638,848 bytes -- the A and current
prognostic carriers, every padded halo, the four `radbc*` radiative-boundary
histories, the held tendencies, the prognostic TKE, the WSM6 persistence and
surface accumulators, and the `physics.asq` water budget. Beside them the
envelope carries the large-step clock, the `cm1.F:3512-3513` domain-location
integral, the dry-mass reference, the closure's held step and the
physics/solve3 counters.

**The swaths.** CM1 writes `sws`..`shs` into every restart and, when the
domain moves (`nrain = 2`), the translated `sws2`..`shs2`
(`restart_write.F:696-758`); `restart_read.F:1064-1126` reads them back. Since
restart profile `gm1.cartesian-restart/v2` (2026-09-23) the
parent restart carries the swaths a run keeps in its own `swath` group:
`swath.<name>` (the frame-relative plane), `swath.<name>2` (the ground-anchored
canvas, whose origin travels as the scalars `swath.ground.origin_x_cells` /
`origin_y_cells` and whose extent is the carrier's shape), the four per-step
series and the per-step vortex index the tornado report reads, and the
per-step domain location (binary64 bits as three exact 22-bit binary32 chunks,
because the envelope admits finite binary32 words only). The receipt's
`swaths` section and the profile's `swaths` layout say what was carried and
from which step (`first_step`; 1 is the whole run). The canvas and the series
are sized from the run's length, so a resume compares only the state part of
the profile (`state_profile_sha256`, which is the v1 profile word for word) and
applies the swaths by their layout when the resumed run opens them: a resume
planned longer or shorter than the run that wrote the restart re-places the
canvas by its recorded origin, and a folded word the new canvas would not
cover is refused, never dropped. A resumed run also sizes its swaths for the
run's own final step now, not for that step plus the steps already taken.
A swath the resumed run keeps and the restart does not carry -- a v1 restart,
or one written with that `output_s*s` word off -- starts at the restart step:
the resume prints `warning: swaths restart here: ...`, writes it as a
`GM1-SWATHS-RESTART-HERE` `warning` event under `--events`, and the frames'
`tornado_diagnostics.swaths.first_step` records where each swath began. A v1
restart still resumes this way when its execution identity matches; a restart
written by code older than restart profile v2 is refused on that identity in any case.
Measured on an RTX 3080: CM1's stock supercell,
resumed from its 900 s restart to 1800 s, publishes the uninterrupted run's
words in every carrier, swaths and ground canvases included.

**What is not carried.** Read-only source uploads and their aliases -- the base
state and the grid metrics. A resumed run rebuilds them from the same
initializer library, sounding and geometry, whose SHA-256 the artifact binds,
and the dry-mass reference derived from that rebuild is compared raw-word
against the captured one before a single carrier is written.

**What a second leg may change.** `README.restart:16-24` is explicit that a
user may change the namelist on a restart, so a resume is admitted against
`CM1Config.restart_identity()` rather than the whole sealed document: the run
clock, the `[restart]` and `[ensemble]` tables and the `[runtime]` execution
table are projected out, and everything else -- including `run.dt_s` -- must
match.

**What was measured.** A 600-step stop-and-resume on a 32x32x40 1 km grid is
bit-identical to the uninterrupted run: same restart arena SHA-256, same
whole-artifact SHA-256, same published frame bytes at 900 s and 1200 s.
`evidence/lifecycle/restart-bit-identical/` carries the documents, the
sidecars and the receipts.

Then the failure the blocker was actually stated as: the same run `kill -9`ed
on its whole process group mid-flight, with no handler and no cleanup. It left
two complete restart file sets and nothing else -- no half-written artifact, no
sidecar without its envelope -- and resuming from the second one reproduced
every later restart and every later frame byte for byte.
`evidence/lifecycle/restart-after-sigkill/` carries that one, including the
kill harness.

The unchecked cases are named in the run receipt under `restart.unvalidated`
and by `gpuwm-cm1 doctor`.

**CM1 restart words with no representation here, and why.**
`src/param.F:891-903` forces `iinit = 0` on a restart so the initializer never
re-fires into the restored atmosphere. This lane reaches the same end by a
different route: a resume builds the runtime exactly as a fresh run does --
warm bubble included -- and then overwrites every mutable registry carrier from
the artifact, so the bubble is among the words replaced. The bit-identity
result is what proves it; nothing initialization put in a mutable carrier
survives the restore.

`restart_format` and `restart_filetype` (`README.namelist:1459-1490`) select
between CM1's binary and NetCDF restart files and between one file and one per
rank. GM1 writes one `CM1FCP1` envelope and is neither, so a key admitting
either value would name a format the run does not produce.

`restart_use_theta` and the `restart_file_*` selectors
(`README.namelist:1490-1530`) choose which extra diagnostic fields CM1's
restart carries, because a CM1 restart carries a chosen subset. This one
carries the whole mutable registry unconditionally, so there is no subset to
select; `README.namelist:1531-1533` notes that none of them is required for a
bit-identical restart, which is the only thing claimed here.

`rstfrq = 0` is CM1's write-one-restart-at-`nstep=0`-and-stop branch
(`src/cm1.F:3110-3114`), the one its own comment marks as useful for ensemble
DA. This lane's capture boundary is the post-solve3 state of a completed large
step, and there is no completed large step at `nstep = 0`, so a zero cadence is
refused by value with that reason rather than silently accepted.


## Run events, run control, the t=0 frame and the frame envelope

Four `cartesian` flags for a program that watches or drives a run (the GUI,
a job-script sidecar). Each is off by default, and with all four at their
defaults the door does what it did before they existed. `--events`,
`--control-dir` and `--history-initial` need `--integrate`, and none of them
may be combined with `--plan`.

```
gpuwm-cm1 cartesian deck.toml --output receipt.json --integrate \
    --history-dir frames --restart-dir restarts \
    --events events.jsonl --control-dir control \
    --history-envelope off --history-initial
```

**`--events PATH`** writes a new file (an existing one is refused) with one
`gm1.run-events/v1` JSON object per line, flushed per line
(`cm1core/run_events.py`; schema `records/GM1-RUN-EVENTS-V1.schema.json`).
Every record has `schema`, `seq` (0, 1, 2, ... with no gap), `kind`,
`time_utc`, `elapsed_s` and `pid`. The kinds, in the order a run writes them:

| kind | when | fields |
|---|---|---|
| `admitted` | every host gate passed, before the device opens | config and its sha256, grid, microphysics, steps, dt, nsound, duration, history interval, frames and restarts expected, the directories, the receipt path, the control directory, the envelope and t=0 choices, host |
| `startup` | once, before the first large step | seconds per start-up stage, NVRTC compile seconds and program count, the GPU's name and UUID |
| `progress` | from the stepping thread at least every 5 s; `phase` is `waiting-for-gpu`, `starting`, `stepping` or `finishing` | step and total, model time and duration, s/step recent and mean, model-s per wall-s, ETA, the last I/O event, GPU telemetry (NVML when importable, else the device's memory counters) |
| `frame`, `restart` | once each is renamed into place | index or number, step, model time, directory, bytes on disk |
| `warning` | something done differently from what was asked | `code`, `message` |
| `refused` | a refusal | `code`, `gate`, `reason`, `remedy`, `option` |
| `end` | always last | `reason` (`completed`, `stopped`, `paused`, `courant-stop`, `refused`, `error`), `rc`, the receipt (or null), step, model time, frames and restarts published (frame 0 included), the signal, the control request, and a `message` (the error, or the `[nest.tornado_criterion]` rule that stopped the run: `stopped` with no request) |

`progress` and the stderr progress line are the same record
(`runtime_profiler.progress_fields`). A heartbeat thread, which reads nothing
but the log, writes a `progress` record with `heartbeat: true` and the last
known values whenever nothing has been written for 8 s (compiling, waiting for
the GPU mutex, joining the last publication), so no two records are more than
10 s apart while the process lives. A log with no `end` record whose last line
is older than 60 s belongs to a process that died without running Python
(SIGKILL, the OOM killer, a lost node). `python -m cm1core.run_events validate
events.jsonl` checks a log with the standard library alone.

**`--control-dir DIR`** is created if absent and refused if it already holds a
`stop` or `pause` file (`cm1core/run_control.py`). The run checks it once per
completed large step: one `stat()` of the directory, and a listing when its
modification time changed or at least every 5 s of wall time. The second rule
is for NFS, where a `stat()` can return the directory's cached attributes for up
to a minute (acdirmax) while opening it to list it revalidates them, so a stop
written from a login node is seen within about 5 s.

- `DIR/stop` ends the run at that step boundary. A publication in flight is a
  completed state and is finished, not discarded; an event that falls due at
  the step the stop is seen on is published first; nothing else is taken. The
  receipt is written with status `stopped-at-run-control-request` and a
  `runtime_benchmark.control_stop` block (when the file appeared, when it was
  seen, the step, the last restart, and the frames published and requested,
  frame 0 counted as the event log counts it), and the process exits 0.
- `DIR/pause` keeps stepping to the next scheduled restart and ends right
  after it is durable, status `paused-at-run-control-request`; `run.irst = 1`,
  `run.rstnum = N` and `--restart-from` continue it. A run with no restart left
  before its last step says so once (a `warning` record and a stderr line) and
  keeps running; a stop file still stops it.
- SIGTERM is the fallback for a process that is not reading the directory. With
  `--events` or `--control-dir` it is turned into the integration's own
  termination for the whole call, so a run stopped while it compiles still
  writes `end` (`stopped`, `rc` 143). During stepping it cancels the publication
  in flight and discards its staging, as it always has.

Measured on the stock supercell deck with WSM6 on the RTX 3080: a stop file
created while frame 4 and restart 1 (step 600) were being written was seen at
step 601, 0.06 s after the file appeared; both were finished, and the run ended
with five frames, one restart and no `.incomplete/`. SIGTERM at the same moment
exited 143 with frames 0-3 intact and frame 4 and restart 1 discarded.

**`--history-envelope off`** writes each frame as its raw arena and tables
only. The `history.cm1fhs` envelope is a second copy of the arena (443,189,440 B
arena and 443,198,452 B envelope at 256x256x80 Morrison), so `off` halves the
frame disk; `gpuwm-cm1 render` reads the envelope and needs `on`. Restarts are
unaffected, and with no restarts and no resume the publisher is not needed at
all. The receipt and the plan carry `history_envelope`.

**`--history-initial`** publishes the initialized state as `frame-000000`
before the first large step, as CM1 writes its `nstep = 0` output
(`cm1.F:3943`). It is taken like every other event, so it is reported first
and reaches disk in model order. A run that declares `output_dbz` publishes
that one frame without `dbz` and `cref`: this runtime computes reflectivity only
inside the microphysics of a step, so at t=0 the carriers hold their -1000 seed
(`cartesian_history.INITIAL_FRAME_REFLECTIVITY_OMITTED`, recorded in the frame's
`reflectivity` block). A resumed run has no initialized state, and the flag is
refused there.

**The field table names the arena relative to the frame.** Column 7 of
`history-field-table.tsv` is `history-carriers.f32`, not the absolute path the
arena had in `.incomplete/` while it was written, so a frame reads the same
after the rename and after it is copied elsewhere. `gm1-history-grads-export`
resolves the name against the frame directory; for a frame published before
this it reads the frame's own arena in place of the vanished staging path. It
names the ground-anchored swath canvases (`sws2`..`shs2`) under
`not_exported` in its receipt: no descriptor of the model grid can place them.
(The frame's publication record now states the canvas origin,
`tornado_diagnostics.swaths.ground_canvas_origin_cells_xy`; the exporter does
not read it yet.)

## Performance profiling

The per-phase host wall clock every receipt has always carried
(`phase_host_wall_seconds_all_steps`) is the time Python spent *enqueuing*
each phase. CuPy launches are asynchronous, so device work is booked to
whichever phase synchronizes next -- a WSM6 wait shows up as
`rk1.pre-acoustic` host time. It cannot say whether a run is limited by the
device, by the host launching kernels, or by I/O. `--profile` records the
three separately:

```
gpuwm-cm1 cartesian <config> --output run.json --integrate \
    --history-dir frames --restart-dir restarts \
    --profile summary --progress-interval 30
gpuwm-cm1 perf run.json
```

| `--profile` | What is recorded |
|---|---|
| `off` | Nothing. The runtime keeps `profiler = None` and takes exactly the code path it took before profiling existed; the receipt has no `performance` key and no progress lines are printed. |
| `summary` (default) | Per phase: device time from a pair of `cupy.cuda.Event` markers on the run's stream, host enqueue time, queue lag at the end of the phase, counted kernel launches, device-to-device, host-to-device and device-to-host copy counts and bytes. Per step: wall, end-of-step synchronization wait, host enqueue time. Every history frame and restart broken into stages with bytes and MB/s. Startup: the Rust initializer, runtime construction stages, every NVRTC program by name, and the first step against the steady mean. A live progress line on stderr. |
| `detailed` | Everything in `summary`, plus per-carrier device-to-host/write/hash rows for the last event of each I/O kind, and NVTX ranges nested step > RK stage > phase and I/O event > I/O sub-step (down to each transfer chunk's device read, write and digest wait), so an `nsys profile -t cuda,nvtx` trace is readable. |

**How device time is measured without adding a synchronization.** The two
markers of each phase are recorded on the stream (`Event.record`), which
enqueues no work that reads or writes a carrier and blocks nothing. They are
resolved with `get_elapsed_time` only after the step's existing
`stream.synchronize()` in `RuntimeClock.step` has returned, when every marker
recorded before it has executed. Immediately after each synchronization a
reference marker is recorded -- the queue is empty then, so the device reaches
it at once -- and the device offset of each phase's end marker from it, minus
the host offset, is that phase's *queue lag*. A phase's device time is the span
of the device timeline between its markers; it includes idle time when the host
is slower than the device, which is why the lag and the synchronization wait are
recorded beside it.

**Launch and copy counts** replace nothing CuPy constructs or returns.
CuPy's Cython code type-checks what its own factories hand back: CuPy 14.2
assigns every `Module.get_function` result to a `cdef function.Function`
(`cupy/_core/_kernel.pyx:113`, `cupy/_core/raw.pyx:129`), so the counting
wrapper the hooks once returned from a subclassed `cupy.cuda.function.Module`
made `cupy.full` raise `TypeError: Cannot convert _CountedKernel to
cupy.cuda.function.Function` on the first step of every profiled run. Kernel
handles are extension types whose `__call__` cannot be patched either. So:

- kernel launches are counted where Python code calls a
  `cupy.cuda.function.Function` or `cupy.RawKernel` object, through a
  `sys.monitoring` CALL event (Python 3.12+, tool id 2, 3 or 4). A call site
  whose callable is not a kernel disables its own event, so the callback keeps
  running only at sites that launch kernels; a site that has launched a kernel
  keeps counting. The instrumentation itself is process-wide, not only at
  launch sites: CALL events are set for the whole interpreter, every code
  object that runs while they are set has its CALL instructions instrumented,
  and every call site pays one callback before it disables itself. On CPython
  3.12 a disabled site is de-instrumented at once and re-specializes, so the
  steady cost is one callback per launch plus a one-time cost per site; on a
  call-dense pure-Python loop it is within timing noise, and the number that
  matters -- the profiled step wall against `--profile off` on the device --
  is what `tools/profile_cartesian_run.py --identity` reports as
  `profiling_overhead` (section below). The price is a known undercount: a
  site whose first callable is
  not a kernel and that launches one later is not counted, and the receipt
  lists that source as uncounted. `tools/oracle/profiler_call_site_identity.py`
  measures it on the real door (host-emulated kernels): the shipped counts
  against a run that never disables a site, phase by phase, and each run's
  hook count against the host stand-in's own tally of every kernel
  `__call__` and every `copyto` it executed (`host_cupy.COUNTS`), which
  raises no CALL event and so is an independent ground truth. Both hold on a
  storm-following nest with child and parent parcels and tornado diagnostics
  (24963 launches in phases; 25044 counted over the run = 16323 kernel
  launches + 8721 copyto calls, in both runs) and on a flat run with
  Eulerian and parcel vorticity budgets, circuits and high-frequency output
  (5610; 5622 = 2870 + 2752) (`tests/test_runtime_profiler_failsafe.py`).
  The comparison does see a loss: with a relay site injected in front of
  every host kernel it failed, 2748 launches against 5610 on the budgets
  deck. Disabled sites survive the tool id, so a
  later profiler in the same process calls `sys.monitoring.restart_events()`
  to count at them again; that call re-arms every tool's disabled events, so
  it is made only after an earlier profiler disabled a site, never by a run's
  first profiler. On Python 3.11, or when every tool id is taken, kernel
  launches are not counted and the receipt says so;
- `copyto`, `asnumpy` and `asarray`/`array` are replaced by forwarding
  functions that return exactly what CuPy's function returned. CuPy's own
  Python code calls `cupy.copyto` through the module too, so `cupy.full` and
  `cupy.full_like` (and `cupy.select`, `cupy.unwrap`) count one launch per
  fill;
- NVRTC compile time comes from wrapping `_NVRTCProgram.compile` in place on
  its pure-Python class, so every program and image is CuPy's own.

All of it is removed when the run ends. CuPy's own ufunc/elementwise launches
(`a + b`, `cupy.where`, `ndarray.fill`, `cupy.sum`) and `ndarray.get/set/item`
run inside Cython and are not counted; the receipt names both lists
(`counted_launch_sources`, `uncounted_launch_sources`).

**Profiling never stops a run.** Every instrument fails safe: a counting hook
(`kernel-launches`, `nvrtc-compile`, `copyto`, `asnumpy`, `asarray`, `array`),
the stream markers (`device-events`), `nvtx`, the `progress` line, or the
`phases`, `steps`, `io` and `startup` bookkeeping. An exception inside one
turns that instrument off for the rest of the run, puts back any CuPy attribute
it replaced, prints one `gm1 profiler: <instrument> turned off after <error>;
the run continues` line on stderr, and is listed in the receipt under
`performance.instrumentation.disabled`. Hooks that could not be installed are
listed under `instrumentation.unavailable`. When kernel launches were not
counted for the whole run (`instrumentation.kernel_launches_counted` false),
the verdict does not call a run `host-launch-bound` and says why. When the
`steps` or `phases` bookkeeping is turned off, `device-events` is turned off
with it: stream markers are recorded around phases and resolved only at the
end-of-step synchronization, so they would otherwise accumulate for the rest
of the run. Independently, at most 16384 marker pairs wait for a
synchronization; a pair past that is dropped and counted in
`unresolved_markers`. An error
CuPy itself raises propagates exactly as it would unprofiled. If the
performance record itself cannot be built, the receipt carries
`performance.record_error` instead of the numbers.

**The verdict.** `performance.verdict.line` is one sentence:

- `GPU-bound` when the end-of-step synchronization wait is at least 25% of the
  step: the device finishes the step's work well after the host has enqueued
  it, so device time is close to the wall step time.
- `host-launch-bound` when that wait is under 10% of the step and at least 100
  kernels are launched per step: the device keeps up with the host.
- `host-bound` for the same small wait with few counted launches, `mixed`
  between the thresholds.
- `I/O-heavy` is appended when frame, restart and field-scan I/O took more than
  20% of the integration's wall clock.

**The I/O table.** Each `history` and `restart` event records, as stages:
`diagnostics.*` (the `&param9`, budget and tornado fields computed for the
frame), `device_staging_enqueue`/`device_staging_sync` (history) or
`device_sync` (restart), `arena.host_window` (the page-locked transfer window),
`arena.d2h`, `arena.write`, `arena.sha256_busy` (both digests' own time) and
`arena.hash_wait` (how long the writer waited for them), `arena.fsync`,
`tables`, `publisher` (the `cm1-forecast-publish` process wall clock) with the
publisher's own `publisher.read_tables`, `publisher.allocate_payload`,
`publisher.read_arena_and_finite_check`, `publisher.hash.carriers_and_payload`,
`publisher.hash.artifact`, `publisher.write`, `publisher.fsync`,
`publisher.link` and `publisher.total`, then `receipt`, `location_record` and
`directory_fsync_rename`; a restart also records `envelope_verify` and
`arena.discard` and no `arena.fsync` (see below), an event committed in the
background records `background.join_wait`, and a restart committed beside a
frame of the same step records `background.turn_wait`. Every event carries `seconds`, the
whole publication wherever it ran, and `blocking_seconds`, what the stepping
thread paid for it (the snapshot plus any join wait); the run's
`io_blocking_seconds` is what the `I/O-heavy` verdict counts, and a background
event on the progress line reads `last io restart 18.4s 4.1 GiB (0.4s
blocking)`. The publisher prints its timings on stderr only when
`GPUWM_CM1_PUBLISH_TIMING=1` is set, which the profiled runtime sets for its
own subprocess; its stdout -- embedded in every publication receipt as
`rust_output` -- is the same bytes either way.

**The progress line** (`--progress-interval SECONDS`, default 30, `0` for none;
needs `summary` or `detailed`) reads

```
gm1 progress: step 4200/14400 t=2100s/7200s | 0.721 s/step recent, 0.718 mean | 0.69 model-s/wall-s | ETA 2h03m | last io restart 97.4s 8.2 GiB | GPU 99% 181W 21.3/24.0GiB
```

The GPU readings appear only when `pynvml` (`nvidia-ml-py`) is importable and
the device is found by PCI bus id; otherwise they are omitted silently. NVML is
never required.

**Overhead of `summary`.** Per large step: two marker records and one or two
elapsed-time reads per scheduled phase (30 phases including the post-step
hooks: about 120 driver calls, each a microsecond or two), about 40 us of
Python bookkeeping (measured on the host stand-in), and 0.17 us per counted
launch for the wrapper. On a 0.7 s step that is well under 0.1%.

**What it changes.** No simulated word and no published byte.
`tools/oracle/cartesian_profiling_identity.py` drives the real
`CudaCartesianRuntime.execute` and `CartesianClosure` through the host stand-in
at `off`, `summary` and `detailed` and compares every published carrier word
(`tests/test_runtime_profiler.py`), and
`tests/test_cartesian_restart.py::test_published_frames_and_restarts_are_the_pre_profiling_bytes`
publishes two frames and a restart at every level through today's writer and
publisher and compares every file with the digests the release head produced
before any of this existed (the one-copy restart directory, below, is the only
declared exception).

### Where a restart's minutes went, and what was removed

`tools/benchmark_publication_io.py` times the host half of a checkpoint on
synthetic host carriers (no device): the shipped arena writer, then the
shipped `cm1-forecast-publish`, with the publisher's peak resident set. On an
8.21 GB payload -- the size of a 512x512x75 WSM6 restart -- the release head's
publisher took 126 s and held 15.3 GiB resident. It read every carrier into a
byte buffer, decoded it into float32 words, copied those a word at a time into a
second payload-sized buffer while both stayed alive, and then hashed the payload
three times in series with a portable SHA-256 -- every carrier, the whole
payload, the whole artifact -- before writing a byte. The Python arena writer
digested every chunk twice on the writing thread and allocated a fresh pageable
host array for every device-to-host chunk. None of it was reported anywhere.

Now, with every published byte and receipt unchanged by these publisher changes:

- SHA-256 runs through the x86 SHA extensions when the processor has them
  (detected at run time; the portable compression remains and is tested
  against it on every length across several block edges).
- The per-carrier digests and the payload digest run in parallel, and the
  whole-artifact digest runs while the envelope is written instead of in a
  pass before it.
- The publisher reads each carrier from the arena straight into its window of
  the envelope payload -- the arena bytes already are the published words -- so
  nothing is decoded and re-encoded and one payload is resident, not two.
- The arena writer reads every chunk into one reusable page-locked window and
  digests it on two worker threads while it is written.

| synthetic checkpoint, same disk | release head | now |
|---|---|---|
| 8.21 GB: publisher wall clock | 125.7 s (65 MB/s) | 32.0 s (256 MB/s) |
| 8.21 GB: publisher peak resident | 15.30 GiB | 8.00 GiB |
| 8.21 GB: publisher user CPU | 68.2 s | 13.7 s |
| 8.21 GB: arena + publisher | 151.1 s | 57.4 s |
| 4.29 GB: publisher wall clock (3 / 2 runs) | 42.2-61.6 s | 17.4-19.7 s |
| 4.29 GB: publisher peak resident | 8.00 GiB | 4.37 GiB |

What is left is the disk. The arena writer was already disk-bound on that host
(6-14 s for 4.29 GB either way; the spread is the disk), and the envelope's own
write and fsync are most of the publisher's remaining time.

### Publication in the background, and one copy of a restart

The GPU no longer waits for any of that. At each output event `integrate`
runs the field scan on the stepping thread as before, then copies the event's
carriers once into a host arena (`forecast_arena.HostArena`: one per artifact
kind -- history, restart, nest child -- reserved page-locked when the run
starts and reused by every event) and hands everything else to one worker
thread (`cm1core.background_publication`): the arena file and its digests, the
tables, `cm1-forecast-publish`, the receipts, every fsync and the rename. Then
it goes back to stepping. What the snapshot reads from the runtime -- the
clock, the config record, the moving-frame integral, the reflectivity and
tornado records, a nest child's placement -- is captured with the words, so a
receipt never describes a later step than its carriers.

- **Queue depth one.** The previous event is joined before the next is taken
  and before `integrate` returns, so an arena is never read and written at
  once and frames and restarts reach disk in `cm1.F`'s order. Between steps the
  runtime also looks whether the worker has finished and settles it at once.
- **A frame and a restart of the same step are written beside each other.**
  Production decks put every restart on a frame step. Both are copied into
  their own arenas at the event, as one event, and the worker commits the
  restart on a second thread while it commits the frame, instead of after it,
  so the event is finished when its slower part is, not after the frame's
  write plus the restart's. The rename order, and what an event that
  succeeds or is cancelled leaves on disk, are unchanged: the restart renames
  into place only after the frame (and its nest child) has
  (`PublicationCancel.await_turn`), and a frame that fails stops the restart
  and discards its staging, so no restart ever names a next frame index whose
  frame was not written. Cancelling the event kills both publishers. The
  receipt's `publication.event_parts` is `beside` (`event_parts_note` spells
  it out), and the restart's I/O stages
  gain `background.turn_wait`, the seconds its finished restart waited for the
  frame's rename (excluded from its `seconds`). A join that waits for both
  parts is counted once in `blocking_seconds`: each second goes to the
  earliest part still committing then. Host memory holds both publishers'
  payload copies at once (about 1.4 GB plus 4.4 GB at 512x512x75) where it
  held the larger one, the disk takes both writes at once, and two
  publishers hash at once (each up to seven SHA-256 threads over its
  carriers plus one over the whole payload, `rust/cm1-state/src/forecast.rs`
  `available_parallelism().clamp(2, 8)`) beside the stepping thread.
- **What running out of disk or memory can cost, and `--event-parts
  in-turn`.** Which part fails is not always what the parts written one
  after the other would give. On a nearly full disk the restart's writes
  (its raw arena and its envelope, about 8.8 GB at the peak at 512x512x75)
  now land while the frame's do, so the frame can meet ENOSPC first; under
  memory pressure the OOM killer can pick the frame's publisher. The frame
  then fails, the restart is stopped and discarded with it, and neither is
  published, where the ordered writer would have published the frame and
  failed only the restart. `cm1core cartesian --integrate --event-parts
  in-turn` (`integrate(concurrent_event_parts=False)`,
  `BackgroundPublisher(concurrent_parts=False)`) writes the parts one after
  another with the same bytes and rename order; the event then costs the
  stepping thread the frame's write on top of the restart's again, and the
  receipt's `publication.event_parts` is `in-turn`. Use it when the disk
  has less free space than a frame plus a restart's arena and envelope
  (about 10 GB at 512x512x75, before the arena is discarded), when host
  memory is below the pinned arenas plus both publishers' payloads (the
  two arenas plus about 5.8 GB at 512x512x75, roughly 12 GB resident
  altogether), or on storage where the two writes contend so badly that the
  event's join wait grew rather than shrank (the benchmark below shows how
  to tell on one disk).
- **Nothing is reported before it is durable.** `progress`, the `frames` and
  `restarts` rows and `last_frame_publication`/`last_restart_publication` are
  filled only from a joined event, after the rename and the root's fsync.
- **Errors.** A worker exception is raised on the stepping thread at the first
  step boundary after it (at the latest, the next event's join); its staging
  directory is already gone and nothing was renamed into place.
- **Signals.** A failure or a first SIGINT while an event is in flight waits
  for it -- it is a completed state, and the synchronous writer would already
  have written it -- then re-raises with a note naming what was published.
  The wait is announced on stderr first, because it can take tens of seconds
  and an operator who sees nothing presses Ctrl-C again:
  `gm1: waiting for frame-000014, restart-000007 to finish publishing (Ctrl-C
  again cancels it and discards its staging)`. A second SIGINT, or SIGTERM
  (raised as `SystemExit(143)` while `integrate` runs, unless the process
  installed its own handler), cancels it and says so (`gm1: cancelling the
  publication of ...`, `gm1: SIGTERM: cancelling ...`): the arena writer stops
  within one 64 MiB window, the publisher's process group is killed and the
  staging directory is removed. A SIGTERM that arrives while an interrupted
  run waits for or cancels its publication is kept and raised once the worker
  has settled, so the process still exits 143. No worker outlives
  `integrate`. Completion is a flag the worker sets, not `Thread.join`: on
  CPython 3.12 a `join` interrupted by a signal marks the still-running thread
  stopped. A worker inside an uninterruptible system call (an fsync against a
  hung NFS server) holds the process until that call returns, whatever signal
  arrives; only SIGKILL ends it, as it would end the synchronous writer.
- **The publisher dies with the run.** It runs in its own session so a
  terminal Ctrl-C does not kill a checkpoint the run is still finishing, and
  so it no longer receives what is sent to the run's process group (a
  terminal hangup, `kill -- -PGID`). On Linux it is started behind a small
  exec wrapper that sets `PR_SET_PDEATHSIG` to SIGKILL, so it is killed when
  the run's process ends however it ends -- SIGKILL, a hangup, the OOM killer
  -- instead of writing on, orphaned, into staging.
- **Taken is published.** Words already copied off the device are published
  even when the event cannot be handed to the worker: when the nest child's
  snapshot fails after its parent frame's (the parent frame is published, as
  the one-call writer renamed it into place before it tried the child), or
  when the worker thread cannot be started. Both are then published on the
  stepping thread and the error is re-raised with a note. SIGTERM discards
  them instead, as it cancels an event in flight.
- **Durability window.** Stepping continues while restart N is written. A
  crash that ends the process outright in that window -- SIGKILL, the OOM
  killer, a segfault, an Xid that kills the process -- loses restart N, which
  the synchronous writer would have finished; the latest durable restart is
  then N-1. The window is at most one output interval, and in practice the
  publication's own wall time (20-45 s for a 4.4 GB restart plus a 1.4 GB
  frame on the node, written one after the other; about the restart's own
  time when they are written beside each other and the two writes do not
  contend for the device). Nothing is reported before it is durable, so the
  receipts and `progress` never name the lost restart. Python-level failures
  and SIGINT wait for it, as above.
- `background_publication=False` commits each event on the stepping thread,
  with the same bytes. It still copies each event into the reserved host
  arenas. The one-call publications that commit at once on the calling
  thread (`publish_history`, `publish_restart`/`write_cartesian_restart`,
  `initial_condition`, `benchmark`, `NestDriver.publish_child_frame`,
  `tools/run_cartesian_storm.py`) reserve no host arena: their words stay on
  the device until the arena writer streams them in 8 MiB page-locked windows,
  as they always were, so host memory never holds a full copy of the payload
  beside the publisher's own.

**One copy of a restart: a restart directory format change (receipt schema
v2).** A restart directory now holds `restart.cm1fcp`, its two tables and its
receipt. `restart-carriers.f32` is still written, as the publisher's input, but
not synced, and it is deleted once the envelope is on disk and its header's
per-array SHA-256 equals every carrier digest the arena writer took (and the
contract, the payload size and the envelope's length agree). Otherwise the
restart is refused and no receipt is written: `integrate` discards its staging
directory and renames nothing into place, and a restart written straight into
its final directory (`publish_restart`, as `tools/run_cartesian_storm.py`
writes them) is left there with its raw arena, tables and envelope but no
receipt, so no resume admits it. A resume reads only the envelope
(`read_cartesian_restart` runs `cm1-forecast-publish restore`), and so does
`tools/nest_downscale_init.py`. History frames are unchanged and keep
`history-carriers.f32`, synced: the frame's field table and location record
name it, and `nest_parcel_backward`, `tools/nest_window_preview.py`,
`tools/nest_downscale_init.py` (the parent frame) and
`tools/benchmark_cartesian_untiled.py` read it directly.

This is a format change, not a correctness fix, and these are exactly the
restart files it changes against the release head:

| file | change |
|---|---|
| `restart-carriers.f32` | no longer in the directory; its words are in `restart.cm1fcp` |
| `restart-publication.json` | `schema` is `gm1.cartesian-restart-publication/v2`; `arena_filename` and `arena.path` are `null`; new `arena.retained` (`false`), `arena.disposition` and `arena.envelope_verification`. Every other key, the arena digests included, is unchanged |
| `restart-field-table.tsv` | same bytes, but its path column names the deleted publisher input, not a file of the restart (an `integrate` restart's table always named the staging path; a restart written in place named its arena, which is now gone). `arena.disposition` says so. Carriers are read only through `cm1-forecast-publish restore` |
| `restart.cm1fcp`, `restart-scheduler-scalars.tsv` | unchanged, byte for byte |

`require_restart_receipt` admits both v1 and v2, so restarts written before
this change still resume. A reader that followed `arena.path` or
`arena_filename` to the raw arena has to check the schema (v2 always carries
`arena.retained`). `tools/nest_downscale_init.py` writes a directory that does
keep the arena, and its receipt says so: schema v2, `arena.retained` `true`,
`arena_filename` and `arena.path` set.

**Comparing restarts.** Compare the arena digests in the receipts
(`arena.sha256`, `arena.carrier_sha256`: SHA-256s of the carrier words, with no
path in them), or restore both envelopes with `cm1-forecast-publish restore`
and `cmp` the restored arenas, which are the release head's arena exactly (the
pinned test rebuilds it that way). Do not `cmp restart-carriers.f32`: it no
longer exists. `cmp restart.cm1fcp` works for two runs of the same tree,
because the envelope is deterministic for identical words (its bytes are
pinned in `PRE_PROFILING_PUBLICATION_SHA256`). Across two trees it does not:
the header records the execution identity, a hash of the package's `.py`
files. A restart event's I/O stages include `envelope_verify` and
`arena.discard` and no `arena.fsync`.

**What it changes.** No simulated word, no history frame byte and no envelope
byte; the restart directory changes as listed above.
`test_published_frames_and_restarts_are_the_pre_profiling_bytes` runs every
profiling level in both modes and requires the release head's digest of every
file, with two declared exceptions: the restart arena is rebuilt from the
envelope's restored words before it is compared, and the receipt must be the
writer's own canonical JSON, byte for byte, and must equal the pinned v1
receipt once its v2 fields are undone.
`tests/test_background_publication.py` proves background and synchronous runs
publish identical trees (six frames and three restarts on reused arenas, and
a nest child's frames and placements), the one-call publications stream from
the device into the same files without a host arena, a run resumed from a
background restart is bit-identical to the continuous one, the stepping thread
pays for the snapshot and not the publication, and a worker exception, a full
disk in the arena writer (`RLIMIT_FSIZE`) or in the publisher, a failed
envelope check, a failed nest child snapshot, a worker thread that cannot
start, SIGINT once, twice and from another process, and SIGTERM each leave
only complete directories, no staging and no worker thread; a killed run's
publisher dies with it. For a frame and a restart of the same step it proves
the restart's publisher runs while the frame's still does, the renames stay
frame then restart when the restart is ready first, the run's last join
waits for the slower part and not the sum, the resumed restart is the
step's state, a failed frame stops the restart beside it (whether it is
still writing or already waiting to rename), a later part's failure spares
the earlier part, cancelling kills both publishers, and a join is not
counted twice in `blocking_seconds`.

`tools/benchmark_publication_io.py --mode background` times one event the way
`integrate` publishes it: the snapshot on the calling thread (a host copy
standing in for the device copy), the commit on the shipped worker, while the
calling thread keeps taking synthetic GIL-holding steps. Same machine and
disk, the shipped publisher, payloads the size of a 512x512x75 restart (51
carriers) and history frame (27 carriers):

| per event, same disk | before (synchronous) | now (background) |
|---|---|---|
| 4.39 GB restart: stepping thread blocked | 14.0-20.7 s (4 runs) | 0.37-0.71 s snapshot, 0 s join |
| 4.39 GB restart: publication wall (worker) | the same 14.0-20.7 s | 18.4-20.2 s |
| 4.39 GB restart: directory | 8.78 GB | 4.39 GB |
| 1.43 GB frame: stepping thread blocked | 3.7-7.1 s (4 runs) | 0.11-0.13 s snapshot, 0 s join |
| 1.43 GB frame: publication wall (worker) | the same 3.7-7.1 s | 6.1 s |
| synthetic step while the worker runs | -- | 0.29-0.32 ms (0.26-0.32 ms idle) |
| envelope SHA-256 | `33403179...` / `23caab6c...` | identical |

The arenas are reserved once per run (2.1-2.3 s pageable and prefaulted for
4.39 GB here; page-locked on a GPU host). On a GPU the snapshot is a
page-locked device copy of the same bytes over PCIe.

A restart on a frame step. `--kind same-step --next-event-s 20` publishes the
restart alone, then the frame and the restart as one event, each time
stepping for 20 s before the next event's join, as `integrate` joins the
previous event before its next snapshot, and samples `MemAvailable` every
0.1 s from the submit to the join. Same host and disk, the same payloads (27
and 51 carriers), three runs per column alternating, `os.sync` before each
scenario. The integration head (`fbd129e`) commits the frame and then the
restart; this tree commits them beside each other by default and one after
the other with `--parts sequential` (`--event-parts in-turn`):

| next event 20 s after the submit | integration head | now, `beside` | now, `in-turn` |
|---|---|---|---|
| restart alone: stepping thread blocked | 3.24, 3.00, 4.27 s | 0.39, 6.73, 1.47 s | 1.20, 6.62, 0.45 s |
| frame + restart: stepping thread blocked | 17.24, 14.95, 21.59 s | 4.32, 9.04, 3.29 s | 22.37, 24.29, 5.02 s |
| frame + restart: join wait | 12.98, 12.67, 13.44 s | 0.00, 3.95, 1.63 s | 16.49, 18.97, 4.43 s |
| what the frame adds to the join wait | 11.11, 10.10, 9.68 s | 0.00, -2.32, 0.99 s | 15.83, 12.77, 4.43 s |
| what the frame adds to the snapshots | 2.89, 1.85, 7.64 s | 3.94, 4.63, 0.83 s | 5.34, 4.90, 0.14 s |
| frame + restart: event commit wall | 33.0, 32.7, 33.4 s (frame 8.3-10.1 s + restart 23.3-24.6 s) | 19.9, 24.0, 21.6 s (the restart; the frame, 7.4-10.3 s, finished first, `turn_wait` 0) | 36.5, 39.0, 24.4 s |
| synthetic step during the frame + restart commit, mean | 0.39, 0.34, 0.51 ms | 0.28, 0.40, 0.50 ms | 0.35, 0.58, 0.36 ms |
| synthetic step during the restart alone, mean | 0.36, 0.28, 0.43 ms | 0.27, 0.28, 0.49 ms | 0.36, 0.43, 0.53 ms |
| synthetic step idle, mean | 0.27, 0.49, 0.44 ms | 0.28, 0.34, 0.47 ms | 0.27, 0.43, 0.53 ms |
| synthetic step during the frame + restart commit, maximum | 20, 11, 49 ms | 6, 13, 74 ms | 5, 51, 7 ms |
| `MemAvailable` drop from the submit, restart alone | 4.5, 5.1, 4.9 GB | 4.6, 5.1, 4.6 GB | 4.8, 5.4, 4.6 GB |
| `MemAvailable` drop from the submit, frame + restart | 4.3, 4.8, 4.7 GB | 4.4, 4.2, 4.8 GB | 4.5, 4.0, 5.2 GB |
| envelope SHA-256 | `33403179...` / `23caab6c...` | identical | identical |

What this set shows, and what it cannot. The join wait for a frame plus a
restart fell from 12.7-13.4 s (the head) to 0.0-4.0 s in every one of the
three `beside` runs, and `in-turn` on this tree reproduces the head
(4.4-19.0 s; the spread is the disk's). The result is disk-dependent: an
earlier set of three runs per tree on the same virtual disk had one
`beside` run whose join wait was 12.0 s against the head's 6.6-7.3 s, the
restart alone taking 24 s through it, and on this disk the two writes can
contend for the device. The deciding measurement is the node's own
`join_wait` comparison (`docs/GM1-GPU-VERIFY-FIXES.md`, section 7b); until
it is recorded the blocked-time claim carries NEEDS-LONG-VALIDATION. The
stepping thread's synthetic step, a GIL-holding host loop, is within the
idle spread in every column: two publishers hashing at once (up to 16
threads, in their own processes) did not measurably slow it on this
32-thread host, and the maximum step's outliers (5-74 ms) appear in every
column. The `MemAvailable` drop is the restart publisher's payload in every
column: the frame's publisher (7-10 s) had finished before the restart's
began reading its arena (after its 4.4 GB arena write), so the two payloads
were not resident together here; the 1.4 GB plus 4.4 GB peak is the bound
for a disk fast enough that they are. The snapshot rows are a pageable
host-to-host copy under this host's memory pressure; on a GPU host it is
the page-locked device copy the section above describes, not measured here.

### Profiling a document on a GPU

`tools/profile_cartesian_run.py` runs a document through the door with
`--profile detailed` for `--steps` measured steps, optionally (`--io`) writes
one frame and one complete restart, optionally (`--identity`) repeats the run
with `--profile off` and compares both published frames word for word, and
runs the same steps at half the horizontal grid. It prints each run's `gpuwm-cm1 perf` summary, the verdicts,
and the scaling ratio of the measured step, split into a fixed per-step cost
and a per-point cost: a ratio near the fourfold point ratio means the device or
its memory bandwidth sets the pace, a ratio near one means the host does.

```
python3 tools/profile_cartesian_run.py storm.toml --steps 20 --warmup 3 --io --identity \
    --workdir /scratch/gm1-profile --sounding input_sounding \
    --gpu-uuid "$GPU_UUID" --lock "$GM1_DIRECTORY_LOCK"
```

With `--identity` the summary also carries `profiling_overhead`: the profiled
run's mean step against the `--profile off` run's, both from the timing
harness's own `runtime_benchmark.mean_step_wall_seconds` (timed around
`step()` with `time.perf_counter` at every level, so it does not depend on the
profiler's accounting). `--profile summary` there measures what the default
level costs a production run.

### Verifying the profiler on a device

Three checks, each under the GPU mutex and each well under 15 minutes,
then one that rides along with the next long run. They were written for the
RTX 4090 host that found the CuPy 14 failure; the first lines are the only
node-specific ones, and they stop rather than run with an empty `$GPU`, a
lock that is not this node's, an interpreter without `sys.monitoring` (the
device test would be skipped, not run, and no kernel launch counted) or a
CuPy that is not the 14 series the failure was seen on. `$PY` is the node's
interpreter (a venv's `bin/python`, as `docs/GM1-GPU-VERIFY-FIXES.md` section
2 sets it), `$TREE` a checkout of this source, `$LIB`/`$PUB` the Rust
artifacts built as in that runbook's section 3, `$DECK` a document that
integrates from its own initial condition, and `$SND` its sounding.

```sh
GPU=$(nvidia-smi --query-gpu=uuid,name --format=csv,noheader | awk -F', ' '/4090/ {print $1}')
test -n "$GPU" && echo "GPU=$GPU" || echo "STOP: no RTX 4090 on this node"
export CUDA_VISIBLE_DEVICES=$GPU
LOCK=   # this computer's GM1 GPU lock directory; never another computer's
test -n "$LOCK" && test ! -e "$LOCK" && echo "lock free" || echo "STOP: LOCK is unset or held"
$PY -c 'import sys; assert sys.version_info >= (3, 12), sys.version'
$PY -c 'import cupy; assert cupy.__version__.startswith("14."), cupy.__version__; print("cupy", cupy.__version__)'
P=~/gm1-profiler-check && test ! -e $P && mkdir -p $P
```

1. The device test (~1 min): CuPy builds its own kernels under the hooks
   (`cupy.full`, the TypeError site), GM1's three launch shapes each count
   exactly once, every hook installs, nothing is turned off.

   ```sh
   CM1_CUDA_TESTS=1 PYTHONPATH=$TREE/src $PY $TREE/tools/perf_probe/with_gpu_mutex.py $GPU $LOCK -- \
     $PY -m pytest -p no:cacheprovider -rfEs -v \
     "$TREE/tests/test_runtime_profiler_failsafe.py::test_device_cupy_builds_its_own_kernels_under_the_hooks_and_gm1_kernels_count" \
     2>&1 | tee $P/device-test.log
   ```

   Expect `1 passed`. `1 skipped` means `CM1_CUDA_TESTS`, CuPy or the device
   was not seen and nothing was verified.

2. The door at both levels, integrating with a frame and a restart (~2 x the
   deck's duration; keep `run.duration_s` to a few minutes of steps):

   ```sh
   for L in summary detailed; do
     mkdir -p $P/$L
     PYTHONPATH=$TREE/src $PY -m cm1core cartesian $DECK --sounding $SND \
       --library $LIB --history-publisher $PUB --integrate \
       --history-dir $P/$L/history --restart-dir $P/$L/restarts \
       --profile $L --progress-interval 10 --output $P/$L/receipt.json \
       --gpu-uuid "$GPU" --lock "$LOCK" 2>&1 | tee $P/$L.log
   done
   $PY $TREE/tools/perf_probe/check_profiled_runs.py door $P
   ```

   Expect two `PASS` lines (steps, more than 100 launches per step, NVRTC
   programs, the verdict) and exit 0. A `FAIL` line names the defect:
   `instrumentation {...}` with a non-empty `disabled` names the instrument
   that turned itself off and the error, and `fail-safe lines in the log`
   quotes the stderr line the run printed when it did.

3. Overhead and identity (~2 x (`--warmup` + `--steps`) steps, plus the
   half-grid run):

   ```sh
   PYTHONPATH=$TREE/src $PY $TREE/tools/profile_cartesian_run.py $DECK \
     --steps 20 --warmup 3 --identity --profile summary --workdir $P/profile \
     --sounding $SND --library $LIB --history-publisher $PUB \
     --gpu-uuid "$GPU" --lock "$LOCK" 2>&1 | tee $P/profile.log
   $PY $TREE/tools/perf_probe/check_profiled_runs.py profile $P/profile
   ```

   Expect `PASS` for `full`, `off`, `half` and `identity`, and an `overhead:`
   line: the `--profile summary` step wall against `--profile off`. That ratio
   is what the default level costs a run on this device; a ratio above about
   1.02 on a 512x512x75 deck is worth reporting with the receipts.

4. The long run. Nothing above runs for hours; the claim that every
   instrument stays on for a two-hour nested run is validated by the next
   such run, which profiles at the default `summary` level. When it ends
   (or dies):

   ```sh
   $PY $TREE/tools/perf_probe/check_profiled_runs.py receipt RUN/receipt.json RUN/stderr.log
   ```

   `PASS` means `instrumentation.disabled == {}` and no `gm1 profiler:` line
   for the whole run; `FAIL` names the instrument and its error, which is
   then the bug to file.

### Where a step's time went, and what was removed

Four cuts leave every published history word and restart carrier as they were
(the first one only once its device gate below passes):

- **The device pool stays cached between phases.** Every pre-acoustic stage,
  and every acoustic, scalar and microphysics release, used to hand every unused
  CuPy pool block back to the driver, only to allocate it again in the next
  phase. The blocks now stay in the pool. What the frees cost is a census
  estimate until the M1 run below measures it: data from the 512x512x75 receipts
  back ~150 ms per step on an RTX 4090, and a fitted model (~18.8 GiB returned
  per step at 15-19 ms per GiB) puts it at 150-290 ms. M1's `free_s` and
  `sync_s` replace those numbers.

  A buffer allocated with `cupy.empty` now holds whatever the pool returns, so
  every such buffer the runtime reaches has to be written before it is read.
  `tools/oracle/cartesian_door_host_run.py --poison nan|big` shows that for
  GM1's own allocations on the host: it drives the real door, fills every
  uninitialized allocation with NaN or -3.0e38, and the published words match
  the unpoisoned run (`tests/test_step_speed_identity.py`). ArWen's column call
  is a host double in that run, so allocations made inside ArWen are not
  covered. (A reading of the gpuwm 2.7.4 wheel, repeated on the pinned 2.8.9
  wheel after its Morrison files changed, finds none in the WSM6
  and Morrison paths: `morrison.py`'s three `cp.empty_like` calls run only when
  a caller omits the scratch and effective-radius arrays, and its `apply`
  passes all of them; the fused preparation kernel 2.8.9 adds writes every
  element of theta, the Exner function and the layer depths. That is a
  reading, not a proof.) **Gate:** `keep` stays
  the default only if the poisoned device runs read PASS on the 4090: the
  shipped defaults with NaN and with large-value poison at 512x512x75 and at
  256x256x75, and both poisons on the mature-stage resume. The report
  prints this as the "fix #1 device poison gate" row.
  `GM1_DYNAMICS_POOL_RELEASE=phase` restores the old frees for a measurement;
  the run records the choice under `last_step_dynamics.device_pool`. It is not a
  document key and does not enter the sealed configuration hash.
- **The final RK stage enqueues each moisture row's advection once.** With
  `pdef=1` (`solve2.F:1451-1455`) CM1's `advs` forms that advection from the
  positive-definite chain (`adv.F:421-445`). The stage used to launch the
  plain fifth-order advection first, and the chain rewrote every word it wrote
  before anything read them: the same x/y faces, `fz(k=2..nk)`, and the
  tendency interior. It is no longer launched. The stage receipts
  (`gm1.cuda-child-stretched-scalar-moist-rk-stage/v3`,
  `gpuwm-cm1.cuda-scalar-moist-rk/v3`) name what formed the advection under
  `moisture_advection`. `GM1_FINAL_STAGE_MOISTURE_ADVECTION=launch` launches it
  again.
- **The acoustic core drops stores nothing reads.** The compact core the lane
  compiles no longer zero-fills `fpk`/`divh`/`hadvp`/`dum3`/`ec`/`ed` at every
  stage, stores `divh`, or stores the update tendency into `ec`
  (`tests/test_acoustic_compact_dead_stores.py` runs b200ad4's compact core
  and this one side by side from the same garbage).
- **`scalar_boundaries` keeps only the `pp3d` fills.** Its rho, theta and q3d
  fills went into three scratch volumes that were released unread. A stretched
  binding's `sigmaf(nk+1)` is also read back once per binding, not once per
  launch.

`tools/perf_probe/run_m0_m4.sh` measures these cuts on a GPU:
`tools/perf_probe/sitecustomize.py` (`GM1_PROBE=off|time|keep|poison-nan|poison-big`)
times, suppresses or poisons from outside the tree, and the script prints the
pass/fail table. Its header holds the runbook: four commands of at most 15
minutes of GPU each, which `BUDGET_S` enforces. On an untiled 512x512x75 or
256x256x75 device run, the table judges the step-time targets it records as
PASS or FAIL. Every other timing row is information, including every
comparison across two trees. A probe that never reached CuPy, a run that
stopped early and a selected run that never started are all FAIL. With
`DOOR_PYTHON=tools/oracle/host_door_python.sh` and `GPU_UUID=none`, the kit also
runs on a CPU.

### CUDA graphs: capture a step once, replay it (`--cuda-graphs`)

A nested step is **host-launch-bound**: a large nested run (640x640x80 tiled parent,
500x500x80 child, time ratio 3) counted ~52,700 launches per parent step and
its device finished each within 0% of the host enqueuing it, so the step
cost Python's 2.4 s, not the card's. Every launch is built, validated and
receipted by Python (tile gathers, per-block WSM6 staging, the Davies zone's
per-side kernels on every acoustic small step), and most are small.

`--cuda-graphs on` records a runtime's whole large-step schedule
(`gm1_runtime.large_step_schedule`) as one CUDA graph by stream capture of
the unchanged Python, and launches the graph for later steps instead of
running the Python again. The parent and the nest child each have their own
graphs (the child one per substep key). `verify` captures every step and
compares each graph, node by node, with the graph `on` would have replayed;
it replays nothing. `off` (the default) takes exactly the path it always
took. Like `--profile` it is an execution control: not a document word, not
in the sealed configuration hash, and it changes no published word.

```
gpuwm-cm1 cartesian deck.toml --integrate ... --cuda-graphs on
```

**What a replay must still do on the host** goes through three doors in
`cm1core/cuda_graphs.py`, which the step code uses everywhere it waits, reads
the device or keeps a counter:

- `synchronize(stream|event)` -- a wait before a registry release, a
  bridge drop, a launch's ready event. Inside a graph the node order on the
  runtime's one stream is the wait; outside a capture it is the wait it was.
  The runtime's stream skips `wait_event` only while capturing (every ready
  event was recorded on that same stream).
- `read_device(array, callback)` -- the closure's invalid-cell count and
  eddy-coefficient maximum, the final-RK EOS check, the roughness
  convergence flag. Under capture the graph copies the words into a slot it
  owns; after every launch and its synchronization the slot is read and the
  same callback runs (a validation that would have stopped the step still
  stops it).
- `bookkeeping(update, step)` -- the completed-step counters the phase order
  checks and the restart writer reads (`closure._held_step`,
  `finish.completed_physics_step`/`completed_solve3_step`, the swath step),
  the nest forcing's counters, the child window's skip counters. It runs now
  and is re-run by every replay with the replayed step number.

**When a graph may replace a step.**

1. *Key.* `graph_key(step)` names every host decision a captured step
   freezes: a nest child's `dtbc` (the time since the boundary tables were
   refreshed; it differs between substeps and repeats across parent steps)
   and the child window's branch. A step whose schedule reads step-varying
   host state or the device mid-step runs in Python by name: reflectivity
   (frame) steps, window-record child steps, budgets, parcels, adaptive `dt`,
   the adaptive moving frame, the two-part closure ramp. The receipt counts
   each reason.
2. *Inventory.* Every device array reachable from the runtime -- path,
   address, shape, strides, dtype -- is hashed before each step. A graph is
   replayed only on the inventory it was captured on, and only if it ends on
   the inventory it began on (a replay runs no Python, so the host objects
   must already name the arrays the device state is in). The broadcast base
   fields (`pi0`/`th0`/`qv0`) stay resident under graphs instead of being
   dropped and re-allocated at every RK stage, so a step ends where it began.
3. *Memory.* Every allocation during a capture comes from one private pool
   per stream (parent and child share it, as they share the default pool
   eagerly). After a capture every live block of it must be held by an array
   some runtime's inventory reaches, or the key is refused.
4. *Confirmation and audit.* `on` trusts a graph only after a second capture
   of the same key and inventory agrees with the first node by node: every
   kernel's name, launch shape and argument bytes (CuPy's `CIndexer` scratch
   words excluded), every copy and fill record, with pointers into the
   capture's own allocations named by allocation, not address. Keys whose
   consecutive captures differ are never replayed. Every 256 replays a graph
   is captured again and held to the same comparison; a difference stops the
   run (`CudaGraphReplayMismatch`) naming the node, kernel and argument.

A capture that meets a host wait or read not routed through the doors fails
loudly (`CudaGraphCaptureError`, naming the phase); CUDA itself refuses such
calls during a capture. The first step each runtime takes runs eagerly so
lazily built workspaces come from the default pool. `--cuda-graphs` is
refused (`CM1-CARTESIAN-CUDA-GRAPHS-UNPROVEN`) for `[chain]` runs, whose
boundary tables are replaced from the host between steps.

One adapter exists outside the doors: the frozen
`cuda_gm1_child_psolver3_open` module (byte-pinned by the child operator
manifest) reads `sigmaf(nk+1)` back from the device on every acoustic stage.
With graphs on, `_require_vertical` is wrapped so it runs whole once per
coordinate object and returns the same result while that object still holds
the same four profile arrays, top and reference spacing. Without graphs the
module runs as pinned.

**Receipt.** `runtime_benchmark.cuda_graphs` has one record per runtime:
counts (eager, captures, confirmations, replays, audits, verified,
verify_mismatches, inventory_misses), `eager_reasons`, `unstable_keys` with
their reasons, every cached graph's key, node count, replays and fingerprint
digest, the shared pool's bytes, and the host seconds spent on inventories,
captures, fingerprints, launches and readbacks. Under `--profile` the
replayed step is one phase, `cuda-graph.replay` (a capture,
`cuda-graph.capture`); launches inside a graph are not Python calls and are
not counted.

**Evidence.**

- *Words.* On the RTX 3080 (sm_86), graph against no graph, every published
  word of two nested decks is bit-identical -- every frame carrier file, every
  scheduler-scalar table, every nest placement record and every frame and
  restart payload hash: a 128x128x40 stretched WSM6 parent (4 tiles) with a
  fixed two-way 100x100x40 child at time ratio 3 (80 parent and 240 child
  steps, two frames, two parent and two child restarts), and the same parent
  with a storm-following child with vertical refinement (55 child levels),
  time ratio 4, the tornado criterion and the child window stream (40 parent
  steps, 160 child steps, 7 moves, 17 window records). A run resumed from the
  first restart with graphs on publishes the words of the same resume with
  graphs off and of the uninterrupted run.
- *Graphs.* `verify` on the same decks: 0 mismatches in 37 parent and 134
  child comparisons (moving deck) and 17 and 55 (fixed deck).
- *Host tests.* `tests/test_cuda_graphs.py` holds every replay boundary on
  the host (the doors eager and under a capture, the inventory, the
  decisions -- warm step, eager keys, confirmation, replay only on a fixed
  point, audit mismatch, unnamed pool block, capture failure -- the canonical
  fingerprint, the keys and counters of the runtime, the door's option).
  `tests/test_cuda_graphs_device.py` (`CM1_CUDA_TESTS=1`) replays real
  kernels against the same kernels run eagerly, word for word, and shows
  `verify` naming a changed scalar argument by kernel and index.

- *At scale, RTX 4090.* That large nested run, resumed from its restart at T+76.5 (24
  parent steps, 72 child steps, the child window stream recording): graph
  against no graph, the frame and restart at the end are identical word for
  word (every carrier file and payload hash, parent and child), at closure
  tile 64 and at tile 128; `verify` at tile 128 made 21 parent and 48 child
  comparisons of 6,924- and 5,096-node graphs with 0 mismatches.

**Speed.** Seconds per parent step, steady state (the first four steps of a
graph run are its captures):

| Deck (card) | off | on | device time of the graphs |
|---|---|---|---|
| 128x128x40 + fixed 100x100x40 child, ratio 3 (RTX 3080) | 0.32-0.80 | 0.11-0.27 | parent 29-58 ms, child 29-47 ms per substep |
| 128x128x40 + moving refined child, window, ratio 4 (RTX 3080) | 0.65-0.70 | 0.40-0.50 | parent 29 ms, child 35 ms per substep |
| 640x640x80 parent + 500x500x80 child, ratio 3, tile 64 (RTX 4090) | 2.38 | 2.15 | parent 657 ms, child 424 ms per substep |
| the same, tile 128 (RTX 4090) | 1.80 | 1.78 | parent 544 ms, child 350 ms per substep |
| the same parent, a child with 103 levels, ratio 12, tile 128 (RTX 4090) | 6.29 | out of memory | -- |

What this measures: on a small grid the step is the host's, and a replay
removes it (2-6x). On the 4090 at the large size the step is already the device's
-- the profiler's `host-launch-bound` verdict there means the device keeps up
with the host, not that it idles for long: the graphs' own device time is
1.93 s of the large run's 2.38 s parent step at tile 64 -- so a replay saves the
rest (10%), and child steps that record the window stream (1 in 4 there)
still run in Python. The larger lever at that size is the closure tile:
`[runtime] tile_nx = tile_ny = 128` (the largest admitted) makes the same
words 1.32x faster without graphs (2.38 -> 1.80 s, and 8.22 -> 6.29 s with the 103-level child,
per parent step; `[runtime]` is outside the restart identity, so a resumed
run may change it). Per phase, the graphs' device time at tile 128 is led by
the held tendencies (124 ms parent, 84 ms child), the final RK scalar stage
(82, 58), WSM6 (61, 40) and the final acoustic stage (57, 36).

**Memory.** Graphs keep the three broadcast base fields resident and reserve
their step scratch for as long as they exist, so a graph run needs more
device memory than the same run without: the large run sampled 25.1-25.2 GB against
21.4-21.9 GB. Its 103-level child at tile 128 (24.6 GB without graphs) ran out of memory
in its first steps with graphs on. Use graphs where the step is host-bound and
the card has room; `off` stays the default.

## Ensembles

With a restart, branching is nearly free, and it is how the question "was this
outcome inevitable?" gets asked. CM1's own ensemble switch is `irandp`
(`README.namelist:692-695`), and `src/init3d.F:1446-1483` is the whole
operation: `tha += amplitude*(2*rand-1)` over the domain.

```toml
[ensemble]
member            = 3          # recorded in the receipt of every leg
seed              = 20260914   # splitmix64; the same seed is the same member
theta_amplitude_k = 0.25       # src/init3d.F:1466's own amplitude
```

Two deliberate differences from CM1, both named in the receipt. The amplitude
is configuration rather than Fortran source -- `src/init3d.F:1466` says "set
magnitude of perturbations in init3d.F". And the stream is seeded: CM1 calls
`random_number`, so a CM1 member cannot be re-run and therefore cannot be
bisected. The generator is splitmix64, five integer operations per word, and
the receipt carries the SHA-256 of the field that was actually added.

The perturbation is applied once, at a completed-large-step boundary: the
initialized state when `irst = 0`, the restored state when `irst = 1`. It
reaches the theta perturbation carriers and nothing else -- the pressure and
density carriers are not re-derived, because at a branch point they carry the
trajectory rather than the base state, so the first acoustic step adjusts an
unbalanced perturbation, which is what an unbalanced perturbation is.

A run that declares no `[ensemble]` table is the unperturbed control, and its
canonical document is byte-for-byte what it was before ensembles existed.

## The environment: CAPE, CIN, LCL, LFC, shear and helicity

CM1 prints the thermodynamic properties of its own base-state sounding for
every moist run and gates that on nothing at all (`src/base.F:3143`), and this
runtime does the same. Every executed run prints
`src/base.F:3175-3180`'s block to stderr and writes it into its receipt under
`initial_sounding_parameters`: the source parcel, LCL, LFC, equilibrium level,
CAPE and CIN for each of `src/getcape.F:65-68`'s three parcels -- surface,
most-unstable and mixed-layer -- computed by the transcription in
`rust/cm1-state/src/getcape.rs` over the model's own column with CM1's surface
level prepended.

CM1 computes no storm motion, bulk shear or storm-relative helicity anywhere,
so three more parameters are added beside them and named by method rather than
left to the reader: the 0-1 km and 0-6 km bulk wind difference, the Bunkers et
al. (2000) internal-dynamics right and left movers, and the 0-1 km and 0-3 km
storm-relative helicity taken against the declared one. The receipt carries the
citation. `tools/report_sounding_parameters.py` prints the same block for a
configuration and a sounding without running anything, and
`tools/analyze_storm_run.py --receipt` prints it beside a run's storm summary,
so a report on a storm that did not form can say what the air was.

CIN follows `src/getcape.F:475`: it is the **positive magnitude** of the
negative area, not a negative number.

The four 2-D fields are CM1's own switches, in an optional `[diagnostics]`
table that only the Cartesian lane may declare:

```toml
[diagnostics]
output_cape = true
output_cin = true
output_lcl = true
output_lfc = true
parcel_source = 3          # getcape.F:65-68; writeout.F:2921 uses 3
adiabat = 1                # getcape.F:73, pseudoadiabatic liquid
mixed_layer_depth_m = 500.0
pressure_increment_pa = 200.0
storm_motion = "bunkers-right"
```

Each switch that is on adds one `(ny, nx)` carrier of that name to the history
frame, computed by `cm1_getcape_fields_v1` once per column of the completed
post-solve3 state, exactly as `src/writeout.F:2895-2932` does. A document that
declares no `[diagnostics]` table is in `writeout.F:2895`'s shipped state: no
column sweep runs, the frame carries the same twelve carriers it always has and
the same 23,808,000-byte payload, and the canonical configuration hash of every
shipped deck is unchanged. `examples/cartesian-environment-fields.toml` is the
deck with all four on.

`tools/run_cartesian_storm.py`, the driver the long integrations actually run
under, builds its own document from flags rather than a deck, so it carries the
same switches: `--output-cape`, `--output-cin`, `--output-lcl`, `--output-lfc`,
`--parcel-source`, `--adiabat`, `--mixed-layer-depth`, `--pressure-increment`
and `--storm-motion`. An invocation that passes none of them projects no
`[diagnostics]` table and produces exactly the configuration hash it always
did.

The renderer draws them: `--product cape|cin|lcl|lfc --nz 1 --level 0` against
the carrier of the same name, on fixed physical scales rather than a per-frame
normalization, so a weakening inflow looks weaker between two frames.

## Binary32 sub-normals on the device

CM1 runs on the host with IEEE gradual underflow. A default-REAL product, sum
or double-to-float conversion whose exact value is below 2**-126 keeps that
sub-normal value (`records/CM1-ARITHMETIC-CONTRACT.md`). A CUDA kernel keeps
it only if NVRTC compiled it with `--ftz=false`, which is also NVRTC's
default. CuPy's own compile path does not allow that.
`cupy.RawModule`, `cupy.RawKernel` and CuPy's elementwise and reduction
kernels are compiled with `-ftz=true` appended after the caller's options.
The last occurrence wins, so those kernels write a zero of the same
sign instead of the sub-normal, and read a sub-normal operand as zero
(`evidence/oracle/nvrtc-gradual-underflow` has the measured table).

Every GM1 kernel is therefore compiled by NVRTC directly and loaded as an
image: `cm1core.nvrtc_module.compile_nvrtc_module` returns an object that
answers `get_function` as a `RawModule` does, and the image compile
functions load the cubin they compiled. The profiler names these programs
by file name (`gm1_nest_live.cu`, `gm1_two_part_sgs.cu`, ...). NVRTC compiles
each one again in every process, because CuPy's on-disk kernel cache is not
used.

What this does not cover:

- CuPy array operations in GM1's Python (`cupy.add`, `a * b`, `cupy.where`,
  `cupy.maximum`, reductions): the runtime's own staging, the eqtset2/WSM6
  bridge, the budget operators and the diagnostics. They still flush.
- ArWen's kernels, which CuPy compiles through its loader. The measurement
  in `evidence/oracle/nvrtc-gradual-underflow` was taken in ArWen's own tree,
  against its loader and two of its component compile sites; which of its
  words that moves is ArWen's to record, not this package's.
- The host stand-in (`tools/oracle/host_cupy.py`) computes all of these with
  IEEE sub-normals. Host/device word identity for them therefore holds only
  where no sub-normal arises. `tools/oracle/host_flush_to_zero.py` runs host
  kernels with the flushed arithmetic when a comparison needs the device's
  words.

The flush can also move normal words. A pdef or advection flux is `m*r`. When
WENO5's double-to-float conversion makes `r` sub-normal and `|m| > 1`, CM1's
flux is a normal word below `|m|*2**-126`, and it becomes zero once `r` is
flushed. Past that, a sub-normal-versus-zero difference reaches a normal word
only through operations that grow it (`sqrt`, a division, a comparison with
zero, a factor above one). In a sum it is absorbed unless the sum itself is
below about `2**-102`. In WENO5's weights, a sub-normal change in `b1..b3`
moves `w1..w3` by a relative amount below `2*2**-126/weps`. That bound is a
statement about `weps = qmag(n)*epsilon` (`solve2.F:1472`, with
`epsilon = 1.0e-18` at `constants.F:57`), and `qmag` is built per `ptype`.
In the three `ptype`s GM1 binds -- 1 (`param.F:3191-3193`), 5
(`:3465-3474`) and 7 (`:3672-3677`) -- the smallest `qmag` is `1.0e-2`, so
the smallest `weps` is `1e-20` and the bound is below `2.4e-18`: at most a
rare one-ulp rounding change in binary64. That is the scoping
`cuda_scalar_moist_rk.CM1_QMAG_F32` and `moisture_weps_f64` already carry.
It is not a claim about CM1 at large. `ptype=26/27` ends its table with
`qmag(:) = 1.e-12` (`param.F:3891`), whose `weps = 1e-30` puts the bound at
`2.4e-8` -- about 0.2 of a binary32 ulp, at which the sentence above does
not hold. GM1 binds no such scheme and nothing here is measured at that
`weps`. Measured,
the 20008 WENO5 stencils the CM1-word tests use give the same word with and
without the flush at both the TKE face `weps` (`100*epsilon = 1e-16`) and
the smallest supported moisture `weps` (`1e-20`), and none of their
reconstructions is sub-normal: the smallest non-zero magnitude is 3.57e-25
and 3.57e-33 respectively, against `2**-126 = 1.2e-38`
(`tests/test_nvrtc_gradual_underflow.py::test_flushing_leaves_every_weno5_reconstruction_word_alone`).

## Validation and remaining work

The 237-level SGS/TKE capacity extension was compared against CPU references
with zero raw-word mismatches. Public WSM6 was exercised on full 237-level
columns with repeatability and finite-state checks. The acoustic storage
comparison covers all three RK selectors and four storage combinations; it
compares every array except the six explicitly omitted column diagnostics,
and requires a one-bit negative control to fail. Held-momentum packing tests
include physical faces, halos, signed zeros, allocation guards, and rejected
overlap.

These operator checks do not replace a complete trajectory comparison.
Next qualification work includes sustained integration, complete initialized
state and trajectory checks against the CM1 dycore authority, and connecting
the existing checkpoint/history lifecycle to this driver. ArWen physics
remains the declared physics seam.

Current development receipts, including failed attempts, are preserved in
`work/cartesian-runtime-20260907-run-v1/`; capacity probes are in
`work/cartesian-capacity-probe-20260907/`.
