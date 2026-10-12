# CM1-NEST-SEAM: grid nesting is an extension beyond CM1 r21.1

Status: **declared extension.** Nothing in this seam is a CM1 port, and no
receipt, plan or document produced by it may be read as native-CM1 behaviour.

## 1. CM1 r21.1 has no grid nesting. Stated by evidence, not by assertion.

The mathematical authority is the CM1 r21.1 release tarball
(`sha256 dc49fe84531056d1ae6249b37a5e3ee453fd96861c3b6bafd63828d92e64edf7`).
In that tree:

* `run/namelist.input` (416 lines, 23 namelist groups) contains **no** nest,
  refinement, ratio or grid-id word.
  `grep -inE 'nest|grid_id|refine|ratio' run/namelist.input` returns exactly
  one line, `:412  ccn = 0.6e9  ! base ccn concentration`, and it matches only
  because *conce-ratio-n* is a substring of "concentration". There is no
  nesting word in CM1's namelist.
* `README`: no occurrence of `nest`.
* `README.namelist`: no occurrence of `nest`. Its lateral boundary section
  (`README.namelist:561-572`) lists the complete set: `1 = periodic`,
  `2 = open-radiative`, `3 = rigid walls, free slip`,
  `4 = rigid walls, no slip`. There is no specified, nudged or relaxed
  lateral boundary in CM1.
* `CHANGES`: no occurrence of `nest` in the whole revision history.
* `src/`: the only matches are the function `get_m_mix_nested` inside the
  imported Thompson radar-reflectivity module (declared
  `src/module_mp_radar.F:32`, defined `:362`, with its own call sites at
  `:342,398,403,429,434,460,465,479` and its diagnostics at
  `:387,408,418,439,449,485,491,493,496`),
  commented-out WRF remnants in the imported GFDL surface layer
  (`src/module_sf_gfdl.F:815,1262,1379,1391,1459,1579,1591`), and two
  comments in the imported radiation driver (`src/radtrns3d.F:62,318`).
  None is a grid nest.

CM1's one moving-mesh capability is **not** a nest: `do_adapt_move`
(`src/azimavg.F:4376-4528`, driven from `src/cm1.F:3641-3679`) translates the
**whole single domain** after a tropical cyclone by adjusting `umove`/`vmove`.
Its ground position is the `domainlocx`/`domainlocy` accumulator of
`src/cm1.F:3512-3513`. GM1 already carries that frame and that accumulator; it
is a different capability from refinement and this seam does not confuse them.

**Consequence.** Every nesting claim GM1 makes is GM1's own. No nested run may
be described as CM1-faithful, and no nested trajectory may be compared against
a CM1 oracle, because CM1 cannot produce one.

## 2. What GM1 nesting takes from CM1, by line

The child is a CM1 grid, so CM1's own rules decide what it may run. These are
ports, cited, and the child is admitted against them by the same code the
parent is:

| Quantity | CM1 r21.1 authority | Where GM1 applies it |
|---|---|---|
| psolver=3 acoustic small step `dtsm = 0.60*min(dx,dy)/360` | `src/misclibs.F:857-868` | `gm1_runtime.cm1_acoustic_small_step_limit_s`, called on the **child's** spacing |
| `nsound`, the nearest integer to `dbldt/dtsm` and at least 4, rounded up to even, +2 if still over | `src/misclibs.F:883-885` | `gm1_runtime.cm1_diagnostic_acoustic_substeps` on the child's step |
| `max_nsound = 192` and the stop above it | `src/misclibs.F:835`, `src/misclibs.F:887-911` | `NestGeometry.clock_defect` |
| advective Courant norm and the 1.50 stop | `src/misclibs.F:2747-2765`, `src/misclibs.F:2714` | `NestGeometry.advective_courant_multiplier`, reported by value |
| `dt` is one default-REAL word | `src/solve2.F:145` | `NestGeometry.clock_defect` via `positive_normal_f32_bits` |
| C-grid staggering that makes face alignment meaningful | `src/cm1.F:744-777` | `NestGeometry` places the child on whole parent cells |
| open radiative lateral boundary the parent uses | `src/bc.F:528-934` (`bcu`, `bcv`) | why the child must stay 3 parent cells clear |
| storm tracker: 100 passes of the 9-point `0.11111111` smoother on the lowest level of `pp3d`, in CM1's own summation order | `src/azimavg.F:4071-4107` | `nest_geometry.cm1_smoothed_pressure` |
| corner interpolation to zeta points and the minimum | `src/azimavg.F:4122-4139` | `nest_geometry.cm1_pressure_centre_cell` |
| the centre is reported as a grid **face** | `src/azimavg.F:4159-4163` | the nest snaps to whole parent cells |
| re-centring cadence `adapt_move_frq` | `src/azimavg.F:4421-4435` | `nest.follow_interval_s` |
| the frame the child inherits | `src/base.F:2661-2674`, `src/init3d.F:32-33` | the child is bound to the parent's `umove`/`vmove` |

## 3. What GM1 nesting adds, and on whose authority

### 3.1 Parent → child interpolation: **this port's own**

Cell-mean-preserving, extremum-limited piecewise linear
(`nest_boundary.limited_slopes` / `interpolate_point`). Centred slopes, one
scalar Barth–Jespersen-style limiter applied to **both** slopes so the
correction stays linear in the sub-cell offset. Two properties, both tested:

* **No new extrema.** Every child value lies between the min and max of the
  five-point parent stencil. A non-negative parent stencil gives non-negative
  child values, which is what keeps the child's microphysics from starting a
  step with negative water at its boundary.
* **The parent cell mean is preserved to the last bit.**
  `NestGeometry.sub_cell_offsets` returns `(2m+1-r)/(2r)` rather than the
  algebraically identical `(m+0.5)/r - 0.5`, precisely so that
  `offset[r-1-m]` is the **exact** binary negation of `offset[m]`;
  `nest_boundary.block_mean_residual` then sums the block in pairing order
  and returns exactly `0.0`. This is not a tolerance.

It is not a transliteration of WRF's SINT, and does not claim to be.

### 3.2 Lateral relaxation: Davies, in WRF's nested discrete form

Davies (1976) QJRMS 102:405–418; Davies (1983) QJRMS 109:225–233. The discrete
weights are WRF's nested branch, `dyn_em/module_bc_em.F lbc_fcx_gcx` (which
carries **no** exponential sponge term; the sponge is the specified-boundary
branch only), applied by `share/module_bc.F relax_bdytend`/`spec_bdytend`.

The reference implementation this port was written against and checked by hand
is the ArWen 2.7.4 tree (staged at the time; since pinned for the column physics):
`gpuwm/core/kernels/spec_bdy.cu` and the float64 mirror
`gpuwm/verify/npref.py:4047-4060` (weights) and `:4062-4139` (application).

**Nothing here depends on the ArWen pin.**
`src/cm1core/engine_pin.py` pins `gpuwm>=2.8.9,<2.8.10`, but its byte-admitted
surface is the column microphysics files only and contains no `nest.cu`,
`spec_bdy.cu` or LBC kernel. GM1 therefore carries its own kernel text
(`nest_boundary.NEST_RELAXATION_KERNEL_SOURCE`, pinned by sha256) and its own
CPU oracle, imports nothing from `gpuwm`, and would keep working if the ArWen
pin never moved.

### 3.3 Two-way feedback: **this port's own**

Block mean of the child's `r*r` interior cells replaces each covered parent
cell (`nest_boundary.restrict_child_to_parent`). The child's own relaxation
rows are excluded, because those rows carry the parent's values and feeding
them back would relax the parent toward itself through the child. The
round trip of a field the child does not modify returns the parent unchanged.

### 3.4 Moving a sub-domain: **WRF's moving nest** (executed: section 5.8)

CM1 moves its whole domain. GM1 moves the nest, in **whole parent cells**, by
WRF's moving-nest rules (`share/mediation_nest_move.F`,
`dyn_em/shift_domain_em.F`), toward the surface vortex WRF's own vortex-centre
algorithm finds on the child, or toward the centre CM1's pressure tracker
reports on the parent. Quantisation is not a convenience: a fractional shift
would take the child's outer `u`/`v` face columns off the parent's face columns
and the interface flux would stop being the parent's own.

The **tracker itself is CM1's, and runs on the device.** One hundred passes of
a nine-point smoother over the parent's lowest level is a field operation, not
orchestration, so `nest_geometry.NEST_PSMTH_KERNEL_SOURCE` carries
`gm1_nest_psmth_pass` and `gm1_nest_zeta_point` and
`device_pressure_centre_cell` ping-pongs them on the device; only the two
integers the nest moves by cross back to the host. CM1's parenthesisation at
`azimavg.F:4092-4096` is preserved literally, including the `0.11111111`
constant CM1 writes rather than `1.0/9.0`, because that parenthesisation fixes
the float summation order, and on a field with two nearly equal pressure
minima, which is exactly what a splitting supercell presents and exactly when
the nest must decide which member to follow, a reassociated smoother reaches a
different minimum. A negative control in
`tests/test_nest_relaxation_kernel.py` reassociates it and requires the raw
comparison to fail.

## 4. The five hard parts, answered

**Conservation across the interface.** The child covers a whole number of
parent cells, so its outermost `u` column is a parent `u` column and its
outermost `v` row a parent `v` row. The normal mass flux specified at the
child's boundary is the parent's own face value, **copied, not interpolated**
(`nest_boundary.boundary_normal_velocity_is_parent_face_value`, and
`parent_face_index` names the face). Whatever mass left the covered region
through that parent face is exactly what enters the child. For tangential and
scalar values the conservation statement is the exact block-mean identity of
§3.1. What is **not** claimed: global mass conservation of the coupled
parent+child system under two-way feedback. The block-mean restriction
conserves the cell mean of each covered parent cell, which is the strongest
statement this pair of operators supports.

**psolver=3's vertically implicit acoustic step.** The implicit part is one
tridiagonal solve per column and **never crosses the lateral interface**: the
child solves its own columns from its own relaxed state, the parent solves its
own. Refinement touches only the horizontal, explicit part, and CM1's own
`dtsmall` answers it: `src/misclibs.F:867` scales `dtsm` with `min(dx,dy)`, so
refining space by `r` and time by `r` leaves `dbldt/dtsm` unchanged and
`src/misclibs.F:883-885` derives the **same** `nsound` for the child as for the
parent. Refining space alone multiplies both the required `nsound` and the
`src/misclibs.F:2714` Courant number by `r`; `[nest]` reports that multiplier
by value and refuses the combinations CM1 itself would stop on.

**Microphysics at the interface.** Every moisture and hydrometeor scalar
crosses through the same extremum-limited operator as `theta`, so a
non-negative parent stencil yields non-negative child boundary values and the
child never begins a step with negative water there. **Not claimed:** any
reconciliation of a drop-size distribution across the spacing change. The
parent does not carry the moments that would make that possible, and the
species are advanced as independent scalars: the standard one-way treatment,
stated rather than glossed.

**Reflection of gravity and acoustic waves.** A specified boundary is
**perfectly reflecting**. The Davies zone damps what reaches it (`fcx` relaxes
toward the parent, `gcx` diffuses the short waves the first-order term would
reflect), but the boundary is **not transparent**, and an updraft's gravity
waves and the psolver=3 acoustic modes do return from it. The mitigation that
works is geometric: place the nest boundary so a reflected wave cannot reach
the feature of interest within the run. `WAVE_REFLECTION_LIMIT` carries this
sentence into every receipt.

**Feedback direction.** Declared by value as `nest.feedback`: `0` one-way,
`1` two-way. Both are implemented; neither is a default.

## 5. The live nest: what executes, on whose authority, and what still refuses

**Status: a fixed nest is implemented and device-executed (5.5 Decision). A
moving nest is implemented and host-tested against the kernel text; it has not
run on a device (5.8).**

A fixed nest is integrated **concurrently** with its parent on one device
(`cm1core.nest_live`, `cm1core.nest_live_kernels`). The three pieces the
previous revision of this section named as missing are closed as follows.

### 5.1 The staggered operator, decided by measurement

The boundary tables and the child's initial condition use
`nest_downscale.downscale_level_staggered` (`u` face/centre, `v`
centre/face, `w` and every scalar centre/centre) in its device form
`gm1_nest_capture`. The worry that kept it out of the boundary path (the
block-mean identity "does not carry over" to a face lattice) was tested rather
than assumed (`tests/test_nest_live_operators.py`):

* every child face that coincides with a parent face has normal offset exactly
  zero and symmetric tangential offsets, so the `r` sub-faces of **every**
  parent face inside the window sum, pair-ordered in the float64
  reconstruction, to exactly `r` times the parent face value;
* consequently the child's divergence integrated over each `r*r` block equals
  the parent's divergence of that cell (measured to 1e-11 of the spacing), so a
  re-imposed boundary tendency carries the parent's own flux through the
  child's boundary;
* a horizontally linear field is reproduced exactly on all four lattices.

Kernel text and CPU reference agree raw-word on every lattice, side, padded
origin and a shuffled thread order (compiled with `g++`), and on the RTX 5090
through NVRTC (`GM1_NEST_DEVICE_TESTS=1`).

### 5.2 Placement: WRF `solve_em.F`, mapped onto CM1's RK3 + psolver=3 step

| WRF (line) | what it does | GM1 placement |
|---|---|---|
| `frame/module_integrate.F:393,416,430,443` | parent `solve_interface`, then `med_nest_force`, then `integrate(nest)`, then `med_nest_feedback` | `CudaCartesianRuntime.step` → `NestDriver.after_parent_step`: tables, `time_refinement_ratio` child steps, feedback |
| `share/interp_fcn.F:2500,2583-2584` | `rdt=1.D0/cdt`; `bdy_txs=rdt*(psca-nfld)`, `bdy_xs=nfld` | `gm1_nest_capture` with `want_tend`: value = child now, tendency = `rdt*(interp(parent new) - child now)` in double, narrowed once |
| `share/mediation_force_domain.F:213`; `dyn_em/solve_em.F:372` | `dtbc=0` when forced; `dtbc=dtbc+dt` before each step | `NestLateralForcing.refresh` / `advance_clock`: sub-step `n` relaxes toward `(n+1)*dtc`; the target is linear across the parent step and equals the interpolated parent at its end |
| `dyn_em/solve_em.F:933-957` (`relax_bdy_dry`, `rk_step==1`), `dyn_em/module_em.F:1047-1080` (`rk_addtend_dry` folds `*_save` into `*_tendf`), `:2258-2276` (`relax_bdy_scalar`) | Davies relaxation once per large step into the tendency every RK stage re-reads | after `finalize_held_tendencies`, from the time-level-`n` state, into `held.uten/vten/wten`, `held.thten1`, `held.qten`, `arrays.ppten` (`gm1_nest_specified_relaxation`) |
| `dyn_em/solve_em.F:979` (`spec_bdy_dry`), `module_small_step_em.F:734-742` (zone excluded from `advance_uv`), `:1341,1351` (`spec_bdyupdate` on `u_2`,`v_2` after `advance_uv`) | specified zone advanced by the boundary tendency alone inside every small step | top of each small step gathers the specified rows (`gm1_nest_spec_gather`); after the pressure-gradient advance `u3d`,`v3d` = gathered + `dts`·tendency (`gm1_nest_spec_apply` mode 0) |
| `:1457,1467` (`t_2`,`mu_2` after `advance_mu_t`), `:1582-1604` (`ph_2`,`w_2` after `advance_w`) | thermodynamic and vertical specified update | after the implicit column step, `w3d`,`pp3d` = gathered + `dts`·tendency |
| `:2277` (`spec_bdy_scalar`), `:2305` (`rk_update_scalar`) | moist specified rows `q = q_old + dt_rk*tend` | after each `scalar_moist(nrk)`: `th3d = tha + dt_rk*tend`, `q3d = qa + dt_rk*tend` on the specified rows (mode 1). psolver=3 advances theta as an RK scalar, not in the acoustic loop |
| CM1 `solve1.F:1748-1751` (eqtset=2 adds `qpten`,`qtten`,`qvten`,`qcten` to the held tendencies), `solve2.F:1561-1576` (subtracts `dt*rate` after the final stage) | no WRF counterpart: WRF has no held diabatic estimate | the specified rows never received the rate (they advance by the boundary tendency alone), so after the final stage's specification `NestLateralForcing.restore_held_diabatic` adds `dt*rate` to their `pp3d`, `th3d`, `qv`, `qc`, and the removal returns exactly WRF's pre-physics value there. Before 2026-09-22 the removal subtracted a rate never added, once per child step, and on condensing or depositing outflow rows built 0.1-1 kg/kg hydrometeor columns in every nested run (qi 0.954, qs 0.378 and qs 0.098 kg/kg in three large El Reno nests) |
| WRF v4.6.1 `dyn_em/solve_em.F:4531-4716` (`IF (config_flags%specified .or. config_flags%nested)`: `spec_bdy_final` on `u_2`, `v_2`, `w_2` (nested only, `:4566`), `t_2`, `ph_2`, `mu_2`, every moist (`:4622-4623`), tracer and scalar field); `share/module_bc.F:2066-2216` | at the end of every step, after microphysics, the specified rows are set to `bdy + dtbc*bdy_tend` ("this code forces boundary values to specified values to avoid drift", `:4531`) | `NestLateralForcing.finalize_specified`, called by the child runtime at the `solve3` phase, before `solve3` copies the new state into the time-level-`n` arrays: the specified rows of `u3d`, `v3d`, `w3d`, `pp3d`, `th3d` and every `q3d` species are set to `value + dtbc*tend` (`gm1_nest_spec_apply` modes 2 and 3; `spec_bdyupdate`'s loop ownership, the y sides owning the corners; a species held at zero or above, after `pdefq`). With `dtbc` the parent step at the last child step, the rows end each parent step at the interpolated parent. Added on 2026-09-24; before it the live nest's specified rows kept whatever microphysics did to them until the next parent step's tables were cut from the child's own words (the ArWen chain had it from its first device pass). Device check: section 5.12 |

Variables. WRF is mass-coordinate: it relaxes `ru`,`rv` (`mu*u`), mu-coupled
`t`, `mu`, `ph`, and `w` (nested only). GM1 relaxes `u3d`,`v3d`,`w3d` directly
(its base density is horizontally uniform on the non-terrain grids this lane
admits, so `rho0*u` and `u` give the same term), `th3d` uncoupled, every `q3d`
species uncoupled, and `pp3d` (π′), which is GM1's only pressure prognostic
and has no separate `mu`/`ph` counterpart. `tke3d` is not forced: WRF's TKE
advance (`solve_em.F:2378-2416`) has no boundary call.

The one choice that is not a transcription: WRF's `mu` relaxation goes through
`mu_tend`, which `rk_tendency` rewrites every stage, so its mass relaxation
acts in RK stage 1 only. CM1's `ppten` is a held tendency built once per step,
so the π′ relaxation added to it acts in all three stages, as WRF's `u`, `v`,
`w`, `t` and `ph` relaxation does. The receipt records this
(`nest_execution.wrf_placement.choice_not_transcription`).

The child does not run CM1's `apmasscon` dry-mass correction: a WRF nest has no
mass fixer and takes its mass through the specified zone, and holding the
child to its own initial mass would add a domain-uniform π′ offset that the
specified zone contradicts on every boundary row. The parent keeps it.

### 5.3 Feedback, initial condition, clocks

`feedback = 1` writes `gm1_nest_restrict` block means (float64 raster sum,
blocks reaching the relaxation zone excluded) into the parent's current and
time-level-`n` carriers, refills the parent's pressure halos and rebuilds
`prs`/`rho`/`rr`/`rf` with the startup `getset` kernel. The child's initial
condition is the parent's state captured onto the whole child lattice (WRF's
`med_nest_initial`), taken after a restored parent is restored and after an
ensemble branch is applied. The child's lateral halos are CM1's open radiative
family whatever the parent's are. The child's clock counts its own steps but
takes its model time and its `domainlocx/y` from the parent at every parent
step: `dt/time_refinement_ratio` times a frame speed is often not a short
binary fraction, and an independently accumulated child integral would trip
`RuntimeClock`'s exactness stop mid-run.

Executed through the door on the RTX 5090 (`python -m cm1core cartesian
... --integrate`): a fresh nested run published parent and `nest-child`
frames at 30 s and 60 s with a `nest_execution` receipt (20 parent steps,
60 child steps, 20 feedback writes, 660 specified small steps); and a
**nest added at resume**: a single-domain parent run wrote restarts at
steps 10 and 20, and the same document with `irst = 1`, `rstnum = 1` and a
`[nest]` table resumed from restart 1, built the child from the restored
parent (`parent_state: restored restart`, `parent_completed_steps: 10`),
placed the child's clock at step 30 and published the child frame at
`model_time_s = 60.0` beside the parent's.

### 5.4 What still refuses, each by what it prevents

| code | when | wrong result prevented |
|---|---|---|
| `CM1-CARTESIAN-NEST-FOLLOW-INTEGRATION` | `follow_storm = true` with a CM1 pressure tracker over a parent whose four lateral boundaries are not all open (code 2) | the tracker's zero-gradient smoother halo is `bc2d` for open boundaries only (`src/bc.F:12-172`); near a periodic or rigid parent edge it would find a minimum CM1 would not and move the child toward it |
| `CM1-CARTESIAN-NEST-FOLLOW-CORRAL` | a tracked move would leave fewer than the corral's parent cells clear of the parent boundary; with `follow_on_edge = "hold"`, only after more than `follow_hold_limit * interval_steps` consecutive parent steps have cancelled one | WRF cancels the move and keeps a nest that no longer follows (`mediation_nest_move.F:849-859`); the receipt would name a storm-following nest refining air its vortex left |
| `CM1-CARTESIAN-NEST-FOLLOW-SPEED` | the surface tracker's maximum lies on an unclamped edge of its search window (`stop`, or the first sample of a run); `hold` holds such a sample instead | the vortex moved further than `follow_max_speed_m_s` allows in one interval; the tracked centre would be the window edge (measured: 80 m off, against 3.5 m when the speed is declared) |
| `CM1-CARTESIAN-NEST-FOLLOW-JUMP` | CM1's pressure centre moves further than `follow_max_speed_m_s * follow_interval_s` in one interval | the tracker has jumped to another pressure minimum; following it would carry the nest off the storm |
| `CM1-CARTESIAN-NEST-FOLLOW-LOST` | the search window holds no finite box mean, under either policy | WRF's own fatal "method failure" (`mediation_nest_move.F:544-548`); every box mean in reach is NaN, so the child's lowest-level winds are non-finite and no kept centre recovers it |
| `CM1-CARTESIAN-NEST-FOLLOW-RELOCATION` | `hold`: `follow_hold_limit` samples were held, and the next one's search over `follow_relocation_radius_m` finds no positive box mean, only flat ones, or its maximum on an unclamped edge | no circulation lies within the declared radius; holding on would publish a storm-following nest refining air its vortex left |
| `CM1-CARTESIAN-NEST-FOLLOW-CARRIER` | the child holds a device buffer with no move rule, or a shifted carrier with no parent counterpart | its words would stay at the old ground position under a child that moved |
| `CM1-CARTESIAN-NEST-FOLLOW-TERRAIN` | a child metric varies horizontally | metrics are not moved; the child would integrate on the metrics of the ground it left |
| `CM1-CFG-NEST-FOLLOW-SMOOTHING` | surface tracker without `follow_smoothing_radius_m`; the key with another tracker; a box under one child cell or wider than the child's interior | no default exists; a single-corner box chases the instantaneous maximum; a box that does not fit reads parent-forced rows |
| `CM1-CFG-NEST-FOLLOW-CORRAL` | `follow_corral_m` under the 3-cell stencil halo, or a placement already inside the corral | a moved child forced from parent halo words; the first move along that axis would stop the run |
| `CM1-CFG-NEST-FOLLOW-ON-EDGE` | `follow_on_edge` other than `stop`/`hold`, or declared without `follow_storm = true` and the surface tracker | an edge, lost or corral case handled by a rule the receipt did not name, or a policy no search window was evaluated under |
| `CM1-CFG-NEST-FOLLOW-HOLD-LIMIT` | `hold` without `follow_hold_limit`, or the limit without `hold` (`CM1-CFG-INTEGER` below 1) | a nest that lost its storm held for the rest of the run under a storm-following receipt; a limit no sample was counted against |
| `CM1-CFG-NEST-FOLLOW-RELOCATION` | `hold` without `follow_relocation_radius_m`, the radius without `hold`, or a radius less than one child cell beyond `(follow_hold_limit + 1) * follow_max_speed_m_s * follow_interval_s`, the reach of the sample that relocates (`CM1-CFG-NUMERIC-*` when not a finite positive number) | a relocation searching nothing beyond the window that just failed, or the whole child where another storm may win; a radius the run never evaluated |
| `CM1-CARTESIAN-NEST-RESTART-ENSEMBLE` | `[ensemble]` resumed from a nested restart that carries its child (5.10) | the branch perturbs the parent only; the restored child would start unperturbed inside a perturbed parent |
| `CM1-CARTESIAN-NEST-RESTART-MISMATCH` | a nested restart whose child receipt records other refinement ratios, spans or child parcel count (or presence) than the document, or a restored placement the document's zones cannot hold (5.10) | the child's words would be read onto another lattice or clock; parcels would end without a termination or be invented; the child would be forced from the parent's own boundary |
| `CM1-CARTESIAN-NEST-INITIAL-CONDITION` | `run.iconly = 1` | the host-only publication would name a child that was never built |
| `CM1-CARTESIAN-NEST-ADAPTIVE-STEP` | `dynamics.adaptive_timestep` | the child would keep the first step's clock while its tables were built for another interval |
| `CM1-CARTESIAN-NEST-ADAPTIVE-FRAME` | `closures.do_adapt_move` | the child's base state would stay in a frame the parent left |
| `CM1-CARTESIAN-NEST-BETA-PLANE` | `dynamics.beta_plane = 1` | `f` would be evaluated about the child's own centre |
| `CM1-CFG-NEST-CHILD-SPACING` | parent spacing / ratio has no binary32 word | the child's faces would drift off the parent's |
| `CM1-CFG-NEST-FOLLOW-UNIT` | `follow_max_cells` declared | a parent-cell count is a different speed on every parent spacing; `follow_max_speed_m_s` replaces it |

`follow_tracker` defaults to `cm1-pressure-held-offset`: CM1's own pressure
centre, with the nest kept at the displacement the document placed it at,
which holds a window placed over the surface vortex (9-11 km from the steady
mesocyclone pressure centre, dca8ab6) on the vortex; `cm1-pressure-centre` is
CM1's own adaptive-move target.

### 5.5 Device validation (RTX 5090, stretch_z = 1, nz = 75, 20 m lowest layer)

All device runs: `tools/run_nest_live_validation.py`, parent 600 m, child
200 m (ratio 3), `stretch_z = 1` column of 75 levels with a 20 m lowest layer,
WSM6, `sgs_model = 1`, spec/relax zones 1/4, outputs on the node's disk.

**(a) Kernels vs CPU references.** `gm1_nest_capture` (all four lattices,
full child and four sides, value and tendency), `gm1_nest_spec_gather` /
`gm1_nest_spec_apply` (both modes, spec zones 1 and 2), `gm1_nest_restrict`
(all lattices): raw float32 words identical to the NumPy references, by `g++`
on the kernel text (66 cases, linear and shuffled thread order) and by NVRTC
on the RTX 5090 (`GM1_NEST_DEVICE_TESTS=1`). The existing
`gm1_nest_specified_relaxation` and tracker device tests, never launched
before, also pass on the 5090.

**(b) Consistency over an unforced environment.** 100 parent steps (300 s),
maximum over every step of |child − parent captured onto the child|:

| environment | u m/s | w m/s | θ K | π′ | qv kg/kg | parent(fb=1) − parent alone: u / θ |
|---|---|---|---|---|---|---|
| periodic, uniform wind, same clock (q=1, nsound 10) | 1.1e-4 | 6.2e-5 | 3.3e-5 | 1.3e-7 | 1.2e-8 | 9.4e-5 / 2.4e-5 |
| periodic, uniform wind, q=3 | 2.7e-4 | 1.8e-4 | 1.7e-4 | 2.6e-7 | 1.2e-8 | 1.5e-4 / 1.6e-4 |
| periodic, WK shear, same clock | 1.1e-4 | 5.6e-5 | 3.7e-5 | 1.4e-7 | 1.2e-8 | 8.4e-5 / 2.2e-5 |
| open parent, uniform wind, q=3 | 2.4e-4 | 1.8e-4 | 1.7e-4 | 2.4e-7 | 1.2e-8 | 1.4e-4 / 1.6e-4 |
| open parent, WK shear, q=3 | 2.6e-4 | 1.8e-4 | 1.8e-4 | 2.5e-7 | 1.2e-8 | 1.5e-4 / 1.5e-4 |

With `feedback = 0` the nested parent is **bit-identical** to the same parent
integrated alone, in every environment. The child is **not** equal to the
captured parent to rounding, and the reason was located rather than assumed:
the single-domain model itself does not integrate a horizontally homogeneous
column identically at 600 m and 200 m. Two periodic single-domain runs, same
clock, TKE zeroed, first diverge in `solve1-held-tendencies` (bisected phase
by phase): the TKE closure's floor and its grid-length mixing scale give
`tketen` 1.0e-4 vs 2.2e-5 and `thten1` 1.3e-5 vs 2.5e-6 K/s at 600 m vs 200 m,
and after one step the columns differ by w 2.5e-5 m/s and θ 3.1e-5 K, the
same size as the child−parent differences above. θ drift after one step
scales with the spacing (1.3e-5 K at 200 m, 3.9e-5 K at 600 m, 9.9e-5 K at
1200 m). `sgs_model = 1` is mandatory in this lane. The unforced child's
columns equal a single-domain 200 m run **bit for bit** (0.0 difference, three
steps), so what the child adds is its own spacing's physics; what remains
between child and parent is that resolution dependence, the pressure
gradients it sets up across the relaxation zone (visible as u departing from
a u both single-domain runs hold exactly), and, with q = 3, the column's
time-step dependence. The rounding-level criterion is therefore unattainable
for any nest in this lane; the measured residual is at the size of the
model's own 600 m-to-200 m column difference and is recorded as such.

A quieter environment was tried and is recorded because it failed: a
constant-θ (300 K), constant-qv (0.1 g/kg), windless `isnd = 7` sounding is
not a rest state for this model (the single-domain parent alone drifts by
θ 0.31 K and w 0.06 m/s in 300 s), so child − captured parent grew to
u 0.07-0.30 m/s, θ 1.2e-2 K. It does not isolate the nest and is not used as
evidence; `feedback = 0` still left the nested parent bit-identical to the
parent alone.

**(c) Reflection.** Parent 72x40 at 600 m, nest span 30x24 (child 90x72 at
200 m), frame-relative wind 12 m/s, against one 216x120 grid at 200 m over the
whole parent domain on the child's clock; errors over the child interior (5
cells in from each edge).

*Skamarock-Klemp inertia-gravity packet* (iinit 11, 0.01 K, 3000 s): the
child's departure from the fine grid is at most w 3.0e-3 m/s (fine window
signal 4.7e-3) and theta 3.7e-3 K (1.2e-2); at every sample it tracks the
PARENT's own departure from the fine grid's block means over the same window
(w 2.9e-3 at the end, ratio child/parent 1.03-1.2 over the run). The nest adds
no measurable error beyond the coarse grid's own; a reflected wave larger than
about 20% of the coarse-grid error would show. feedback 0 and 1 agree to 3
significant figures.

*Warm bubble* (iinit 1, +2 K, WK sounding, 1800 s): the bubble triggers deep
moist convection (w 37 m/s) and the child and fine-grid storms diverge as
convective trajectories do (w max diff 30 m/s once the updraft is inside the
window, 0.004 m/s before it arrives). This is not a reflection measurement and
is recorded only as the case the task named; the packet case above is.

**(d) Test suite.** Against the suite before this change: no new failures; two tests that pin evidence-inventory
hashes of `cuda_psolver3_terrain_open.py` were re-pinned to the file the
acoustic hook changed.

**Decision.** `CM1-CARTESIAN-NEST-INTEGRATION` is lifted. Criterion (b) as
first written -- child equal to the captured parent to rounding -- was the
wrong criterion: the TKE closure's held tendencies depend on grid spacing, so a
child is not supposed to integrate like its parent. The criterion that follows
from the physics is that the unforced child equals a single-domain run at the
child's own spacing, and it does, bit for bit. Kernels (a) and reflection (c)
passed. What the live nest still does not do -- run with an adaptive step, an
adaptive frame or a beta plane -- stays refused by name in 5.4. Moving during
integration is implemented in 5.8, restarts of a nested run in 5.10.


### 5.6 Vertical refinement: a child on its own, finer deck

**Status: implemented and host-checked (CPU references, `g++` on the kernel
text); not yet executed on a device.**

Until this revision the child copied the parent's column word for word. A
child may now declare its own CM1 vertical deck over the parent's model top:
`[nest]` keys `child_nz`, `child_stretch_z` (1, 3 or 4) and the lengths that
branch reads (`child_str_bot_m`, `child_str_top_m`, `child_dz_bot_m`,
`child_dz_top_m`, or `child_levels_m`); the one-way downscale takes the
child's own document through `tools/nest_downscale_init.py --child-config`.
Absent, nothing changes: the `child_vertical` key is not projected, and the
canonical hash of every shipped nest document is identical to the tree before
the keys existed. Implementation: `cm1core.nest_vertical`.

**Admission** (`nest_vertical.vertical_refinement_defect`, refused as
`CM1-CFG-NEST-VERTICAL-REFINEMENT` / `CM1-NEST-DOWNSCALE-VERTICAL`):

| refused | wrong result prevented |
|---|---|
| child model top is another binary32 word | parent air above the child's lid, or child levels CM1's constant extension would fill from the parent's last level |
| a child layer thicker than every parent layer it overlaps | the child averaging away structure the parent resolves and feedback returning it as resolved |
| child deck infeasible under CM1's own generator (`CM1-CFG-NEST-CHILD-VERTICAL-DECK`) | the same refusal a parent `[grid]` gets, renamed onto the `nest.child_*` key |
| `child_*` keys without `child_nz`/`child_stretch_z` (`CM1-CFG-NEST-CHILD-VERTICAL`) | a declared deck silently falling back to the parent's |

**Parent → child: CM1's rule in WRF's order.**

| authority | what it decides | GM1 |
|---|---|---|
| CM1 `src/restart_read.F:564-575`, `:600-609` | `kold` search over the old deck's ghosted `sigma0(0:nk+1)` / `sigmaf0(0:nk+2)`; REAL `rz` | `build_cm1_vertical_map`, a literal transliteration |
| CM1 `src/restart_read.F:3037-3039`, `3302-3406` (`prep_for_interp`) | sequential plane reads; first level duplicated into the lower plane, lower = upper once no old level is left | same function, the state machine literally |
| CM1 `src/restart_read.F:3437-3444`, `3458-3465` (`do_interp`) | eight REAL weight products widened to DOUBLE, summed in written order, narrowed once | `gm1_nest_vinterp`, evaluated at `rx = ry = 0` with all eight terms |
| CM1 `src/restart_read.F:1207` | `tkea` is not read on a new grid | the downscale tool leaves the child's own TKE; the live nest keeps capturing TKE through `sigmaf`, as WRF interpolates its nest's TKE |
| WRF `external/RSL_LITE/force_domain_em_part2.F:126`, `:141-147`; `interp_domain_em_part2.F:132` | vertical interpolation on the intermediate (parent-resolution) grid **before** the horizontal one, for forcing and for the initial condition | the vertical step fills a window of parent columns (`VerticalWindow`, the donor range plus one point), then the unchanged `gm1_nest_capture` reads it with donors at offset 1 |
| WRF `share/interp_fcn.F:1910-2013` | u, v, t, moist on half levels; w, ph on full levels | u, v, θ, π′, q on `sigma`; w (and TKE) on `sigmaf` |
| WRF `share/interp_fcn.F:2073-2106` (`inter_wrf_copy`) | monotone cubic in log-pressure altitude | **not taken**: WRF's eta levels move with column mass, GM1's are fixed heights, and CM1's linear `sigma` rule is what CM1 applies between two of its own decks |
| WRF `share/module_check_a_mundo.F:2724-2742` | `rebalance = 1` required | **not taken**: CM1's `interp_on_restart` does no rebalance; the child's base state is built by `base.F` on its own levels and the interpolated perturbations are left as interpolated |

The window keeps the capture's clamps disengaged exactly as the whole parent
lattice does (`PARENT_STENCIL_HALO`), so on identical decks the refined
composition is the copy path bit for bit, tested through both kernels.
Below the parent's first scalar level the 5 m child's 2.5 m and 7.5 m levels
take CM1's constant extension with the ghost-derived `rz` (0.625, 0.875) on
two copies of parent level 1.

**Child → parent: GM1's choice.** WRF refuses feedback with vertical nesting
(`share/module_check_a_mundo.F:2711-2722`). GM1 admits it with
`gm1_nest_restrict_vertical`: each parent layer `k` becomes
`Σ_m ρ0_m ov(m,k) B_m / Σ_m ρ0_m ov(m,k) / r²`, with `B_m` the existing
float64 raster block sum of child layer `m` and the weights normalized per row
in float64. Because the child layers tile `[0, ztop]`,
`Σ_k M_k P_k = Σ_m ρ0_m Δz_m C_m` with `M_k = Σ_m ρ0_m ov(m,k)`: the child's
column integral of every mass-specific carrier (dry air, water, heat,
momentum) is the parent's, in the child's mass measure (tested to 1e-13
relative in float64, and to the float32 narrowing of each parent word through
the 3-D reference). `w` is restricted over `rf0`-weighted dual cells
`[zh(f-1), zh(f)]`, with its ground and top faces the child's own coincident
faces copied. **Not claimed:** that `M_k` equals the parent's own `ρ0 Δz`; the
two decks sample one base state at different heights.

**What a 5 m lowest layer does to the clock, by CM1's own rules.**

* `psolver = 3` is vertically implicit: `src/misclibs.F:857-868` sets
  `dtsm = 0.60*min(dx,dy)/360` with no `dz`, so the 5 m layer changes neither
  `dtsm` (0.0333 s at 20 m) nor `nsound`. (Under `psolver = 2`,
  `:846-855` would take `0.50*min(dx,dy,dz)/360 = 0.00694 s`, and
  `:883-885` would give `nsound` 14, still over, so 16 at `dt = 0.1 s`; that
  solver is not admitted in this lane.)
* The advective stop does change. `calccfl` (`src/misclibs.F:2584-2718`) forms
  `sqrt((ū dt/dx)² + (v̄ dt/dy)² + (w̄ dt/Δz_k)²)` (Δz through `mh`,
  `:2627-2629`) and stops at 1.50 (`:2714`):

  | step | lowest layer | w alone at 1.50 | w with 60 m/s horizontal wind |
  |---|---|---|---|
  | live nest, 0.5/5 = 0.1 s | 20 m (parent deck) | 300 m/s | 294 m/s |
  | live nest, 0.1 s | **5 m** | **75 m/s** | **73.5 m/s** |
  | downscale, 0.125 s | 20 m | 240 m/s | 232 m/s |
  | downscale, 0.125 s | **5 m** | **60 m/s** | **58.1 m/s** |

  Every child layer below 170 m is 5 m thick, so the bound holds through the
  lowest 170 m. `vertical_implicit_diffusion = true` in both decks, so the
  subgrid vertical diffusion adds no explicit limit.
* Memory, by `cartesian_door.nest_allocation_plan`: the 300x300 child's peak
  listed phase payload is 2.86 GiB at 102 levels against 2.11 GiB at 75
  (×1.356). The four vertical window scratches (102-103 levels over the
  62-63 point window) are 6.05 MiB together.

**Decks.** A live nest (100 m parent on 75 levels with a 20 m lowest layer,
20 m child on 102 levels with a 5 m lowest layer, feedback 1) and the one-way
child on the same deck.

**Evidence (host).** `tests/test_nest_vertical_refinement.py`, 51 tests: the
map against a second, 1-based transliteration of CM1's search and state
machine; identical decks interpolating to the same words; admission and every
refusal by name; a smooth column (`θ = 300 + 4 sin(2πz/3000) + 0.003z`) within
the linear-interpolation bound `h²/8·max|f″|` at every child level (constant
extension bounded by `|f′|` times the distance); `gm1_nest_vinterp` and
`gm1_nest_restrict_vertical` kernel text against their references on all four
lattices, linear and shuffled thread order; identical decks through
vinterp+capture and through the vertical restriction reproducing
`gm1_nest_capture` and `gm1_nest_restrict` raw-word; column mass and heat
conservation; the downscale tool's transfer on synthetic arenas for both
paths; the config grammar, canonical projection and door binding.

**Device-only, not yet done** (no GPU was available for this revision):

1. NVRTC compilation of `NEST_VERTICAL_KERNEL_SOURCE` and a raw-word comparison
   of `gm1_nest_vinterp` / `gm1_nest_restrict_vertical` against the references
   on the device (the `g++` evidence above covers the text, not NVRTC).
2. The child runtime built on a 102-level deck: `IdealizedSource` base state
   on the child's own levels, the column capacity profile for `nz = 102`, and
   the stretched operator families on a 5 m bottom layer.
3. A consistency run in the 5.5(b) style: a horizontally homogeneous
   environment with the child on the 5 m deck, compared against a
   single-domain 20 m run on the same deck (the criterion 5.5 established),
   with `feedback = 0` still leaving the parent bit-identical; and with
   `feedback = 1`, the parent's column integrals of `ρ0 θ` and `ρ0 q` over the
   covered window before and after each feedback write.
4. The downscale tool end to end on a real parent frame and a 102-level
   `iconly` child restart (needs the Rust publisher built).

**Found, not changed.** `gm1_restart_grid_interp.build_cm1_restart_axis_map`
gives new levels below the first old level `fraction = 0`; CM1
(`restart_read.F:572` with `sigma0(0)`) and the Rust producer
(`child_atmosphere.rs`, fraction from `old[0]`) take `rz` from the ghost level
on duplicated planes. The two agree except where
`d·f32(1-rz) + d·rz` rounds away from `d` in double before narrowing, which the
Rust producer's own parity evidence would expose; `nest_vertical` follows CM1.

### 5.7 What was executed before this revision

The rows below are unchanged from the host-checked estate this seam landed
with, and remain true:

| Piece | Where | How it is checked |
|---|---|---|
| `[nest]` grammar, each key refused by key and value | `config.py` | `tests/test_nest_config_to_call_site.py` |
| geometry, placement, donor maps, face alignment | `nest_geometry.py` | same |
| the child's clock, by CM1's own `dtsmall` | `nest_geometry.py` + `gm1_runtime.py` | same |
| conservative extremum-limited interpolation | `nest_boundary.py` | exact block-mean identity, ratios 3/5/7/9 |
| Davies relaxation, packaged CUDA text | `nest_boundary.NEST_RELAXATION_KERNEL_SOURCE` | `g++` raw words vs CPU oracle, three negative controls; now also launched on the 5090 |
| storm tracker, host and device | `nest_geometry.py` | same face; now also launched on the 5090 |

The device tests stay gated by `GM1_NEST_DEVICE_TESTS=1` and the GPU
mutex, for the reason given when they were written: CuPy may be installed on
machines that run production storms.

### 5.8 The moving nest (`cm1core.nest_move`, `cm1core.nest_move_kernels`)

**Status: implemented; verified on the host against the packaged kernel text;
never run on a device.** `follow_storm = true` with `--integrate` is admitted.

**Authority.** WRF's moving nest. `share/track_driver.F` is WRF's point-track
output, not a vortex tracker; the vortex centre is `time_for_move2` under
`VORTEX_CENTER`.

| WRF (line) | what it does | GM1 |
|---|---|---|
| `frame/module_integrate.F:442-449` | `med_nest_feedback`, then `med_nest_move` | `NestDriver.after_parent_step`: tables, child steps, feedback, then `MovingNest.after_parent_step` |
| `share/mediation_nest_move.F:471-487` | search window: previous centre ± `60*vortex_interval*max_vortex_speed/dx`, whole nest the first time | `follow_max_speed_m_s * follow_interval_s / dx_child`, REAL bounds truncated to INTEGER |
| `:527-542` | first strict extremum in `DO j / DO i` order | `gm1_nest_vortex_centroid` (serial), mirrored to a maximum |
| `:556-606` (`a=0, b=1`) | centroid of `vmax - height` over `rsmooth/dx`; unchanged when every weight is zero | centroid of `zbar - vmin` over `follow_smoothing_radius_m/dx_child` |
| `:659-704`, `:122-127` | `move = int(disp/pgr)` clamped to one parent cell; `vc -= move*pgr`; EM refuses larger moves | `wrf_move_cells`, every parent step, both axes |
| `:816-859` | cancel a move that ends within `corral_dist` of the outermost boundary | `CM1-CARTESIAN-NEST-FOLLOW-CORRAL` stops the run instead; with `follow_on_edge = "hold"` the move is cancelled per axis as WRF cancels it (`:849-859`) for at most `follow_hold_limit * interval_steps` consecutive parent steps |
| `:482-485`, `:544-548`, `:590` | the window is bounded by speed over the time since the previous centre and not checked; nothing found is fatal; all-zero weights leave the centre | `stop`: `CM1-CARTESIAN-NEST-FOLLOW-SPEED` / `-LOST`; `hold`: keep `vc` on a non-positive or flat maximum, an edge maximum, or a centroid beyond reach (`lost-hold` / `edge-hold` / `too-fast-hold`, in that order), the window and reach widening by one interval's travel per held sample and every `hold` window one child cell past its reach, then relocate within `follow_relocation_radius_m`; `-LOST` under both, and under `hold` also on a found maximum whose centroid is not finite |
| `dyn_em/shift_domain_em.F:52-80`, `frame/loop_based_{x,y}_shift_code.h` | shift every state array by `parent_grid_ratio` per parent cell; `imask` marks the exposed strip (`:64-72`) | `gm1_nest_shift` over each carrier's lattice |
| `mediation_nest_move.F:188`, `:282-292` | `med_interp_domain` over `imask == 1`; copy level 2 into level 1 there | `gm1_nest_capture` onto the moved lattice, then the shift overwrites all but the exposed strip; `*a` and `*3d` each from their own words and their own parent carrier |
| `share/interp_fcn.F:2583-2584` at the next `med_nest_force` | boundary tables from the moved nest | `NestLateralForcing.refresh` with the moved geometry |
| `Registry/Registry.EM_COMMON:2354-2358` | `vortex_interval`, `max_vortex_speed`, `corral_dist`, `track_level` | `follow_interval_s`, `follow_max_speed_m_s`, `follow_corral_m`; the lowest scalar level |

**The tracker, and why it is this one.** `wrf-surface-circulation` runs WRF's
algorithm on the child's own lowest-level winds with the tracked field changed
from `track_level` height to the **area-averaged vertical vorticity**: CM1's
corner vorticity (`src/parcel.F:1064-1066`) averaged over a
`(2h+1)^2` corner box, `h = floor(follow_smoothing_radius_m / dx_child)`, which
is the circulation around the box divided by its area (`gm1_nest_zeta_box`).
Section 5.4 records (dca8ab6) that CM1's pressure centre sits 9-11 km from the
surface vortex and that the instantaneous |zeta| maximum was a gust-front
filament 18-36 km off on eight of twelve frames; the box mean is what keeps a
short-lived sub-vortex, whose circulation is small beside the box's, from
capturing the tracker, and `h >= 1` is enforced. Admissible centres keep the
whole box off the specified and relaxation rows. CM1's pressure trackers stay
selectable and drive the same move rule.

Departures from WRF, each stated in the receipt (`departures_from_wrf`):
the tracked field (above); WRF's `irad = 6` nest-cell clamp (`:671-680`) is
omitted -- it never changes the move for `parent_grid_ratio <= 6` and makes a
ratio-7-or-more nest immovable; WRF's `+0.5` (`:602-603`) is not added because
the corner lattice already is its staggered centre frame; centroid sums are
float64; a corral-cancelled move stops the run; and a maximum on an unclamped
search-window edge stops the run (`CM1-CARTESIAN-NEST-FOLLOW-SPEED`), which
WRF does not check.

**The hold policy** (`nest.follow_on_edge = "hold"`, surface tracker only;
`nest.follow_hold_limit` and `nest.follow_relocation_radius_m` required with
it). Under `stop`, one tracker sample out of reach -- occlusion, cyclic
re-formation, a sub-vortex orbiting the parent circulation faster than the
declared speed -- or one corral hit ends the whole job at that sample, and
everything since the last restart with it (a nested run writes restarts, 5.10,
but a stop is still a stop). Once a centre exists, `hold` classifies every follow-interval
sample before taking its centroid, in this order: `lost-hold` (largest box
mean not positive, or all centroid weights zero -- first, because such a
window's first strict maximum is its `(is, js)` corner, an edge no vortex
reached), `edge-hold` (maximum on an unclamped window edge) or `too-fast-hold`
(centroid beyond the search reach from the previous centre on an axis). The
reach is `follow_max_speed_m_s * follow_interval_s` times the follow
intervals since the last accepted centre: WRF bounds its window by
`max_vortex_speed` over the time since its previous centre (`:482-485`),
which is always one interval in WRF because WRF takes every centroid, while
a held sample keeps an older centre. Under `hold` every search window (the
held search and the relocation) is one child cell wider than its reach
(`HOLD_WINDOW_SLACK_CELLS`), because WRF's edges are `INT` of `vc +- reach`
and the maximum is a corner: without the slack a corner maximum within reach
on the side the vortex moves toward lands on the truncated edge, and a
vortex at 19 or 19.5 m/s under a 20 m/s reach was edge-held on 10 and 25 of
41 samples (measured below). The too-fast test and the relocation radius
keep the reach itself; `stop` keeps WRF's edges bit for bit. What the
widened window guarantees, stated exactly: a vortex whose displacement from
the last **accepted** centre is within `held + 1` reaches on each axis is
found again. That is not a statement about the vortex's speed. The
displacement is measured from that centre's centroid, so a centroid that
lagged the vortex when it was accepted (a box clipped by the admissible
boundary, or the plateau bias of a box many times the core) is carried into
every widened window, and at or over the declared speed the window grows no
faster than the vortex: near the declared speed expect alternating
edge-holds, and over it a relocation every `follow_hold_limit + 1` intervals
or a stop by name (measured below). A held sample leaves `vc` in place, so
the nest holds its position about the last accepted centre (no move toward
the held sample), and is listed with its parent step, model
time, window, maximum, value, centroid, kept centre and a reason sentence; its
tracker row carries `held = true`. A clean sample resets the count and
following resumes. When `follow_hold_limit` samples have been held and the
next would be held too, the tracker **relocates**: it searches the same
box-mean plane over the kept centre plus or minus `follow_relocation_radius_m`,
which must reach one child cell past that sample's `(follow_hold_limit +
1)`-interval reach (a second launch of `gm1_nest_vortex_centroid` on the plane
already on the device; no kernel was added), takes WRF's first strict maximum
and centroid
there, and lists a `relocation` event with the triggering sample, the window,
the old and new centres and the displacement; its tracker row carries
`relocated = true`. The nest then moves toward the new centre one parent cell
per parent step (`:122-127`), never jumping. The run stops only when holding
cannot continue safely: a window with no finite box mean (`-LOST`, as under
`stop`: the child's winds are non-finite), a found maximum whose centroid is
not finite (`-LOST` under `hold` only, on a sample or a relocation, before
any centre is taken: a NaN box mean within one smoothing radius of the
maximum makes `vmin`, the weight sum and the centroid NaN, `:556-606`;
`stop` takes that centroid as WRF does, and the NaN centre ends the run one
parent step later without a name, the path before the key), a relocation
that finds no positive
circulation, only flat box means, or its maximum on the relocation window's
edge (`CM1-CARTESIAN-NEST-FOLLOW-RELOCATION`), or more than `follow_hold_limit
* interval_steps` consecutive parent steps whose move the corral cancelled.
Moves into the corral are cancelled per axis exactly as WRF cancels them
(`move_cd_x = 0` for X and `move_cd_y = 0` for Y, each only when that axis
moves into the corral, `:849-859`), so the nest keeps following on the free
axis; each such parent step is a `corral-hold` event. The receipt lists every
relocation, up to 4096 held samples and, separately, up to 4096 corral holds
(one per parent step, so a long corral episode cannot unlist a held sample or
a relocation), counts the rest, and records per-kind counts and the longest
consecutive runs; move rows name the accepted or relocated sample they follow
(`tracked_at_parent_step`) and the held samples since; `departures_from_wrf`
states the five hold departures. Absent or `stop`, every path, tracker row and
receipt is byte-identical to the code before the key (digest measured at
b200ad4 and pinned in `tests/test_nest_move.py`).

**The smoothing radius, measured against the failure it was suspected of.** A
2-hour moving 20 m nest stopped after 90 s with `CM1-CARTESIAN-NEST-FOLLOW-SPEED`
(`follow_max_speed_m_s = 20`, `follow_interval_s = 15`, `R = 200 m`): the
strongest box mean sat on the window edge. `tests/test_nest_move.py` section 12
reproduces that on synthetic lowest-level winds -- a 100 m parent, a 5 km 20 m
child, a 1 km-core parent circulation translating at 5.8 m/s and a 30 m-core
sub-vortex orbiting it 350 m out at 21 m/s. With `R = 200 m` the first sample's
maximum is the box holding the sub-vortex and the circulation's shoulder, 266 m
from the parent core, and `stop` ends the run on the window edge at 75 s with
the parent circulation still inside the window. Under `hold` (limit 4, radius
1520 m, the smallest admitted) the same run survives ten minutes holding no
sample at all -- the maximum `stop` refused at 75 s lay 14.65 cells from the
kept centre, inside the 15-cell reach on the `INT`-truncated window edge,
and `hold`'s windows reach one cell past it; with WRF's edges (the slack set
to zero) it holds 6 samples, none consecutive, and relocates nothing --
**but `hold` does not follow the parent circulation**: on every one of the
41 samples the tracked centre is the box holding the sub-vortex, 219-272 m
(219-319 m with WRF's edges) from the parent circulation, and the nest
centre is up to 357 (440) m from it. It survives because
this synthetic sub-vortex orbits the parent circulation indefinitely. (Before
the reach widened with held samples, the same run held 21 samples and
relocated 5 times, each relocation landing back on the sub-vortex's box, with
the nest parked for 60 s at a time and up to 696 m off.) Whether the tracker
should default to a wider box, as WRF's `rsmooth = 100000.` m -- 25/9 of its default 36 km reach
(`mediation_nest_move.F:342,556`, `Registry.EM_COMMON:2354-2355`) -- suggests,
was measured rather than assumed. On the reproduction, `R` = 300, 600 and 840 m
(WRF's ratio) all follow for ten minutes, within 191, 51 and 33 m. On a
translating 60 m-core vortex, `R` = 200 m centres it within 3 m (53 m with
0.02 s-1 corner noise) and 300 m within 87 m, while 600 and 840 m stop the run
at the first follow sample: every box near the vortex holds its whole
circulation, the box mean is a plateau, and its first strict maximum falls on
the plateau's edge. WRF's ratio does not carry over because WRF takes its
extremum on the raw field and only its centroid is wide, whereas `R` sets both
here. `R` belongs between the orbit radius of the sub-vortices the nest must
not chase and a few core radii of the vortex it must centre -- a property of
the storm, not of the reach -- so `follow_smoothing_radius_m` keeps no default;
a nest meant to follow a tornado's parent circulation through multiple-vortex
phases wants `R` of the order of that circulation's core (600-1000 m here), and
`hold` is the survival path when a sample still falls out of reach, not a way
to follow the right circulation: with `R` = 600 or 840 m on the 60 m-core
vortex, `hold` keeps the run alive (1 and 10 held samples) with the centre
still about 300 and 470 m off.

**The widened reach, measured against the pinned window it replaced.** A
300 m-core vortex translating at 15 or 18 m/s under the reproduction's 20 m/s
and 15 s reach (15 child cells), with no circulation for one sample at 60 s
or for four samples (60-105 s), `hold` limit 4 and a 1520 m radius: every
missed sample is one `lost-hold`, the next sample is accepted inside the
widened window, nothing relocates, and the tracked centre is within 4.1 m of
the vortex afterwards. With the window left at one interval's reach (the
code before this revision), the 15 m/s vortex missed once lies 30 cells from
the kept centre at 75 s, sits on the pinned window's edge for 3 more samples
(box means 0.21, 0.075, 0.011 as it leaves), and relocates with the tracked
centre 898 m off; with the 1 km radius admitted before this revision that
relocation found nothing and stopped the run at 120 s, 60 s after `stop`
would have. Missed for five samples, one more than the limit, the relocation
finds no circulation and the run stops by name at 120 s.

**The window's slack and the guarantee, measured near and over the declared
speed.** A 1 km-core vortex translating east at 17, 19, 19.5 and 20 m/s
under the same 20 m/s, 15 s reach, `R` = 600 m, limit 4 and a 1520 m radius.
With the slack no sample is held at any of the four speeds, the tracked
centre stays within 4 m and the nest within 390 m, and the accepted centres
still never move faster than the declared speed
(`tools/check_nest_following.py` passes). With WRF's `INT`-truncated edges
instead (`HOLD_WINDOW_SLACK_CELLS` set to zero): 17 m/s holds nothing; 19
m/s (5 per cent under) is edge-held on 10 of 41 samples with the nest parked
up to 660 m off; 19.5 m/s on 25 of 41 -- every other interval, up to 968 m
off; 20 m/s spends the limit and stops with `-RELOCATION` at 150 s; and
`stop` refuses the 19 m/s vortex at 45 s, when the kept centre at 128.28
puts the east edge at int(143.28) = 143 and the vortex at 142.75, 14.47 cells
away, has its corner maximum there. The lag the guarantee excludes: a 300
m-core vortex at 19.5 m/s under `R` = 840 m, whose box the admissible
boundary clips early in the run. With WRF's edges the first sample is
edge-held, the accepted centres that follow lag the vortex by 4 cells at 105
s and 21 at 180 s, each cycle's widened windows grow at the vortex's own
speed from a centre further behind, and the run stops with `-RELOCATION` at
480 s after 24 edge-holds with the tracked centre 2.1 km off; with the slack
it is edge-held 4 times, none consecutive, and stays within 293 m for ten
minutes. Over the declared speed only the radius helps: the 1 km-core vortex
at 20.5 m/s (2.5 per cent over) is edge-held on every sample, the widened
window never outpaces it, and the smallest admitted radius, 1520 m, stops
the run at the first relocation, 90 s, with the maximum on the relocation
window's edge (75 s with WRF's edges); 2000, 2500 and 3040 m each survive
ten minutes with 7 relocations (8 with WRF's edges) and the nest up to 2 km
off, while a limit of 8 with 3040 m relocates nothing and parks the nest 3.5
km off. `CM1-CARTESIAN-NEST-FOLLOW-RELOCATION`'s remedy therefore names the
radius first and says when the speed, not the limit, is the other knob.

**For a deck that stopped as the 2-hour 20 m nest did** (`R` = 200 m, 20 m/s,
15 s): the fix is the smoothing radius -- `R` of the order of the parent
circulation's core, about 600 m on the reproduction -- with `hold` as the
backup for the samples that still fall out of reach, and a relocation radius
with margin over the smallest admitted, `(follow_hold_limit + 1) * 300 m +
20 m` (1520 m for a limit of 4): at least twice it, 3040 m, since the
smallest admitted radius has no margin (a vortex 2.5 per cent over the
declared speed stops the run at its first relocation, above) and a longer
hold limit only forces a larger radius without catching a faster vortex.
The price is measured too: a 600 m box sits about 300 m off a compact 60 m-core
vortex, so the nest follows the parent circulation rather than centring a
tornado inside it -- well inside a child several kilometres across, but
not a tornado-centre track. `hold` alone with `R` = 200 m would keep such a
run alive while tracking the sub-vortex. Neither has run on the storm; item 7
below is the device check.

**The carriers.** Every registry buffer of the child is named: *shifted* (the
padded winds and pressures and their `*a` levels, `rru/rrv/rrw`, `ppx`,
`ppten`, `th`, every `q` species, the held tendencies, TKE, the microphysics
persistence including the surface precipitation accumulators, `znt/xland/
mavail/tsk`, the `output_s*s` swaths), *rebuilt* (`prs/rho/rr/rf` by `getset`
after the move; reflectivity and `uh` at publication), *invariant* (metrics,
checked horizontally uniform on the device before the first move; base
profiles; `uh/vh/ruf/rvf`), *edge history* (`radbc*`), or *step workspace*
(buffers the restart profile does not carry either). Anything else refuses by
name before a word moves. Every restart-carried child buffer is shifted,
rebuilt or edge history (`tests/test_nest_move.py`).

*Edge history, stated rather than hidden.* `radbc*` are the child's own
open-halo phase speeds along its own edges and are left in place. irbc=1 and
irbc=2 rebuild them from the winds before any read; for irbc=4 they are
re-produced after the next RK pressure stage (`solve2.F:1258-1271`), so the
first stage after a move fills the child's halo from the pre-move edge history.
Those halo words feed only rows the specified zone overwrites every small step.
Not measured.

**On a child with its own vertical deck (5.6).** The exposed strip is filled
through the same vertical-then-horizontal path the boundary tables and the
initial condition use: `NestVerticalCoupling.relocate` re-places each stagger's
parent window over the moved footprint (its extent depends only on the span),
`gm1_nest_vinterp` fills it on the child's levels, and `gm1_nest_capture`
reads it. Surface planes need no vertical step. The next parent step's tables
and feedback read the relocated windows and the moved geometry.

**Feedback across a move.** Before the move, from the unmoved footprint; one
parent step later, from the moved one. Parent cells the nest leaves keep the
child's last block means; parent cells it enters receive block means of the
parent's own interpolation.

**Verified on the host** (`tests/test_nest_move.py`, no CUDA context):

* `gm1_nest_shift`, `gm1_nest_zeta_box`, `gm1_nest_vortex_centroid`: raw words
  identical to the CPU references by `g++` on the packaged text, all four
  lattices, padded owners, one- and two-parent-cell moves, linear and shuffled
  thread order; flat and all-NaN centroid cases; the launchers' argument order
  driven through the same text.
* The move leaves every retained point the unmoved child's word and every
  exposed point the parent's capture, raw word for raw word, on all lattices.
* **A moving nest over a translating field is the interpolated parent, bit for
  bit**: uniform flow over eight moves, and a translating non-uniform feature
  over five, on all lattices. Section 5.5's caveat decides the design: an
  integrated child departs from the captured parent by its own spacing's
  closure, so the case isolates what the move itself adds, which is nothing.
* **Feedback mass across a diagonal move**: blocks covered before and after are
  bit-identical, cells left keep the child's fed-back means, cells entered
  differ from the parent by at most 2 float32 ulps, and the parent total changes
  by no more than that bound summed over the entered cells.
* **Tracker**: a sub-vortex with 2.4 times the main vortex's point vorticity
  and a thirtieth of its circulation does not capture it (centre within half a
  child cell of the main vortex); a stronger decoy outside the speed-bounded
  window does not capture a tracker that has a previous centre; a 12 m/s vortex
  followed with 20 m/s declared was tracked to 3.5 m (child dx 20 m) with the
  nest centre lagging at most 300 m (one 30 s interval of travel is 360 m) over
  17 one-cell moves until the corral stopped it by name. With 10 m/s declared
  the unchecked centre was 80 m off; that case now refuses.
* `MovingNest.execute` over real registry names and shapes, run through the
  `g++`-compiled capture and shift kernels: every shifted carrier equals the
  reference, every other buffer is untouched, geometry and forcing move, halos
  and getset are rebuilt once.
* The hold policy on the host: an edge hit held with no move toward it and
  then resumed, inside the window widened by the held sample, with the
  counter reset; beyond-reach, non-positive and flat samples held with the
  window widening per held sample (and accepted under `stop`); a window with
  no positive box mean held as `lost-hold`, not `edge-hold`, although its
  maximum is a window corner; a sample whose window does not span the
  intervals since the last accepted centre not classified; a NaN window
  refused under `hold` as under `stop`, and a found maximum whose centroid
  is not finite refused by name under `hold` on a sample, before the first
  centre and on a relocation, while `stop` takes the NaN centre as before;
  every `hold` window one child cell past its reach (the same maxima and
  centroids found as in `stop`'s windows on a translating vortex run into
  the corral) and the 17-20.5 m/s and 300 m-core measurements above, with
  the slack and with WRF's edges; the limit spent relocating to the
  strongest circulation in the radius, the nest then moving one parent cell
  per step toward it, and relocations that find a negative, flat or edge
  maximum refused by name; relocations listed however many corral holds
  preceded them, each capped kind counted past its cap; a
  corral move cancelled on its blocked axis while the free axis moves, then
  refused after `follow_hold_limit * interval_steps` parent steps; a
  translating vortex run into the corral kept integrating under `hold` where
  `stop` ends it, with `stop`'s tracker rows and moves before the corral;
  `stop` (absent or declared) against the pinned pre-change digest; the
  orbiting sub-vortex reproduction, the vanishing translating vortex against
  the pinned-window control, and the smoothing-radius measurements above, on
  a box-mean plane proved word-for-word equal to the reference; the
  relocation radius refused below `(follow_hold_limit + 1)` reaches plus one
  child cell and admitted at it.
* Configuration keys refused by key and value; single-domain example identities
  unchanged (compared against the base tree); the door admits a following nest
  past every nest gate and refuses the pressure tracker over a periodic parent.

**Needs device validation** (none of this has run):

1. NVRTC compile of `NEST_MOVE_KERNEL_SOURCE` and raw-word agreement with the
   references on the device (`GM1_NEST_DEVICE_TESTS=1`).
2. `tools/run_nest_live_validation.py moving`: forced moves on a running
   nested pair over the homogeneous periodic environment of 5.5(b) -- exposed
   strip equal to the captured parent word for word after each move, retained
   interior equal to the shifted pre-move child, child minus captured parent
   over the run against the fixed nest's 5.5(b) numbers, and parent dry mass
   with `feedback = 1` against a fixed nest. Written, never executed.
3. The carrier inventory on a real child: `CM1-CARTESIAN-NEST-FOLLOW-CARRIER`
   fires if the runtime registers a buffer the table does not name.
4. A tracked storm: the surface tracker on a live supercell child, its
   `tracker_events` against the archived 100 m frames of dca8ab6, and a choice
   of `follow_smoothing_radius_m` measured rather than guessed.
5. The irbc=4 edge-history read after a move, and the cost of a full-lattice
   capture per carrier per move.
6. The hold policy on the device: `tests/test_nest_move.py::
   test_the_hold_policy_reads_the_tracker_launch_words_on_a_device`
   (`GM1_NEST_DEVICE_TESTS=1 python3 -m pytest -q -p no:cacheprovider
   tests/test_nest_move.py -k tracker_launch`) runs `MovingNest._track_surface`
   with NVRTC-compiled `gm1_nest_zeta_box` and `gm1_nest_vortex_centroid` on a
   vortex that is followed, edge-clipped, relocated through a second centroid
   launch on the same device plane, then followed again, and checks that every
   downloaded word set is the CPU reference's and that `hold` holds, relocates
   and resumes. The same body passes on the host through the `g++`-compiled
   kernel text. The policy itself is host arithmetic on words the tracker
   downloads once per follow interval (twice on a relocation); no kernel was
   added.
7. The smoothing radius and `hold` on the deck that stopped: from the same
   parent restart (resumed with `--restart-from`; pass `--restart-dir` as
   well, since a nested run now writes its own restarts, 5.10, and the hold
   state crosses them), 300 s of model time three
   ways -- `R` = 200 m with `hold` (`follow_hold_limit = 4`,
   `follow_relocation_radius_m = 3040.0`: twice the smallest admitted 1520
   m), `R` = 600 m with no `follow_on_edge`, and `R` = 600 m with the same
   `hold` keys -- each passing only with no refusal of any kind, a
   receipt whose accepted tracker centres move no faster than the declared
   speed over the intervals between them, at most one relocation whose
   displacement lies within the relocation radius plus `R`, and the `R` = 200
   m centres compared sample by sample with the `R` = 600 m centres
   (`tools/check_nest_following.py`; on the reproduction `R` = 200 m with
   `hold` holds no sample and relocates nothing, the sub-vortex capture is a
   179-295 m separation on all 41 samples, while `R` = 600 and 840 m, both
   on the parent circulation, agree within 15-25 m). Wall time recorded, not
   estimated.

### 5.9 The tornado criterion (`cm1core.nest_tornado_criterion`)

**Status: implemented; the kernel text is held raw-word identical to its host
reference by `g++`, the live sampling path (CuPy words on a stream, one
`get(stream=)` read) runs on the host through `tools/oracle/host_cupy.py`, and
the orchestration, ledger, sidecar and stop rules run on the host through the
shipped `integrate`; never run on a device.** Optional sub-table
`[nest.tornado_criterion]`, empty or with stop-rule keys; absent, nothing is
sampled and no key is projected, so every sealed nest document keeps its
identity. **Integration only:** the timing harness (`benchmark`, the door
without `--integrate`) would sample every step, warm-up included, with no
sidecar to publish the samples to, so the door refuses the table there
(`CM1-CARTESIAN-NEST-TORNADO-CRITERION-INTEGRATION`) and `benchmark()` raises.

**Authority.** CM1 r21.1 has no nesting and makes no tornado claim anywhere in
`src`, so the reductions, the episode rule and the verdict are GM1's. What is
CM1's is every formula a sample reads:

| CM1 (line) | formula | GM1 |
|---|---|---|
| `src/solve3.F:864-865` (`sws`) | `sqrt((umove+0.5*(ua(i)+ua(i+1)))**2 + (vmove+0.5*(va(j)+va(j+1)))**2)` | ground-relative wind maximum over the disk's scalar points; the frame-relative maximum beside it, reported only |
| `src/parcel.F:1066-1067`, `src/solve3.F:894-895` (`svs`) | `(v(i)-v(i-1))*rdx - (u(j)-u(j-1))*rdy` at the corners | the centre and the core count, as `vorticity_budget.corner_vertical_vorticity` evaluates it (`/dx`) |
| `src/cm1.F:3512-3513` | `domainlocx/y` | ground position of the centre = domain location + child origin + corner times `dx` |

**The criterion is fixed** (`CRITERION`, identity `CRITERION_SHA256` over the
constants and the kernel text). No threshold is a key: `config.py` refuses one
written into the table by name (`CM1-CFG-NEST-TORNADO-CRITERION-FIXED`), so a
TORNADO verdict means the same thing in every receipt carrying the same
identity and nobody can tune it after a run.

**Where and what.** `NestDriver.after_parent_step`, every parent step, after the
child's steps, its synchronise and the domain-location copy, **before**
feedback and any move. One serial kernel, `gm1_nest_tornado_sample`, reads the
child's completed `u3d`/`v3d` at 10 m AGL (float64 weights between the child
scalar levels bracketing 10 m), finds the centre as the first strict maximum of
corner vorticity over the corners at least 1000 m from every child edge
(`DO j / DO i` order, non-finite corners skipped), and reduces over the 300 m
disk: ground-relative and frame-relative wind maxima and `ncore`, the corners
with vorticity at least half the maximum. The host reads nine words. The kernel
is taken from its module (`get_function`, where CuPy compiles) when the nest is
attached, so an NVRTC failure refuses the run at startup, not at the first
parent step.

**Host rules** (`CriterionLedger`). A sample qualifies when `zeta_max >= 0.30
s-1`, the ground-relative wind maximum `>= 35 m/s` (a non-finite maximum fails
its condition: the sidecar row carries it as `null`, so a replay could never
see it pass, and a blown-up wind is no evidence of a vortex), the core radius
`dx*sqrt(ncore/pi) >= 40 m` and the centre's ground position moved `<= 60 m/s
* dt + 40 m` since the previous sample (the first sample of an integration
passes; one after a sample with no centre fails). Consecutive qualifying
samples spanning `>= 60 s` first to last are an episode; TORNADO when one
exists, genesis at its first sample. `stop_after_episode_s` requests a stop
once an episode has ended, no qualifying run is in progress and that long has
passed; `stop_if_no_tornado_by_s` at that model time with no episode and no
qualifying run in progress; `config.py` refuses a cutoff at or after
`run.duration_s` (`CM1-CFG-NEST-TORNADO-STOP`), whose request could only fall on
the final event. "No qualifying run in progress" is held at every sample, not
only at the one that raised the request: a qualifying sample while a request
is pending withdraws it (`stop_requests_withdrawn` counts them and
`last_withdrawn_stop_request` keeps the last, with the step, the model time
and `STOP_WITHDRAWN_BY_RUN`), so a vortex forming one sample after the request
is not cut at the next frame, up to a history interval later. A run that
becomes an episode restarts the after-episode interval and retires the cutoff;
a run that ends short of one re-raises the request at its first
non-qualifying sample. The replay reaches the same withdrawals from the rows.
A request is honoured at the next history event that is not
the final scheduled event, after its frame is published, with a
`criterion_stop` block (its `frame_directory` the frame row's, or the formatted
directory past `frame_row_budget`, as `courant_stop`) and status
`STOPPED_AT_NEST_TORNADO_CRITERION_STOP_RULE`. A request still pending at the
final scheduled event (one raised in the last history interval) ends nothing:
the requested duration was integrated, the status is the completed
integration's, `criterion_stop` is `null`, and the block's
`stop_request_disposition` names the request unused
(`STOP_UNUSED_AT_DURATION`; `STOP_UNUSED_AT_COURANT` when CM1's Courant stop
ended the run first; `STOP_HONOURED` when it ended the run).

**Outputs.** `<history>/nest-child/tornado-criterion.jsonl`: a header (criterion,
plan, `time_refinement_ratio`, columns) and one compact array per parent step,
keyed by the parent's `completed_steps` and carrying the placement it was
sampled on (`parent_i0`, `parent_j0`), flushed as written; a run
that dies keeps the file with every sample it took (an integration that failed
before its first sample removes it, so the retry's root stays admissible). The
`nest_tornado_criterion` receipt block carries the verdict, genesis and
declaration times, condition counts, the longest qualifying run, the episodes
(peaks, start/end ground positions, mean centre velocity), the sidecar's
SHA-256, rows and bytes, and `sample_wall_seconds` (count, total, mean,
maximum): the host wall the parent step waited on the sample, measured around
the launch and the blocking read with the child synchronised just before, so
the per-step cost of the serial kernel is a measured number in every nested
receipt and needs no control run. `tools/tornado_criterion.py replay` re-decides every
row and the verdict from the sidecar alone; `tools/tornado_criterion.py frames`
recomputes each published child frame's sample with the host reference and
compares it with the row of its parent step. A child frame's placement names
the CHILD's step, `time_refinement_ratio` times the parent's (the invariant
`NestDriver` checks every parent step), so the row is the one at placement step
divided by the header's ratio, and its model time must equal the placement's
(the child clock takes the parent's at every parent-step boundary). The row's
`parent_i0`/`parent_j0` are compared with the placement's as integers: a frame
published after a nest move is counted, not compared, and so is a frame with no
row (reconstructing the origin from the ground position in floating point would
misclassify frames whenever the domain location carries low-order bits).

**Restarts.** The ledger is nest child restart state (5.10): every value a
later sample's decision reads -- the previous centre, the qualifying run in
progress with its peaks, the last episode's end, the pending stop request --
and every whole-run count the verdict reports -- the closed episodes, genesis,
the condition counts, the withdrawals -- are written bit-exactly into the
child restart's scalar table (`CriterionLedger.checkpoint`, `nest.tornado.*`,
binary64 as hex bits so an untouched `-inf` peak and a signed zero survive)
and read back before the resumed run's first sample
(`CriterionLedger.restore`, refusing a missing, mistyped or other-schema row
by name). A qualifying run that straddles the restart therefore becomes an
episode at 60 s, not 60 s after the resume, and the resumed receipt's ledger
is the uninterrupted run's. The resumed sidecar's header carries the same
rows (`carried_from_restart`), so `replay` seeds its ledger from them and
re-decides this leg's rows against the same history; the receipt block
carries `carried_from_restart` (samples, verdict at the restart, the run in
progress, the pending request) and its `verdict_scope` says so. A restart
written by a nest that declared no criterion carries no ledger; a document
that declares one at resume starts the ledger there, and the resume record
(`child_restored_from_restart.tornado_criterion.ledger`) says which. Proved on
the host double through the child restart's own scalar table
(`test_a_resumed_nested_run_carries_the_ledger_through_the_child_restart`):
qualifying to 61 s, quiet to 70 s, qualifying again, restarted at 80 s with a
10 s run in progress; the resumed run declares the second episode at 130 s
as the uninterrupted run does, every ledger field of its receipt equals the
uninterrupted run's, its replay agrees, and its next child restart carries
the same `nest.tornado.*` rows; a ledger started fresh at the resume declares
one episode at 140.5 s instead.

**Verified on the host** (`tests/test_nest_tornado_criterion.py`, no CUDA
context): the kernel text raw-word identical to the reference by `g++`, with
NaN and infinity, clipped disks, ties and an all-non-finite search; the
reference against solid-body rotation interpolated in height; synthetic
Rankine vortices on a 20 m child in an (8, 6) m/s frame: a translating vortex
becoming a tornado 60 s after it first qualifies, a vortex that qualifies only
on the ground-relative wind, a core under 40 m, a weak vortex in strong flow, a
stronger vortex inside the edge margin never the centre, and a jump between two
vortices ending the run; a qualifying run surviving a one-parent-cell move
of the nest under a translating vortex (the corner index drops by the ratio,
the ground position does not jump, `M` holds, the frame written at the step
of the move counted and not compared, the frames on either side identical);
every threshold inclusive at its value; both stop
rules, and a pending request withdrawn by the run that begins one sample
after it (ledger, and through the shipped `integrate`: the next frame ends
nothing and the vortex becomes a TORNADO); a non-finite maximum failing its condition live and in the replay alike
(an infinite wind inside the disk); the driver's sampling order; the kernel
taken once, at construction, and a compile failure raised there; the live
`sample` (no override) on `host_cupy` owners allocated at
`terrain_open_sound_shapes`' padded `u3d`/`v3d` shapes, one launch and one
stream read per sample, words equal to the reference; the stop, a stop past
`frame_row_budget`, a request pending at the final event (cutoff at the
duration, and raised in the last interval) finishing as a completed
integration, the undisturbed run and a failed run through the shipped
`integrate` on the host double; the timing harness refused in `benchmark()` and
at the door; the offline replay, and the frame recompute at
`time_refinement_ratio = 5` with a float32 frame velocity and `dt = 0.4 s`
(every frame on the sampled placement compared, a moved frame and a frame off
a parent-step boundary counted); configuration refusals by key and value, and
the nested example's identity unchanged.

**Needs device validation** (none of this has run; each check is minutes):

1. NVRTC compile of `NEST_TORNADO_KERNEL_SOURCE` and raw-word agreement with
   the reference on the device:
   `GM1_NEST_DEVICE_TESTS=1 CM1_R21_1_PRISTINE=<pristine> python3 -m pytest -q
   -p no:cacheprovider tests/test_nest_tornado_criterion.py -k device`.
2. A live FIXED nested run with the table declared, on the live nest above
   (120 s, frames at 60 and 120 s; no `--restart-dir` is passed, so no restart
   is written),
   about 5-10 minutes on the RTX 5090:
   `mkdir <run> && gpuwm-cm1 cartesian <deck> --integrate --history-dir
   <run>/history --output <run>/receipt.json`, then `python3
   tools/tornado_criterion.py replay <run>/history/nest-child/tornado-criterion.jsonl`
   (`rows_identical: true`, 240 samples) and `python3 tools/tornado_criterion.py
   frames <run>/history/nest-child` (`frames_compared: 2`,
   `frames_identical_to_jsonl: 2`, `frames_without_a_row: 0`), with the receipt's
   `nest_tornado_criterion.device_launches` 240. The per-step cost is the
   receipt's own `nest_tornado_criterion.sample_wall_seconds.mean` against
   `step_wall_seconds.mean` (the door's receipt keeps the integration record
   under `runtime_benchmark`): `python3 -c "import json,sys; r=json.load(open(sys.argv[1]));
   b=r.get('runtime_benchmark', r); c=b['nest_tornado_criterion']['sample_wall_seconds'];
   s=b['step_wall_seconds']; print(c['mean'], s['mean'], c['mean']/s['mean'])" <run>/receipt.json`.
   The kernel is one serial device thread over the 201 x 201 interior corners
   (eight float32 loads each) and the disk, launched and read back every
   parent step; estimated, not measured, at 4-15 ms per parent step on this
   6 km child, about six times that on a 12 km one. If the share exceeds 5 % of the parent
   step, the centre search moves to a block-parallel deterministic argmax (each
   row's first strict maximum in raster order, then the first row's winner),
   the disk reductions staying serial; that changes `NEST_TORNADO_KERNEL_SHA256`
   and so `CRITERION_SHA256`, and the new text is re-proved raw-word identical
   through the `g++` harness extended to run every thread of the block.
3. A live FOLLOWING nested run with the table declared, the moving-nest paths
   the El Reno use case needs (the sample taken before the move, the ground
   position continuous across `parent_i0`/`parent_j0` jumps,
   `frames_after_a_move`, and the sample's cost beside the tracker's). The
   analytic deck cannot exercise them: from its horizontally homogeneous
   initial state the child's surface vorticity is noise for the first 120 s,
   with no vortex for the tracker to follow (its maximum could as easily sit
   on a window edge, `CM1-CARTESIAN-NEST-FOLLOW-SPEED`, as move the nest).
   Take a moving-nest deck of a real-sounding storm (one resumed from the
   parent's spawn restart with `follow_tracker = "wrf-surface-circulation"`,
   `follow_max_speed_m_s = 20`, `follow_smoothing_radius_m = 200`; it is not
   shipped), copy it, and set `run.duration_s` to
   the restart's model time plus 120, `run.history_interval_s = 60`,
   `nest.follow_interval_s = 15`, with `[nest.tornado_criterion]` declared and
   empty (`restart_interval_s` at most `duration_s`; without `--restart-dir` it
   writes nothing, and with it every restart carries the criterion's ledger,
   5.10). At most
   15 minutes on the RTX 5090 given the 2.26x step budget:
   `mkdir <run> && gpuwm-cm1 cartesian <moving-deck> --integrate --history-dir
   <run>/history --output <run>/receipt.json && python3 tools/tornado_criterion.py
   replay <run>/history/nest-child/tornado-criterion.jsonl && python3
   tools/tornado_criterion.py frames <run>/history/nest-child`. Expected:
   `rows_identical: true`; `parent_i0`/`parent_j0` changing across rows at
   least once (`python3 -c "import json,sys; rows=[json.loads(l) for l in
   open(sys.argv[1])][1:]; print(sorted({(r[2], r[3]) for r in rows}))"
   <run>/history/nest-child/tornado-criterion.jsonl` prints more than one
   placement; the receipt's `nest_execution.storm_following.moves` say the
   same); `frames_read` 2,
   `frames_after_a_move >= 1` when a frame's step is a move's step, and
   `frames_identical_to_jsonl` equal to the frames on an unmoved placement,
   `frames_without_a_row: 0`; `sample_wall_seconds.mean` of the same order as
   in check 2. A run that ends in `CM1-CARTESIAN-NEST-FOLLOW-SPEED` keeps its
   sidecar (the rows to the refusal) and the replay still runs; a run with no
   move proves only the fixed paths again and is extended, not counted.
4. NEEDS-LONG-VALIDATION: a verdict on a real tornadic case needs an hours-long
   nested run (and a following nest over it), which none of the above is; the
   per-step cost of check 2 over such a run (the share can only be read from
   its own receipt's `sample_wall_seconds`) is part of the same validation.

### 5.10 Restarts of a nested run (`cm1core.nest_restart`)

**Status: implemented; resume identity proved on the host with the packaged
kernel text; never run on a device.** `CM1-CARTESIAN-NEST-RESTART-WRITE` is
retired: `--restart-dir` on a nested run writes a restart for both domains at
every restart time, and `run.irst = 1` restores the child from its own restart
instead of rebuilding it from the parent.

**Authority.** WRF's per-domain restart.

| WRF (line) | what it does | GM1 |
|---|---|---|
| `share/mediation_integrate.F:347-354` | only the parent initiates restart writing, so every domain is written at one time | the parent's restart event (`cm1.F:3186`, `cm1.F:4246`) takes the child's snapshot at the same completed parent step, after that step's child steps, feedback, move and parcel hand-off |
| `:1157-1182` `med_restart_out` | this domain's restart, then each child's | `restart-NNNNNN/restart.cm1fcp` (unchanged) and `restart-NNNNNN/nest-child/restart.cm1fcp`, both by `commit_cartesian_restart`, assembled in one staging directory and renamed into place together; the parent's receipt is written last and declares the child (`nest_child`) by the digests of its committed envelope header (Rust's per-carrier SHA-256 of every word), contract and scalar table (`child_identity`), so the child beside a parent is the publication written with it, not another run's at the same step |
| `:861-892` `med_nest_initial` on restart | the nest reads its own restart (`input_restart`) | `NestDriver(restoring=True)` at the restored placement, then `restore_nest_child` word for word |
| `Registry/Registry.EM_COMMON:438-441` | state rows `vc_i`, `vc_j` ("need for restarts of moving nests") | `nest.follow.vc_i/vc_j` (f64 bits), the pressure trackers' centre and held offset |
| `Registry/Registry.EM_COMMON:2318-2319`; `share/mediation_nest_move.F:162-163,176-177` | `i_parent_start`, `j_parent_start`: `rconfig` (namelist) entries flagged `rh` (restart and history I/O), not state rows, which WRF updates on every move | `nest.parent_i0/j0` |
| `:897-913` | exact only on a `vortex_interval` boundary | the tracker state is carried and the cadence is the absolute parent step, so any completed parent step is exact |

**What the child restart carries.** The child runtime's own restart profile --
every carrier a single-domain restart carries for its grid (padded owners with
halos, `ppx`, `ppten`, mass fluxes, radiative edge history, `rho/prs/rr/rf`,
the `*a`/`*3d` pairs, the eqtset=2 rate carriers, the held tendencies into which
the Davies relaxation is added, TKE, the microphysics persistence), the
standard scalars (`mass1`, `asq`, the cadence tokens, the moving-frame set) --
plus the parcel positions and owning grid of `[lagrangian.nest_child]`, and the
coupling scalars: placement, ratios, spans, zones, the time ratio and parent
step, the driver, lateral-forcing, move and parcel counters, and the tracker.
Under `nest.follow_on_edge = "hold"` (5.8) the tracker's hold state as well
(`nest.follow.hold.*`): the consecutive held samples, which set the next
sample's search reach and decide when it relocates, the consecutive
corral-cancelled parent steps, which decide when the run stops by name, and
the whole run's hold-event counts and longest runs, which the receipt reports
(`hold_counts`, `max_consecutive_*`, with the restart's share under
`carried_from_restart`). A restart written under `stop` carries only the flag;
resumed under `hold` it starts with no sample held, which is what its tracker
had, and a `hold` restart resumed under `stop` drops the count, since a stop
tracker never holds; the resume record (`storm_following.hold_state`) says
which. With `[nest.tornado_criterion]` declared, the criterion's ledger
(`nest.tornado.*`, 5.9).

**What is rebuilt, and why that is exact.** The boundary value/tendency tables
and `dtbc` are rewritten by `med_nest_force` at the top of every parent step
before a child step reads them (`NestLateralForcing.refresh` writes every word
and resets `dtbc`); the specified-zone gathers at the top of every small step;
a refined deck's windows at every capture, and its maps and weights from the
same decks and base state; the parcel velocity snapshots at the next
`solve2.F:1290-1301` capture.

**Refused at resume, each by what it would do.** Before a device opens, from
the receipts alone, when the document declares a `[nest]`: a declared child
restart whose receipt is missing, unreadable, of another profile or another
step, not the publication its parent declared (`child_identity`), or without
its coupling state (`CM1-CARTESIAN-RESTART-RECEIPT`); a child receipt whose
`nest_coupling.state` records other refinement ratios or spans than the
document, another child parcel count, parcels the document does not declare
or the reverse, or a restored placement the document's zones cannot hold
(`CM1-CARTESIAN-NEST-RESTART-MISMATCH`). After the device opens: a child
envelope whose header is not the one its parent declared, or a torn one, found
on the host before the parent's words reach the device; the same coupling
checks again on the envelope's own scalar table; another child grid or
vertical deck (the child profile, as for the parent); a child ground position
or step that is not the restored parent's. These stop the run as
`CM1-CARTESIAN-RUN-FAILED`, naming the defect. Admitted and recorded in
`child_initialization` and `child_restored_from_restart`
(`document_changes`): a changed `spec_zone`, `relax_zone` or `feedback`, which
decide how the restored child is forced and fed back from then on, not what
its words mean. Admitted and recorded under `storm_following`: changed follow
settings (the carried tracker continues), a changed tracker (it starts WRF's
first time through; the move and tracker-event counts still continue), a
following nest added at resume, or `follow_storm` dropped at resume (the nest
stays at the restored placement). A document with no `[nest]` resumes the
parent alone from its own restart and says so (`nest_child_restart` in the
resume record), whatever the child directory holds or fails to hold: the
child is neither read nor admitted, so a child that is gone never bars the
parent from continuing. The coupling scalar table is priced before the first
step (`integrate`), so a tracker without a restart code would stop a run
there, not at its first restart; the codes are the trackers `nest_move`
declares, in order.

**Proved (host, `tools/oracle/nest_restart_identity.py`,
`tests/test_nest_restart.py`).** The real door, Rust initializer and publisher,
kernels compiled from the packaged CUDA text, on a 45x45x20 parent with a
following 33x33 child (ratio 3, time ratio 2), `bbc = 3`, the surface tracker
every 3 parent steps, restarts every 3 steps and 32 child parcels, with every
uninitialized allocation of each resumed leg poisoned with NaN. For every model
time after the resume point the resumed leg publishes exactly the frames and
restarts the uninterrupted leg publishes, parent and child, and every carrier
(by receipt digest and by the words restored from each envelope), every scalar
table, every parent and child restart's `scheduler_scalars` (tracker centre,
placement, the whole run's move and tracker-event counts, the last track,
lateral, feedback and parcel counters, `nwrite`/`nrst`; none excluded), every
file `cm1-forecast-publish restore` writes back out of each parent and child
restart envelope (arena, field table, `restart-scheduler-scalars.tsv`; by
digest), the parcel track frames and hand-off events, the whole move rows and
tracker events and the move count are identical. Each resume is *exercised*:
a restored tracker centre, at least one tracker event and at least one move
after the resume point (the base legs track 3 and 2 times and move 5 and 2;
the refined legs track 3 and 2, move 4 and 1; the pressure legs track 4 and 4,
move 1 and 1).

| deck | resumed from | carriers | moves after the resume |
|---|---|---|---|
| surface tracker, parent's vertical deck | restart 1 (step 3), 2 (step 6) | 462, 308 | steps 4-8, 7-8, the first from the carried centre and the carried last track |
| the same child on its own deck (`child_nz = 30`, `child_stretch_z = 4`, 100 m levels under the parent's 150 m; vertical windows and `restrict_vertical` feedback) | restart 1, 2 | 462, 308 | steps 4-7, 7 |
| `cm1-pressure-held-offset`, restarts every step | restart 1 (step 1), 2 (step 2) | 1344, 1240 | step 3, from the carried held offset and pressure centre |
| the surface tracker under `hold` (80 m/s reach, `follow_hold_limit = 1`, 280 m relocation radius) | restart 1 (step 3), 2 (step 6) | 462, 308 | steps 4-8, 7-8; the step-3 sample is held on its window edge and restart 1 is written with it held, so the step-6 sample relocates from the carried count (one relocation, 6 moves, identical hold-event counts) |

Negative controls: a child rebuilt from the parent changes 397 carriers; a
surface tracker started fresh changes the move rows; a held-offset tracker
started fresh at step 2 never moves and changes 1028 carriers; a `hold`
tracker resumed with its count dropped (`fresh-hold`, what a restart without
the hold rows would give) holds the step-6 sample instead of relocating and
changes the move rows; a resume with
another relax zone and no feedback is admitted, recorded as
`document_changes`, and changes the words. Before the counts were carried, the
same comparison found every child restart after a resume short of the
interrupted run's moves and tracker events, and the first resumed move row
without its track.

**Owed on the device.** `tools/oracle/nest_restart_identity.py --device
--deck-kind device` (192x192x75 stretched parent, 200x200x75 child at 20 m,
90 s, restarts every 30 s, resume from restart 1; a fresh `--workdir` per
attempt): NVRTC words, the page-locked child arena and background
publication. It passes (exit 0) only when the resume is identical and
exercised -- a restored `vc_i`/`vc_j`, a tracker event and a move after
step 60; identical without both exits 3, NEEDS-LONG-VALIDATION.

Not claimed: a two-part SGS closure (`sgs_model = 4` with `t2p_avg`) resumes
with CM1's restarted flag (`src/cm1.F:2663`, `src/turb.F:1460-1466`) in both
domains, as a single-domain restart does; that is CM1's restart, not the
uninterrupted step. The stretched tornado-grade example
(`examples/cartesian-stretched-tornado-two-part-sgs.toml`) runs that closure,
so for it a resumed nested run is CM1's restarted trajectory, not the
uninterrupted one bit for bit. The large El Reno nested decks (fixed and
moving nests) all ran `sgs_model = 1`, the
closure the identity covers; it has not been checked on them. The identity
above is proved for `sgs_model = 1` (every host deck and the device deck) and
claimed for no other closure. NEEDS-LONG-VALIDATION: the vertical-refinement and
pressure-tracker resumes are proved on the host decks above and have not run
on a device.

### 5.11 The vortex-following child window (`cm1core.nest_child_window`)

**What the measurements said.**  Three real attempts on the 24 May 2011
environment (100 m parent, two-way 20 m nest, RTX 5090; the tornado reaches
117-121 m/s ground-relative, `zeta` 4-6 s-1, `p'` -104 to -128 hPa) established
that neither existing path reaches the core:

| attempt | seeding | result |
| --- | --- | --- |
| 100k inline child parcels, seeded T+60 over 5x5 km, 10-280 m AGL, saved every 3 s | forward lattice | by T+75, 92% had left the child (median height 2.1 km); **one** parcel within 300 m of the tornado |
| 140k parcels seeded T+65 in the source region the offline backward tool suggested | forward lattice | 1,441 passes within 300 m below 1 km, but almost all during the T+67-70 developing vortex; ~10 during the mature tornado |
| 135k parcels seeded T+70 over 6x6 km, 10-210 m AGL, saved every 1.5 s | forward 40 m lattice | 296 passes within 300 m, but the **maximum vorticity any parcel ever recorded is 2.0 s-1** against the grid's 4-6 s-1, and only 46 ever had `zeta > 1` s-1 below 500 m |
| `cm1core.nest_parcel_backward` through 30 s child history frames | offline backward | all 9,000 parcels reached the child boundary within 10 min; the tool's own position-error estimate was **800-1200 m median**, which is twenty core radii |

The last row is not a defect of the tool: its estimate is the Lagrange
remainder of a linear-in-time wind, `(T^2/8) max|u_tt|`, and it scales with the
square of the output interval.  No post-hoc reading of 30 s frames can be made
accurate near a 100 m/s core, and a 40 m forward lattice does not sample a
40-60 m core.

**What was built.**  A sub-volume of the live child that **follows the vortex**
and is written **every child step**, holding the model's own advecting velocity
words rather than the completed state a history frame holds.  The offline
integrator gained a second wind source, `model-step`, that steps backward
through those records one at a time with no interpolation: the same one frozen
wind for both RK2 stages that `src/parcel.F` itself uses, so the backward step
is the mirror of the model's own step.  The time-interpolation term is then not
estimated but absent.

**What is CM1's**, cited and not re-derived:

| word | CM1 authority |
| --- | --- |
| the advecting velocity the stream carries | `src/solve2.F:1290-1301`, `src/solve3.F:438-469`, at `nrk == nrkmax` |
| the step it belongs to | `src/solve3.F:768-769` (once per completed step) |
| the `k = 0` ghost of `u` and `v` | `src/parcel.F:59-96` (reads levels 1-3 only, so it is local to the bottom of any block that reaches the ground) |
| the sampler the offline integrator runs on it | `src/parcel.F:153-461`, unchanged, on the sub-volume's own deck |
| the log layer below `zh(1)` and its `zntmp` | `src/parcel.F:470-515`; `src/turb.F:1161,1233-1237`; `src/cm1.F:2192-2194` |
| the scalar-point stencil the budget is sampled with | `src/parcel.F:1365-1447` |
| the vorticity components | `src/misclibs.F:1542-2100` `calcvort`, flat-Cartesian branch |

**What is GM1's**, and what each choice prevents:

* The window is a **pure slice** of the child's padded carriers.  Nothing is
  interpolated, no boundary rule is applied, and there is no kernel.  That is
  bought with two refusals: the window *and its three ghost columns each way*
  must lie inside the child's free interior (else it would publish the parent's
  interpolated rows as the child's own), and its top must be strictly below the
  child's (else the level above it would be `tbc`'s free-slip extrapolation, a
  lid the child's interior does not have).
* `k0` is not a key: the window always reaches the ground, because the air a
  tornado ingests is in the lowest few hundred metres.
* The centre is the tornado criterion's own (5.9) from the previous parent
  step; a sample that finds none holds the last centre and the record says so;
  before the first centre is found nothing is written and that is counted.
  The criterion samples **before** the feedback and before any move, so its
  corner index belongs to the placement the child was integrated on, not to
  the one the window is about to be cut from.  The centre is therefore taken
  from the sample's own ground position -- which already carries that
  placement's origin -- put back in the parent-grid-origin frame by removing
  the domain's ground position.  Re-deriving it from the current geometry
  would move the centre by exactly the cells the nest just moved, silently,
  because the corner index is a perfectly good index on both lattices.
* A centre that would push the window past the free interior is **clamped**,
  not refused, and every record carries its own index origin and ground origin,
  so nothing is ever placed by inference.
* A **move** re-places the deck, not the window.  A following nest (5.8) moves
  the child by whole parent cells between parent steps; the window's centre is
  in the parent's frame, so what changes is its *index* into the moved lattice.
  The child deck's own `xf(1)`/`yf(1)` are put on the new geometry at the same
  moment, because a record's published ground origin is
  `child xf(1) + i0*dx`.  Without that, every trajectory read out of the stream
  after the first move would be put on the wrong ground -- the one failure a
  moving nest makes silent, since the words themselves would still be the
  child's.  `tests/test_nest_child_window.py` holds the ground origin fixed
  across a two-parent-cell move while the index shifts by two.
* One append-only arena with a fixed record stride, one TSV index row per
  record with its SHA-256, and one receipt.  Ten records a second is six
  hundred directories a model minute.
* The stream is bounded in model time (`start_s`, `end_s`) and in bytes: the
  door computes what the declared geometry costs per model minute and refuses
  it against `budget_bytes_per_model_minute`, so a run cannot discover its disk
  cost by filling the disk.
* The only word the cut cannot carry is `parcel.F:59-96`'s `k = 0` ghost of `u`
  and `v`, formed on the window's own block, and a reflected `w` ghost
  (`w[0] = -w[2]`) that **no trajectory reads** -- `parcel.F:251-260`'s `kflag`
  is at least 1 -- and that exists only so the budget's lowest-level `dw/dz` is
  not formed from a word CM1 leaves undefined.

**The analysis side** (`cm1core.nest_window_budget`,
`tools/nest_window_trajectories.py`): seed an equal-area disk *inside* the
core rather than a lattice across it, integrate backward through the stream,
and publish beside every position the measured round-trip residual, the
sensitivity amplification `prod(1 + |dt| L_n)` and the position error estimate
built from them.  Then sample the vorticity budget at every position with
`parcel.F`'s scalar-point stencil and check that it closes.  **Friction is the curl of the held tendency** (stream schema v2): the window
carries the step's held momentum tendency `uten1`/`vten1`/`wten1` (`solve1.F`),
and the door (`CM1-CARTESIAN-NEST-WINDOW-DAMPING`) admits a window
only below the Rayleigh layer and without `hrdamp`, and this lane admits no
large-scale forcing but `lspgrad = 1` -- so inside the window that tendency is
the subgrid stress of `turbtend.F` (`solve1.F:1124-1160`) plus any `idiff`
diffusion, plus `lspgrad = 1`'s per-level words, and nothing else.
**Rotation** (stream schema v3, `icor = 1` on the f plane): CM1 adds the
Coriolis force to the stage tendency (`solve2.F:633-653`), never to the held
one, so the v3 header records `fcor`, the frame velocity and, with
`lspgrad = 1`, `solve1.F:590-612`'s per-level words; the budget re-forms the
force from `uadv`/`vadv` in CM1's float32 order on CM1's faces, publishes it
(`coriolis_accel_x/y`) and its curl (`coriolis_x/y/z`: `f du/dz`, `f dv/dz`,
`-f (du/dx + dv/dy)`), and takes the `lspgrad` words out of friction and
publishes them once, as `lspg`.  The beta plane (`-CORIOLIS-BETA-PLANE`) and
any other `lspgrad` (`-LSPGRAD`) are refused.  (The whole-grid `*_hturb`/`*_vturb` split still
does not exist in a nested run; it is not needed, because the window never
contains the rows where the held tendency holds anything but friction.)  The
solenoid uses the total density `rho_d(1+qv+qhyd)`.  The closure is checked on
all three components; its residual holds the implicit dissipation of the
odd-order advection, which no held array carries, and the discretisation and
sampling error.

**Schema v3: Morrison's species.**  A `morrison-mp10` child writes
`gm1.nest-child-window-stream/v3`: `qhyd`'s slot holds four species words --
`qc`, `qr`, `qis = qi + qs` and `qg` (hail when `ihail = 1`,
`morrison.F:169-170,381-384`) -- each found **by name** in `param.F:3414-3473`'s
table, so the viz can tell cloud from rain from ice from hail and form
reflectivity at the window's cadence.  The receipt adds `schema_version`,
`field_names` and the species' `qa` rows; the door prices the three extra ring
fields against `budget_bytes_per_model_minute`; a component that lacks a
species, or a live `qa` that is not the component's own `numq`, is refused.
`WindowStream.fields` rebuilds `qhyd = ((qc + qr) + qis) + qg` in float32, so
the budget and the backward integrator read either schema unchanged; the
regrouping differs from `parcel.F:745-775`'s sequential sum by float rounding
only.  Every other non-rotating component keeps schema v2 byte for byte.
**One v3, two additions.**  Rotation and species were added on separate
branches under the same name; merged, v3 is one schema that carries either or
both, each declared in the receipt (`rotation` when `icor = 1`, `species` for
Morrison), and every v3 receipt carries `schema_version` and `field_names`.  A
Morrison child with Coriolis writes one v3 stream
with both.  The reader refuses a v2 receipt with either header, and a v3
receipt with neither.

**Host verification** (`tests/test_nest_child_window.py`,
`tests/test_nest_window_budget.py`; no CUDA context is opened, and there is no
kernel in either module -- the cut is a strided copy on either lane):

* the cut is an **index identity** against the whole child's own carriers, for
  every field over its own staggered extent, ghost columns included;
* the `k = 0` ghost is `parcel.F:63-72`'s own extrapolation of the window's own
  block, and the `w` ghost is the reflection;
* every configuration refusal fires by name, and the sealed projection carries
  every window key only when the table is declared;
* the disk cost the door states is the record's own size times its cadence;
* a rewritten arena is caught by the record's own SHA-256;
* backward through the stream reproduces the **closed-form** trajectory of a
  stretching vortex to 0.006 m over 10 s of a 15 m/s vortex, where the same
  trajectory through records four times apart errs by 0.70 m and eight times
  apart by 1.75 m -- the measurement that says why every child step.  A stream
  actually written at `stride_child_steps > 1` is **refused** by the
  `model-step` source, because a record is still the average over one child
  step while the interval between two of them is several: the source would
  freeze that one step's wind across steps it was never the average of, and
  its receipt would say the time-interpolation error is zero when it is first
  order in the record interval.  The test above reaches that counterfactual
  only by stripping the records' own step numbers, and says so;
* the budget **closes**: in that flow tilting, the divergence term and the
  solenoid are identically zero and `stretching_z = zeta * a = D(zeta)/Dt`
  exactly, and the residual is 1.3e-5 of the term scale in median over 1,250
  trajectory positions;
* a material circuit carried backward through the same stream conserves its
  circulation to under 1% (Kelvin), and its seed value is the closed-form
  circulation of the discrete N-gon.

**Not claimed.**  Nothing here has run on a device.  The stream's cost in step
rate (the device-to-host copy on the stepping thread, the write on a worker)
is measured by the run and published (`device_copy_seconds`,
`blocked_seconds`); no number for it is claimed in advance.

### 5.12 The end-of-step reset of the specified rows (`spec_bdy_final`)

WRF ends every `solve_em` step, after microphysics, by setting the specified
rows to their boundary values: `IF (config_flags%specified .or.
config_flags%nested)` calls `spec_bdy_final` (WRF v4.6.1
`dyn_em/solve_em.F:4531-4716`, "this code forces boundary values to specified
values to avoid drift", `:4531`; the routine `share/module_bc.F:2066-2216`) on
`u_2`, `v_2`, `t_2`, `ph_2`, `mu_2`, `w_2` (nested only, `:4566`) and every
moist (`:4622-4623`), tracer and scalar field, to `bdy + dtbc*bdy_tend`. The
chain's first device pass flagged that the live nest did not (only the ArWen chain had it); it now does.

GM1 (`nest_live.NestLateralForcing.finalize_specified`, called by the child
runtime at the `solve3` phase, before `solve3` copies the new state into the
time-level-n arrays): the specified rows of `u3d`, `v3d`, `w3d`, `pp3d`, `th3d`
and every `q3d` species are set to `value + dtbc*tend` in single precision,
the product then the sum (`gm1_nest_spec_apply` modes 2 and 3; the loop
ownership of `spec_bdyupdate`, the y sides owning the corners). One kernel
serves the nest (WRF's nested form, every field) and the ArWen chain (WRF's
specified form: u, v, π′, θ and the transferred species, not w;
`arwen_chain_runtime`). GM1's choices, recorded in the run's receipt: a species
target is held at zero or above (the reset runs after `pdefq`, and the product
can round a few 1e-12 below zero where both ends hold none); π′ stands for
WRF's t, ph and mu together, as in the stage specification (5.2).

Host tests (`tests/test_nest_spec_bdy_final.py`): the nested lane covers every
carrier, w included; the specified lane covers u, v, π′, θ and the species and
not w; the rows and nothing else change (not the relaxed rows, not the interior,
not the time-level-n arrays); the reset runs once per child step, just before
`solve3`; modes 2 and 3 equal their CPU reference word for word (g++, every
lattice and side, spec zone 1 and 2, linear and shuffled thread order), and on a
device with `GM1_NEST_DEVICE_TESTS=1` (passed on the RTX 3080). The hold-tracker
restart deck of `tests/test_nest_restart.py` is re-pinned: the reset moved its
vortex track by a fraction of a cell; resume identity is unchanged.

Device check (RTX 3080; `GM1-ARWEN-CHAIN.md` §9.1): a WK82 bubble whose
updraft runs through a 200 m child's west side, 30 min, the specified rows
compared with the parent captured onto the child's lattice (WRF's target at the
end of a parent step) every 5th parent step. With the reset, every carrier is
at the target to float rounding (u 1.9e-9, w 6.0e-8 m/s, θ 7.5e-9 K, π′
7.3e-12, qv 0, qc 3.6e-12 kg/kg); with the reset disabled (the nest before the reset),
θ was up to 0.66 K and qv and qc up to 0.24 g/kg off.

## 6. Claim limit

This seam integrates one refined child grid concurrently inside one Cartesian
parent, with the parent supplying the child's lateral boundary values and
tendencies and optional two-way feedback, and moves that child with a tracked
surface vortex (or CM1's pressure centre) by WRF's moving-nest rules (5.8).
The child may carry a finer vertical deck over the parent's model top (5.6).
Both paths are host-checked and neither has run on a device. The tornado
criterion (5.9) reports whether GM1's fixed criterion was met on the child's
grid-volume 10 m winds; that is a verdict under GM1's own rules, not a damage
rating or an observed tornado, and it is host-checked only. A nested run
writes restarts for both domains and resumes bit-identically on the host
(5.10), the `hold` tracker's state included; that restart identity is not yet
device-verified. The child window (5.11) publishes the child's own
advecting velocity words over a following sub-volume at the child's own step,
and the offline integrator that reads it removes the time-interpolation error
rather than estimating it; its trajectories and its vorticity budget are
checked against closed forms on the host, and its budget publishes friction
as the curl of the held tendency only where the door has excluded damping from
the window, and on a rotating child adds the Coriolis force it re-forms from
the stream's winds and header (schema v3). The seam makes no forecast-skill claim for a nested run, and
no native-CM1 comparison of any kind, because CM1 r21.1 has no nesting to
compare against.
