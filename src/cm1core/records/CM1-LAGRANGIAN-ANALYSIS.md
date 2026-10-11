# Inline Lagrangian and budget analysis

What this record covers: the `[lagrangian]` configuration table, the code
behind it, what is CM1 r21.1's and what is GM1's own, and exactly what has
been validated and how.

## Why it is inline

Trajectories, vorticity budgets and circulation are normally computed *post
hoc* from model output, which forces a dump every few seconds -- terabytes --
and still loses the path between dumps.  GM1 already holds the state on the
device, so these are computed at the model timestep for the cost of a small
kernel, with no extra history at all.

Nothing in this table writes a prognostic carrier.  Turning all of it on
changes no simulated word; the run receipt says so explicitly
(`lagrangian.changes_the_simulated_storm = false`).

## The authority, line by line

| GM1 module | CM1 r21.1 |
| --- | --- |
| integrator (`nest_parcels.reference_rk2` and its kernel text, on the grid's own deck) | `src/parcel.F:10-620` `parcel_driver` |
| RK2 update, below-ground clamp, periodic exit | `src/parcel.F:521-593` |
| face searches, staggered trilinear stencil and weights | `src/parcel.F:153-461,1693-1711` |
| vertical ghost fill (bbc/tbc) | `src/parcel.F:59-130` |
| semi-slip (`bbc = 3`) log layer below `zh(1)` | `src/parcel.F:470-515` (driver), `:1515-1583` (interp), `get2d :1719-1787`; `zntmp` from `src/solve3.F:769`, `src/cm1.F:2192-2194,4011`, `src/turb.F:1161,1233-1237`, `src/bc.F:12-172` |
| recorded attributes, `u`/`v`/`w` included | `src/parcel.F:656-1590` `parcel_interp` on `u3d`/`v3d`/`w3d` (`src/cm1.F:4010-4020`) |
| attribute inventory | `src/parcel.F:1898-2130`, offsets `src/param.F:6385-6500` |
| the advecting velocity for `psolver=3` | `src/solve3.F:438-466`, saved at `src/solve2.F:1290-1301`, averaged at `src/sound.F:806-832` |
| call site | `src/solve3.F:768-769` |
| seeding formula | `src/init3d.F:355-411` |
| horizontal origin (`[initialization].iorigin`, shared by the door and the runtime) | `src/param.F:6646-6664,7212-7213,7406-7407,7743-7744` |
| track file layout | `src/parcel.F:1626-1690`, descriptor `:2145-2210` |
| vorticity components | `src/misclibs.F:1542-2100` `calcvort` |
| pdcomp decomposition | `src/pdcomp.F:10-420`, inversion `src/poiss.F:10-220` on `src/param.F:5178-5205`'s mirrored modes (`src/init3d.F:1668-1692`), names `src/writeout.F:6252-6340,8478-8580`, selector `src/param.F:727-729` |
| high-frequency output | `src/hifrq.F:21-1790`, selectors `src/cm1.F:2645-2647` |

## Two defects the differential found

`src/solve2.F:1290-1301` copies the acoustic small-step time average into
`uten1`/`vten1`/`wten1` at `nrk == nrkmax`.  That copy is not bookkeeping: the
very next thing CM1's large step does is refill `rru`/`rrv`/`rrw` with the
**mass flux** `rho0*u` for the scalar advection, which is what `src/adv.F`
wants.  GM1 does the same refill at `cuda_advection5.py:330`.

The first version of this port read `arrays["rru"]` at `solve3` time and
advected every parcel with `rho0*u`.  Against CM1 that is a height-dependent
error: the ratio GM1/CM1 measured **exactly the base density** -- 1.1185 at
250 m, 0.9326 at 2250 m, 0.4513 at 9250 m, against CM1's own published
`rho` of 1.1112, 0.9315, 0.4513 at those parcels.  It produces trajectories
that look entirely plausible and are wrong by a factor of two in the upper
troposphere.  `LagrangianStage.capture_advecting_velocity` is the fix and
`tests/test_lagrangian_door.py` pins it.

The second is the corner-vorticity index.  `parcel.F:1064-1066` builds `zvort`
at C-grid corners from `v(i,j)-v(i-1,j)` and `u(i,j)-u(i,j-1)`; the first
slice of the padded carrier was one cell west and south of that.  Fixing it
moved `zvort`'s `max|d|/peak` on the same differential from 0.491 to 0.0378 --
a 13x improvement on a field that is itself at the noise level in this deck,
so only the differential could see it.  `corner_vertical_vorticity` is now one
shared function and `tests/test_vorticity_budget.py` proves it exact for a
linear wind.

## What is GM1's own, and marked so

* **seeding modes** -- CM1 hard-codes one 60x60x10 net (`init3d.F:381-387`).
  GM1 exposes that as parameters and adds a box and a vorticity-threshold
  mode.  `seeding = "grid"` with the defaults *is* CM1's formula.
* **backward trajectories** -- the same RK2 with `-dt`.
* **the vorticity budget** -- CM1 r21.1 computes no tilting, stretching,
  solenoidal or frictional tendency anywhere.
* **material circuits and circulation** -- CM1 has no circuit diagnostic.
* **the high-frequency window** -- `hifrq.F` writes the whole domain, and
  `dohifrq`/`hifrqfrq` are hard-coded in `cm1.F` and reachable from no
  namelist at all.

## Validation

| capability | how | result |
| --- | --- | --- |
| parcel trajectories | **differential against CM1 r21.1**, `examples/oracle/parcel-advection-32.differential.toml`, horizon chosen from the advective timescale `dx/U` | 8.56 cell crossings, 8564 m mean displacement; worst of 2250 parcels 2.1 m, RMS 3-D separation 0.34 m (2.5e-4 of displacement) |
| parcel attributes | same run, every attribute both models write | `x y z u th t prs qv rho` all `max|d|/peak <= 2.7e-3`; `qc qr qi qs qg tke` identically zero in both |
| RK2 integrator + sampler + weights | analytic: solid-body rotation, for which the discrete map is exactly `(1 - theta^2/2 + i*theta)` per step | amplitude and phase reproduced to 2e-5 relative over 40 steps |
| uniform advection | analytic | exact to 1e-3 m over 30 steps |
| vorticity components and budget | analytic: a random **linear** velocity field, for which every centred difference is exact | every component and all nine velocity-gradient terms to 1e-5 relative; identically zero for solid-body rotation |
| solenoid | analytic cross-product on a linear rho/p pair | 1e-4 relative in the centred interior |
| circulation | analytic: the exact N-gon value `2*omega*area` | 1e-6 relative |
| Kelvin's theorem | conservation identity, circuit carried by the real RK2 | conserved to 2e-4 over 60 steps |
| contracting circuit | analytic `r(t) = r0 exp(-a t)` plus Stokes on a constant-vorticity field | both to 1e-3 |

### Not resolved by the differential

`w`, `b`, `vpg` and `zvort` are near-zero fields in the warm-bubble deck the
differential uses (`zvort` peaks at 1.3e-4 s^-1 with `icor=0` and no storm),
so the comparison measures two independently rounded dycores rather than the
diagnostic.  Whole-domain: `w` 2.3e-2, `b` 2.2e-2, `vpg` 0.19, `zvort` 3.8e-2 of peak.
Interior-only (8 km margin): `b` 1.1e-3, `vpg` 5.7e-2, `w` 1.1e-2.  A rotating
case would resolve these; this one does not, and the divergence table in
`evidence/oracle/parcel-advection-32/` says so per attribute.

## What each selector actually produces

A run with the table on writes one `lagrangian-track/` directory beside the
history frames:

| selector | on disk |
| --- | --- |
| `iprcl` | `parcel-attributes.f32` (frames x npvals x nparcels, little-endian f32, CM1's `parcel.F:1668` variable order transposed once and stated in the table), `parcel-attribute-table.tsv`, `parcel-track.json` |
| `circuits` | `circulation-circuits.json`, one row per `prclfrq_s` |
| `vorticity_budget` | `budget-<step>/`, 3 components + 15 terms |
| `output_pdcomp` | `pdcomp-<frame>/` (`pipb`, `pipdl`, `pipdn`, and `pipc` when `icor=1`) and `pdcomp-w-<frame>/` (`pgradb`, `pgraddl`, `pgraddn`, `pgradc`, nz+1 w levels), one pair per history frame |
| `dohifrq` | `hifrq-<step>/`, the declared window of 15 carriers |

Every one of those receipts carries its own `unvalidated` marker **on disk**,
because a marker that exists only in a returned dictionary is a marker nobody
reading the artifact will see.

`evidence/lagrangian/full-selector-run-32/` is a real RTX 5070 Ti run with all
five on, and its README states the measured values and why each is physically
right for that deck.

## Unvalidated, and where the marker appears

Every item below is **selectable now** and carries a machine-readable marker
in the run receipt (`lagrangian.unvalidated`) and in `gpuwm-cm1 doctor`
(status `UNVALID`, one finding each):

* `vorticity_budget` -- no CM1 counterpart exists; analytic validation only.
* `circulation` -- no CM1 counterpart exists; analytic validation only.
* `pdcomp` -- CM1's own decomposition is inverted on the device with CM1's
  open-lateral method (even reflection into a doubled transform,
  `src/param.F:5178-5205`, `src/poiss.F:80-114`).  Forcings, fluxes, `divx`,
  `thv0` and coefficients match gfortran-built pristine `pdcomp.F` bit for
  bit and the inversion matches `poiss.F`/`singleton.F` to FFT round-off
  (`tests/test_pdcomp_open.py`); no storm run has been differenced against
  CM1's history output, and `pipdn` inherits this lane's `advu`/`advv`/`advw`.
* `high_frequency_window` -- unreachable from any CM1 namelist, so there is no
  CM1 run to compare a window against.
* `backward_trajectories` -- CM1 has no backward mode.
* `vorticity_threshold_seeding` -- CM1 has no threshold mode.

## Which switches CM1 keeps

The sixteen `prcl_*` words are a request.  `src/param.F:6257-6282` overrides
it from the rest of the namelist before `:6385-6500` lays out one column, and
`lagrangian_parcels.parcel_attribute_layout` is the one place GM1 does the
same, for the parent stage and the nest child's tracker alike, from the run's
`ParcelSwitchContext` (`parcel_switch_context(runtime_config)`):

| rule | CM1 | switches turned off |
| --- | --- | --- |
| `iptra = 0` (GM1 carries no passive tracers) | `param.F:6257-6259` | `prcl_ptra` |
| `imoist = 0` | `:6260-6264` | `prcl_q`, `prcl_nc`, `prcl_dbz`, `prcl_qsat` |
| `ptype = 1` or `4` | `:6266-6268` | `prcl_dbz` |
| `idm = 0` (WSM6) | `:6269-6271` | `prcl_nc` |
| `sgsmodel <= 0 .or. ipbl <= 0` (every `cm1setup = 1` deck) | `:6273-6276` | `prcl_km`, `prcl_kh` |
| `.not. iusetke` | `:6277-6279` | `prcl_tke` |
| `bbc /= 3` | `:6280-6282` | `prcl_sfc` |

`bbc` is read after `:873-889` forces it to 3 whenever `sfcmodel >= 1`,
`oceanmodel = 2` or `ipbl >= 1`.  No admitted GM1 run reaches that rewrite (the
door refuses `sfcmodel >= 1` without `bbc = 3` by name, refuses
`oceanmodel = 2`, and holds this lane to `cm1setup = 1`, where `ipbl >= 1` is
refused), but the resolver applies it, and the oracle's
`bbc-forced-by-surface-layer` case holds it to CM1.  `sgsmodel` is read after
`:1074-1085` rewrites it to 0 without `idoles`, and `iusetke` as
`:4989-4997`/`:5107-5108` set it.  The layout then gates again:
`prcl_q`'s ice rows and `prcl_qsat`'s `qsi` need `iice = 1`, and `prcl_km`/`kh`/
`tke` lay out nothing unless `cm1setup >= 1 .or. ipbl >= 1` (`:6443`).
`iptra = 1` and `terrain_flag` are refused by name: their columns (`pt1..`,
`zs`, `sigma`) have no carrier or integrator here.

The sealed `selectors` keep the request; every layout record
(`gpuwm-cm1.lagrangian-parcels/v3`) carries the resolved `switches`, the
`requested_switches`, the rule in `switch_overrides`, any
`switches_without_columns`, and the `switch_context` it read.
`evidence/oracle/parcel-switch-resolution/cm1-parcel-switch-resolution.json`
is what the CM1 r21.1 binary itself printed and wrote for seventeen namelists,
one per branch (`tools/oracle/parcel_switch_resolution_oracle.py`), and
`tests/test_parcel_switch_resolution.py` holds GM1's resolved switches, column
names, units and offsets to it.

## Attributes with no carrier in this lane

Named in `PARCEL_UNFILLED_ATTRIBUTES`, published as the `-1.0e30` sentinel and
listed in the track table's `filled` column:

The Eulerian budget's `friction_*` terms are published as exactly zero and
named in `BUDGET_UNFILLED_TERMS` for the same reason: GM1's retained momentum
tendency (`held.uten`/`vten`/`wten`) is CM1's *complete* held RK tendency --
subgrid turbulence, Rayleigh damping and the large-scale terms together
(`src/solve1.F`) -- not the frictional part alone, so publishing its curl
under the name `friction` would attribute damping and forcing to friction.
`cpu_vorticity_budget` takes an isolated tendency from any caller that has
one, and `tests/test_vorticity_budget.py` proves the term is the curl it
claims to be.

Parcel attributes with no carrier:
`qsl`, `qsi` (`cm1libs.F`'s `rslf`/`rsif` fits are not transcribed); `dbz` (the
ArWen WSM6 reflectivity is not retained past its physics block); on a
`bbc = 3` deck, `znt` (the semi-slip step reads `closure.surface.znt`, but the
`prcl_sfc` column is not recorded from it) and `ust` (surface-layer tile
scratch, not a retained carrier).  `kmh`, `kmv`, `khh`, `khv` would be unfilled
too (the Cartesian closure allocates them as held-tendency phase scratch,
`cuda_cartesian_closure.py:336-339`), but CM1 lays them out only for
`cm1setup = 4` with `ipbl >= 1`, which this lane refuses, so no track carries
them.

## CM1 budgets and the vorticity budget

`output_ubudget`/`vbudget`/`wbudget`/`thbudget`/`qvbudget` (`[output]`) are
CM1's own words and publish CM1's own names; the term list is
`src/param.F:5302-5875`'s decision for the run, resolved in
`cm1_budget_terms.py`.  `cartesian_budgets.py` takes each term at CM1's own
placement on the step whose state a frame carries (`src/cm1.F:3268-3296`):

| term | CM1 | how GM1 takes it |
| --- | --- | --- |
| `*_hadv`, `*_vadv` | `adv.F` doingubud/doingsbud | the fused advection kernel replayed with the other direction's mass flux zeroed and the directional divergence share |
| `*_hidiff`, `*_vidiff` | `adv.F` gethidiff/getvidiff | `tot - ndiff`, `ndiff` from `flx6`/`vadv_lwr_flx_4`/`_2` |
| `*_hturb`, `*_vturb` | `turbtend.F` turbu/v/w/s | the SGS column kernels replayed with the vertical (resp. horizontal) stresses and kmv/khv zeroed; the implicit column solve collapses to an exact zero |
| `*_rdamp` | `solve1.F:489-529` | the tile's Rayleigh output |
| `ptb_diss`, `ptb_rad` | `solve1.F:920-1048` | the held-theta increment across the operator |
| `ub/vb_cor` | `solve2.F:694-714` | the rotation kernel into a zero tendency |
| `wb_buoy` | `solve2.F:896-920` | the buoyancy kernel into a zero tendency |
| `*_pgrad` | `solve2.F:925-956,1196-1245` | `(x3d-xa)/dt` minus the saved explicit tendency |
| `ptb_div` | `solve2.F:1331-1340` | `-t11*thterm` of the final stage |
| `ptb_mp`, `qvb_mp` | `mp_driver.F:151-170,1670-1690` | `(after-before)/dt` around pdefq and the scheme |

Every replay launch goes through `BudgetWriteGuard`, which binds arguments by
the kernel source's own parameter names and refuses any writable argument that
is not a budget-owned array, on the device path as well as on the host.
`tests/test_cm1_budgets.py` runs the packaged kernel text on the CPU
(`tools/oracle/cuda_kernel_host_emulator.py`) and proves: every split sums to
the fused tendency to binary32 rounding, on uniform and stretched columns; the
u/v/w budgets close against `(x3d-xa)/dt`; theta and qv close against the
scalar stage's own update; and a final RK stage run with every budget on
leaves every model word identical to the run with budgets off.  Closure is
also measured by value in every run receipt (`budgets.closure`).

Refused by name (`CM1-OUTPUT-BUDGET`): inside a live nest (the Davies
relaxation and specified zone have no CM1 term, and feedback overwrites the
parent's state after its step); with a PBL scheme (`*_pbl` not captured);
`radopt>=1` for `ptb_rad`; `testcase=10` nudging; the Morrison component for
the theta/qv budgets (`morrison.F`'s process rates do not cross the seam).
No budget state is written to a restart: `restart_write.F:2083-2092` writes
diag arrays only under `restart_file_diags`, which CM1 ships `.false.`.

`lagrangian.vorticity_budget_history` and `lagrangian.parcel_vorticity_budget`
are GM1 extensions.  Every curl term is `calcvort` (`misclibs.F:1618-2056`,
transcribed word for word against `cuda_cm1_output_diagnostics`) applied to
one momentum budget term, so `xvortb_tend` is the tendency of CM1's own
`xvort` to rounding; baroclinic generation is `xvortb_buoy`/`yvortb_buoy`
(the curl of `wb_buoy`) and friction is `*vortb_hturb + *vortb_vturb`.  The
stretching/tilting/divergence/advection decomposition is the continuous
identity at scalar points, evaluated on the winds the final-stage advection
used; its disagreement with the curl of the model's own advection terms is
published as `*vortb_advres`, by value.  When the stage ran on a
`vorticity_budget` frame step, that frame's `friction_*` slots are the curl of
the `hturb`/`vturb` split rather than zero.

## Sealed identity

`[lagrangian]` is optional as a whole.  A document that omits it projects
nothing for it, so every already-recorded `plan_sha256` is unchanged; this is
pinned by `test_lagrangian_door.py::test_a_document_without_the_table_keeps_its_exact_sealed_identity`,
which names the four shipped examples and their hashes.  A document that
*declares* it projects every key it resolves, declared or defaulted, so the
sealed bytes name the whole diagnostic configuration.

## Parcels in the live nest child (`[lagrangian.nest_child]`)

GM1 EXTENSION: CM1 r21.1 has no nesting.  `cm1core.nest_parcels` integrates a
declared box of parcels on the child's own lattice and vertical deck at the
child's step, with the child's own `solve2.F:1290-1301` average, and records
the same attributes as parent parcels plus a `grid` row.

| piece | authority / rule |
| --- | --- |
| step, sampler, clamp | `parcel.F:59-130,153-461,521-568`, `tri_interp :1693-1711`; the deck's own `zf`/`zh` search (`:251-260`) |
| halos of the advecting average | `solve3.F:498-508`, `bc.F` `bcu`/`bcv`/`bcw` |
| attributes | `parcel_interp :732-876,1061-1072,1134-1510`, `constants.F:112-133`, `base.F:3209`, `param.F:6327` |
| coordinates | metres from the **parent grid origin** (parent `xf(1)`, `yf(1)`, ground), parent's translating frame; `seed_frame = "parent-grid-origin"` is required |
| child boundary | GM1: hand-off at parent-step boundaries (after feedback and any move) between the child's free interior and the parent; whole CM1 steps on each grid; per-sample `grid` code |
| termination | only where `parcel.F` has no answer: left its lattice between boundaries, stencil past the deck ghosts, non-finite |
| refusals | no `[nest]`; seeds outside the child's free interior; `prclfrq_s` not a whole number of parent steps; terrain; rigid parent laterals |
| semi-slip | `bbc = 3` admitted: each grid's `closure.surface.znt` is captured with its advecting average and `bc2d`-padded into `zntmp`; child frames on a `bbc = 3` child carry `placements/frame-NNNNNN.znt.f32` (SHA-256 in the placement) for the offline reader |
| output | `nest-child-lagrangian-track/` (`parcel-attributes.f32`, table, `parcel-events.tsv`, `parcel-track.json`); `nest-child/placements/frame-NNNNNN.json` beside each child frame |
| extension point | `NestParcelTracker.register_hook` (`ParcelAttributeHook`), for per-parcel vorticity-budget terms |

The kernel text is held raw-word to the vectorized host reference by
`tests/test_nest_parcels.py` through `tools/oracle/nest_parcel_oracle.py`
(g++, `-ffp-contract=off`, linear and shuffled thread order), on stretched and
uniform decks, with and without the semi-slip branch (`logf` bound to the host
libm on both sides).  The semi-slip branch, `pad_roughness` and
`parcel_interp`'s log-layer `u`/`v`/`zvort` are further held to statement-by-
statement scalar transcriptions of `parcel.F:470-515`, `:1515-1583` and
`bc.F` `bc2d`.  On a device, `tools/oracle/nest_parcel_device_parity.py`
reports the same comparison; its `bbc = 3` cases pass through CUDA's `logf`
and have not been run.

## One integrator, and what it corrected

Parent `[lagrangian]` parcels used to run their own kernel.  They now run the
nest transcription above on the parent's deck, which corrected five
deviations from `parcel.F`, each pinned by a test:

| was | now | authority | test |
| --- | --- | --- | --- |
| `bbc = 3` accepted, trilinear wind through an unfilled ghost below `zh(1)` | log layer over `zntmp` | `parcel.F:470-515` | `test_lagrangian_parcels.py::test_a_semi_slip_stage_steps_with_zntmp_and_refuses_without_it` |
| `w` zeroed at storage `nz+2`, one level above the top face | `wa(nk+1)` zeroed | `parcel.F:124-128` | `::test_the_top_face_w_zero_is_cm1_level_nk_plus_1` |
| first RK stage wrapped and clamped | neither | `parcel.F:526-537` vs `:553-593` | `::test_the_first_rk_stage_is_neither_wrapped_nor_clamped` |
| closed-form vertical search on a uniform `dz` | scan of the grid's `zf` | `parcel.F:251-260` | `::test_a_stretched_deck_is_searched_on_its_own_faces` |
| recorded `u`/`v`/`w` from the advecting average; scalars on a float64 clamped stencil with uniform `dz` | `parcel_interp` on the completed state | `parcel.F:1134-1590`, `cm1.F:4010-4020` | `test_lagrangian_door.py::test_a_recorded_frame_carries_the_attributes_and_names_the_unfilled_ones` |
| runtime frame `iorigin = 1` while the door checked `[initialization].iorigin` | both read `RuntimeConfig.iorigin` | `param.F:7212-7213` | `test_lagrangian_door.py::test_the_door_and_the_runtime_share_the_initialization_frame` |

Also: halos of the parent's advecting average now follow `bc.F` for the
grid's own laterals (periodic included) rather than a one-cell open copy, and
the runtime passes its `tbc` rather than a literal 1.

Evidence recorded before these corrections and not re-measured:
`evidence/oracle/parcel-advection-32/` (the CM1 differential: uniform deck,
free slip, parcels far below the lid, so the search and `w`-top corrections
do not reach it; the midpoint-wrap correction reaches only parcels within one
step of a lateral face, and the recorded-attribute sampler change reaches
every published `u`/`v`/`w` and scalar column, so its worst-parcel 2.1 m and
attribute ratios are the old integrator's) and
`evidence/lagrangian/full-selector-run-32/` (see its README).

`cm1core.nest_parcel_backward` (`tools/nest_backward_trajectories.py`) is an
**offline diagnostic**: backward RK2 through saved child frames with a
linear-in-time wind and a Lagrange-remainder error estimate accumulated by a
discrete Gronwall recursion (Seibert 1993; Stohl 1998; Dahl et al. 2012).
UNVALIDATED against CM1, which has no backward or offline mode.

## The vortex-following child window (`[lagrangian.nest_child_window]`)

GM1 EXTENSION.  The full account is `CM1-NEST-SEAM.md` section 5.11; what
belongs here is why this table exists in `[lagrangian]` at all and what it
changes about the two paths above.

**Nothing else in `[lagrangian]` reaches the child.**
`cm1core.nest_live.child_runtime_config` builds the child's contract with
`lagrangian = None`, so the parcel stage, the Eulerian `vorticity_budget`, the
material `circuits`, `output_pdcomp` and the `dohifrq`/`hifrq_i0..k1` window
all run on the **parent** grid only -- and that window is a fixed index box at
a declared cadence, which is neither vortex-following nor at the child's step.
Before this change the only `[lagrangian]` table that reached the child at all
was `nest_child`.

Neither path reaches a tornado core.  Forward child parcels on a 40 m lattice
do not sample a 40-60 m core: three measured attempts (100k, 140k, 135k
parcels) put at most a few hundred within 300 m of the vortex and **not one
ever recorded more than half the grid's own vorticity**.  Backward
trajectories through saved 30 s child frames do reach it, but their own
position-error estimate was 800-1200 m median, and that estimate is the
Lagrange remainder `(T^2/8) max|u_tt|` -- it scales with the *square* of the
output interval, so no post-hoc reading of 30 s frames can be made accurate
there.

`[lagrangian.nest_child_window]` writes a sub-volume of the live child that
**follows the vortex**, at **every child step**, holding the model's own
advecting velocity -- the `solve2.F:1290-1301` acoustic small-step average the
integrator above is already handed at `solve3.F:768-769`.  Nothing in it is
re-derived: every field is a strided copy of the child's padded carrier, and
the only words formed are `parcel.F:59-96`'s `k = 0` ghost of `u` and `v` and a
reflected `w` ghost no trajectory reads.

| piece | authority / rule |
| --- | --- |
| the words carried | `solve2.F:1290-1301`, `solve3.F:438-469` (`uadv`/`vadv`/`wadv`), the child's live `rho`/`prs`, and `znt` when `bbc = 3` (`turb.F:1161,1233-1237`) |
| when | at the child's `nrk == nrkmax`, the same moment in the same step as `NestParcelTracker.capture_child` |
| the bottom ghost | `parcel.F:59-96`, local to the bottom of any block that reaches the ground (`nest_parcels.fill_velocity_bottom_ghost`) |
| coordinates | metres from the **parent grid origin**, `frame = "parent-grid-origin"` required, exactly `[lagrangian.nest_child]`'s frame |
| where | GM1: the tornado criterion's centre (`CM1-NEST-SEAM.md` 5.9) from the previous parent step, taken from that sample's own ground position (it was sampled before the move, so its corner index belongs to the old placement), held when a sample finds none, clamped at the child's free interior less three ghost columns |
| refusals | no `[nest]`; terrain; a `bbc` with no `parcel.F` lower branch; `follow = "tornado-criterion"` without `[nest.tornado_criterion]`; `end_s` past `run.duration_s`; a window that cannot fit the free interior with its ghosts; a top at or above the child's own; bytes per model minute over `budget_bytes_per_model_minute` |
| a nest move | the child deck's `xf(1)`/`yf(1)` are put on the new geometry at the parent-step boundary, so the record's ground origin (`child xf(1) + i0*dx`) follows the move while the centre stays where the parent's frame put it |
| output | `<history>/nest-child-window/`: one append-only `window-stream.f32`, one `window-records.tsv` row per record with its SHA-256, one `window-stream.json` |

`wind_source = "model-step"` (`cm1core.nest_parcel_backward`) reads that
stream and reverses one record per step with **no time interpolation**: the
same one frozen wind for both RK2 stages `parcel.F` itself uses.  It is refused
on a stream written with `stride_child_steps > 1`, by name: a record is the
average over one child step, so across a strided interval the wind is frozen
over steps it was never the average of and the receipt's "zero
time-interpolation error" would be false.  At stride 1 the
time-interpolation term is absent, not estimated.  What is published
instead is a measured `round_trip_residual` -- every parcel re-integrated
forward from where it ended -- a measured `sensitivity_amplification`
`prod(1 + |dt| L_n)`, and a `position_error_estimate` that is the residual plus
half a window cell carried back through that amplification.  The round trip
alone is a **lower** bound and the receipt says so: where the flow is locally
linear, RK2 with `-dt` and RK2 with `+dt` are inverses to `O(dt^4)`.

`cm1core.nest_window_budget` (`tools/nest_window_trajectories.py`) samples the
vorticity budget at every trajectory position with `parcel.F:1365-1447`'s own
scalar-point stencil, on the window deck's own `zh` (`misclibs.F:1628`'s
`rdz*mf` metric, because every tornado-grade child is stretched), and checks
that it closes.  **Friction is the curl of the held tendency** (stream schema v2): the window
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

UNVALIDATED against CM1 -- it has no nesting, no sub-volume writer, no
backward mode and no vorticity budget.  Validated analytically against a
closed-form stretching vortex whose vorticity equation closes exactly
(`tests/test_nest_child_window.py`, `tests/test_nest_window_budget.py`).
