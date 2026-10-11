# CM1 21.1 cadence map

This map describes the selected first-milestone branch. It separates a
source observation from an executable claim: every `oracle missing` item is a
runtime refusal, even when the call order is unambiguous in Fortran. Two
startup cadence points are ratified: `O-GRID-BASE-V1` writes once after the
single `base` call and before `init3d`; `O-POST-GETSET-V1` writes once after
the single `getset` call and before time-loop preparation.

## Startup initialization cadence

The driver calls `init3d` once at `src/cm1.F:2390-2403`, optionally reads a
restart at `2660-2667`, and calls `getset` once at `2884-2904`. The selected
official zero-step profile has `irst=0`, so no restart read intervenes.
`getset` copies A carriers into current carriers, forms pressure/density and
`rr/rf`, establishes `mass1`, saves `ppx`, and refreshes selected TKE state at
the exact ranges pinned in `CM1-SOURCE-MAP.md`.

`O-POST-GETSET-V1` writes at exact authority line `src/cm1.F:2905`, phase
`post_getset`, call ordinal 1. It is
invoked exactly once per run on rank zero. All 46 complete f32 allocations and
the source-adjacent input/layout streams are identical across O0, O2, and an
O2 repeat. Its cadence control changes only the selected warm-bubble record's
call ordinal from 1 to 2 and is uniquely rejected. This ratifies the captured
combined startup result, not a separate post-`init3d` state and not any
time-dependent cadence. A later startup `sfc_and_turb` pass at
`src/cm1.F:3718-3788` prepares omitted closure state before the first solve;
the captured named fields happen to remain unchanged for this configuration,
but the deck is not the complete first-solve state.

## Outer step

For each large step, `src/cm1.F:3311-3475` executes:

1. `solve1`: compute terms held fixed through all RK stages and acoustic
   substeps.
2. `solve2`: perform low-storage RK stages, advection, buoyancy, pressure
   gradient, acoustic substeps, equation-set updates, and final-stage mass
   correction.
3. When moisture is enabled, dispatch native `mp_driver` after `solve2`.
   This is native-context evidence only. The product replaces the component
   at an explicitly measured seam; it does not reuse `ptype`.
4. `solve3`: finish communications/state bookkeeping and post-physics work.

The exact retained-tendency lifetime across this sequence is part of restart
identity. It cannot be reconstructed from the visible prognostic fields
alone.

## Low-storage RK and split-explicit pressure

`src/solve2.F:294-299` sets

```text
dttmp = dt / (nrkmax + 1 - nrk)
```

For the target `nrkmax=3`, the stage durations are `dt/3`, `dt/2`, and
`dt`. These are sequential low-storage updates, not three independent
weighted tendencies. The port must preserve the state that each stage reads
and overwrites; matching only the three fractions is insufficient.

The pressure dispatch at `src/solve2.F:1080-1173` is branch-specific:

- `psolver=2`/`7` calls `sounde`, the reduced bootstrap lane.
- `psolver=3` calls `sound`, the product lane.

Within `sound` (`src/sound.F:10-857`) the large-stage update is subdivided
into acoustic steps. Setup/counting occurs near lines 88-99, preparation near
191, substeps begin near 253, vertical implicit-system setup begins near 484,
the solution/new-`w` path begins near 659, and completion is near 839. No
operator may be moved across these boundaries merely to reduce kernel count.

## Advection and WENO

Scalar and velocity advection are invoked from `solve2` for each RK stage.
The selected centered fifth-order operators and WENO5 helpers have different
branch/cadence semantics.

For `advwenos=2`, `src/adv.F:75-78` activates scalar WENO only when both the
local WENO flag is set and `nrk==nrkmax`. Corresponding velocity-family checks
are at lines 958-960, 1894-1896 and 2828-2830. The official supercell profile
uses scalar WENO selector 2 and velocity WENO selector 0. Therefore:

- stages 1 and 2 use the selected fifth-order scalar advection without WENO;
- stage 3 may use WENO5 at eligible points;
- velocity advection does not silently inherit scalar WENO.

An oracle must include a smooth field and a deliberately non-smooth field.
If turning the WENO flag on produces no measured change in the non-smooth
case, the oracle deck is invalid.

## Final-stage-only work

`src/solve2.F:1618-1656` starts the dry-mass correction only on the last RK
stage. Per-level sums are accumulated/reduced in double precision in the
native distributed path; lines 1973-2030 finish the reduction, use an explicit
64-bit gridpoint count, and update pressure/density. The single-GPU port must
state a fixed reduction tree and compare its correction separately before
using a trajectory norm.

The bounded `eqtset=2` retained-tendency remove/snapshot/rebuild/recapture
lifecycle and public WSM6 call now pass CPU/sm86 gates, including two
full-domain corrected-real transitions and exact transition-local resume. The
complete `src/solve2.F:1561-1611` composition with live dycore tendencies and
multi-step water/energy budgets remains red; the microphysics-only transition
does not validate that coupled cadence.

`idiss` is a different cadence: `src/solve1.F:744-848,917-975` prepares the
moist heat capacity and inserts the adjacent-face `dissten` average pre-RK,
before the prior-physics held-tendency consume. `O-EQTSET2-IDISS-V1` proves
the supplied-input consumer native/sm86 raw-word exactly. The selected startup
pass now wires it only after `calcnm -> calcdef -> tkekm -> tkebc` has supplied
live z-face `dissten`; the full corrected-real first-step composition passes on
sm86. It must not be moved into the post-final-RK WSM6 bridge. `efall=1` would
instead execute inside the post-microphysics driver before the ptype-5 rebuild
and remains strictly refused absent public species terminal velocities.

## Boundaries, damping, moving frame and SGS

Boundary arrays are not generic halos. Scalar, pressure, `u`, `v`, `w`, TKE,
and acoustic substeps use family-specific routines in `src/bc.F`. Fourth-
order radiative lateral work (`radbcew4`, `radbcns4`) and its corner order
must be captured independently. Rigid bottom/top behavior is checked on
z-faces, including the extra `w` level.

Upper damping, moving-frame terms, and SGS/TKE are scheduled through retained
pre-RK work and/or stage tendencies selected in `solve1`, `solve2`, `turb.F`
and `turbtend.F`. `O-TERRAIN-DEFORMATION-SGS-V1` proves the selected
terrain/all-open startup order through the local TKE source. The bounded
evolved-closure lane now applies `gettau -> turbu/turbv/turbw` once in
`solve1`, applies `turbt` once in `solve1`, then requires a separate
`advw/RK/bcw` call immediately after each RK stage's post-acoustic mass-flux
and `getdivx` rebuild. `tkea` remains the fixed large-step baseline until the
stage-3 result is committed in `solve3`; `tketen` remains held. The
cadence-ambiguous all-three-stage entry point refuses. A native primitive deck
and analytic CPU/sm86 deck are raw-word exact, and the real `300 x 300 x 75`
gate executes the order with source-live fields. The real T0 `tkea` remains
exact zero after clipping, and the mass-flux rebuilds deliberately hold T0
momentum fixed, so later-step/coupled-trajectory, scalar turbulence, MPI halo,
storm, and forecast claims remain absent.

## History and restart cadence

History output is observational and must not change the model trajectory.
Restart is semantic state: prognostic arrays, clocks, stage/step identity,
held tendencies, radiative-boundary history, SGS/TKE state, component state,
configuration/source identities and the deterministic-reduction contract all
belong to the checkpoint inventory. A restart gate compares uninterrupted and
checkpoint/resume runs with identical history-write schedules.

The `CM1W6R1` proof now covers 21 transition-local caller, held-tendency, and
WSM6 component arrays plus completed-transition cadence: the restored fresh
arena's second WSM6 transition is raw-word identical to the uninterrupted
second transition. Reflectivity is an exactly-once output-due transient and is
correctly absent from that checkpoint.

The Cartesian lane now has the complete inventory as a `CM1FCP1` restart
(`src/cm1core/cartesian_restart.py`): the padded prognostic owners **with
their halos**, the held tendencies, the TKE triple, the eqtset=2 rate
carriers, the radiative lateral-boundary history, the ArWen WSM6 persistence
and rain accumulators, and the scalars CM1 writes beside them --- `nstep`,
`mtime`, `nwrite`, `nrst`, the dry-mass reference `mass1`
(`restart_write.F:412`), the `asq` moisture budget (`restart_write.F:657`) and
the `domainlocx`/`domainlocy` integral (`restart_write.F:399-400`). The base
state, grid metrics and uniform grid coefficients are rebuilt from the pinned
sounding and geometry rather than carried, which is what
README.namelist:1512-1516 records for CM1's own restart files. The restart
gate this section asks for exists and passes on the host
(`tests/test_cartesian_restart.py`): a run stopped halfway, written through
the real Rust boundary and continued lands on the same words, and the same
frames under the same indices, as one that was never interrupted. The
device-side form of the same gate is `tests/test_cuda_cartesian_restart.py`,
written and GPU-gated.

Current status: the call-site/cadence ledger is partial and `R-016` remains
open. The bounded post-base sample and complete captured post-`getset` startup
state both pass O0/O2/repeat raw-bit oracles (`max_ulp=0`) plus cadence negative
controls. Bounded CUDA operator and WSM6 cadence slices exist, but the complete
source-ordered large-step cadence and native-vs-port trajectory remain absent,
so every forecast door still refuses before backend allocation.
