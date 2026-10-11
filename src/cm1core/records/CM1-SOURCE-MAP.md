# CM1 21.1 source map

Authority: UCAR `cm1r21.1.tar.gz`, SHA-256
`dc49fe84531056d1ae6249b37a5e3ee453fd96861c3b6bafd63828d92e64edf7`.
Every line number below addresses the pristine extraction of those bytes. A
source row being mapped does not mean that it has a native oracle or a CUDA
implementation; those are separate columns and gates.

## Cadence vocabulary

| Token | Exact meaning in the CM1 driver |
|---|---|
| `S0` | Once during process startup, before `init3d`. |
| `S1` | Once after `init3d`/optional restart, before the time loop. |
| `L` | Once per solved large step inside `cm1.F:3165-3475`. |
| `R` | Once in each `nrk=1..nrkmax` pass of `solve2.F:296-2052`; the selected profile has three passes. |
| `F` | Final RK pass only, `nrk==nrkmax`. |
| `A(nrk,n)` | Once per `sound` small step `n=1..nloop` within one RK pass. |
| `RST-R` | Startup only when `irst==1`. |
| `RST-W` | After a solved step only when the restart clock sets `dorestart`. |

## Partial operator, call-site, and cadence ledger

This ledger contains the reviewed rows below; it is not an every-operator
inventory and `R-016` remains open. Ranges pin the currently reviewed source
and cadence sites. Missing rows and every missing executable deck remain
runtime gates even when a cited row is source-exact.

| Operator ID | Authoritative implementation | Driver/caller and branch | Cadence | Oracle / port gate |
|---|---|---|---|---|
| `GRID-HALO` | `src/cm1.F:744-777` | Startup calls `wenocheck`, selects `ngxy` from advection/WENO order, fixes `ngz=1`, then derives `ib:ke`. | `S0`, before grid allocation | `O-GRID-BASE-V1` captures the derived bounds as inputs and full recorded 1-D extents only. It does not cover complete 3-D family halos; CUDA allocation remains open (grid and state). |
| `GRID-PARAM` | `src/param.F:6640-7451`, `7458-7726`, `7789-7812`, `8189-8209` | `param` is called at `src/cm1.F:1236-1250`; the cited x/y ranges form horizontal coordinates, weights, and reciprocal metrics, and the later ranges form uniform-z and reciprocal vertical metrics for the official profile (`stretch_[xyz]=0`). | `S0`, once | The complete captured 1-D coordinate/metric extents pass raw-bit comparison in `O-GRID-BASE-V1`; 3-D metric fields are only five sampled columns. CUDA implementation missing. |
| `BASE` | `src/base.F:25-3348` | One call at `src/cm1.F:1462-1471`, after base-array allocation at `1381-1416`, before all `init3d` allocation/work beginning at `1475`. | `S0`, once | **Bounded native oracle:** fixture `O-GRID-BASE-V1`, instrument write site `cm1.F:after-1471-before-1475`, 34 inputs and 6,601 outputs, zero-bit O0/O2/repeat spread, all four controls detected. Only five surface points and five 3-D columns over full vertical extents are sampled; no complete 3-D base/halo or CUDA claim. |
| `INIT3D` | `src/init3d.F:60-2024` | One call at `src/cm1.F:2390-2403`, after full state allocation; the official `iinit=1` path initializes wind/base pairs, the warm-bubble `tha`, and moisture state before later startup work. | `S0`, once | `O-POST-GETSET-V1` captures the combined result after the following `getset`; it is complete for its 46 named allocation payloads but is not an isolated post-`init3d` attribution or complete first-solve-state deck. CUDA comparison remains open (grid and state). |
| `RESTART-READ` | `src/restart_read.F:19-2684` | `src/cm1.F:2660-2667`, only when `irst==1`. | `RST-R` | Semantic inventory only; state/cadence identity deck missing, open (restart and output). |
| `GETSET` | `src/init3d.F:2032-2872` | One call at `src/cm1.F:2884-2904`, consuming `restarted`; `a -> 3d` copies are at `2280-2333`, pressure/density and `rr/rf` formation at `2362-2476`, `mass1` at `2483-2530`, `ppx` at `2535-2575`, and selected TKE work at `2688-2760`. | `S1`, once | **Bounded native oracle:** `O-POST-GETSET-V1`, exact write site `cm1.F:2905`, captures all bytes of 46 named f32 allocations (123,070,752 bytes/run), including stored u/v/w extra faces, plus `mass1` f64. `prs` is source-live only on its physical interior and its captured halo is allocation-zero storage; `rf0` and other named omissions remain gaps. O0/O2/repeat are raw-bit identical and all four controls are uniquely detected. Native-only; no CUDA/port or time-dependent claim. |
| `LARGE-STEP` | `src/cm1.F:3165-3513` | While `mtime<timax`, `dosolve` increments `nstep`, then calls `solve1` (`3313-3351`), `solve2` (`3357-3401`), optional native MP (`3405-3427`), and `solve3` (`3432-3475`); time/domain location advance at `3507-3513`. | `L` | Call order exact; one-complete-step instrument deck missing. Native MP remains context only. |
| `RETAINED-PRE-RK` | `src/solve1.F:77-1791` | Called only from `src/cm1.F:3313-3351`; creates tendencies explicitly held through RK/acoustic work. | `L`, before all `R` work | Source exact; retained-tendency deck missing, open (dry dynamics, boundaries and closures). |
| `SGS-TKE-PRE-RK` | `src/turb.F:1601-1638,4065-4506,4536-4545`; `src/turbtend.F:700-1044`, `1052-1808`, `1816-2465`, `2473-2885`, and `18-434` | `gettau` dispatch at `turb.F:1601-1638`; `turbt` at `solve1.F:362-369`; `turbu/v/w` at `1124-1165`; scalar `turbs` at `1208-1237`, under their SGS/TKE/configuration branches. | `L`, pre-RK; one call per enabled field/family | `O-EVOLVED-SGS-TKE-PRIMITIVES-V1` plus the analytic CPU/sm86 exact deck admit selected square-grid terrain `gettau`, momentum stress divergence and general nonuniform-`tkea` `turbt` on physical arrays. The real full-domain T0 gate is finite/source-live but produces clipped-zero TKE; scalar turbulence, native halo/MPI regions, and later-step coupled composition remain open (boundaries and closures). |
| `SGS-TKE-RK` | `src/solve2.F:1708-1803`; `src/adv.F:2778-3610`; `src/adv_routines.F:204-430,2359-2557,4038-4394`; `src/bc.F:942-1143`; commit `src/solve3.F:560-590` | Each stage copies held `tketen`, calls `advw(stag=4)` using that stage's rebuilt `rru/rrv/rrw/divx`, updates from fixed `tkea` with `dt/(3,2,1)`, clips below `1e-10`, and applies `bcw`; `solve3` commits stage 3. | `R` after each post-acoustic `getdivx`; commit once after `nrk=3` | Explicit per-stage CUDA API and CPU reference are raw-word exact on the analytic four-open deck; WENO is stage 3 only. The full real T0 cadence gate rebuilds flux before every call but deliberately does not evolve momentum/acoustic state between rebuilds. No coupled trajectory or mid-stage restart claim. |
| `RAYLEIGH-HELD` | `src/solve1.F:414-454` | Branch `irdamp==1 or hrdamp==1`; official target uses upper Rayleigh `irdamp=1`. | `L`, pre-RK | Exact branch; layer/tendency ablation deck missing, open (boundaries and closures). |
| `EQTSET2-DISSIPATION` | `src/solve1.F:744-848`, `917-975` | Moist heat-capacity preparation when `eqtset==2` and `idiss==1 or rterm==1`; with `bbc=1`, heating uses the adjacent z-face average `0.5*(dissten(k)+dissten(k+1))` and inserts into `thten1` before prior-physics tendency consume. | `L`, pre-RK | **Bounded native/CUDA oracle and real composition:** `O-EQTSET2-IDISS-V1` proves the supplied-input consumer raw-bit exact; `O-TERRAIN-DEFORMATION-SGS-V1` plus the full `300 x 300 x 75` sm86 gate binds its live z-face `dissten` immediately upstream for the source-initialized first step. Evolved TKE/SGS, complete tendency composition, later steps, and trajectory remain open (boundaries and closures, physics seam). |
| `PRIOR-PHYSICS-TENDENCY` | `src/solve1.F:1696-1777`; recapture at `src/solve3.F:319-337` | Reuses held `qpten/qtten/qvten/qcten` before RK, then replaces them after current physics. | Consume `L` pre-RK; recapture `L` post-RK/physics | Exact lifetime; injected-tendency/restart deck missing, hard red. |
| `RK3` | `src/solve2.F:294-299`, loop end `2052` | `DO nrk=1,nrkmax`; `dttmp=dt/(nrkmax+1-nrk)`, hence `dt/3`, `dt/2`, `dt` for `nrkmax=3`. | Three sequential `R` passes per `L` | Exact cadence; per-stage/full-step deck missing, open (dry dynamics). |
| `MOIST-COEFF-BUOYANCY` | `src/solve2.F:314-416` | Inside every RK pass; `394-416` is the `eqtset==2` moist heat-capacity/buoyancy branch. | `R` | Exact branch; prescribed-state term deck missing, hard red. |
| `CORIOLIS-MOVING-FRAME` | `src/solve2.F:631-720` | For the selected Cartesian f-plane branch (`icor=1`, `axisymm=0`, `betaplane=0`), stagger-interpolated `v3d+vmove` and `u3d+umove` enter the u/v tendencies; diagnostic copies are final-stage-only. | Tendency application `R`; diagnostics `F` | Selected-profile source branch mapped; Coriolis/constant-moving-frame ablation deck missing, open (boundaries and closures). |
| `MOMENTUM-ADV` | `src/adv.F:911-1839` (`advu`), `1847-2770` (`advv`), `2778-3610` (`advw`) | Calls at `src/solve2.F:885-892`. | `R`; u/v/w once each pass when the dynamical branch is live | Exact family/cadence; staggered operator decks missing, open (dry dynamics). |
| `BUOYANCY-APPLICATION` | `src/solve2.F:895-921` | Interpolates scalar-point buoyancy `dum8` with `c1/c2` onto interior w faces and adds it to `wten`; the optional diagnostic copy is final-stage-only. | Tendency application `R`; diagnostic `F` | Application site mapped; prescribed-state term and diagnostic-cadence decks missing, open (dry dynamics). |
| `SCALAR-ADV-PRESSURE` | `src/adv.F:14-903` | `advs` call at `src/solve2.F:1065-1070` for the pressure/buoyancy helper field. | `R` | Exact call; operator deck missing, open (dry dynamics). |
| `SCALAR-ADV-THETA` | `src/adv.F:14-903` | `advs` call at `src/solve2.F:1394-1399` with `tha/th3d/sten`. | `R` | Exact call; smooth/WENO deck missing, next target. |
| `SCALAR-ADV-MOISTURE` | `src/adv.F:14-903` | Moisture loop calls at `src/solve2.F:1490-1495` for vapor and `1510-1515` for other `q` species, when `imoist==1`. | `R` per active moisture species | Exact calls; species/operator decks missing, open (dry dynamics). |
| `SCALAR-ADV-TKE` | `src/adv.F:14-903` | `src/solve2.F:1808-1853`, only for enabled MYNN TKE advection branch. | `R` when branch live | Exact call/branch; deck missing, open (dry dynamics, boundaries and closures). |
| `SCALAR-ADV-TRACER` | `src/adv.F:14-903` | `src/solve2.F:1904-1946`, once per passive tracer when `iptra==1`. | `R` per tracer | Exact call/branch; outside first fixture but mapped, deck missing. |
| `WENO-SELECTOR` | `src/adv.F:75-78`, `958-960`, `1894-1896`, `2828-2830` | Scalar WENO also requires `wflag==1`; selector 1 is every stage, selector 2 only `nrk==nrkmax`; velocity families use their own selectors. | Tested at entry of each applicable advection call; selector 2 is `F` only | Exact branch; smooth/non-smooth and selector-off negative deck missing, open (dry dynamics). |
| `WENO5-HORIZONTAL` | `src/adv_routines.F:204-430`; reconstruction function `6166-6199` | Dispatch sites: scalar `adv.F:137-140`, u `1024-1027`, v `1962-1965`, w `2876-2879`; `weno_order==5 and doweno`. | Each eligible advection invocation; official scalar selector 2 means `F` | Exact helper/call sites; native helper deck missing, open (dry dynamics). |
| `WENO5-VERTICAL` | `src/adv_routines.F:2359-2557`; boundary-order fallbacks at `2421-2445`, `2548-2554` | Dispatch sites: scalar `adv.F:459-462`, u `1265-1268`, v `2202-2205`, w `3202-3205`; `weno_order==5 and doweno`. | Each eligible advection invocation; official scalar selector 2 means `F` | Exact helper/call sites; bottom/top branch deck missing, open (dry dynamics, boundaries and closures). |
| `PRESSURE-DISPATCH` | `src/solve2.F:1085-1173` | `psolver=1` `soundns`; `2/7` `sounde`; `3` `sound`; `4/5` `anelp`; `6` `soundcb`. | One dispatch per `R` | Exact branch table; psolver 2 and 3 decks missing, open (dry dynamics). |
| `SOUND3-COUNT` | `src/sound.F:84-103` | `sound` call at `solve2.F:1122-1136` only for `psolver==3`; derives `nloop,dts` separately for `nrk=1,2,3`. | Once per `R`, before acoustic loop | Exact equations; cadence deck missing, open (dry dynamics). |
| `SOUND3-PREP` | `src/sound.F:191-248` | Stage-1 and later-stage state preparation branches immediately before small steps. | Once per `R` | Exact branch; intermediate-state deck missing, open (dry dynamics). |
| `SOUND3-SUBSTEP` | `src/sound.F:255-837` | `DO n=1,nloop`; stateful open radiation calls at `261-267` when `irbc==2`. | `A(nrk,n)` | Exact loop/call sites; substep deck missing, open (dry dynamics, boundaries and closures). |
| `SOUND3-VIMPLICIT` | `src/sound.F:659-730` | Forward elimination at k=2/interior/top and backward substitution for new `w`. | Once per `A(nrk,n)` | Exact solver; coefficient/RHS/residual deck missing, open (dry dynamics). |
| `SOUND3-PRESSURE` | `src/sound.F:739-803` | Pressure update each acoustic step; `bcp(ppd)` only when `n<nloop`. | Once per `A(nrk,n)`; boundary fill between substeps | Exact update/cadence; deck missing, open (dry dynamics, boundaries and closures). |
| `SOUND3-HELD-PPX` | `src/sound.F:842-853` | Saves pressure tendency `ppx` only when `nrk==nrkmax`. | `F`, after final acoustic loop | Exact branch; retained/restart deck missing, open (dry dynamics, restart and output). |
| `PGRAD-DIAGNOSTICS` | `src/solve2.F:1196-1248` | Post-pressure-solver pressure-gradient/buoyancy diagnostic accumulation. | `R`, immediately after pressure dispatch | Exact site; term deck missing, open (dry dynamics). |
| `RADIATION4` | `src/bc.F:2054-2109`, `2117-2172` | Calls at `solve2.F:1261-1269` when `irbc==4` and the relevant lateral edge is present. | `R`, after pressure work | Exact edge/cadence; edge/corner deck missing, open (boundaries and closures). |
| `EQTSET2-STAGE` | `src/solve2.F:1273-1342` | Forms divergence and moist energy-conserving theta stage work for `eqtset==2`. | `R` | Exact branch; injected-tendency energy deck missing, hard red. |
| `EQTSET2-FINAL-REMOVE` | `src/solve2.F:1561-1611` | Removal/snapshot branch requires `nrk==nrkmax`, `eqtset==2`, moist physics and `ptype>=1`; EOS rebuild follows. | `F` | The CM1-owned public WSM6 bridge passes bounded source-ordered CPU/sm86 lifecycle gates and two full-domain corrected-real transitions with exact 21-array restart/resume. Composition with the dycore and complete forecast state remains red. |
| `EQTSET2-POST-MP` | `src/mp_driver.F:570-617`; terminal-velocity dependence corroborated at `src/kessler.F:485-522` | Native `efall=1` adds fallout/gravitational energy from species terminal velocities before the `ptype=5,eqtset=2` pressure/theta rebuild. | `L`, post-final-RK microphysics | The public WSM6 bridge passes the rebuild and real-data transition gates with `efall=0`. `efall=1` is strictly refused because public WSM6 does not expose the required `vterm` fields; no surrogate is inferred. |
| `MASS-REDUCE-START` | `src/solve2.F:1618-1656` | Requires `apmasscon==1 and nrk==nrkmax`; accumulates FP64 per-level sums and starts MPI all-reduces when compiled with MPI. | `F`, before stage boundary work | Exact branch; deterministic reduction deck missing, open (boundaries and closures). |
| `STAGE-BOUNDARY` | `src/bc.F:180-1143` (`bcs/bcp/bcu/bcv/bcw`) | Calls at `solve2.F:1665-1676` for u/v/w each stage; rho/pressure/theta/moisture at `1681-1706` only when `nrk<nrkmax`. | Velocity `R`; scalar/pressure stages 1 and 2 only | Exact family/cadence; edge/corner/rigid-face decks missing, open (boundaries and closures). |
| `MASS-REDUCE-FINISH` | `src/solve2.F:1973-2030` | Waits for reductions, forms `mass2`, uses the 64-bit product of `nx`, `ny` and `nz`, corrects `pp3d/prs/rho`. | `F`, after scalar/tracer work | Exact equations; deterministic tree/permutation deck missing, open (boundaries and closures). |
| `FINISH-STEP` | `src/solve3.F:77-1143` | Called at `src/cm1.F:3432-3475` after `solve2` and optional native MP; boundary/communication sites include `411-432`, `495-505`, and `606-704`; moving-surface work `784-1130`. | `L`, post-RK/physics | Exact call order; post-step trajectory/restart deck missing, open (dry dynamics, boundaries and closures). |
| `SGS-NEXT-STEP` | `src/turb.F:12-2971`; `calcnm/calcdef/tkekm/tkebc` dispatch `1326-1389`; `calcdef` `4711-5423`; `tkekm` `2977-3211`; `tkebc` `3219-3367` | `sfc_and_turb` called at `src/cm1.F:3737` after the solved-step dynamics/physics path and during the startup pass before the first solved step; selected internal order is `calcnm -> calcdef -> tkekm -> tkebc`. | `L`, prepares closure state for the next step, under enabled branches | `O-TERRAIN-DEFORMATION-SGS-V1` proves selected terrain/all-open deformation through local coefficients, dissipation, and TKE source; CPU/native and CUDA are raw-bit exact on the analytic operator deck. The corrected-real `300 x 300 x 75` first-step gate supplies live `dissten` to `idiss`. Evolved closure state, native halo/MPI regions, later steps, and coupled trajectory remain open (boundaries and closures, physics seam). |
| `RESTART-WRITE` | `src/restart_write.F:10-2148` | `src/cm1.F:4246-4256`, only when the end-of-step restart clock sets `dorestart`; clock is advanced first. | `RST-W` | Exact call/clock; uninterrupted versus resume identity deck missing, open (restart and output). |

## Ratified native slices and claim boundary

`development-repository:evidence/oracles/grid-base-v1/` and
`development-repository:evidence/oracles/post-getset-v1/` identify the two
ratified numerical slices in this map. Each was produced from fresh authority
copies with GNU Fortran O0 and O2 builds and repeated under O2. Their derived
ASCII contracts are identical across the applicable runs, and each comparator
rejects one-cell input-bit, location, cadence, and output corruptions. The
comparison contract is explicitly raw-bit with `max_ulp=0`; no numerical or
null tolerance is used.

The post-base fixture ratifies captured bounds/configuration, complete recorded
1-D coordinate/metric extents, and five sampled surface points and 3-D columns
over their full vertical extents. The post-`getset` fixture additionally
ratifies every stored word of 46 named allocations at `cm1.F:2905`:
base/derived fields, wind/pressure/theta state pairs, ten ordinal moisture
species in both `a/3d` arrays, and TKE `a/3d` arrays. It includes the `u0`/u
`ie+1`, `v0`/v `je+1`, and w/TKE `ke+1` storage. The `a/3d` pairs are equal by
explicit startup copy, not independent physical coverage; `ppx` is saved
acoustic pressure-tendency state. `qv0` is captured base persistence but is not
a `getset` argument. `prs` lacks a `getset` halo fill, and `rf0`, `rr0`,
`rrf0`, `qc0`, `qi0`, surface base fields, and later startup turbulence state
are omitted. The fixture therefore does not provide complete first-solve
state, complete 3-D grid/surface/metric arrays, isolated attribution between
`init3d` and `getset`, semantic ArWen species mapping, any RK/advection/
pressure or evolving boundary operator, a complete step, restart identity,
CUDA/port comparison, or product parity. Every missing row or deck remains a
runtime refusal.

Native CM1 physics sources are scientific context and oracle producers only.
The product lane keeps source-mapped CM1 dynamics/closures and calls an
explicitly versioned ArWen component through a CM1-owned adapter. No `ptype`
integer is silently translated into an ArWen choice.
