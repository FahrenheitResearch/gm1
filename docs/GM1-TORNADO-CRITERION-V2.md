# GM1 tornado criterion v2 (pre-registered)

| | |
|---|---|
| Status | **PRE-REGISTERED, FROZEN, amended once before scoring.** No GM1 run had been scored with v2 at the first freeze or at the amendment. |
| First pre-registration commit | `fc063af5149e996b161583191772638236b9964a` ("Pre-register tornado criterion v2"), `2026-09-22T18:01:07Z` |
| Amended and re-frozen (the version in force) | commit `198de287e3e304e32172003ad699c1d27e4211b3` ("Amend tornado criterion v2 before scoring: ..."), `2026-09-22T19:27:37Z` (commit time, UTC). See "Pre-registration record". |
| Constants SHA-256 | `807b2cc83dcef09bf8067ab6854a0851a910217e9c7fae972c709af83282532e` (`v2rule.CRITERION_V2_SHA256`; the first freeze's `c3a1a881…adde0` is superseded and is not a v2 verdict hash) |
| Code SHA-256 | `tools/criterion_v2/FROZEN.json`: `v2rule.py` `87f49b8d6008bd9668f6dc392e1c06138ac814ad99e4a426aa69ef60e95d8e27`, `score.py` `d46e453e324c801396ec96717ae4eabaebf6c938a144a28fd45274ec7c7f4fcc`. `score.py` refuses to run when either file differs. |
| Code | `tools/criterion_v2/v2rule.py` (the rule), `tools/criterion_v2/score.py` (the scorer) |
| Tests | `tests/test_criterion_v2_rule.py`; synthetic evidence in `tools/criterion_v2/results/synthetic-v2-prefreeze/` |
| Replaces | v1's core-radius rule (`src/cm1core/nest_tornado_criterion.py`, which is left unchanged). v1 verdicts are kept and reported beside v2 verdicts. |
| Verdicts | `docs/GM1-TORNADO-VERDICTS-V2.md`, which also says what a reader can check and what cannot be checked yet. Commit hashes on this page name commits of GM1's private development repository, where the freeze was made; run labels (E100V, E100W, E100N, E100T) name development runs whose output is not published. |

## Why v2 exists

v1 counts a sample only when its core radius is at least 40 m. The core radius is
`dx*sqrt(n/pi)`, where n is the number of grid corners with vorticity of at least half the peak. The
synthetic diagnosis in commit 99ec53c (`tools/criterion_v2/results/synthetic-v1-diagnosis/`) shows
that this rule depends on the grid:

* **Coarse grids pass anything.** One corner alone gives 56 m at 100 m spacing.
* **Fine grids fail when small features are present.** Once satellite vortices are resolved, they
  set the peak, and the "core" then measures a satellite instead of the tornado. The same flow
  passes at 20 m and fails at 8, 4 and 2 m.
* **Two-celled vortices are penalised.** A thin two-celled ring needs about twice the radius of
  maximum wind (RMW) that a Rankine vortex needs to pass.
* **The centre jumps.** v1 takes the first strict maximum as the centre, which lies on the ring of
  a two-celled vortex, so the centre hops 100 m or more between samples.

v1 also samples at 10 m AGL, and that part is kept. Using the lowest model level instead would
change the wind by a factor of 0.46 to 1.54, depending on the grid.

## The rule

Every number below is a key of `CRITERION_V2`, and the table after the rule gives the source of
each. The rule is cyclonic only: it looks for positive vorticity and positive tangential wind.

### Where the rule samples

1. **Height.** The rule uses frame-relative u and v at **10 m AGL**. They are interpolated linearly
   in height between the two scalar levels that bracket 10 m, using the run's own level heights
   (explicit child levels, a stretched grid or `zhval`). The lowest level is never used in place of
   10 m. If the lowest scalar level is above 10 m, the run is UNRESOLVED.
2. **Grid gate.** If the horizontal spacing is more than **25 m**, the run is UNRESOLVED ("grid too
   coarse to judge"). Samples are still computed and written.
3. **Nest spin-up.** A nest child's first **300 s** after it is spawned carry only what the parent
   resolved (its initial state and inflow are interpolated from the parent). Samples in that time
   are judged at the **parent's** spacing wherever this rule says dx (S condition, marginal band).
   The scorer needs the spawn time and the parent spacing for every child or window source.

### Finding the vortices in each sample

4. **First guesses.** The rule takes CM1 corner vorticity (`parcel.F:1066-1067` form) of the 10 m
   winds and averages it over a square of fixed physical half-width **100 m**. It takes up to **64**
   of the strongest positive local maxima that are at least **200 m** apart and lie in the *centre
   region*: at least **300 m** inside the model's free interior, which excludes the specified and
   relaxation zones of a nest child. It ranks them by the peak azimuthal-mean tangential wind about
   each (the objective of step 5), and refines the best **16**.
5. **Centre.** Each first guess is moved to the point that **maximises the peak of the
   azimuthal-mean tangential wind**. This is a deterministic 8-neighbour pattern search with steps
   of 64, 32, 16, 8, 4, 2 and 1 m, up to 64 moves per step. The search profile has a radial step of
   min(4 m, dx) and 96 azimuths. For an axisymmetric vortex this maximum is the unique axis, so the
   centre does not jump on a two-celled ring or a flat top, and small satellites do not move it.
   Refined centres within max(2 dx, 2 m) of one already kept are the same vortex. **Every** kept
   candidate is classified and tracked (step 11); the strongest is the sample's row in the TSV.
6. **Profile.** About the centre, the azimuthal-mean tangential wind is Vbar(r). Each wind component
   is interpolated bilinearly from its own staggered lattice onto **360** azimuths. Radii run in
   steps of min(2 m, dx/2), from **2 dx** out to R_avail = min(**1000 m**, distance to the edge of
   valid data). From the profile:
   * Vmax is the largest Vbar; RMW is the radius where Vbar equals Vmax.
   * The profile is **closed** when Vbar falls to **0.9 Vmax** or less somewhere in
     (RMW, min(**3 RMW**, R_avail)].
   * **Shape**: on the RMW circle, the tangential wind Vt(θ) is split into azimuthal wavenumbers.
     Wavenumber 1 is a uniform flow across the vortex (its motion relative to the frame) and is only
     reported. The wavenumber-2 amplitude |V2| (elongation) is compared with Vmax.
   * **Mean core vorticity** 2 Vmax/RMW (Stokes: the mean vorticity inside the RMW).
7. **Hazard wind.** This is the maximum ground-relative 10 m wind speed within **1.5 RMW** of the
   centre. It is taken at scalar points, with face winds averaged to the point as in CM1 `sws`
   (`solve3.F:864-865`), plus umove and vmove.
8. **Conditions.** They are listed in the TSV `conditions` column in the order E R C A Z W S. An
   upper-case letter holds, `-` failed, and a lower-case letter (R, Z, W only) is **marginal**:
   failed by less than the discretisation uncertainty δ = **0.5 dx/RMW** of the measured value
   (value ≥ threshold × (1 − δ)).
   * **E**: the refined centre lies in the centre region.
   * **R**: Vmax ≥ **20 m/s**.
   * **C**: the profile is closed.
   * **A**: |V2| ≤ **0.5 Vmax** (the flow goes round the centre, not along a line).
   * **Z**: 2 Vmax/RMW ≥ **0.1 s⁻¹**.
   * **W**: the hazard wind is at least **35 m/s**.
   * **S**: RMW ≥ **4 dx**.
9. **Candidate class.**
   * **Q** (qualifies): all seven hold.
   * **U** (cannot be judged on this grid): E, C and A hold; R, Z and W hold or are marginal; and S
     fails or one of R, Z, W is marginal.
   * **N**: anything else.

### Deciding the run from the time series

10. **Links.** A candidate in one sample links to a candidate in an earlier sample when the samples
    are at most **60 s** apart and adjacent, or at most **3 s** apart with samples between them (a
    brief dip in a window stream is bridged). A link is
    * **plausible** when the ground centre moved no more than **60 m/s × dt + 100 m**;
    * an **identity** link when it is plausible and, in the storm-following frame (the ground move
      minus the domain's own motion (umove, vmove) × dt), the centre moved no more than the **sum of
      the two RMWs**: the two cores overlap, so they are one vortex, not two.

    The ground position is plane position + child origin + domainloc; for window streams it is
    origin + umove × t.
11. **Tracking.** Every candidate is tracked: a chain is a sequence of candidates joined by links,
    and for each candidate the longest-lasting chain ending there is found (dynamic programming over
    all candidates). A second vortex never breaks the first one's chain, and two vortices are never
    stitched into one by identity links.
12. **Duration.** An **episode** is an identity-linked chain of Q candidates lasting at least
    **120 s** (first to last model time). The longest identity-linked Q chain is reported against
    v1's 60 s beside it, which decides nothing.
13. **Run outcome**, checked in this order:
    1. **UNRESOLVED** if the grid gate (step 2) or the height gate (step 1) fails.
    2. **INSUFFICIENT_CADENCE** if no stretch of the record with every interval ≤ 60 s lasts 120 s.
    3. **TORNADO** if at least one episode exists. Genesis is the first sample of the first episode.
    4. **UNRESOLVED** if an identity-linked chain of Q or U candidates lasts at least 120 s.
    5. **INSUFFICIENT_CADENCE** if a chain of Q (or Q or U) candidates lasts 120 s only through
       plausible links: qualifying vortices were present, but the output interval cannot show that
       one vortex persisted.
    6. **NO_TORNADO** otherwise.

### Duration and output cadence

| Output | What 120 s means | How the verdict must be stated |
|---|---|---|
| 0.5 s window stream | ≥ 241 records of one tracked vortex, dips of ≤ 3 s bridged | Each record is a sample. |
| 30 s frames | ≥ 5 consecutive frames of one tracked vortex | "30 s output" |
| 60 s frames | ≥ 3 consecutive frames (the Flournoy et al. 2020 practice: 2 min at 1-min output) of one tracked vortex | "60 s output". A vortex that moves more than its core diameter between frames in the storm-following frame gives INSUFFICIENT_CADENCE. |
| > 60 s frames (E100W's child frames are **75 s**), one frame, or a record < 120 s | cannot be judged | INSUFFICIENT_CADENCE |

The TORNADO caption always names the cadence, and `verdict.json` carries a
`cadence_statement`.

## Thresholds and their sources

| Key | Value | Source / reasoning |
|---|---|---|
| `height_agl_m` | 10 m | The EF scale's reference height. Also the height used by Coffer et al. 2017 and 2023, Flournoy et al. 2020, Snook et al. 2019 and Markowski 2020/2024 (the Markowski ζ is at the lowest level). v1 already used 10 m. The v1 diagnosis shows the lowest level is 0.46–1.54 × the 10 m wind, depending on the grid (neutral log profile). |
| `max_lowest_scalar_level_m` | 10 m | 10 m must be interpolated, not extrapolated. |
| `max_grid_spacing_m` | 25 m | Markowski 2024 (JAS 81) footnote 3: "TLV [tornado-like vortex] is probably more appropriate than tornado when horizontal … grid spacings larger than ∼25 m … are used". With the 4 dx rule this also means 4 dx ≤ 100 m, below the median observed tornado radius of 125–250 m (Wurman et al. 2021 PNAS: median diameter 250–500 m). Markowski's vertical figure (~10 m) is *not* a gate here. |
| `resolved_rmw_over_dx` | 4 | Synthetic test (`snapshot.tsv`, 16 sub-grid phases, every grid). At RMW = 4 dx the measured Vmax is within 1.7 % of the analytic value for Burgers–Rott, 3.9 % for Sullivan, 9.4 % for Rankine and 12 % for a thin ring. At 3 dx the errors are 2.9 / 6.6 / 13.1 / 16.3 %, and at 2 dx 6.2 / 12.6 / 19.2 / 25.4 %. Physical cross-checks: a core diameter of 8 dx is just above the effective resolution of about 7 dx (Skamarock 2004, MWR; recalled, not re-read), and 4 dx is twice the "≈2Δx" contraction radius that Markowski 2024 calls the bare minimum. |
| `discretisation_uncertainty_dx_over_rmw` | 0.5 | The worst-case measured-Vmax deficit in `snapshot.tsv` (thin ring: 25.4 % at 2 dx, 16.3 % at 3 dx, 12 % at 4 dx) is ≈ 0.5 dx/RMW. A threshold failed by less than that cannot be told from a pass on this grid, so the sample is U, not N (a vortex whose true Vmax is 21.5 m/s reads 19.8–20.0 m/s at 20 m). This covers interpolation error on exact fields only; GM1's own numerical smoothing (effective resolution about 6–7 dx) makes a model vortex near 4 dx read lower still, which is one more reason a 20 m verdict near the thresholds is weak. |
| `nest_spinup_s` | 300 s | Procedural, not from a paper. A child's initial state and inflow are interpolated from the parent, so its first samples cannot carry structure finer than the parent's (an interpolated 100 m vortex passed at the child's 8 m in the review). 300 s is 2.5 × the 120 s duration, so no episode can rest on spin-up samples alone. Chosen knowing only the spawn times (E100V, E100W at T+70; E100N at T+60), not any verdict. |
| `profile_inner_radius_over_dx` | 2 | The profile inside 2 dx is sub-grid (Markowski 2024 uses 2Δx as the smallest radius a vortex can contract to). This also keeps a single-cell spike from producing a large tangential wind at r ≈ 0. |
| `first_guess_box_half_width_m` | 100 m | A fixed physical box (Mashiko & Niino 2017 used 50 × 50 m for sub-vortices, and Rotunno & Markowski 2025 code option 5 uses a box-mean maximum of ζ). It only seeds step 5. |
| `first_guess_pool`, `first_guess_candidates`, `first_guess_separation_m` | 64, 16, 200 m | The pool is ranked by the step-5 objective, so a straight shear line (whose box-mean ζ can exceed a small tornado's and whose plateau gives a row of maxima every 200 m) cannot crowd a vortex out; the line's Vbar is ΔU/π, a tornado's is its Vmax. 16 refined candidates leave room for several vortices. These affect only which vortices are examined, not whether one qualifies. |
| `centre_search_*` | steps 64 → 1 m; search profile min(4 m, dx) and 96 azimuths | Fixed physical steps, so the centre is found to 1 m on every grid. A centre defined by maximum tangential wind or circulation is the tropical-cyclone convention, and it is Rotunno & Markowski's option 5 by Stokes. Synthetic centre error is ≤ 2.8 m on every grid (`snapshot.tsv`). |
| `edge_margin_m` | 300 m | Replaces v1's 1000 m. On a 2 km child (E100V's; the choice uses that child's geometry, not any verdict), 1000 m left a single corner, which made the live criterion degenerate. 300 m is v1's own analysis disk, so that disk lies in the free interior, and it is 3 parent cells of a 100 m parent. |
| `profile_dr_*`, `profile_azimuths` | min(2 m, dx/2), 360 | Resolution of the diagnostic only. |
| `profile_max_radius_m` | 1000 m | The mobile-radar standard: velocity difference ≥ 40 m/s across a couplet < 2 km in diameter (Alexander & Wurman 2008; Marquis et al. 2012; Kosiba et al. 2013, as quoted by Houser et al. 2022, MWR 150). |
| `mean_tangential_wind_min_m_s` | 20 m/s | The same radar standard: ΔV ≥ 40 m/s is a rotational velocity of 20 m/s. For an axisymmetric vortex the couplet half-difference equals the peak tangential wind. Vbar is the total rotation about the axis, as radar ΔV is: it includes vorticity the vortex has gathered from a shear line it sits on. Per Nolan et al. 2017 (JAS 74) the mean maximum tangential wind and its radius change only slightly from 20 m to 5 m grids. |
| `closure_fraction`, `closure_radius_over_rmw` | 0.9, 3 | A vortex has a closed wind maximum; solid-body rotation and a broad vortex whose profile still rises at the data edge do not. A straight shear line (Vbar = ΔU/π at every radius, ratio 1.00) is rejected here and, since the amendment, by A as well. For a modified-Rankine outer decay r^−α, Vbar(3 RMW)/Vmax ≤ 0.9 needs α ≥ 0.1; observed tornado outer exponents are about 0.5–0.7 (Wurman et al. 2021 use 0.6), and isolated analytic profiles give 0.33 (Rankine) to 0.52 (α = 0.6) at 3 RMW. The first freeze's 0.85 within 2 RMW rejected tornadoes inside a low-level mesocyclone, whose rotation keeps Vbar high beyond the RMW: Burgers V35 R100 inside a Burgers V30 R600 mesocyclone reads 0.88 at 3 RMW (0.89 at 2 RMW). With sheets now rejected by A, closure only has to certify a maximum inside the data, so the fraction was relaxed to 0.9 and the radius extended to 3 RMW. |
| `wavenumber2_max_fraction` | 0.5 | Analytic: a straight vortex sheet crossing the RMW circle has Vt = (ΔU/2)|sin θ|, so |V2|/Vbar = 2/3 whatever its strength. A finite sheet segment has net circulation, so its far field closes like a vortex's; C cannot reject it, A does. A field that is part sheet (share f of Vbar) and part axisymmetric vortex has |V2|/Vbar ≈ 2f/3, so 0.5 requires the axisymmetric part to supply at least a quarter of Vbar. Axisymmetric vortices and ones with three or more symmetric satellites have |V2| ≈ 0 (synthetic < 0.1). |
| `mean_core_vorticity_min_s_inv` | 0.1 s⁻¹ | Scale separation: mesocyclone vorticity is O(0.01 s⁻¹) and tornado-core vorticity O(0.1–1 s⁻¹); 0.1 s⁻¹ is the order-of-magnitude boundary. It rejects a near-ground mesocyclone that meets the radar ΔV (20 m/s at RMW 600–750 m is 0.056–0.07 s⁻¹, and with 15–20 m/s of storm motion it meets W too). Equivalent: at Vmax = 20 m/s the RMW must be ≤ 400 m. |
| `hazard_radius_over_rmw` | 1.5 | A translating vortex's strongest ground-relative wind is at its RMW; at 1.5 RMW a Rankine vortex's own wind is 0.67 Vmax. 1.5 RMW keeps that and excludes gusts beyond it (the first freeze's 2 RMW reached 2 km, so an RFD surge 900 m from a broad vortex's centre supplied its hazard wind). The maximum within 300 m is also reported for comparison with v1. |
| `ground_relative_wind_min_m_s` | 35 m/s | Coffer et al. 2017 ("midrange EF0"), Flournoy et al. 2020 ("low-end EF0"), and v1. The EF0 range is 29–38 m/s. The 29 m/s result (Roberts et al. 2016; Markowski 2020/2024; Snook et al. 2019; Coffer et al. 2023) can be read from `ground_wind_max_m_s`. |
| `centre_speed_max_m_s`, `centre_jump_allowance_m` | 60 m/s, 100 m | A plausible link: 60 m/s is v1's value, about twice the fastest observed tornado translation; 100 m replaces v1's 40 m (2 dx at 20 m). At 30 s or 60 s output this bound (1.9 km, 3.7 km) exceeds a 2 km child, so it cannot tell one vortex from another; only identity links make episodes. |
| `identity_move_max_over_rmw_sum` | 1 | Two distinct vortex cores cannot lie closer than the sum of their RMWs without sharing one wind maximum, i.e. being one vortex. So a move of at most RMW₁ + RMW₂ between samples, in the frame moving with the domain (whose umove, vmove are set to the storm motion), is the same vortex; a larger move may be a different one. |
| `bridge_gap_s` | 3 s | The EF scale's wind is a 3-s gust: a dip shorter than the damage scale's own averaging time does not end a tornado. It applies only when samples are finer than 3 s (window streams). |
| `sustain_s` | 120 s | The most common literature duration: Coffer et al. 2017 and 2023, Flournoy et al. 2020 (2 consecutive minutes), Schenkman et al. 2012 (≥ 2 min). v1's 60 s is reported beside it. |
| `max_sample_interval_s` | 60 s | Flournoy et al. 2020 judged 2 min from 1-min output. Coarser output cannot show 120 s with at least 3 samples. |

The following are **reported but do not gate the verdict**: circulation 2π·RMW·Vmax; the ratio of
mean vorticity inside RMW/4 to that inside the RMW (< 1 suggests two cells); the wavenumber-1
amplitude; the point maximum of corner ζ within the RMW, which is v1's kind of number; minimum
10 m p′ within the RMW (frames only, from `prs − prs0`); the frame-relative wind maximum; and the
maximum within 300 m.

## Outcomes and caption wording

| Outcome | Caption (use exactly) |
|---|---|
| TORNADO | "simulated tornado (GM1 criterion v2, *{dx}* grid, *{cadence}* output)", e.g. "simulated tornado (GM1 criterion v2, 4 m grid, 60 s output)" |
| NO_TORNADO | "vortex did not meet GM1's tornado criterion (v2)" |
| UNRESOLVED | "grid too coarse to judge (GM1 criterion v2)" |
| INSUFFICIENT_CADENCE | "output too sparse or too short to judge duration (GM1 criterion v2)" |

`verdict.json` carries `outcome`, `caption`, `reason` (for UNRESOLVED and INSUFFICIENT_CADENCE),
`episodes`, `genesis_model_time_s`, the longest identity-linked Q and Q∪U chains and the longest Q
chain through plausible links, `reported_60s_rule`, the class, condition, marginal and flag counts,
the constants and their SHA-256, the SHA-256 of `v2rule.py` and `score.py` and whether they match
`FROZEN.json`, `protocol_conforming`, the nest spin-up used, the scorer's git commit, the source,
and the cadence statement. `samples.tsv` has one row per sample (its strongest candidate) and
`candidates.tsv` every refined candidate. For a nest child it adds `v1_beside`: the replay of the
live v1 sidecar when there is one, otherwise v1's own host sample (`reference_sample`) recomputed on
every child frame and decided by v1's ledger at frame cadence, labelled as such (E100N has no
sidecar).

## What v2 does not claim

* **A TORNADO is a statement about the model, not about the atmosphere.** It means a simulated
  vortex met this rule.
* **NO_TORNADO means no *resolved* vortex qualified.** A vortex with RMW < 4 dx is at best U, and
  when badly under-resolved it can read below the marginal band and count as N. On a 20 m grid, v2
  cannot see a tornado with RMW < 80 m (E100W, E100N).
* **Near the thresholds the grid matters.** Measured Vmax and point winds read low by up to
  0.5 dx/RMW (12 % at 4 dx, 7 % at 4.5 dx for Rankine) on exact fields, more on model fields. The
  marginal band turns such a sample into U; it cannot make a 20 m verdict as strong as a 4 m one.
* **Persistence at frame cadence is only as good as the identity links.** At 30 s or 60 s output a
  TORNADO needs the vortex to stay within one core diameter in the storm-following frame from frame
  to frame. A vortex that drifts faster relative to the domain gives INSUFFICIENT_CADENCE even if it
  is one long-lived tornado; the window stream then decides. Conversely a *different* vortex that
  happens to sit where the last one was (e.g. sub-vortices of a multiple-vortex parent whose orbit
  aliases with the output interval) is taken for the same vortex. Between frames nothing is
  observed.
* **Total rotation, not the vortex's "own" rotation.** Vbar includes the vorticity a vortex has
  gathered from a shear line it sits on, as radar ΔV does. A compact misovortex on a strong gust
  front with Vbar ≥ 20 m/s, mean core vorticity ~1 s⁻¹ and ≥ 35 m/s ground-relative wind for 120 s
  is a TORNADO under v2 (one level cannot tell a gustnado from a tornado; that needs the vortex's
  depth and its connection to the updraft).
* **Broad weak tornadoes near the Z floor.** A vortex with Vmax 20–22 m/s and RMW up to 400 m meets
  Z; with storm motion it meets W. The radar standard would call it a tornado; some would call it a
  small mesocyclone.
* **Markowski 2024's vertical figure is not gated.** His footnote says "tornado-like vortex" is more
  appropriate when Δz > ~10 m. The 20 m decks have Δz = 20 m near the ground (lowest level 10 m).
  v2 samples at 10 m on them without extrapolation, but a TORNADO on such a grid is a
  "tornado-like vortex" in Markowski's terms.
* **Vertical interpolation is linear in height.** Between levels 4/12 m or 5/15 m it reads a log
  wind profile 2–3 % low at 10 m. E100V has a level at 10 m and E100W/E100N have their lowest level
  at 10.0 m, so it does not affect the current runs.
* **Point hazard winds depend on the grid.** They are instantaneous point values, not 3-s gusts.
  Nolan et al. 2017 found peak winds rise from 20 m to 2.5 m grids.
* **Vortices bigger than the data are missed.** A vortex whose RMW exceeds about R_avail/3 may not
  close inside the data and counts as N, flagged `extent-limited`. E100V's window (0.9 km) limits
  R_avail to about 450 m, so a vortex there with RMW beyond about 150–300 m may be extent-limited.
* **Only cyclonic vortices on flat ground are covered.** Anticyclonic tornadoes are not detected,
  and the scorer refuses non-zero terrain.
* **Window streams carry the acoustic-average advecting winds** (`uadv`, `vadv`), not the completed
  `u3d`/`v3d`. Their verdicts are about those winds.
* **A grid-noise vortex** (RMW < 4 dx with strong winds) gives UNRESOLVED, captioned "grid too
  coarse to judge", not NO_TORNADO.
* **Sources that were not re-read.** The Skamarock 2004 figure is recalled. Only Houser et al.
  2022's quotation of the radar papers was read. The Rotunno & Markowski 2025 paper text was not
  read; only its released code was. The vorticity scales behind the Z floor are textbook orders of
  magnitude, not one paper's threshold. These sources support the choices but no threshold rests on
  them alone.

## Pre-registration protocol

1. The constants, the rule code, the scorer and the synthetic evidence are committed before any GM1
   run is scored with v2. The commit hash, UTC time, constants SHA-256 and the two file SHA-256
   (`FROZEN.json`) are in the header above.
2. The scoring stage runs `score.py` on **every** run and source listed below, reads every sample
   (`--every 1`, no `--end`), and reports each v2 verdict with the v1 verdict beside it, whatever
   they are, stating the source (frames or window) and cadence. A verdict with
   `protocol_conforming: false` or `matches_frozen: false` is exploratory, not a v2 verdict.
3. Nobody tunes a threshold after seeing a run verdict. If a frozen rule is later found to be
   broken, the fix is a documented v3 with its own hashes, and both verdicts are kept.
4. `verdict.json` must carry `criterion_v2_sha256 = 807b2cc83dcef09bf8067ab6854a0851a910217e9c7fae972c709af83282532e` and file hashes equal to
   `FROZEN.json`.

Pre-registered sources and arguments (paths on the machine that holds the data; spawn times from the
decks; parent spacing 100 m for every child):

| Run | Source | Arguments |
|---|---|---|
| E100V | child frames (60 s) | `E100V/frames/nest-child --nest-start-s 4200 --parent-dx 100` |
| E100V | window (0.5 s, T+76–82) | `E100V/frames/nest-child-window --nest-start-s 4200 --parent-dx 100` |
| E100V | parent (100 m, standing in for E100P after T+70) | `E100V/frames --deck E100V.toml --start 4200` (UNRESOLVED by the grid gate; its samples are reported) |
| E100W | child frames (**75 s**) | `E100W/frames/nest-child --nest-start-s 4200 --parent-dx 100`: INSUFFICIENT_CADENCE by construction; its window (0.5 s, T+72–82) is scored when it is available |
| E100N | child frames (30 s) | `E100N/frames/nest-child --nest-start-s 3600 --parent-dx 100` |
| E100T | child frame (one) | `E100T/frames/nest-child --nest-start-s <spawn time from its deck> --parent-dx 100`: INSUFFICIENT_CADENCE by construction |
| 2 m run | child frames and window when produced | `--nest-start-s` and `--parent-dx` from its deck |

## Synthetic evidence at the freeze

`tools/criterion_v2/synthetic_v2.py` generated `results/synthetic-v2-prefreeze/` (105 case and
grid combinations, 0 mismatches against the expected outcomes written before the run):

* **Same verdict on every grid (20, 8, 4 and 2 m): TORNADO.** Rankine R100 V50 (flat top);
  Burgers–Rott R100 V60; Sullivan R100 V60 (two-celled); thin ring R120 V50; Burgers R100 V100 and
  Sullivan R100 V100 with three orbiting RMW-8 m satellites (v1 passed these at 20 m and failed
  finer); Burgers R150 V50 inside a Burgers R700 V30 mesocyclone; Burgers R100 V35 inside an R600
  V30 mesocyclone.
* **Near the thresholds.** Rankine R90 V21.5 and ring R100 V21.5: TORNADO at 8, 4 and 2 m; at 20 m
  only UNRESOLVED or TORNADO are accepted (they read 19.8–20.1 m/s there: marginal).
* **Resolvedness guard.** Burgers R40 V60 (10 s output) is UNRESOLVED at 20 m and TORNADO at 8, 4
  and 2 m. Burgers R100 V60 on a 100 m grid is UNRESOLVED. A Burgers R30 V35 tornado 600 m from a
  ΔU = 55 m/s shear line is UNRESOLVED at 20 and 8 m and TORNADO at 4 and 2 m (the line's plateau
  of vorticity maxima does not crowd it out).
* **Negative controls are NO_TORNADO on every grid:** single-cell grid spike; infinite shear line
  ΔU = 80 m/s; a 1 km sheet segment ΔU = 80 m/s (|V2|/Vbar = 0.67); weak Burgers R150 V15;
  mesocyclones Burgers R1500 V25, Rankine R750 V21 and Burgers R600 V21 (the last two with 20 m/s
  storm motion meet R, C and W and fail only Z); a Rankine R500 V20.5 and an R200 V22 in still air
  whose only ≥ 35 m/s wind is a gust 900 m and 400 m away; a strong vortex lasting 30 s; a strong
  vortex 150 m from the free-interior edge.
* **Persistence not observable: INSUFFICIENT_CADENCE on every grid:** a different 30 s-lived
  vortex in each 60 s frame, the same at 30 s frames, and Burgers R40 V60 at 30 s output (it drifts
  115 m per frame in the storm-following frame, more than its 80 m core diameter).
* **Scorer on synthetic output** (tests): child frames with explicit 4 m-then-stretched levels and a
  moving placement give TORNADO, the ground centre continuous across placement moves within 2 m;
  10 m is read, not the lowest level; a window stream gives TORNADO; 100 m parent frames give
  UNRESOLVED; one frame, 90 s or 75 s frames, or a record with a gap give INSUFFICIENT_CADENCE; a
  child scored inside its spin-up is judged at the parent's 100 m and gives UNRESOLVED; v1 is
  recomputed from frames when there is no sidecar; the scorer refuses a nest without its spawn
  time, and code that differs from `FROZEN.json`.
* **Reviewer reproducers kept as tests**: a record with a gap; a second, flickering, stronger vortex
  beside a 240 s tornado in a 0.5 s stream (TORNADO, 240 s); three vortices 2 km apart in three
  frames and A→B→A (not stitched); identity judged in the storm-following frame; 0.5 s dips bridged
  and a 5 s gap not; the hazard wind's umove sign; the sheet, mesocyclone and satellite shape and
  intensity values.

## Running it (scoring stage, not part of the pre-registration)

```
nice -n 19 ionice -c3 python3 tools/criterion_v2/score.py <run>/frames/nest-child --out <dir> --label <run> \
    --nest-start-s <spawn s> --parent-dx 100
nice -n 19 ionice -c3 python3 tools/criterion_v2/score.py <run>/frames/nest-child-window --out <dir> \
    --nest-start-s <spawn s> --parent-dx 100
nice -n 19 ionice -c3 python3 tools/criterion_v2/score.py <run>/frames --deck <deck.toml> --out <dir> --start 4200
```

`--workers` accepts 1–6; use no more than the machine can spare. The scorer is numpy-only,
reads two model levels per frame through memmaps, and needs no GPU. `--every`, `--end`, or `--start`
on a child or window make the verdict `protocol_conforming: false`.

## Pre-registration record

**First freeze** (`fc063af`, 2026-09-22T18:01:07Z, constants `c3a1a881…adde0`). Three adversarial
reviews of the frozen rule, run on synthetic fields only (no GM1 run scored; of the runs, only
placement, receipt and deck metadata were read), found false TORNADO verdicts, false NO_TORNADO
verdicts, a cadence bug, tracking bugs and integrity gaps. Because no run had been scored, the rule
was **amended before scoring** rather than superseded by a v3. Each amendment is listed with the
finding that forced it; every amended threshold is argued above from physics, the literature or the
analytic/synthetic fields, not from any GM1 verdict.

| # | Finding (reproducer) | Amendment |
|---|---|---|
| 1 | A 1 km straight vortex-sheet segment (ΔU 64–80 m/s) passed as TORNADO: a finite sheet has net circulation, so its far field "closes" like a vortex's. | New condition **A**: wavenumber-2 amplitude of Vt on the RMW circle ≤ 0.5 Vmax (a sheet gives 2/3). |
| 2 | Broad low-level mesocyclones (Rankine R750 V21, Burgers R600 V21, R400–600 with 20 m/s motion) passed: no vorticity or size floor, and storm motion supplies most of the 35 m/s. | New condition **Z**: mean core vorticity 2 Vmax/RMW ≥ 0.1 s⁻¹. |
| 3 | The hazard wind came from an RFD-like gust 900 m from a broad vortex's centre (2 RMW reached 2 km). | Hazard disk 2 RMW → **1.5 RMW**. |
| 4 | A tornado inside a mesocyclone failed closure (ratios 0.852–0.89 within 2 RMW): the 0.85 fraction had been set on isolated vortices only. | Closure radius 2 RMW → **3 RMW** and fraction 0.85 → **0.9** (a synthetic Burgers V35 R100 in a V30 R600 mesocyclone still read 0.88 at 3 RMW; sheets are now rejected by A). |
| 5 | Only the per-sample strongest vortex was followed: a second qualifying vortex made the centre hop and broke a real tornado's run (NO_TORNADO with every record Q at 0.5 s), and A→B→A was chained into one episode. | **Every candidate is tracked** (longest chain by dynamic programming); duplicate centres merged. |
| 6 | At 30 s / 60 s output the continuity bound (1.9 / 3.7 km) exceeded a 2 km child: a different short-lived vortex in each frame gave TORNADO. | **Identity links**: an episode needs consecutive centres within RMW₁ + RMW₂ of each other in the storm-following frame; qualifying chains joined only by plausible links give INSUFFICIENT_CADENCE. |
| 7 | Five first guesses were filled by a strong shear line's plateau of box-vorticity maxima; a small tornado 600 m away was missed (verdict flipped with float ties at 2 m). | A pool of up to **64** separated maxima, ranked by the peak azimuthal-mean tangential wind about each; the best **16** refined. |
| 8 | Near-threshold resolved vortices (true 21.5 m/s at RMW 4.5 dx) read 19.8–20 m/s at 20 m and gave NO_TORNADO while finer grids gave TORNADO. | **Marginal band**: R, Z, W failed by less than 0.5 dx/RMW of the value make the sample U, not N. |
| 9 | One 0.5 s record below threshold every 100 s broke a window-stream run that 60 s frames called TORNADO. | Gaps of ≤ **3 s** (the EF-scale gust duration) are bridged. |
| 10 | A 100 m-grid vortex interpolated onto an 8 m child passed at the child's dx; a child's first frames carry only parent structure. | **Nest spin-up**: a child's first 300 s are judged at the parent spacing; the scorer requires the spawn time and parent spacing. |
| 11 | Code bug: samples at 0, 60, 150, 240, 330, 420 s (all Q) gave NO_TORNADO; the cadence test needed only one ≤ 60 s interval anywhere. | INSUFFICIENT_CADENCE unless a stretch with every interval ≤ 60 s lasts 120 s. Reproducer is a test. |
| 12 | The hash covered the constants only; `score.py` was not hashed; scoring choices (`--every`, `--start`, `--end`) were free; W and the sign of umove had no test (a sign flip passed every test). | `FROZEN.json` holds the SHA-256 of `v2rule.py` and `score.py` and the scorer refuses on mismatch; `protocol_conforming` and the pre-registered argument table; tests for W, the umove sign, W-only failures. |
| 13 | E100N has no live v1 sidecar, so no v1 verdict would stand beside v2. | The scorer recomputes v1 from child frames with v1's own host sample and ledger (labelled as frame cadence). |
| 14 | E100W's child frames are 75 s apart, not 60 s as the first freeze's text said. | Documentation corrected: they give INSUFFICIENT_CADENCE by construction; E100W's verdict must come from its window. The 60 s limit (Flournoy et al. 2020) is not changed for it. |

Findings **not** adopted as rule changes, and why:

* A compact misovortex (own 12 m/s) on a ΔU = 30 m/s gust front with 15 m/s outflow reads Vbar 21,
  mean core vorticity 1.0 s⁻¹, 42 m/s ground-relative, |V2|/Vbar 0.35, and gives TORNADO at 4 and
  8 m. The rule, like radar ΔV and every model criterion cited, measures total rotation; the split
  into "own" and "background" rotation is not observable. Recorded under "What v2 does not claim".
* Sub-vortices of a weak parent whose orbit aliases with the 60 s output (each sub-vortex lands
  where another was) are taken for one vortex. Each sub-vortex qualifies on its own; the aliasing is
  a limit of frame cadence, recorded above.
* Burgers R400 V22 with 20 m/s motion still gives TORNADO (Z = 0.11 s⁻¹). Recorded as the Z floor's
  boundary, not moved.
* Linear-in-height interpolation reads a log profile 2–3 % low mid-layer; no current run is
  affected (levels at exactly 10 m). Recorded.
* The 300 m edge margin used E100V's child geometry (not a verdict). Recorded in the table.

**Amended freeze**: commit `198de287e3e304e32172003ad699c1d27e4211b3` at `2026-09-22T19:27:37Z`, constants `807b2cc83dcef09bf8067ab6854a0851a910217e9c7fae972c709af83282532e`, `v2rule.py`
`87f49b8d6008bd9668f6dc392e1c06138ac814ad99e4a426aa69ef60e95d8e27`, `score.py` `d46e453e324c801396ec96717ae4eabaebf6c938a144a28fd45274ec7c7f4fcc`. Still no GM1 run scored.
