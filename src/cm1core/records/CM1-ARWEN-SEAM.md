# CM1 to ArWen column-microphysics seam

This record covers both column components GM1 can bind: CM1 `ptype=7` through
ArWen `mp_physics=6` (WSM6) and CM1 `ptype=5` through ArWen `mp_physics=10`
(Morrison two-moment). The adapter is one object,
`cm1core.arwen_column_component.ArwenColumnComponent`, parameterized by a
`ColumnComponentSpec`; the two bindings are
`cm1core.arwen_wsm6_component` and `cm1core.arwen_morrison_component`.

## Admitted estate

The dependency window is `gpuwm>=2.8.0,<2.8.1`, with `gpuwm-data>=2.8.0,<2.8.1`
beside it (gpuwm 2.8.0 itself requires `gpuwm-data==2.8.0`). The
byte-admitted estate is the published
`gpuwm-2.8.0-py3-none-manylinux_2_28_x86_64.whl`, SHA-256
`6a8749c8acc8050de25857fb698ed5070fafeac49a002e127dfc664b14ff3577`. The
pure-Python `gpuwm-2.8.0-py3-none-any.whl` carries the same bytes for every
admitted file.

Source identity is recorded as the distribution artifact rather than a git
commit on purpose. 2.5.8 was admitted from a tracked gpuwm checkout, so its
identity was commit `f47f177d4b2e0da95568d70230f5ce9785e632c7`; the 2.7.4 and
2.8.0 wheels record no upstream commit anywhere in their metadata, and writing
one down anyway would be a fabricated provenance claim. What was actually
hashed is the wheel, so the wheel is what the pin names. The superseded 2.7.4
pin was `gpuwm-2.7.4-py3-none-manylinux_2_28_x86_64.whl`, SHA-256
`defe2d64ad97a95f3578c1d5145c4b36ed68f08b00c77218d644cc1a824f3afd`.

Runtime admission checks the 17 exact SHA-256 values in `cm1core.engine_pin`
without importing CuPy or opening a CUDA context. Any missing or changed byte
returns `CM1-ARWEN-ESTATE-DRIFT`.

Morrison reaches two further files of the same pinned 2.8.0 estate, lazily
imported by `gpuwm.core.microphysics.apply` at `mp_physics=10`:

| Path | SHA-256 |
|---|---|
| `gpuwm/core/kernels/morrison.cu` | `a563005eb0992bc925d1c80f58d173dad1bfb385824be1bae5f9f90cc6ecb829` |
| `gpuwm/core/morrison.py` | `e01202401d8a14db6c52206ad56deeafd106a4acf8cc0794e1f68fbe06c49177` |

`gpuwm/core/morrison_constants.py` was already inside the WSM6 surface because
`gpuwm/core/microphysics.py` imports it at module scope. The two files above
are a **separate** mapping, `engine_pin.AUDITED_MORRISON_EXTRA_SURFACE`, not an
addition to the WSM6 one: `engine_pin.audited_surface()` with no component
returns exactly the WSM6 surface, so a WSM6 run's audited surface SHA-256 does
not depend on Morrison. A component with no entry in
`engine_pin.COMPONENT_EXTRA_SURFACE` raises rather than executing unaudited
bytes.

### What the 2.7.4 to 2.8.0 move did to the executed WSM6 and Morrison surfaces

The answer is recorded in full in the estate-delta audit for 2.8.0, which
`tools/audit_arwen_wsm6_estate_delta.py` regenerates on any host with no GPU
from the two published wheels (`--base-root` the extracted 2.7.4 wheel,
`--admitted-root` the extracted 2.8.0 wheel). The audit record itself is
project evidence and is not part of the distribution.

**Nothing that decides a WSM6 answer moved.** The seven files that carry the
WSM6 physics (the table below) are byte-identical between 2.7.4 and 2.8.0,
and so are `gpuwm/core/state.py`, `gpuwm/physics_registry.py` and
`gpuwm/physics_vertical_contract.py`. The assembled NVRTC source is identical
at the default tier and at every capacity tier (the digests in the table
below are the 2.8.0 digests as well). The `wsm6-mp6` registry object is
byte-identical too (`85a0e92dd2f278e9adb27cd55728c3ca6eb26513f45589b88743cf6f54ef9579`
in both).

Seven paths of the 2.8.0 WSM6 surface differ from 2.7.4. None reaches the
WSM6 executed path:

| Changed path | What changed | Why it does not reach WSM6 |
|---|---|---|
| `gpuwm/core/microphysics.py` | classic Thompson (mp=8) gains WRF's entry zeroing at R1, a column-exit mask and L_qc/L_qr presence handling; the NSSL2 (mp=18) arm steps its production binding | `apply` is textually identical, and the `_dispatch_scheme` path for `mp_physics == 6` (every test ahead of that arm, and the arm) is textually identical; the whole dispatch body differs only in the NSSL2 fallthrough |
| `gpuwm/core/kernels/__init__.py` | both module compile routes call `_compile_observed`, which wraps `module.compile()` in `gpuwm.kernel_compile_notice.observe_module_compile` | the assembled source handed to NVRTC is unchanged (measured above); the wrapper only reports whether a compile wrote to the kernel cache |
| `gpuwm/kernel_compile_notice.py` | new in 2.8.0; joins the admitted surface because every module compile now executes it | a filesystem-only progress notice; it imports no CUDA and touches no model array |
| `gpuwm/core/refl.py` | one Thompson error message | `launch_refl10cm_wsm6` and `consume_refl_10cm` are textually identical, and the mp=6 presence check still resolves to `qr`/`qs`/`qg` |
| `gpuwm/config.py` | comments, and three new `RunConfig` fields (`min_time_step_sound`, `relax_timescale_s`, `relax_w`) | all three carry defaults; no field was dropped and no existing default or annotation changed |
| `gpuwm/core/physics.py` | driver-side changes, including a refactor of the WSM6 loop count that computes the same count | the WSM6 column call the CM1 seam makes goes through `gpuwm.core.microphysics.apply`, not the driver |
| `gpuwm/physics_registry_v2.json` | Morrison's number-species `state_name` fields, other schemes' consumers and warnings, one radiation option removed | the `wsm6-mp6` object is byte-identical |

The delta audit verifies thirteen mechanical executed-path checks (the
`apply` body, the WSM6 dispatch path, the WSM6 launcher module, the
reflectivity launcher and consumer, the mp=6 species list, the `RunConfig`
field table, the ring-guard family, the two registry-constraint checks and the
WSM6 vertical bounds). All thirteen pass, so the record's
`verdict.wsm6_answers_can_move` is `false`.

For Morrison, `gpuwm/core/kernels/morrison.cu`, `gpuwm/core/morrison.py` and
`gpuwm/core/morrison_constants.py` are byte-identical between 2.7.4 and 2.8.0,
the assembled Morrison NVRTC source is identical, and the `mp_physics == 10`
dispatch path is textually identical. The Morrison registry row changed only
its `consumers` metadata.

The admission gate, the compile path and the dispatch module did change bytes,
so the shipped presets (all Morrison) and the WSM6 lane are re-run on a GPU
before any 2.8.0 result is claimed; the release notes (`docs/RELEASE-NOTES.md`,
"Engine pin") give each scheme's revalidation status on 2.8.0.

### History: what the 2.5.8 to 2.7.4 move did to the executed WSM6 surface

This section records the earlier move, when 2.7.4 superseded 2.5.8. Its
audit record is project evidence and is not part of the distribution.

**Nothing that decides a WSM6 answer moved.** Seven of the sixteen admitted
files are byte-identical between 2.5.8 and 2.7.4, and they are precisely the
seven that carry the physics:

| Byte-identical file | What it holds |
|---|---|
| `gpuwm/core/kernels/wsm6.cu` | the WSM6 column kernel itself |
| `gpuwm/core/kernels/refl.cu` | the 10 cm reflectivity kernels |
| `gpuwm/core/kernels/common.cuh` | the shared device preamble |
| `gpuwm/core/wsm6.py` | the WSM6 launcher and field preparation |
| `gpuwm/core/wsm6_constants.py` | WSM6 coefficients and capacity tiers |
| `gpuwm/core/constants.py` | the thermodynamic constants and `CUDA_DEFINES` |
| `gpuwm/core/morrison_constants.py` | shared reflectivity coefficients |

The decisive number is the assembled NVRTC source. `gpuwm.core.kernels`
hands CuPy a single string built from the `CUDA_DEFINES` preamble, `common.cuh`
and `wsm6.cu`; `gpuwm.certify.kernel_manifest.record_module` digests exactly
that string. Reassembled from both estates it is the same in both, at the
default tier and at every capacity tier the CM1 seam can select:

| WSM6 tier | Assembled source SHA-256 | 2.5.8 vs 2.7.4 |
|---|---|---|
| default (`get_kernel`, no define) | `1a6d20da0d450f235227fe609bdb12b368d96aec5ac231752074ff4dd9cc50e6` | identical |
| `WSM6_KMAX=64` | `43bf2ac4f1ff0df6a91b4cdf4dcc19dc5b9903e2faab29935f11767efe5c6c23` | identical |
| `WSM6_KMAX=80` | `3b53f2197faacb9196892e549a2d1e9eb5f876da4e4d04e510f0541a65b20c0a` | identical |
| `WSM6_KMAX=128` | `3b839812f609c963dd76f13c703d6f3533b5fa0d5c4f946a87335187059e64ca` | identical |
| `WSM6_KMAX=256` | `fc8d6e792573506692ebe8dc1a59761eb81f658341d087fa1df84df9acbba450` | identical |

The first row is the one a shallow column actually compiles:
`gpuwm.core.wsm6.launch_wsm6` takes the plain `get_kernel` route whenever the
selected capacity is the shallow tier, so no `WSM6_KMAX` define is inserted and
`wsm6.cu`'s own `#ifndef` fallback of 64 applies. The explicit `WSM6_KMAX=64`
row is therefore a string no lane compiles; it is hashed anyway so the table
covers the whole ladder rather than the part that happens to be reachable.

The default-tier value is the same `1a6d20da...` the 2026-08-29 WRF v4.6.1
adapter oracle recorded. **The bytes NVRTC compiles for WSM6 under 2.7.4 are
the bytes that oracle graded.** The existing oracle and the existing
full-domain transition evidence are therefore not invalidated by the move.

Nine of the sixteen did change. Each was diffed hunk by hunk against the WSM6
executed path and none of them reaches it:

| Changed file | What changed | Why it does not reach WSM6 |
|---|---|---|
| `gpuwm/core/microphysics.py` | `_RING_STATE_FIELDS` became a union derived from the registry's per-scheme `ring_guard` rows; one hunk adds arguments to the Thompson saturation/evaporation calls | `apply` and `_dispatch_scheme` are textually identical; the WSM6 arm is unchanged; the derived union is a strict superset that still contains exactly WSM6's ten fields, and the ring guard only runs on specified/nested domains, which the CM1 seam does not configure |
| `gpuwm/core/refl.py` | the per-scheme required-field lists moved from four inline literals into one published `REFL_10CM_INPUT_SPECIES` table; three schemes that compute their own dBZ now refuse by name | `launch_refl10cm_wsm6` and `consume_refl_10cm` are textually identical, and the mp=6 presence check resolves to the same three species, `qr`/`qs`/`qg` |
| `gpuwm/config.py` | 14 new `RunConfig` fields (adaptive time step, eta levels, New Tiedtke closure) | every one has a default; no field was dropped and no existing default or annotation changed, so the 14 keyword arguments the CM1 adapter passes construct the same configuration |
| `gpuwm/physics_registry_v2.json` | the `wsm6-mp6` object gained a `consumers` block and two constraint keys | additive metadata; the one new refusal forbids `mp28_aerosol_source` in `{climatology, synthetic}`, and that field defaults to `auto` and is not among the settings the CM1 adapter passes |
| `gpuwm/physics_vertical_contract.py` | rows added for Milbrandt-Yau, WDM6, P3 and New Tiedtke; the legacy shortwave layer cap became runtime-sized | `WSM6_VERTICAL_LEVEL_BOUNDS` is still `(2, 80)` |
| `gpuwm/core/kernels/__init__.py` | `gf` and `ntiedtke` were added to the shared-header allow-list; a Noah-MP compile route was added | `wsm6` is absent from the allow-list in both versions, so its assembled source is unchanged, which the table above measures rather than assumes |
| `gpuwm/core/state.py` | `ScratchArena.view` accepts a same-width dtype; `DomainState` gained `set_base_geopotential` | a float32 request takes the identical path, and the CM1 seam supplies its own minimal state object rather than ArWen's `DomainState` |
| `gpuwm/core/physics.py` | cumulus, YSU and column-tendency coupling | no hunk mentions WSM6 or `mp_physics == 6` |
| `gpuwm/physics_registry.py` | new option rows and accessors | no hunk mentions WSM6 |

The `wsm6-mp6` registry object is therefore **not** byte-identical any more:
it was `bbe121da85b78e3674852b990c1a7b752cba798b3c727a7c314e9ea34d83f124`
under 2.5.8 and is `85a0e92dd2f278e9adb27cd55728c3ca6eb26513f45589b88743cf6f54ef9579`
under 2.7.4. That is a real delta and it is named here rather than waved
through; what it contains is descriptive metadata and one refusal that the CM1
configuration cannot reach.

That audit also verified thirteen mechanical executed-path checks: the
`apply` body, the dispatch body, the WSM6 launcher module, the reflectivity
launcher and consumer, the mp=6 species list, the `RunConfig` field table (no
field dropped, no existing default or annotation changed; the 14 fields 2.7.4
adds all carry defaults), the ring-guard family (covers WSM6, and is a strict
superset of 2.5.8's), the two registry-constraint checks, and the WSM6
vertical bounds. All thirteen pass, so the record's
`verdict.wsm6_answers_can_move` is `false`.

### History: what the 2.5.8 to 2.7.4 move did to the executed Morrison surface

The answer is different, and it is stated separately rather than folded into
the WSM6 verdict. `gpuwm/core/morrison.py` is byte-identical across the move.
`gpuwm/core/kernels/morrison.cu` is **not**: it was
`58eb091a705a476226b07cd82b4b7d8cf03d549c97c92cee312b19ef53e732b3` under 2.5.8
and is `a563005e...` under 2.7.4, 1181 lines against 1312, and the change is
numerical:

| Hunk | What it does |
|---|---|
| finalize kernel gains `qv[idx] = qvk;` | 2.5.8 never wrote the updated vapour mixing ratio back out of the finalize kernel. Every Morrison call under 2.5.8 would have discarded its own vapour update. |
| `morr_bound` takes `pres` and rebuilds PGAM's reference density as `pres/(287.15*temp)` | 2.5.8 scaled the frozen wrapper density by `RD/287.15`. WRF reads that reference density from the CURRENT `T3D` at `module_mp_morr_two_moment.F:1558` and `:3920`, after melting and after the tendency apply; the frozen `rhoa` of `:1325` is not it. |
| `morr_bound_one` only rewrites the number moment when the slope was actually clipped | an unclipped slope now round-trips exactly instead of being rebuilt from its own cube. |
| `morr_cloud_freezing`, `morr_limit_cold_cloud`, `morr_limit_cold_rain` | contact/immersion freezing moved to log-space gamma moments with a double-precision fallback, and the cloud and rain donor limiters gained wide-range arms, because the inherited FP32 `lambda**6` overflows at ordinary cloud slopes and silently erases freezing. |
| deposition nucleation bounded by the vapour excess over ice saturation | removes a path that drove `qv` negative at model-top levels. |

**No GM1 answer moves because of this.** Morrison has never been bound in this
port before; the first GM1 run that executes it executes 2.7.4. The delta is
recorded because a future re-pin has to be read against it, and because the
2.5.8 `morrison.cu` is not a fallback anything may select.

### Neighbouring records

The earlier audits that admitted 2.5.8 and 2.7.4 are kept as project
history; nothing admits against them.

`ARWEN-CAPABILITY-SURVEY.md` and its generated JSON inventory the rest of the
pinned estate: all 94 translation units, their drivers, entry points,
registry constraints and vertical bounds. GM1 consumes one component from it;
that record exists so the next lane does not have to rediscover the estate.

## CM1 r21.1 authority for this seam

The mathematical authority is the CM1 r21.1 release source at
`work/native-reference/cm1r21.1`. The seam substitutes ArWen's GPU WSM6 for
CM1's own, so these are the lines it has to match and the lines any future
re-pin has to be re-read against:

| CM1 r21.1 | What it fixes |
|---|---|
| `src/cm1.F:3405` | microphysics runs only when `imoist == 1` |
| `src/cm1.F:3416` | exactly one `mp_driver` call per large step, after `solve2` returns |
| `src/param.F:2723` | `ptype = 7` is WSM6; `src/param.F:3632` is its init arm |
| `src/mp_driver.F:1472` | the `ptype == 7` dispatch arm |
| `src/mp_driver.F:1474-1479` | the six `pdefq` positive-definite passes that precede the scheme, `qv` at floor `0.0` and the five hydrometeors at `qsmall` |
| `src/mp_driver.F:1488-1500` | the column pack WSM6 is handed: `t`, `qv`, `qc`, `qi`, `qr`, `qs`, `qg`, `den`, `p`, `delz` |
| `src/mp_driver.F:1516-1559` | the `mp_wsm6_wrapper` argument list, including the thermodynamic constants `g`, `cpd`, `cpv`, `rd`, `rv`, `t0c`, `ep1`, `ep2`, `xls`, `xlv0`, `xlf0`, `den0`, `denr`, `cliq`, `cice`, `psat` |
| `src/mp_driver.F:1563-1569` | the write-back: temperature to `dum1` and the six species straight back into `q3d`, in place |
| `src/mp_driver.F:1569-1573` | CM1's ptype=7 finish: theta from the new temperature at *fixed* `pp3d`/`prs`, and `rho` rediagnosed -- the constant-pressure closure, because WSM6 heats at constant pressure |
| `src/param.F:936-948` | CM1 forces `eqtset=1` for ptype 4, 7, 50-53 and 55: it never runs WSM6 at eqtset=2 |
| `src/mp_driver.F:1631` | `refl10cm_wsm6` is CM1's own reflectivity diagnostic for this scheme |

### The heating convention the eqtset=2 rebuild needs

CM1's eqtset=2 finish (`src/mp_driver.F:603-617`) rebuilds pressure, Exner and
theta at *fixed density* from the scheme's new temperature.  That is CM1's
pairing only for a temperature heated at *constant volume*: CM1's own Morrison
and Thompson replace `L/cpm` by `(L - Rv*T)*cvm` and `Lf*cvm` under eqtset=2
(`src/morrison.F:1640-1650`, `src/thompson.F:2865-2920`, `:3184-3200`), and
Morrison adjusts to saturation with the iterated constant-volume `satadj2`
(`src/morrison.F:2441-2480`, `:4012-4051`, `:6402-6504`) instead of WRF's
one-step `PCC`.

Every ArWen column scheme is WRF's, and heats at constant pressure.  Until
2026-09-23 the Cartesian lane handed that temperature straight to the fixed-rho
rebuild, so theta kept only `cv/cp + Rv*T/L`, about 0.78, of every phase
change's latent heat, and CM1's own supercell deck grew a storm about 40 %
weaker than CM1's (`evidence/oracle/supercell-fidelity`).
`cm1core.cuda_cartesian_moist_heating` now converts the component's step to
CM1's eqtset=2 heating before the (unchanged, sealed) rebuild:
`T = Tb + (cpm_scheme*(T - Tb) + Rv*Tb*dqv)*cvm`, with `cvm` of the column's
entry state, and for Morrison CM1's own `satadj2` after it.  WSM6 takes the
same constant-volume conversion; CM1 has no eqtset=2 WSM6 to hold it to, so
its receipt says `cm1_configuration: false`.  A new component admits only with
a recorded heating convention (`GM1-MOIST-HEATING-CONVENTION-UNKNOWN`).

The older tilestream and product lanes (`eqtset2_wsm6_bridge.py`) are sealed
evidence lanes and still pair WSM6's constant-pressure heating with the
fixed-rho rebuild.

The cadence the adapter declares -- one public call after the completed RK
stages per large step -- is `src/cm1.F:3416`. The carrier family it binds is
`src/mp_driver.F:1488-1500`. What GM1 does **not** claim is native-CM1
trajectory identity: ArWen's WSM6 is transcribed from WRF v4.6.1, graded
against a WRF v4.6.1 oracle, and is a substitution, not CM1's own Fortran.
## CM1's `ihail`, and what each setting does

`ihail` is one integer in CM1's `&param2` namelist (`src/input.F:95,272`).
CM1 refuses anything outside `0..1` and supplies no default
(`src/param.F:1567-1580`), and it passes the value straight through to
whichever scheme is selected: as the first argument of `graupel_init` at
`src/param.F:3407` for `ptype=5`, and `call mp_wsm6_init(..., hail_opt=ihail,
...)` at `src/param.F:3697-3706` for `ptype=7`. `README.namelist:505-506`
names only `ptype=2,5`; the r21.1 source is the authority and it reaches WSM6
too, so GM1 follows the source.

GM1 previously hard-coded `wsm6_hail_opt=0` at the call site, so every storm
it has ever run had a graupel-like dense ice category. The flag is now
`physics.ihail` in the TOML document, threaded through `RuntimeConfig.ihail`
and `cm1core.dense_ice` to the single `RunConfig` keyword each component takes
it in. It is an optional key, and **its undeclared value is now CM1's own**:
`1`. CM1 supplies no compiled default, but it ships one, in both
`run/namelist.input:56` and `run/config_files/supercell/namelist.input:55` --
the very deck this lane exists to reproduce. This port bound `0` until that
move, so every receipt recorded before it names a graupel-like category CM1's
own decks do not select. **The move changes the words of every WSM6 run**,
including its 10 cm reflectivity, which `gpuwm/core/refl.py:638-644` also
keys on `wsm6_hail_opt`. A document that wants the previously recorded words
declares `physics.ihail = 0` explicitly, and the run records the resulting
constants by value either way.

A component whose scheme has no dense-ice branch to select -- either Thompson
-- refuses a declared `physics.ihail` by name
(`GM1-DENSE-ICE-SELECTOR-ABSENT`) rather than recording a value the scheme
never reads. WDM6 does read one, through WRF's shared `hail_opt` key, and it
is the *same* five constants: `gpuwm/core/wdm6_constants.py:39` imports
`rimed_ice_constants` from `wsm6_constants`. Its undeclared default is `0`,
because CM1 has no WDM6 `ptype` and therefore no answer to move to. WDM6 is
transcribed but not selectable; see the blocked table below.
it in.

It is an optional key, and what an undeclared one binds is **per component**,
because CM1 supplies no default at all (`src/param.F:1567` stops the model
outside 0..1) and the two components have different histories.  `wsm6-mp6`
binds `0`: that is the branch this lane has executed since it first bound
WSM6, every already-measured run's words depend on it, and ArWen's own
`wsm6_hail_opt` default is `0`.  `morrison-mp10` binds `1`: GM1 has never
executed a Morrison column, so there is no continuity value to prefer, and CM1
ships `ptype=5` with `ihail=1` in both the distributed namelist
(`run/namelist.input:55-56`) and the supercell deck this lane exists to
reproduce (`run/config_files/supercell/namelist.input:54-55`); WRF's own
`morr_rimed_ice` default is `1` for the same scheme.  Binding `0` there would
silently run CM1's reference supercell with graupel where CM1 runs hail --
exactly the fall-speed error this selector exists to end.  A component with no
recorded default is refused by value (`GM1-DENSE-ICE-DEFAULT-COMPONENT`).  The
run records the resulting constants by value either way.

What the two settings are, transcribed in `cm1core.dense_ice`:

| | WSM6 `ihail=0` (graupel) | WSM6 `ihail=1` (hail) |
|---|---:|---:|
| `n0g` (m-4) | `4.0e6` | `4.0e4` |
| `deng` (kg m-3) | `500` | `700` |
| `avtg` | `330.0` | `285.0` |
| `bvtg` | `0.8` | `0.8` |
| `lamdagmax` (m-1) | `6.0e4` | `2.0e4` |

What the move cost, measured rather than asserted. CM1 r21.1's own compiled
WSM6 and GM1's seam were each run on the same 120 s near-saturated column
seeded with 3 g/kg of graupel at 4.1-5.0 km, once per branch. The horizon is
the term's own: one 250 m layer at about 6 m/s is 40 s, so 120 s is three
crossings.

| | CM1 r21.1 | GM1 (ArWen WSM6) |
|---|---:|---:|
| `ihail=0` dense-ice centroid fall | 613.4 m (5.11 m/s) | 458.9 m (3.82 m/s) |
| `ihail=1` | 1131.9 m (9.43 m/s) | 938.4 m (7.82 m/s) |
| hail / graupel | **1.845** | **2.045** |

Both move the same way by the same order, an 11% difference in the response:
`physics.ihail` reaches the branch CM1's `ihail` reaches. The measurement also
turned up something it was not looking for, and the receipt records it rather
than explaining it away: on that column CM1's WSM6 converts 2.42 kg m-2 of
vapour in 120 s and nearly doubles the graupel burden, while ArWen's converts
almost none. The absolute fall speeds differ by about 20% as a direct
consequence -- a heavier graupel population has a larger slope parameter -- and
nothing in this pass established the cause. Receipt:
`evidence/micro/wsm6-dense-ice-ihail/ihail-differential.json`; harnesses:
`tools/oracle/wsm6_dense_ice_sedimentation_harness.f90` and
`tools/oracle/wsm6_dense_ice_sedimentation_gm1.py`.

Authority `src/mp_wsm6.F:86-100`. The fall speed those feed is
`vt2g = pvtg * rslope**bvtg * denfac` with `pvtg = avtg*gamma(4+bvtg)/6`
(`:152`), `rslope = 1/lamdag` and `lamdag = (pi*deng*n0g/(rho*q))**0.25`
(`:155,939,1551`). Both branches share `bvtg`, so the gamma factor cancels and
the ratio is exact for every `rho*qg`: **the hail branch falls about 2.03x
faster than the graupel branch**. A hundred times fewer, denser, larger
particles instead of many small slow ones.

| | Morrison `ihail=0` (graupel) | Morrison `ihail=1` (hail) |
|---|---:|---:|
| `AG` | `19.3` | `114.5` |
| `BG` | `0.37` | `0.5` |
| `RHOG` (kg m-3) | `400.` | `900.` |

Authority `src/morrison.F:397-412,441-444`, with `CG = RHOG*PI/6` at `:476`.
The relation is `V = A*D**B` (`src/morrison.F:397`), so at a one-centimetre
particle **hail falls about 3.26x faster than graupel**.

Dense-ice fall speed sets how far precipitation is carried from the updraft
before it reaches the ground, which is where the cold pool forms, which is
what the low-level mesocyclone then develops over. `cm1core.dense_ice`
computes these ratios from the transcribed constants rather than asserting
them, and every component metadata, receipt and runtime record carries the
selected constants, the unselected ones and that sentence.

### `ihail` proved into the compiled kernel, with no GPU

`tools/oracle/morrison_dense_ice_fallspeed_oracle.py` closes the last link of
the `ihail` claim on a host with no card. It reads
`gpuwm/core/kernels/morrison.cu` out of the admitted estate, refuses unless its
SHA-256 is the byte `engine_pin` pins, lifts `morr_bound_one` and
`morr_terminal_velocity` **verbatim** by brace matching, and compiles them with
`g++` behind a shim that makes the CUDA spellings mean their IEEE binary32
selves (`real` -> `float`, each `__f*_rn` -> the single rounded operation it
names, `-ffp-contract=off`). It then compiles a Fortran harness transcribed from
CM1's own `src/morrison.F:4463-4539` -- the graupel slope bound, `UMG`/`UNG`
and the `20*dum` realistic-limit cap -- and grades one deterministic deck.

The result, sealed at
`development-repository:evidence/oracles/morrison-dense-ice-fallspeed-v1/receipt.json`:

* the compiled kernel text and CM1 agree to a worst relative difference of
  `2.716e-7`, about one binary32 ulp;
* the one operation that is not bit-reproducible is named rather than hidden:
  CM1 forms the slope as `(CONS2*N/Q)**(1./DG)` and the kernel as
  `cbrtf(six_c*number/q)` on the identical argument -- `six_c == CONS2 ==
  GAMMA(1.+DG)*CG == PI*RHOG` exactly -- and `powf` and `cbrtf` are separate
  glibc routines that round independently;
* the two `ihail` branches do not produce the same words, and every column that
  carries dense ice falls faster under hail, by a factor between 1.9 and 3.2
  over the deck.

`tests/test_morrison_dense_ice_fallspeed_oracle.py` drives that path from
`examples/cartesian-supercell-morrison-hail.toml`: the document's `ihail = 1`
becomes `RuntimeConfig.ihail`, becomes the transcribed constants, and those
constants are what the compiled kernel evaluates `V = A*D**B` with. No CUDA
context is opened.

The sealed GPU oracle harness `tools/run_arwen_wsm6_oracle.py` still refuses
anything but `hail_opt=0`. That is not a limit on the component: the harness
is byte-pinned by `evidence/validation/arwen-wsm6-adapter-oracle-v1.json`,
whose measurements were captured at `hail_opt=0`, and re-pinning it without
re-measuring on a GPU would make the receipt claim bytes that never produced
its numbers. The hail branch has no measured component oracle yet.

## Morrison (CM1 `ptype=5`) binding

CM1's own supercell deck is `ptype=5` with `ihail=1`
(`run/config_files/supercell/namelist.input:54-55`).

**Scalar axis.** `src/param.F:3417-3473` gives `numq = 10`: the six WSM6 mass
mixing ratios followed by four number concentrations `nci, ncs, ncr, ncg`,
with `nqv=1`, `nql1..nql2=2..3`, `nqs1..nqs2=4..6`, `nnc1..nnc2=7..10`,
`iice=1` and `idm=1`. The `numq = 11` table at `src/param.F:3476-3582` is
unreachable in an unmodified r21.1 because `src/morrison.F:325` hard-codes
`INUM = 1` and `GRAUPEL_INIT` returns it through `cm1inum`
(`src/morrison.F:314,656`), so it is not transcribed.
`cm1core.moisture_species` owns both tables, including CM1's `qmag`
(`1.0e-2` for every mass row, `1.0e6` for every number row), which is what
`src/solve2.F:1472` scales the WENO positive-definite epsilon by.

**Name mapping.** CM1 orders its number concentrations `nci, ncs, ncr, ncg`
and hands them to `MP_GRAUPEL` in that order (`src/mp_driver.F:546-558`
against the argument list at `src/morrison.F:687,701-705`). ArWen's launcher
takes `nc, nr, ni, ns, ng`. Nothing is bound positionally: `MORRISON_SPEC`
carries the map `nci->ni, ncs->ns, ncr->nr, ncg->ng` and every consumer goes
through it.

**Cloud-droplet number.** At `inum=1` CM1 does not transport it. It reassigns
`NC3D(K)` to `NDCNST` times 1.E6 over `RHO(K)` inside the column loop before the size
distributions are formed (`src/morrison.F:1830-1833` and `:2770-2773`), and
`src/morrison.F:1673` makes `RHO(K)` the model's prognostic density rather
than the diagnosed `PRES/(R*T)` of the WRF original. The admitted ArWen kernel
instead refreshes `nc` at the END of a call, from its own diagnosed density
(`gpuwm/core/kernels/morrison.cu:1310`). GM1 therefore owns `nc` as a
component carrier and writes CM1's statement itself, from CM1's `alt = 1/rho`,
immediately before every dispatch
(`ArwenMorrisonComponent.prepare_component_owned_carriers`). That restores
CM1's statement order and CM1's density; the kernel's own end-of-call write is
then overwritten before it is next read.

**`ndcnst`.** `gpuwm/core/kernels/morrison.cu:124` and `:1310` write the conversion as
the literal `250.0e6f / rhoa`, which is CM1's `&param3` default and its
supercell deck's value. `physics.ndcnst` is exposed and any other value is
refused by name (`GM1-MORRISON-NDCNST`), because the alternative is a run
whose configuration and receipts record a droplet concentration that no column
ever activated. Declaring it against a single-moment component is refused too
(`CM1-CFG-CARTESIAN-NDCNST-SCOPE`).

**Persistence.** Thirteen retained carriers against WSM6's eleven:
`h_diabatic`, `effc`, `effr`, `effi`, `effs`, `nc`, and the same seven surface
accumulators. `effr` is Morrison's rain effective radius, which WSM6 does not
diagnose. Every empty category reports 25 microns
(`src/morrison.F:5153-5178`), which is the value a fresh column carries.

**Column depth.** `gpuwm/core/morrison.py` compiles its sedimentation kernel
at 64 and 256 column slots, so the Morrison binding admits `2 <= nz <= 256`
with no overlay. The explicit WSM6 column-capacity overlay patches
`gpuwm/core/wsm6_constants.py` and `gpuwm/physics_vertical_contract.py`, which
the Morrison launcher never reads, so selecting it against Morrison is refused
(`CM1-MORR-CAPACITY-PROFILE`, `CM1-CARTESIAN-CAPACITY-PROFILE-SCOPE`).

**Reflectivity.** `gpuwm/core/refl.py:767-776` consumes `qr, nr, qs, ns, qg,
ng` at `mp_physics=10` with `qv` as the vapour argument; `qc`, `qi` and CM1's
`nci` are not arguments to it. The component metadata names both sets. Note
that the Morrison reflectivity launcher takes `morr_rimed_ice` too, so
`ihail` moves the published dBZ as well as the fall speed.

**Divergences from native CM1 at this seam, beyond the WSM6 ones above.**
ArWen's Morrison diagnoses its own air density as `p/(RD*T)` inside the
kernel, where CM1 r21.1 uses the model's prognostic `RHO3D`
(`src/morrison.F:1672-1673`, the WRF line commented out). GM1 cannot change
that without editing the pinned dependency, so it is recorded, not claimed
away. No independent Morrison component oracle has been measured: the tiled
launch record carries `native_oracle_receipt_sha256: null` for this component
rather than borrowing WSM6's.

## Every ArWen microphysics scheme, and where each one stands

ArWen's public dispatch (`gpuwm/core/microphysics.py` `apply`) admits
`mp_physics` 1, 6, 8, 9, 10, 16, 18, 28 and 50. GM1 binds four.

| GM1 component | scheme | CM1 `ptype` | ArWen `mp_physics` | `numq` | state |
|---|---|---:|---:|---:|---|
| `wsm6-mp6` | WSM6 | 7 (`param.F:3632`) | 6 | 6 | measured |
| `morrison-mp10` | Morrison 2-moment | 5 (`param.F:3400`) | 10 | 10 | **unvalidated** |
| `thompson-mp8` | Thompson | 3 (`param.F:3322`) | 8 | 8 | **unvalidated** |
| `thompson-aerosol-mp28` | Thompson aerosol-aware | none | 28 | 11 | **unvalidatable against CM1** |

All four dispatch. Each was run once through the CM1-owned adapter on one
120 s seeded near-saturated column on an RTX 5070 Ti and returned a finite,
non-negative column whose dense ice fell: WSM6 938.4 m, Morrison 1147.9 m,
both Thompsons 1011.9 m
(`evidence/micro/wsm6-dense-ice-ihail/column-component-smoke.json`). That is
not a validation and the register says so; it is the check that a carrier map
or a scalar count is not silently wrong, and it is the reason a scheme that
looked bindable is in the blocked table below instead.

`cm1core.component_validation` is the one place that says which is which, and
its words travel verbatim into every run receipt and into `gpuwm-cm1 doctor`,
where an unmeasured component prints as `UNVALID` carrying the token
`GM1-UNVALIDATED-CAPABILITY` and the list of what was not checked. Nothing is
hidden and nothing is refused for being unmeasured.

Five schemes are compiled and reachable inside ArWen and **not** selectable
from GM1. Each is held by one named piece of GM1 work, recorded with the exact
change in `cm1core.column_component_blockers`, and a document that names one
is refused with that blocker rather than with the word "unsupported":

| component | scheme | blocker |
|---|---|---|
| `kessler-mp1` | Kessler (CM1 `ptype=1`) | six-mass-species post-physics |
| `milbrandt2-mp9` | Milbrandt-Yau 2-moment | six-mass-species post-physics (it carries a seventh mass row, hail) |
| `nssl2-mp18` | NSSL 2-moment (CM1 `ptype=26/27`) | the public seam needs a `PhysicsDriver` and a production binding |
| `wdm6-mp16` | WDM6 | ArWen's WDM6 refuses without WRF's `XLAND` land/sea mask |
| `p3-mp50` | P3 one-category (CM1 `ptype=50`) | six-mass-species post-physics |

`wdm6-mp16` is in that table **because the column was dispatched**, not because
anyone reasoned about it: its nine-scalar axis, carrier map and shared WSM6
rimed-ice constants are all transcribed and tested
(`cm1core.arwen_scheme_components.WDM6_SPEC`, kept), and ArWen refuses at
`gpuwm/core/wdm6.py:156` -- *"mp_physics=16 (WDM6) needs WRF's XLAND land/sea
mask for its per-column autoconversion threshold (module_mp_wdm6.F:607-614) and
the physics driver carries none; refusing to substitute an all-land or
all-water mask"*. CM1's idealized Cartesian lane has no land/sea field at all,
so this is a lane question rather than a species-table one.

One blocker now does the rest.

**six-mass-species post-physics.** `gm1_tiled_post_wsm6_eos_qa_f32_v1` copies
a compile-time constant six mass rows (`for(unsigned s=0;s<6u;++s)`) and its
binding requires `qa` to be exactly `(6, nk, ny, nx)`. On the Cartesian lane
that `qa` is discarded block scratch (solve3 publishes the domain `qa`), but
the kernel still reads six `q3d` rows to fill it. CM1's own tables are
not all six long: `ptype=1` is three (`param.F:3171`) and `ptype=50` has four
mass rows before its number and rime rows (`param.F:4102-4108`). Making that
loop bound a kernel parameter is the whole fix, and six stays bit-identical
because the loop is a copy with no arithmetic. Milbrandt-Yau needs it for the
opposite reason: it carries a seventh mass row, hail.

**column `w` carrier -- built.** Classic Thompson
(`gpuwm/core/microphysics.py:551`), aerosol-aware Thompson
(`gpuwm/core/microphysics_aerosol.py:436,463`) and Milbrandt-Yau
(`gpuwm/core/milbrandt2.py:254`) all read `state.w` -- WRF's `grid%w_2` lower
full-level slice, not a mass-level average. `ColumnComponentSpec` now declares
`extra_interface_carriers`, the tile inventory stages them, and
`cuda_cartesian_finish` copies the dycore's `w3d`, which is the same quantity
on exactly ArWen's `(nz+1, ny, nx)` stagger, read-only and underived. A
component that declares none has a byte-identical inventory to the one it had.
This is what a measured dispatch bought: classic Thompson looked bindable
without it and was not.

Aerosol-aware Thompson needed one thing more, which the same measurement
found: `nwfa2d` and `nifa2d`, WRF's two *two-dimensional* surface aerosol
emission rates (`gpuwm/core/microphysics_aerosol.py:220`). They are
component-owned SURFACE persistence, declared through
`component_owned_surface_carriers`, not rows on the transported axis -- a
scheme that carried them as column scalars would advect an emission rate. GM1
allocates both at zero because CM1's idealized lane models no surface aerosol
source, and the receipt says so.

NSSL is the one that is not a species-table question at all: ArWen reaches it
through `apply_nssl2_production(state, cfg, dt, hooks, ...)` with its own hook
contract, and the plain public `apply` refuses `mp_physics=18` without a
`DomainState` `PhysicsDriver` and an `NSSL2ProductionBinding`. CM1's own
`ptype=26/27` axis is built dynamically from `lhab` and the hail/CCN switches
(`param.F:3736-3810`), so its transcription is real work too.

### Schemes CM1 has and ArWen does not

`param.F:3141-4509` also dispatches `ptype` 2 and 4 (Goddard-LFO and
GSR-LFO), 6 (Rotunno-Emanuel), 51, 52 and 53 (the other three P3 packages)
and 55 (Jensen ISHMAEL). ArWen ports none of them, and `gpuwm/config.py`
refuses 51/52/53 by name with that reason. They are out of reach from this
seam in principle, not by omission.

## What is generic and what is not

`cm1core.cuda_gm1_tilestream_column` owns the device-side tile staging and
launch for both components; `cm1core.cuda_gm1_tilestream_wsm6` keeps its
public names and delegates, so there is one implementation, not two. The
pinned post-physics kernel does `mp_driver.F:603-617` (the eqtset=2
prs/pp3d/th3d rebuild, which reads only `q3d(nqv)`) over a contiguous
mass-prefix window of any admitted `q3d`, so it keeps its six-species contract
whatever the component transports. The six-row `qa` that kernel also writes is
per-block scratch, which the Cartesian lane never publishes:
`solve3.F:624-643` (`CartesianFinish.solve3`) equates `qa` to `q3d` for every
`n=1,numq`, number rows included, and CM1 calls `mp_driver` before `solve3`
(`src/cm1.F:3416`, `:3432`), so `qa` holds the step's start from microphysics
until solve3 -- which is what the qv budget's step tendency reads. The same
mass-prefix rule applies to
`calcnm` (`src/turb.F:5521-5535` excludes number concentrations from `qt` by
name) and to the positive-definite repair (`src/mp_driver.F:505-520` shows the
`asq(7)` call commented out in the authority, so only the six mass scalars are
repaired). Scalar advection and scalar turbulence run over all `numq`, as
`src/solve2.F:1435` and `src/solve1.F:1213-1231` do.

## The Cartesian lane executes every bound component

Until this change every non-WSM6 document was admitted by the configuration
layer and then refused at startup: the capability was present and unreachable.
Five places assumed six scalars or one scheme:

| where | what it did | fix |
|---|---|---|
| `cuda_cartesian_runtime._warm_public_component` | always built the WSM6 component, so `CM1-COMPONENT-ID` refused any other id | warms `column_component_spec(physics.component)` with that scheme's own carrier set (Thompson's `w` included); the WSM6 branch is the old construction verbatim |
| `cuda_cartesian_startup` | required `q3d` to be `(6, ...)` | takes the run's `numq`; getset's density reads `q3d(nqv)` only (`src/init3d.F:2393-2396`) |
| `CartesianDynamics.pre_acoustic` -> `cuda_moist_stage_bridge` | handed the whole `q3d` to a six-row contract | hands `mass_prefix_q3d()`, a contiguous view of rows 0..5; the bridge keeps refusing ten rows |
| `cuda_gm1_tilestream_post_wsm6._require_wsm6_launch` | accepted only the WSM6 launch status | that module is sealed by its device evidence receipt and is **not** edited; `cuda_cartesian_finish` still calls it for WSM6, and calls `cuda_gm1_tilestream_post_column` for every other component: the same compiled kernel, device-inventory and snapshot checks, with a launch check that admits exactly one non-WSM6 identity triple whose axis leads with CM1's six mass rows (a relabelled triple is refused) |
| `cartesian_restart` / `cartesian_dynamics_state_shapes` | persisted WSM6's persistence list and a literal six scalar rows | persists the component's own persistence and `numq` rows; restore also refuses a live allocation of another extent |

A CPU host double of the physics block loop also found that
`CartesianFinish._allocate_persistence` gave aerosol-aware Thompson's two
surface emission carriers mass volumes (it keyed on `surface_scratch`, not
`surface_persistence`); that is fixed.

### Species classification

`cm1core.moisture_species.require_cm1_mass_prefix` is the one gate: every
admitted axis carries `qv,qc,qr,qi,qs,qg` as rows 0..5 and only number
concentrations after them. What each consumer reads, by CM1's own loop bounds:

| consumer | CM1 | rows |
|---|---|---|
| buoyancy, theta-rho, eqtset=2 cpm/cvm | `src/solve2.F:317-412,895-921` (dum2 over `nql1..nql2`, dum3 over `nqs1..nqs2`) | mass only |
| post-physics EOS | `src/mp_driver.F:603-617` | `q3d(nqv)` only |
| positive-definite repair | `src/mp_driver.F:505-520` (`asq(7)` commented out) | 1..6 |
| `calcnm` total water | `src/turb.F:5521-5535` | `nqv`, `nql1..nql2`, `nqs1..nqs2` |
| startup density | `src/init3d.F:2393-2396` | `q3d(nqv)` only |
| dry-mass ledger | `src/solve2.F:1618-1645` | none |
| advection, turbulence, boundaries, `qa` | `src/solve2.F:1435`, `src/solve1.F:1213-1231`, `src/solve3.F:624-643` | all `numq` |

### Nest injection

`tools/nest_downscale_init.py` maps the child's `q3d` rows onto the parent
frame's `q3d` (mass) and `nc3d` (number) carriers by CM1 scalar name
(`cm1core.nest_downscale.plan_species_transfer`). It refuses
(`CM1-NEST-DOWNSCALE-SPECIES`) a child mass species the parent lacks, a parent
mass species the child cannot carry, and a child number concentration the
parent lacks. The one exception is explicit: `--missing-number-concentrations
scheme-lambda-bound`, admitted for `morrison-mp10` only, writes exact zero and
lets Morrison's own slope bound rebuild each number from its mass on the first
call (`src/morrison.F:1876-1885,1947-1956,1971-1980,2796-2805`;
`gpuwm/core/kernels/morrison.cu:99-113,146-154`). That number is the scheme's
largest-mean-size bound, not a measured distribution, and the receipt says so.
A row-for-row copy into a wider child axis, which used to leave `nci..ncg` at
zero silently, is refused. The live nest refuses a parent and child on
different components.

**GPU status: pending.** Everything above is proven on CPU host doubles
(`tests/test_cartesian_species_generic_runtime.py`). No Morrison Cartesian step
has run on a device yet.

## Implemented public contract

`cm1core.arwen_wsm6_component` owns the CM1-facing semantic adapter. It calls
only public `gpuwm.core.microphysics.apply` for component id `wsm6-mp6`; it
does not import private `gpuwm.core.wsm6`, copy WSM6 CUDA source, or invoke the
full MPAS phase1+phase2 seam.

The v1 carrier contract is C-contiguous CUDA binary32 with `k=0` at the lowest
model level:

| Family | Carriers | Shape |
|---|---|---|
| scalar volume | `thp`, `p`, `alt`, `qv`, `qc`, `qr`, `qi`, `qs`, `qg` | `(nz, ny, nx)` |
| scalar base | `thb` | `(nz,)` |
| interface base | `phb` | `(nz+1,)` |
| interface perturbation | `php` | `(nz+1, ny, nx)` |

The adapter validates semantic id, integer dimensions, finite positive
geometry and timestep, `2 <= nz <= 80`, exact shapes, binary32 dtype,
C-contiguity, no mutable-carrier aliases, one CUDA device and one stream before
dispatch. `thp` and all six water species are updated in place. It allocates
and retains component-owned `h_diabatic`, `effc`, `effi`, `effs`, `rainnc`,
`rainncv`, `snownc`, `snowncv`, `graupelnc`, `graupelncv` and `sr` arrays.
Those arrays are the WSM6 component restart inventory; caller-owned CM1
prognostics remain a separate restart inventory.

Execution stays on the caller's current CUDA stream, performs no component
host copy or explicit synchronization, and returns a recorded ready event.
Later calls must use the same device and stream so retained physics state
cannot race. On output-due calls, the adapter passes `refl_10cm_due=True`, then
calls only public `gpuwm.core.refl.consume_refl_10cm`. It verifies the returned
binary32 `(nz,ny,nx)` dBZ scratch identity, consumes the stash exactly once,
and treats the result as a transient diagnostic rather than restart state.
For WSM6, ArWen's measured reflectivity mapping consumes `qv`, `qr`, `qs`, and
`qg` at post-scheme temperature/pressure; `qc` and `qi` are not arguments to
that WSM6 reflectivity diagnostic.

The Cartesian production lane reaches this consumer. `[physics] output_dbz`
(CM1 `README.namelist:1366-1369`) raises `refl_10cm_due` on every tile block
of the steps a frame is published at, exactly as CM1 raises `getdbz`
(`cm1.F:3214-3237`); the tile-local dBZ block is drained into the domain
carrier before the component rebinds, and CM1's `calccref` column maximum
(`misclibs.F:2259-2292`) is folded from it on the device. The formula stays
ArWen's: WRF `refl10cm_wsm6` (`mp_wsm6.F90:2275-2444`) with
`module_mp_radar.F:74-190` `radar_init`, inside the byte-pinned
`gpuwm/core/kernels/refl.cu` and `gpuwm/core/refl.py`. This port derives no
reflectivity of its own and refuses `output_dbz` for any component with no
admitted scheme-matched formula, as `param.F:1583-1603` does by `ptype`.

## Measured component oracle

On 2026-08-29 the adapter ran three eight-level, 30-second product-mode columns
(warm cloud, cold cloud and graupel) on an NVIDIA GeForce RTX 3080, compute
capability 8.6. Each result was compared with the frozen direct WRF v4.6.1
WSM6 oracle. All gates passed:

| Quantity | Worst observed | Gate |
|---|---:|---:|
| temperature | `4.84375e-7 K` | `2e-4 K` |
| mixing ratio | `9.35498e-10 kg kg-1` | `2e-8 kg kg-1` |
| surface precipitation | `8.94562e-8 kg m-2` | `5e-6 kg m-2` |
| frozen ratio | `4.91349e-7` | `5e-5` |
| stable-regime effective radius | `0.00175447 micron` | `0.03 micron` |
| all-regime effective radius | `15.01000023 micron` | `16.1 micron` |

No hydrometeor was negative below `-1e-12 kg kg-1`. The assembled WSM6 NVRTC
source SHA-256 was
`1a6d20da0d450f235227fe609bdb12b368d96aec5ac231752074ff4dd9cc50e6`.
The canonical evidence is
`development-repository:evidence/validation/arwen-wsm6-adapter-oracle-v1.json`;
the reproducible driver is
`development-repository:tools/run_arwen_wsm6_oracle.py`.

## Full corrected-real transition and restart proof

On 2026-08-29 the bridge ran two public WSM6 transitions over the full
`300 x 300 x 75` corrected-real initialized-product physical domain on the
same RTX 3080. The first transition requested output-due reflectivity exactly
once. Its 6,750,000 values were finite, ranged from `-35` to
`10.287071228027344 dBZ`, and were excluded from the checkpoint. Every one of
the six caller-state groups changed at least one binary32 word.

All 21 transition-local caller, held-tendency, and component arrays were
staged exactly and persisted through Rust `CM1W6R1` write/inspect/restore. A
fresh CUDA arena resumed from the restored words produced a second transition
raw-word identical to the uninterrupted second transition. The complete
receipt is
`development-repository:evidence/validation/initialized-product-wsm6-full-v1/full-domain-sm86.json`;
the two real-data PNGs live beside it. This closes the bounded transition-local
restart and reflectivity-consumer gates, not the complete forecast-restart
inventory.

## Energy-term boundary

`O-EQTSET2-IDISS-V1` and the matching sm86 receipt prove the exact supplied-
input dissipative-heating consumer at CM1's pre-RK `solve1` cadence. It is not
called from the post-final-RK WSM6 bridge. `O-TERRAIN-DEFORMATION-SGS-V1`
now proves the selected terrain/all-open `calcdef -> tkekm -> local TKE-source`
chain, and a guarded full corrected-real `300 x 300 x 75` sm86 execution binds
the resulting live z-face `dissten` directly into that consumer for the source-
initialized first large step. The next bounded lane implements selected
`gettau` and momentum stress divergence plus general nonuniform-`tkea`
`turbt` and stage-interleaved TKE advection/RK/commit. Its native oracle is
primitive-bounded, its complete analytic CPU/sm86 outputs are raw-word exact,
and its real T0 run clips TKE to exact zero while holding the dycore state
fixed between stage-local mass-flux rebuilds. This is not a later-step or
coupled trajectory. Real-product `efall` remains fixed at zero;
`efall=1` is strictly refused because CM1 requires species terminal velocities
that the public WSM6 seam does not expose.

## Gate boundary

The former `CM1-ARWEN-SEAM-MISSING`, reflectivity-consumer, bounded transition-
local restart, and first-step live-`dissten` producer blockers are retired.
Product preflight still refuses while scalar turbulence, later-step coupled
SGS/TKE, the complete CUDA dycore/closure/boundary composition, complete
forecast-restart inventory, and trajectory gates are open. The full-domain
PNGs are real initialized-product operator evidence, not a coupled supercell
forecast or tornado simulation.

## The ArWen estate does not supply nesting

GM1 pins `gpuwm>=2.8.0,<2.8.1` (`engine_pin.GPUWM_REQUIREMENT`) and the
byte-admitted surface is the column microphysics, reflectivity, constants and
registry files only. It contains no `nest.cu`, no `spec_bdy.cu` and no
lateral-boundary kernel: **the pinned estate supplies no nesting to this port,
and no ArWen nesting kernel is admitted**.

ArWen **2.7.4** does carry a complete WRF-style nesting apparatus (`nest.cu`
(SINT/SINTB interpolation, `bdy_interp1`, terrain blending, feedback copies),
`nest_microphysics.cu`, `spec_bdy.cu` (Davies specified/relaxation),
`lbc_flow.cu`, `lbc_state.cu`, `lbc_time.cu` and `openbc.cu`, with Python
drivers `nest.py`, `nest_interp.py`, `nest_lifecycle.py`, `nest_relocation.py`
and `nest_spawn.py`) implementing one-way forcing and an experimental
two-way feedback at odd refinement ratios, measured at 3:1. When this section
was written the 2.7.4 tree was a staged, unpinned reference; the seam has since
moved to 2.7.4 and then to 2.8.0 for the column physics, and none of these
nesting files was added to the admitted surface.

GM1's nesting therefore takes **nothing at runtime** from ArWen. It carries its
own CUDA text (`nest_boundary.NEST_RELAXATION_KERNEL_SOURCE`, sha256-pinned),
its own CPU oracle, and its own geometry and coupler, and imports `gpuwm`
nowhere in that path. What it took from the 2.7.4 tree is the same thing it
would have taken from the WRF source: the discrete form of the Davies
relaxation weights and their application, cross-checked by hand against
`gpuwm/core/kernels/spec_bdy.cu` and `gpuwm/verify/npref.py:4047-4139`, and
cited as such in `CM1-NEST-SEAM.md`. GM1's nesting would keep working
unchanged if the ArWen pin never moved.
