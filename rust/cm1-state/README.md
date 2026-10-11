# cm1-state

`cm1-state` is the dependency-free Rust ownership boundary for two deliberately
separate doors: a strict structural-zero-step container verifier and one
sealed, source-anchored scientific initializer.

## Structural container

The structural door strictly validates:

- `CM1SZS1\0` magic and little-endian `u64` JSON-header length;
- the 65,536-byte header cap and 67,174,416-byte whole-envelope cap, including
  filesystem metadata preflight before an input file is read;
- Python canonical sorted-key, compact UTF-8 JSON;
- the fixed zero-step, state, and state-plan v1 schemas;
- exact lane geometry: dry is `16x16x16` at `1000/1000/1000 m`; product and
  native context are `120x120x40` at `1000/1000/500 m`;
- all coordinate rows, duplicated stagger geometries, and the complete ordered
  deferred-state gate inventory;
- fixed `ua,u3d,va,v3d,wa,w3d,ppi,pp3d,tha,th3d` order;
- little-endian float32, C-order `(z,y,x)` shapes and native bounds;
- contiguous offsets, exact lengths, plan hash, and every payload SHA-256;
- every payload word: exact `+0.0f` bits inside the confirmed interior and
  exact `0x7fc00000` bits everywhere outside it, even after valid rehashing;
- absence of padding or trailing bytes.

The structural container remains sentinel-only. It makes no scientific-
initialization, CUDA-residency, history, or restart claim.

```text
cargo run -- inspect INPUT.cm1szs
cargo run -- verify INPUT.cm1szs
cargo run -- roundtrip INPUT.cm1szs OUTPUT.cm1szs
```

`roundtrip` refuses an existing output path. Successful output is reparsed and
must retain the input artifact SHA-256 byte for byte.

The pinned Python-produced dry 16-cube compatibility fixture is 372,467 bytes,
contains 358,688 payload bytes, and has artifact SHA-256
`1482fada63702183932283b2f7bd3fbb85df3b9e5c6d7e85588fcf954bf6ede4`.
The Rust test suite builds those bytes through the Python producer, validates
them, reconstructs them, and requires byte identity.

## Scientific initializer

The scientific door evaluates CM1 r21.1 formulas in Rust binary32 for the
sealed `120x120x40` product profile (`dx=dy=1000 m`, `dz=500 m`). It does not
read or embed native field values. Its bounded implementation is:

- flat, centered, uniform `xh/xf/yh/yf/zh/zf` coordinates;
- `isnd=5` Weisman-Klemp base thermodynamics with the authoritative 20-pass
  hydrostatic/moist iteration;
- `iwnd=2` winds, `umove=12.5 m/s`, `vmove=3 m/s`, and the exact open-boundary,
  global-corner, vertical-extrapolation, and repeated GETSET fill cadence;
- a declared `init3d.F` initiation and `ibalance=0` zero pressure
  perturbation.  `iinit` 0, 1, 2, 3, 5, 11 and 13 are transcribed with the
  geometry each branch reads (`init3d.F:450-1416`), including
  `init3d.F:1422-1434`'s constant-relative-humidity moisture adjustment; the
  remaining branches are refused by value.  The sealed product profile above
  is the official `iinit=1` warm bubble, and the declared-geometry entry point
  reproduces it bit for bit;
- `initiation_admission.rs`, a `getcape.F:14-495` transcription that decides,
  from the declared sounding and the declared geometry alone, whether the
  perturbation can lift its own air to its level of free convection.  CM1 has
  no such gate: it integrates whatever `tha` `init3d.F` writes, so a
  perturbation that cannot convect costs a full run to discover;
- the nine captured base fields `pi0,prs0,rho0,thv0,th0,rth0,qv0,u0,v0` and
  ten core fields `ua,u3d,va,v3d,wa,w3d,ppi,pp3d,tha,th3d` through the first
  nonrestart GETSET call.

Every field file is complete raw little-endian `f32` in native Fortran storage
order (i fastest, then j, then k), with native allocation bounds. Horizontal
and vertical coordinate files are lossless 1-D reductions of this flat grid;
they are not a terrain-capable coordinate representation. `manifest.json` is
deterministic LF-only JSON and records bounds, shapes, byte counts, hashes,
source anchors, and the claim limit. Output directories and PNG paths must not
already exist.

```text
cargo run -- initialize-scientific NEW_OUTPUT_DIR
cargo run -- compare-scientific NATIVE_POST_GETSET_DIR NEW_OUTPUT_DIR
cargo run -- render-scientific-comparison NATIVE_POST_GETSET_DIR NEW_OUTPUT.png
```

`compare-scientific` validates the capture's `*post-getset-layout.tsv` (its
streams are `<prefix>post-getset-<field>.f32le` with the layout's own prefix), then
compares identical native indices without regridding. It reports field shape,
exact-bit count, maximum ULP and absolute difference for the whole allocation,
physical interior, and boundary separately. It also writes `comparison.json`
and a labeled, Rust-native `comparison.png`. On the pinned Windows O0 capture,
17 of 19 fields are full-allocation bit-exact; only `tha/th3d` differ, at 24 of
666,792 words each, with maximum 2 ULP / `5.960464477539063e-8 K`. This measured
libm result is reported, not tuned into the formulas or promoted to a portable
bit-identity guarantee.

Primary CM1 r21.1 source anchors are:

- `src/constants.F:111-158` (default thermodynamic constants);
- `src/param.F:6640-6695,7327-7408,7695-7745,7789-7812` (flat centered grid);
- `src/base.F:359-450,2041-2193,2317-2360,2660-2718,3204-3212` (WK base,
  wind, moving frame, fills, and reciprocals);
- `src/bc.F:528-934,1467-1711` and `src/comm.F:6511-6881` (staggered physical
  bands and MPI/global-corner semantics);
- `src/init3d.F:204-324,445-482,1733-1743,2140-2577` (initialization and
  first GETSET cadence);
- `src/cm1.F:2390-2403,2660-2743,2884-2907` (single calls, nonrestart guard,
  and post-GETSET observation point).

The claim stops before any timestep. It does not cover alternate namelists,
terrain or stretched grids, forecasts, restarts, CUDA residency, or GPU
execution. Moisture carrier families, TKE, and derived `rho/rr/rf/prs/ppx`
diagnostics remain explicitly deferred from this slice even though their
native captures can be compared in later source-mapped extensions.

## Deterministic checkpoint and history slice

The Rust path can seal the currently implemented scientific state in two
dependency-free, single-file envelopes:

```text
cargo run -- checkpoint-scientific NEW_OUTPUT.cm1ckpt
cargo run -- restore-checkpoint INPUT.cm1ckpt
cargo run -- history-scientific NEW_OUTPUT.cm1hist
cargo run -- verify-history INPUT.cm1hist
```

Each command emits compact, sorted-key canonical JSON containing the artifact,
header, and concatenated-payload SHA-256 identities. Writers use a create-new
same-directory temporary file, `sync_all`, create-new hard-link publication,
and removal of only their owned temporary name. Existing targets are never
overwritten, and readers preflight regular-file and size limits before reading.

`CM1CKP1` stores all six coordinate reductions and all 19 base/core arrays
currently owned by `ScientificState`, including full native allocations and
halos. Restore regenerates the sealed state independently and compares every
array, so changing a payload and updating its hashes still refuses. This is an
exact first-nonrestart-post-GETSET startup checkpoint only. It is not yet a
forecast restart: moisture carriers, TKE/SGS state, derived
`rho/rr/rf/prs/ppx`, held/stage tendencies, radiative-boundary history,
component persistence, scheduler/restart clocks, domain translation, and the
deterministic reduction identity are explicitly deferred.

`CM1HST1` stores physical-domain coordinates plus 13 observational base/current
fields at model time zero; halos, A-time-level duplicates, internal reciprocal
base state, and forecast-only families are omitted by schema. It is a verified
Rust history byte path, not a forecast trajectory, NetCDF-compatibility, or
weather-render claim.

## Generic forecast envelopes

The separate `forecast` Rust API extends the byte path to caller-supplied
complete-step forecast state without changing `CM1CKP1` or `CM1HST1`. A
`ForecastContract` names every required float32 array (shape, native bounds,
meaning, units, and staggering), typed scheduler scalar, and opaque component
blob schema/size ceiling. `ForecastFrame` supplies the step, model time, `dt`,
array words, scalar values, and component bytes.

The API sorts contract inventories by name and refuses missing, extra, or
duplicate arrays, scalars, and component blobs. It also refuses shape/type/blob
schema drift, non-finite clocks or f64 scalars, mismatched config/execution
identities, noncanonical headers, corrupt payloads, trailing bytes, symlinks,
oversized artifacts, and existing outputs. Float values are preserved by raw
bits (including signed zero and subnormals); scheduler f64 values and clocks
are represented by 16 lowercase hexadecimal bits.

`CM1FCP1` and `CM1FHS1` are separately versioned generic checkpoint/history
envelopes. They use the same atomic no-overwrite publication contract and emit
canonical content receipts. The executable example performs a real write/read
round trip of arrays, all four scalar types, and an opaque ArWen WSM6 blob:

```text
cargo run --release --example forecast_fixture -- NEW.cm1fcp NEW.cm1fhs
```

This is the deterministic host serialization/restore API, not live-model
wiring. Copying state from CUDA, restoring device allocations, proving that a
specific CM1 runtime contract is semantically complete, and uninterrupted
versus resumed trajectory identity remain deferred to the forecast driver.

## Real HRRR to CM1 initialization

The `hrrr` API accepts an exact Rust-decoded HRRR float32 window and refuses
any missing, extra, short, non-finite, or nonphysical carrier. Its fixed input
inventory includes all 50 hybrid levels of `PRES`, `HGT`, `TT`, `SPFH`,
grid-relative `U/V`, five hydrometeors, and `VVEL`, plus surface terrain and
direct `REFC`, `REFD-1000`, and `REFD-4000`. There is no analytic bubble,
random perturbation, horizontally uniform replacement, zero-w substitution,
or reflectivity-from-hydrometeors path.

Rust owns the spherical Lambert forward/inverse coordinates, local target
geodesics, grid-relative-to-earth wind rotation, bilinear interpolation,
linear-Z reflectivity interpolation, HGT-bracketed vertical interpolation,
log-pressure interpolation, omega-to-geometric-w conversion, thermodynamic
conversion, target terrain, CM1 x/y/z staggering, domain-mean base profiles,
and perturbation carriers. The f01 frame uses the same contract as f00 and is
marked boundary-ready at its exact forcing offset.

The optimized example defaults to the `300x300x75`, `100x100x200 m` profile;
the center latitude and longitude remain mandatory arguments:

```text
cargo run --release --example hrrr_cm1_init -- \
  DECODED_SERIES_ROOT NEW_OUTPUT_DIR CENTER_LAT CENTER_LON
```

`initial.cm1fcp`, `boundary-f01.cm1fhs`, and the final
`initialization-receipt.json` commit marker are all create-new publications.
The boundary frame can be rendered without an intermediate science/export
format by passing `boundary-f01.cm1fhs`, its independently sourced forecast
contract SHA-256, and a named array such as `refc_dbz` or `w3d` to
`cm1-render --format cm1fhs`.
Both large envelopes are read back through the generic forecast verifier
before the initialization receipt is published. GRIB2 entropy decoding is an
explicit upstream Rust boundary; this module accepts only its strict decoded
window artifact and performs no Python science.

This slice is not yet a runnable terrain forecast. Live CUDA allocation copy,
terrain-following metric/contravariant-velocity construction, native halo and
boundary-zone population, dynamical pressure rebalance, forcing-time
interpolation, and injection into the production forecast driver remain
deferred and are named in every initialization receipt.
