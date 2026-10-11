# ArWen WSM6 256-level capacity build

This profile lets GM1 execute the 400 by 400 by 237, 75 m configuration.
This is a CM1-owned dependency build profile, not a modification to the shared
ArWen checkout or the preserved audited overlay.

The profile is `gm1.arwen-wsm6-column-capacity-256/v1`. Its base is the byte-
audited ArWen estate named by `cm1core.engine_pin.GPUWM_AUDITED_SOURCE_IDENTITY`.
The existing default admission, 80-level ceiling, 17-file source pins and
canonical component metadata remain unchanged. Selecting the new profile
requires its exact modified source bytes; a base tree cannot satisfy the
extended profile, and an extended tree cannot satisfy default admission.

## Rebased on ArWen 2.8.0

**The profile is not retired, because 2.8.0 does not fix the defect it
exists for either.** `gpuwm/core/wsm6_constants.py` and
`gpuwm/physics_vertical_contract.py` are byte-identical between 2.7.4 and
2.8.0, so stock 2.8.0 still compiles `WSM6_DEEP_KMAX = 80` and still declares
`WSM6_VERTICAL_LEVEL_BOUNDS = (2, 80)`. Both baseline digests and both
capacity-build digests in the table below are therefore unchanged by the
move; the patch applied to the 2.8.0 base reproduces them byte for byte. The
assembled WSM6 NVRTC source is identical between 2.7.4 and 2.8.0 at every
tier, so the resource measurements below still describe what 2.8.0 compiles.

What the move does change is the base the profile names: its
`base_source_identity` and `base_version` now name the 2.8.0 wheel, and the
unchanged part of the surface gains `gpuwm/kernel_compile_notice.py` and the
2.8.0 digests of the six files that moved outside WSM6's executed path
(`CM1-ARWEN-SEAM.md`). The canonical profile document, and so its SHA-256,
moves with them.

## Rebased on ArWen 2.7.4 (2026-09-13)

**The profile is not retired, because 2.7.4 does not fix the defect it
exists for.** `gpuwm/core/wsm6_constants.py` is byte-identical between 2.5.8
and 2.7.4, so stock 2.7.4 still compiles `WSM6_DEEP_KMAX = 80` and the tier
sequence `(64, 80)`; `gpuwm/physics_vertical_contract.py` still declares
`WSM6_VERTICAL_LEVEL_BOUNDS = (2, 80)`. A 237-level column is still refused by
a stock estate, so "fixed means default" does not apply yet. If a future
ArWen admits deeper WSM6 columns natively, this profile is retired that day.

Both patch targets keep exactly the shape the delta assumes: all four anchor
strings occur exactly once in the 2.7.4 tree, and `patch_capacity_source`
re-verifies that inventory on every build. The consequences of the rebase are
asymmetric and both are recorded in the table below:

- `gpuwm/core/wsm6_constants.py` is byte-identical across the two versions, so
  its baseline digest AND its capacity-build digest are both unchanged. The
  extended WSM6 tier compiles the same source it compiled under 2.5.8.
- `gpuwm/physics_vertical_contract.py` gained rows for Milbrandt-Yau, WDM6, P3
  and New Tiedtke and made the legacy shortwave layer cap runtime-sized, so
  both its baseline and its capacity-build digest moved. None of those rows is
  WSM6's.

The assembled WSM6 NVRTC source is unchanged at every tier this profile
compiles, which is measured rather than assumed:
`WSM6_KMAX=128` is
`3b839812f609c963dd76f13c703d6f3533b5fa0d5c4f946a87335187059e64ca` and
`WSM6_KMAX=256` is
`fc8d6e792573506692ebe8dc1a59761eb81f658341d087fa1df84df9acbba450`
under both 2.5.8 and 2.7.4. The offline `sm_120` frame and register
measurements below were taken against those exact source strings and are
therefore still the measurements of what 2.7.4 compiles.

## Exact permitted delta

`gpuwm/core/wsm6_constants.py` changes its maximum column capacity to 256 and
its compile-time tier sequence from `(64, 80)` to `(64, 80, 128, 256)`. Its
comment describing the maximum number of cached modules changes accordingly.
All coefficient functions, constants used by the equations and the tier-
selection function body remain byte/AST unchanged.

`gpuwm/physics_vertical_contract.py` changes only
`WSM6_VERTICAL_LEVEL_BOUNDS = (2, 80)` to `(2, 256)`. This makes preparation-time
admission agree with the kernel launcher. The WSM6 registry entry contains no
independent vertical ceiling and remains unchanged.

| File | Baseline SHA-256 (2.7.4 and 2.8.0) | Capacity-build SHA-256 | Moved from 2.5.8 |
|---|---|---|---|
| `gpuwm/core/wsm6_constants.py` | `4d110ed7f5a792f48d925fd0ff8e974305d870a11ebb31437ff4e6fce6d3c62d` | `a1bf764f2a7553bdb148524463ada69e26c855482c7db76cbc2f6374ac08aa2a` | no, both identical |
| `gpuwm/physics_vertical_contract.py` | `e18e7756b77be3170de1f34f49dbf7e489afb78995be072d251d1b36b5aeb1d4` | `3fcf990ed68c09a3b594e2ee35393f2992a2b73ec97c0f6fafbdfa9f7aa72a8c` | yes, both |

The superseded 2.5.8 pair was baseline
`866c9a70e886e72b4a8d88435a9c62279de25a64eb25e66f4fccd361a8f84fb4` and
capacity-build `602e28a4264a7acdcec58d4486932775af9babfb34c031bdc303f47284ad8038`
for `gpuwm/physics_vertical_contract.py`. Nothing admits against those any more.

The other 15 source pins remain exact, including the WSM6 CUDA source,
thermodynamic constants, public microphysics dispatch, WSM6 launcher,
reflectivity source, state and registry. No vertical partitioning, extra
sedimentation boundary, numerical approximation, coefficient substitution or
runtime monkeypatch is introduced.

The existing public `gpuwm.core.microphysics.apply` path already asks
`wsm6_level_tier(nz)` for the shallowest fitting tier and compiles its generic
CUDA source through the existing `WSM6_KMAX` integer define. A 237-level column
therefore compiles with capacity 256 and still executes all 237 levels within
one thread. Grids with at most 64 or 80 levels select their original tiers.

## Reproducible build and source admission

`cm1core.engine_capacity.build_wsm6_capacity_overlay(source_root, destination)`
verifies all 17 baseline source pins, copies the package and its pinned
version metadata into a fresh directory, applies the closed two-file delta, verifies
the result, and atomically publishes the directory. It excludes stale Python
bytecode, refuses existing destinations, and records both complete file-
manifest hashes. It never writes the source tree.

The CPU-only command is:

```text
python3 tools/build_arwen_wsm6_capacity_overlay.py --source-root tmp/arwen-estate-gpuwm-2.8.0 --destination work/arwen-wsm6-capacity-256-v3/overlay
```

`work/arwen-wsm6-capacity-256-v3/overlay` is the tree built on the 2.8.0
base, and its `gm1-wsm6-capacity-overlay.json` receipt records `base_version`
2.8.0, `gpu_used: false`, `runtime_monkeypatch: false`,
`source_tree_modified: false` and exactly the two changed files. The
superseded `...-v2` tree (2.7.4 base) and `...-v1` tree (2.5.8 base) admit
nothing now.

Start a fresh process with that overlay on `PYTHONPATH`, ahead of any other
ArWen installation. Pass `capacity_profile=WSM6_256_CAPACITY_PROFILE` to the
component factory or tiled WSM6 launcher, including for shallow-grid
regression runs within that process. Engine inspection, component metadata
and execution receipts bind the selected surface and capacity profile.

## Resource measurements and claim boundary

Offline NVRTC 13.0 plus ptxas compilation for `sm_120`, using the existing
`--std=c++17` source assembly, measured:

| Compile-time tier | Stack frame per thread | Registers | Spill stores/loads |
|---|---:|---:|---:|
| 64 | 7,216 bytes | 102 | 0 / 0 |
| 80 | 9,008 bytes | 102 | 0 / 0 |
| 128 | 14,384 bytes | 102 | 0 / 0 |
| 256 | 28,720 bytes | 102 | 0 / 0 |

The existing ArWen preflight model of 112 additional bytes per level predicts
all four measured tiers exactly; its code is unchanged. The 256 tier adds
19,712 bytes per thread over the 80 tier. This can create substantial driver-
owned local-memory backing beyond the CuPy allocation pool. Actual backing
allocation and available VRAM still require measurement on the target GPU.

No CUDA context or GPU launch was used for these compiler measurements.
The run-local sources, PTX, cubins and compiler logs were retained beside the
superseded `...-v1` build, with `offline-resource-receipt.json` recording their
hashes; that tree is not distributed. The
measurements themselves still describe 2.7.4, because the assembled source
strings they were taken from are the byte-identical ones in the table above.

ArWen 2.7.4 corroborates the 64-level row independently: its own
`gpuwm/core/kernel_frame_recordings.py` reads `wsm6` at **7,216 bytes** on all
three recorded compile platforms (sm_120/NVRTC 13.0.48, sm_120/NVRTC 13.3.33
and sm_86/NVRTC 13.0.48), which is the same number the offline compilation
measured. That row is reproduced in `ARWEN-CAPABILITY-SURVEY.json` under
`kernels.wsm6.local_frame_bytes` and asserted in
`tests/test_arwen_capability_survey.py`. ArWen publishes no row for the 128 or
256 tiers, because those tiers exist only in this profile.

CPU tests prove deterministic patching, unchanged coefficient function ASTs,
exact source admission, crossed-profile refusal, the public vertical contract,
tier boundaries, unchanged default metadata, and profile propagation through
the public component and tiled launcher. They also follow the selection the
whole way: `tests/test_engine_capacity.py` parses the TOML text, reads
`runtime.capacity_profile` through the cartesian door, and checks that the
default text admits the stock surface at `(2, 80)` while the profile text
admits the patched surface at `(2, 256)` -- then reassembles the WSM6 NVRTC
source from the estate's own bytes and confirms the default lane still hashes
to `1a6d20da...`, the string the WRF v4.6.1 adapter oracle graded. Extended-column GPU numerical
oracles, measured driver backing, complete steps and sustained performance are
separate gates; this decision does not claim that they have passed.
