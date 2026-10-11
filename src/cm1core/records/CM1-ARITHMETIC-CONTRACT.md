# CM1 arithmetic contract v1

This contract fixes what must be measured before a CUDA kernel can count as a
CM1 operator. It is intentionally stricter than “use FP32.”

## Storage, calculation and reductions

- Prognostic/tendency arrays follow the native default-real intent and are
  represented as IEEE-754 binary32 on device unless a source/oracle row says
  otherwise.
- Coordinates, base-state arrays and constants are not automatically folded
  to literals. Their storage and point-of-use rounding are captured by the
  operator fixture that first consumes them.
- Dry-mass per-level/global sums and any conservation diagnostics that decide
  admission use binary64 accumulation. The CUDA reduction tree is fixed by
  implementation version and recorded in every run receipt.
- Index, branch-selector, substep and clock arithmetic uses explicit integer
  widths. Overflow checks occur before GPU allocation.
- Fused multiply-add, reciprocal approximation, flush-to-zero, denormals-are-
  zero and compiler fast-math behavior are off by default. Binary32 subnormal
  inputs/outputs are preserved unless the measured native point of use proves
  otherwise. Enabling or changing one creates a new arithmetic profile and
  requires fresh helper plus trajectory gates.

## Compile route and binary32 sub-normals

- CuPy compiles `cupy.RawModule`, `cupy.RawKernel` and its own elementwise
  and reduction kernels with `-ftz=true` appended after the caller's options
  (CuPy 14.2 `cupy/cuda/compiler.py`, `_compile_with_cache_cuda`). NVRTC's
  user guide gives `--ftz` the default `false` but does not say which
  occurrence of a repeated option wins; measured, the last one does, so a
  `--ftz=false` handed to a `RawModule` has no effect. Measured on a device
  (`evidence/oracle/nvrtc-gradual-underflow`), such a kernel turns a
  sub-normal result into a zero of the same sign. It also reads a sub-normal
  operand as zero in products, `__f*_rn` intrinsics, `fminf`, comparisons and
  double-to-float conversions. NVRTC called directly with `--ftz=false`,
  with the image loaded through `cupy.cuda.function.Module`, keeps IEEE
  gradual underflow for all of these. The cause is the instruction modifier
  the compiler emits, not the device: on the same card the flushing routes'
  SASS carries `.FTZ` on its float instructions and the direct route's
  carries none, and a kernel whose body is inline PTX written without `.ftz`
  keeps the sub-normals even when CuPy compiled it with `-ftz=true`. So there
  is no run-time, driver or environment setting that restores them.
- Every GM1 kernel text is compiled the second way: by
  `cm1core.nvrtc_module.compile_nvrtc_module`, or by an image compile
  function that calls `_NVRTCProgram` with `--ftz=false` and loads the cubin
  it compiled. `tests/test_nvrtc_gradual_underflow.py` refuses any production
  `RawModule`, `RawKernel`, `ElementwiseKernel`, `ReductionKernel` or
  `fuse`. It also holds the host stand-ins to the device measurement: the
  kernel emulator is the direct route, and the same text under SSE FTZ|DAZ
  (`tools/oracle/host_flush_to_zero.py`) is CuPy's route.
- Still open (risk R-020): the CuPy array operations GM1's Python still
  performs (`cupy.add`, `a * b`, `cupy.where`, `cupy.maximum`, reductions),
  and every ArWen kernel, are compiled by CuPy and flush. Their device words
  equal the host stand-in's NumPy words only where no binary32 sub-normal
  arises. The flush is not limited to sub-normal words. A sub-normal
  multiplied by a factor above one is a normal word in CM1, and zero after
  the flush. In the pdef flux test cases, 4 qv and 5 qc face words below
  `|m|*2**-126` move this way.

## Ordering and branch law

- Fortran expression association, live-region update order and low-storage
  overwrite points are part of the oracle. Algebraically equivalent rewrites
  are not presumed trajectory-equivalent.
- WENO epsilon, weights, clamping, min/max and sign behavior are taken from
  the pinned helper at point of use. A carrier-field clamp elsewhere is not a
  substitute.
- Lookup tables retain their native index expression, bound policy, category
  order and payload hash. Interpolation, index clamping or constant folding is
  changed only under a table-edge oracle and a newly versioned profile.
- Boundary corners have a named update order. Parallel writes that race at a
  corner are forbidden even if typical inputs happen to agree.
- Split-explicit substep count and `dt` partition are computed once by the
  sealed scheduler and passed explicitly. Kernels do not derive an alternate
  count from rounded values.
- Physics outputs are tendencies over a declared interval, never guessed as
  replacement states. CM1 applies them at a source-mapped cadence.

## Comparison hierarchy

Each operator oracle records exact input/output bytes, dtype, shape, strides,
compiler/runtime identity and the source line range. Gates are evaluated in
this order:

1. shape, staggering, halo and finite-value identity;
2. branch/cadence identity with a negative control that must change output;
3. exact equality where native arithmetic permits it;
4. otherwise componentwise `max_abs`, `max_rel`, ULP distribution and named
   conservation residual with tolerances derived from measured native repeat
   spread, never selected after observing the port;
5. coupled short trajectory with per-term ablations;
6. restart-continuation identity.

No global RMSE can hide a wrong boundary, vertical level, species or stage.
Tolerances remain `null` in the oracle manifest until native repeat data
exists. A null tolerance is a failed/not-run gate, not an invitation to use a
default such as `1e-5`.

## Determinism receipt

Every numerical run must seal: package version; CM1 authority; ArWen version
and exact executed-source manifest; config hash; kernel source/PTX or cubin
hashes; Python/CuPy/NVRTC/CUDA/driver/GPU identities; compiler flags; launch
geometry; reduction profile; history/restart schemas; and output hashes.

Current profile status: declared, not validated. No production kernel has yet
passed this contract.
