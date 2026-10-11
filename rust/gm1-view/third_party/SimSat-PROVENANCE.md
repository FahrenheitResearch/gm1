# SimSat positive log-SH3 sky: provenance

GM1 View's sky irradiance is adapted from the SimSat project's positive
log-SH3 sky implementation. The basis and evaluation ordering in
`src/lighting.rs` (`SIMSAT_*`), the shaders that evaluate it
(`src/gm1_volume.wgsl`, and the `she_*` material paths that read the sky), the
SimSat top view, infrared and water-vapour layers in `src/science_layers.rs`,
and the data file `assets/t300-sky-radiance-log-sh3.f32le` derive from it.

SimSat is the work of the same project as GM1. The adapted code and data are
distributed here as part of GM1 View under GM1's licence, the Apache License,
Version 2.0 (the repository's `LICENSE`), and NOTICE credits the SimSat
project. The digests in `src/lighting.rs` and `src/science_layers.rs` pin the
SimSat sources and data rows these were adapted from.
