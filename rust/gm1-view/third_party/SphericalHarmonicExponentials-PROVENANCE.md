# Spherical Harmonic Exponentials provenance

GM1 View's Spherical Harmonic Exponentials (SHE) files port or adapt the
MIT-licensed `Avelina9X/SphericalHarmonicExponentials` implementation at exact
commit `3c14ddd5709705eebe00ec1305c891bc5f7d038b`. The licence is reproduced in
`SphericalHarmonicExponentials-LICENSE.txt` beside this file.

| File | What it takes from upstream |
|---|---|
| `src/she.rs` | FP32 source-order port of `CommonSH.hlsli` and the shading entry points |
| `src/she_fp32.wgsl` | WGSL port of `CommonSH.hlsli` |
| `src/she_glossy_upstream.wgsl` | WGSL port of the upstream-default FP32 glossy compute chain (prefilter, accumulator, sum, solver) |
| `src/she_glossy.rs` | the glossy fit's collector: GGX importance sampling and the radical-inverse sequence in upstream's source order |
| `src/she_material.wgsl` | upstream's FP32 basis and coefficient order, for the diffuse material |
| `src/she_glossy_material.wgsl` | upstream's glossy shading math with a 33-term fit |
| `src/she_admitted_terrain.wgsl` | upstream's glossy shading math for GM1 terrain |
| `src/she_material.rs` | projects the sky into upstream's L2 coefficient order by upstream's cubemap quadrature |
| `src/she_glossy_material.rs`, `src/she_glossy_upstream.rs`, `src/she_admitted_terrain.rs` | drive the shaders above and pin the upstream sources' SHA-256 |
| `src/bin/gm1_view_she_fp32_oracle.rs`, `src/bin/gm1_view_she_glossy_fit.rs`, `src/bin/gm1_view_she_glossy_upstream.rs` | run the ported shaders and the ported fit against upstream's math and record the upstream commit and licence with the results |
| `assets/t300-specular-she-provisional-fp32.f32le`, `assets/gm1-she-gm1-alpha32-fp32.f32le` | fit coefficients computed with the ported fit (data, no upstream code) |

Primary upstream authorities:

- `Shaders/CommonSH.hlsli`, SHA-256 `2ffeb4fcc001e57ba61b3f825177d70ed6283ebd5b710c39204acc16460aa56c`
- `Shaders/Graphics/SH_Shade32PS.hlsl`, SHA-256 `3a4cf0fce13db21fc2a53a4a8affc09d5f4a750a4e3b89afa89d502f500356d1`
- `Shaders/Compute/SH/SH_DiffusePrefilterCS.hlsl`, SHA-256 `55603e8112938dbb421a456d9355a970b230c4952ec94f307d608efdf75c34c4`
- `Shaders/Compute/SH/SH_SpecularPrefilterAccCS.hlsl`, SHA-256 `e2ccf1af80d9136260b6cdb547dec81ab9464ed905a54342cd155f53cbff3168`
- `Shaders/Compute/SH/SH_SpecularSolverCS.hlsl`, SHA-256 `2aa237dc5ebb77c60db8d8beeb2861c5dbb73cceb08dddec7186dc37b9916824`

The SHE material path is display-only. SimSat positive log-SH3 remains the sky
irradiance authority, and exact GM1 meteorological fields are not modified.
