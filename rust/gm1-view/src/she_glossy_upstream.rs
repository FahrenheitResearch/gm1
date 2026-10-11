//! Admission constants and held-direction gate for the pinned upstream-default
//! FP32 glossy SHE compute chain.

use crate::{
    she::SheSpecularCoefficients,
    she_glossy::{
        GlossyErrorMetrics, SimSatSkyRadiance, glossy_error_for_coefficients,
        glossy_error_for_radiance,
    },
};
use bevy::math::Vec3;

pub const WGSL_SOURCE: &str = include_str!("she_glossy_upstream.wgsl");
pub const COLLECTOR_RESOLUTION: u32 = 256;
pub const GGX_SAMPLES: u32 = 4096;
pub const ALPHA_LEVELS: [f32; 4] = [0.2, 0.466_666_67, 0.733_333_35, 1.0];
pub const HELD_RESOLUTION: u32 = 17;
pub const HELD_GGX_SAMPLES: u32 = 4096;
pub const HELD_ALPHA_LEVELS: [f32; 4] = [0.22, 0.41, 0.68, 0.95];

pub const UPSTREAM_PREFILTER_SHA256: &str =
    "cbdde396d7246b88f89d3cc1860c05ecf3830e0fe0f60301b90215111f8454cb";
pub const UPSTREAM_ACCUMULATOR_SHA256: &str =
    "e2ccf1af80d9136260b6cdb547dec81ab9464ed905a54342cd155f53cbff3168";
pub const UPSTREAM_SUM_SHA256: &str =
    "3e53d3fdb126a009aac09a4ce1b3f1e7c78ee497abe48ddd0ed4eeebe1a0dc3b";
pub const UPSTREAM_SOLVER_SHA256: &str =
    "2aa237dc5ebb77c60db8d8beeb2861c5dbb73cceb08dddec7186dc37b9916824";

pub fn held_error(coefficients: &SheSpecularCoefficients) -> Result<GlossyErrorMetrics, String> {
    glossy_error_for_coefficients(
        SimSatSkyRadiance::sealed_t300()?,
        coefficients,
        HELD_RESOLUTION,
        HELD_GGX_SAMPLES,
        HELD_ALPHA_LEVELS,
    )
}

pub fn held_error_for_radiance<R>(
    radiance: &R,
    coefficients: &SheSpecularCoefficients,
) -> Result<GlossyErrorMetrics, String>
where
    R: Fn(Vec3) -> Vec3,
{
    glossy_error_for_radiance(
        radiance,
        coefficients,
        HELD_RESOLUTION,
        HELD_GGX_SAMPLES,
        HELD_ALPHA_LEVELS,
    )
}

#[cfg(test)]
mod tests {
    use super::{ALPHA_LEVELS, COLLECTOR_RESOLUTION, GGX_SAMPLES, WGSL_SOURCE};
    use naga::valid::{Capabilities, ValidationFlags, Validator};

    #[test]
    fn upstream_default_wgsl_chain_is_well_formed_and_not_reduced() {
        assert_eq!(COLLECTOR_RESOLUTION, 256);
        assert_eq!(GGX_SAMPLES, 4096);
        assert_eq!(ALPHA_LEVELS.len(), 4);
        assert!(WGSL_SOURCE.contains("@workgroup_size(1024, 1, 1)"));
        assert!(WGSL_SOURCE.contains("fn solve_unregularized"));
        let module =
            naga::front::wgsl::parse_str(WGSL_SOURCE).expect("parse upstream-default WGSL");
        Validator::new(ValidationFlags::all(), Capabilities::all())
            .validate(&module)
            .expect("validate upstream-default WGSL");
    }
}
