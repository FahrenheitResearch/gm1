//! Real-material use of the pinned FP32 Spherical Harmonic Exponentials port.
//!
//! The source lighting is the independently sealed SimSat physical sky row.
//! Its diffuse irradiance is projected with the same cubemap quadrature used by
//! upstream `SH_DiffusePrefilterCS.hlsl`; no authored lighting coefficients are
//! introduced here.  Cloud-volume scattering remains a separate renderer.

use crate::{
    lighting::{PositiveSky, T300_SKY_ROW_SHA256, T300_SUN_ENU_F32_BITS},
    she::{SheDiffuseCoefficients, compute_sh_basis_l2, reconstruct_diffuse},
};
#[cfg(not(debug_assertions))]
use bevy::asset::{AssetPath, embedded_asset, embedded_path};
use bevy::{
    pbr::{Material, MaterialPlugin},
    prelude::*,
    reflect::TypePath,
    render::render_resource::{AsBindGroup, ShaderType},
    shader::ShaderRef,
};
use sha2::{Digest, Sha256};

pub const SHE_MATERIAL_SHADER: &str = "she_material.wgsl";
pub const SHE_DIFFUSE_FACE_RESOLUTION: u32 = 128;
pub const SHE_DIFFUSE_SOURCE: &str = "sealed SimSat T+300 positive log-SH3 irradiance projected by upstream cubemap solid-angle quadrature";

#[derive(Clone, Debug)]
pub struct T300DiffuseProjection {
    pub coefficients: SheDiffuseCoefficients,
    pub sha256: String,
    pub face_resolution: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct DiffuseProjectionError {
    pub compared_rgb_words: usize,
    pub max_absolute: f32,
    pub rms_absolute: f32,
    pub max_relative: f32,
    pub rms_relative: f32,
    pub max_luminance_absolute: f32,
    pub rms_luminance_absolute: f32,
    pub max_luminance_relative: f32,
    pub rms_luminance_relative: f32,
}

/// Project the sealed physical sky into upstream's FP32 L2 coefficient order.
pub fn project_t300_simsat_diffuse_l2(face_resolution: u32) -> T300DiffuseProjection {
    assert!(face_resolution > 0);
    let sky = PositiveSky::t300_physical();
    let mut coefficients = [Vec3::ZERO; 9];
    let inverse_texels = 1.0 / (face_resolution * face_resolution) as f32;
    for face in 0..6 {
        for y in 0..face_resolution {
            for x in 0..face_resolution {
                let uv = Vec2::new(
                    (x as f32 + 0.5) / face_resolution as f32 * 2.0 - 1.0,
                    -((y as f32 + 0.5) / face_resolution as f32 * 2.0 - 1.0),
                );
                let direction = cubemap_face_direction(uv, face).normalize();
                let irradiance = simsat_irradiance_for_she_direction(sky, direction);
                let solid_angle =
                    4.0 * inverse_texels / (1.0 + uv.x * uv.x + uv.y * uv.y).powf(1.5);
                for (coefficient, basis) in
                    coefficients.iter_mut().zip(compute_sh_basis_l2(direction))
                {
                    *coefficient += irradiance * (basis * solid_angle);
                }
            }
        }
    }
    let coefficients = SheDiffuseCoefficients(coefficients);
    T300DiffuseProjection {
        sha256: diffuse_coefficients_sha256(&coefficients),
        coefficients,
        face_resolution,
    }
}

pub fn t300_reference_irradiance(world_normal: Vec3) -> Vec3 {
    let sky = PositiveSky::t300_physical();
    // Upstream's SH convention is z-up; Bevy is y-up.
    let she_direction = Vec3::new(world_normal.x, world_normal.z, world_normal.y).normalize();
    simsat_irradiance_for_she_direction(sky, she_direction)
}

pub fn t300_projected_irradiance(world_normal: Vec3, projection: &T300DiffuseProjection) -> Vec3 {
    let she_direction = Vec3::new(world_normal.x, world_normal.z, world_normal.y).normalize();
    reconstruct_diffuse(she_direction, &projection.coefficients)
}

fn simsat_irradiance_for_she_direction(sky: PositiveSky, direction: Vec3) -> Vec3 {
    let world = Vec3::new(direction.x, direction.z, direction.y);
    let sun_enu = T300_SUN_ENU_F32_BITS.map(f32::from_bits);
    let sun_world = Vec3::new(sun_enu[0], sun_enu[2], sun_enu[1]).normalize();
    let up = Vec3::Y;
    let toward_sun = (sun_world - up * sun_world.dot(up)).normalize();
    let cross_sun = up.cross(toward_sun);
    let sun_frame = Vec3::new(world.dot(toward_sun), world.dot(cross_sun), world.dot(up));
    Vec3::from_array(
        sky.irradiance(sun_frame.to_array().map(f64::from))
            .map(|v| v as f32),
    )
}

fn cubemap_face_direction(uv: Vec2, face: u32) -> Vec3 {
    match face {
        0 => Vec3::new(1.0, uv.y, -uv.x),
        1 => Vec3::new(-1.0, uv.y, uv.x),
        2 => Vec3::new(uv.x, 1.0, -uv.y),
        3 => Vec3::new(uv.x, -1.0, uv.y),
        4 => Vec3::new(uv.x, uv.y, 1.0),
        5 => Vec3::new(-uv.x, uv.y, -1.0),
        _ => unreachable!("cubemap face is constrained to 0..6"),
    }
}

pub fn diffuse_coefficients_sha256(coefficients: &SheDiffuseCoefficients) -> String {
    let bytes = coefficients
        .0
        .iter()
        .flat_map(|coefficient| coefficient.to_array())
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    format!("{:x}", Sha256::digest(bytes))
}

#[derive(Clone, Copy, Debug, ShaderType)]
pub struct SheSkyMaterialUniform {
    /// x: 0=exact SimSat reference, 1=projected FP32 L2; y: exposure.
    pub control: Vec4,
    pub albedo: Vec4,
    pub sun_direction: Vec4,
    pub sky_amplitude: Vec4,
    pub sky_log_coef: [Vec4; 16],
    pub diffuse_coefficients: [Vec4; 9],
}

impl SheSkyMaterialUniform {
    fn t300(mode: f32, projection: &T300DiffuseProjection) -> Self {
        let sky = PositiveSky::t300_physical();
        let sun_enu = T300_SUN_ENU_F32_BITS.map(f32::from_bits);
        Self {
            control: Vec4::new(mode, 0.08, 0.0, 0.0),
            albedo: Vec4::new(0.76, 0.79, 0.82, 1.0),
            sun_direction: Vec3::new(sun_enu[0], sun_enu[2], sun_enu[1])
                .normalize()
                .extend(0.0),
            sky_amplitude: Vec4::new(
                sky.amplitude[0] as f32,
                sky.amplitude[1] as f32,
                sky.amplitude[2] as f32,
                0.0,
            ),
            sky_log_coef: std::array::from_fn(|index| {
                Vec4::new(
                    sky.log_coef[index][0] as f32,
                    sky.log_coef[index][1] as f32,
                    sky.log_coef[index][2] as f32,
                    0.0,
                )
            }),
            diffuse_coefficients: std::array::from_fn(|index| {
                projection.coefficients.0[index].extend(0.0)
            }),
        }
    }
}

#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
pub struct SheSkyMaterial {
    #[uniform(0)]
    pub uniform: SheSkyMaterialUniform,
}

impl SheSkyMaterial {
    pub fn t300_reference(projection: &T300DiffuseProjection) -> Self {
        Self {
            uniform: SheSkyMaterialUniform::t300(0.0, projection),
        }
    }

    pub fn t300_projected(projection: &T300DiffuseProjection) -> Self {
        Self {
            uniform: SheSkyMaterialUniform::t300(1.0, projection),
        }
    }
}

impl Material for SheSkyMaterial {
    fn fragment_shader() -> ShaderRef {
        #[cfg(debug_assertions)]
        {
            SHE_MATERIAL_SHADER.into()
        }
        #[cfg(not(debug_assertions))]
        {
            AssetPath::from_path_buf(embedded_path!("she_material.wgsl"))
                .with_source("embedded")
                .into()
        }
    }

    fn enable_shadows() -> bool {
        false
    }
}

pub struct SheSkyMaterialPlugin;

impl Plugin for SheSkyMaterialPlugin {
    fn build(&self, app: &mut App) {
        #[cfg(not(debug_assertions))]
        embedded_asset!(app, "she_material.wgsl");
        app.add_plugins(MaterialPlugin::<SheSkyMaterial>::default());
    }
}

pub fn t300_projection_error(
    projection: &T300DiffuseProjection,
    samples: u32,
) -> DiffuseProjectionError {
    const LUMA: Vec3 = Vec3::new(0.2126, 0.7152, 0.0722);
    let mut max_absolute = 0.0_f32;
    let mut squared_absolute = 0.0_f64;
    let mut max_relative = 0.0_f32;
    let mut squared_relative = 0.0_f64;
    let mut max_luminance_absolute = 0.0_f32;
    let mut squared_luminance_absolute = 0.0_f64;
    let mut max_luminance_relative = 0.0_f32;
    let mut squared_luminance_relative = 0.0_f64;
    let mut compared = 0usize;
    for index in 0..samples {
        let fraction = (index as f32 + 0.5) / samples as f32;
        let z = 1.0 - 2.0 * fraction;
        let radius = (1.0 - z * z).sqrt();
        let azimuth = index as f32 * 2.399_963_1;
        let she_direction = Vec3::new(radius * azimuth.cos(), radius * azimuth.sin(), z);
        let world_normal = Vec3::new(she_direction.x, she_direction.z, she_direction.y);
        let reference = t300_reference_irradiance(world_normal);
        let projected = t300_projected_irradiance(world_normal, projection);
        for channel in 0..3 {
            let absolute = (projected[channel] - reference[channel]).abs();
            let relative = absolute / reference[channel].abs().max(f32::MIN_POSITIVE);
            max_absolute = max_absolute.max(absolute);
            squared_absolute += f64::from(absolute * absolute);
            max_relative = max_relative.max(relative);
            squared_relative += f64::from(relative * relative);
            compared += 1;
        }
        let reference_luminance = reference.dot(LUMA).abs();
        let projected_luminance = projected.dot(LUMA);
        let luminance_absolute = (projected_luminance - reference_luminance).abs();
        let luminance_relative = luminance_absolute / reference_luminance.max(f32::MIN_POSITIVE);
        max_luminance_absolute = max_luminance_absolute.max(luminance_absolute);
        squared_luminance_absolute += f64::from(luminance_absolute * luminance_absolute);
        max_luminance_relative = max_luminance_relative.max(luminance_relative);
        squared_luminance_relative += f64::from(luminance_relative * luminance_relative);
    }
    DiffuseProjectionError {
        compared_rgb_words: compared,
        max_absolute,
        rms_absolute: (squared_absolute / compared as f64).sqrt() as f32,
        max_relative,
        rms_relative: (squared_relative / compared as f64).sqrt() as f32,
        max_luminance_absolute,
        rms_luminance_absolute: (squared_luminance_absolute / samples as f64).sqrt() as f32,
        max_luminance_relative,
        rms_luminance_relative: (squared_luminance_relative / samples as f64).sqrt() as f32,
    }
}

pub const SHE_SKY_SOURCE_SHA256: &str = T300_SKY_ROW_SHA256;

#[cfg(test)]
mod tests {
    use super::{
        SHE_DIFFUSE_FACE_RESOLUTION, SheSkyMaterialUniform, diffuse_coefficients_sha256,
        project_t300_simsat_diffuse_l2, t300_projection_error,
    };
    use bevy::render::render_resource::ShaderType;

    #[test]
    fn physical_simsat_projection_is_finite_and_stable() {
        let projection = project_t300_simsat_diffuse_l2(SHE_DIFFUSE_FACE_RESOLUTION);
        assert!(
            projection
                .coefficients
                .0
                .iter()
                .all(|coefficient| coefficient.is_finite())
        );
        assert_eq!(
            projection.sha256,
            diffuse_coefficients_sha256(&projection.coefficients)
        );
        let error = t300_projection_error(&projection, 4096);
        // L2 is intentionally a compact nine-coefficient display projection;
        // preserve and report its measured approximation error rather than
        // pretending it is the 16-term SimSat reference.
        assert!(
            error.max_relative < 0.12,
            "maximum/RMS relative error {}/{}",
            error.max_relative,
            error.rms_relative
        );
        assert!(
            error.rms_relative < 0.04,
            "maximum/RMS relative error {}/{}",
            error.max_relative,
            error.rms_relative
        );
        assert_eq!(error.compared_rgb_words, 4096 * 3);
        assert!(error.max_absolute.is_finite());
        assert!(error.rms_absolute.is_finite());
        assert!(error.max_luminance_absolute.is_finite());
        assert!(error.rms_luminance_absolute.is_finite());
        assert!(error.max_luminance_relative.is_finite());
        assert!(error.rms_luminance_relative.is_finite());
    }

    #[test]
    fn material_uniform_layout_matches_wgsl() {
        assert_eq!(std::mem::align_of::<SheSkyMaterialUniform>(), 16);
        assert_eq!(std::mem::size_of::<SheSkyMaterialUniform>(), 464);
        assert_eq!(SheSkyMaterialUniform::min_size().get(), 464);
    }
}
